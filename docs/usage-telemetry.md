# Usage telemetry

LLMProxy can append one record per request to a Redis Stream. Because the proxy
sits after load balancing, failover and caching, it is the only place that knows
which endpoint actually served a request, how many attempts it took, and whether
the proxy cache answered it without calling upstream at all.

Telemetry is off unless `general_settings.usage_stream` is present.

## Configuration

```yaml
general_settings:
  usage_stream:
    enabled: True
    stream_key: "llmproxy:usage"
    max_len: 1000000
    caller_headers:
      - x-coral-agent
      - x-coral-run-id
      - x-coral-feature
```

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Set to `false` to keep the block but stop emitting. |
| `stream_key` | `llmproxy:usage` | Redis Stream key to append to. |
| `max_len` | `1000000` | Approximate cap (`XADD MAXLEN ~`). Redis trims to roughly this many entries, so a stalled consumer cannot grow the stream without bound. |
| `caller_headers` | the three above | Inbound request headers copied onto each record. Matched case-insensitively; values are trimmed to 256 characters. |

Writes are best effort. If Redis is unavailable the failure is logged as
`usage_record_write_failed` and the proxied request is unaffected.

## Caller attribution

The proxy does not know which agent or workflow issued a request, so callers
label their own traffic with headers. Any listed header that is present is
copied verbatim into the record's `caller` object; requests without them are
still recorded, just unattributed.

With the OpenAI Python SDK:

```python
client.chat.completions.create(
    model="gpt-5",
    messages=[...],
    extra_headers={"X-Coral-Agent": "posting_classifier", "X-Coral-Run-Id": run_id},
)
```

With Pydantic AI, set `extra_headers` in `ModelSettings`, or attach a shared
`httpx.AsyncClient` whose event hook injects them per run.

## Record shape

Each stream entry has a single field, `payload`, holding a JSON object:

```json
{
  "request_id": "6f1c...",
  "recorded_at": 1790063501.482,
  "api_surface": "responses",
  "model_group": "gpt-5",
  "endpoint_model": "gpt-5",
  "endpoint_id": "a1b2c3d4e5f60718",
  "endpoint_base_url": "https://example.openai.azure.com",
  "streaming": true,
  "reasoning_effort": "high",
  "status_code": 200,
  "cache_hit": false,
  "attempts": 1,
  "latency_ms": 2841,
  "caller": {"x-coral-agent": "posting_classifier", "x-coral-run-id": "run-1"},
  "usage": {
    "input_tokens": 1204,
    "output_tokens": 316,
    "total_tokens": 1520,
    "cached_tokens": 960,
    "cache_write_tokens": 0,
    "reasoning_tokens": 192
  }
}
```

Notes on individual fields:

- `model_group` is what the caller asked for; `endpoint_model` and
  `endpoint_id` are what actually served it. They differ whenever a model group
  fans out across several deployments.
- `attempts` counts endpoints tried for this request. A value above 1 means
  failover occurred; `0` means the cache answered without any upstream call.
- `cache_hit` marks a replayed response. The `usage` on those rows is what the
  cached body reported, not tokens billed again — **price only rows where
  `cache_hit` is false**, and use the cache-hit rows to quantify what the cache
  saved.
- `usage` is `null` when the upstream never reported it: failed requests, and
  chat streams from clients that did not set `stream_options.include_usage`.
- `reasoning_effort` is read from `reasoning.effort` (Responses) or
  `reasoning_effort` (Chat Completions), and is `null` when unset.
- `error` is present only on failures, truncated to 500 characters.

Prompt and response content is never recorded — only counts and labels.

## Consuming the stream

Use a consumer group so records survive a consumer restart:

```python
import json
import redis

client = redis.Redis(host="localhost", port=6379, decode_responses=True)

try:
    client.xgroup_create("llmproxy:usage", "ingest", id="0", mkstream=True)
except redis.ResponseError:
    pass  # group already exists

entries = client.xreadgroup("ingest", "worker-1", {"llmproxy:usage": ">"}, count=500)
for _stream, messages in entries:
    for entry_id, fields in messages:
        record = json.loads(fields["payload"])
        ...  # persist it
        client.xack("llmproxy:usage", "ingest", entry_id)
```

Because `max_len` trims the stream, a consumer that stays down long enough will
lose the oldest records. Size `max_len` to comfortably cover your worst expected
consumer outage.
