# Usage telemetry

LLMProxy can append one record per request to a Redis Stream. Because the proxy
sits after load balancing, failover and caching, it is the only place that knows
which endpoint actually served a request, how many attempts it took, and whether
the proxy cache answered it without calling upstream at all.

Requests the proxy refuses are recorded too, down to one that names no model.
Only a body that is not JSON, or is larger than `max_request_body_bytes`, has no
record: it is turned away before it is read.

Telemetry is off unless `general_settings.usage_stream` is present.

## Configuration

```yaml
general_settings:
  usage_stream:
    enabled: True
    stream_key: "llmproxy-telemetry:usage"
    max_len: 100000
    caller_headers:
      - x-coral-agent
      - x-coral-run-id
```

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Set to `false` to keep the block but stop emitting. |
| `stream_key` | `llmproxy-telemetry:usage` | Redis Stream key to append to. It must stay outside the cache namespace (`cache_params.namespace`, default `llmproxy`): `DELETE /cache` removes every key under it. |
| `max_len` | `100000` | Approximate cap (`XADD MAXLEN ~`). See [sizing](#sizing). |
| `caller_headers` | none | Inbound request headers copied onto each record. Matched case-insensitively; values are trimmed to 256 characters. A name that suggests a credential (`authorization`, `cookie`, an API key, a token, a secret or a password) is rejected at load time, because the stream keeps the value as long as the record. |

Writes are best effort. If Redis is unavailable the failure is logged as
`usage_record_write_failed` and the proxied request is unaffected.

### Sizing

Reading a stream does not remove its entries, and the proxy sets no expiry on
it, so it grows to about `max_len` entries and is trimmed there. A record takes
roughly 0.75 KB, so the default holds about 75 MB. The cap is also how far a
consumer can fall behind: entries trimmed before it reads them are lost. Size
it to cover the longest consumer outage you want to survive, within what the
Redis instance can spare.

`max_len` is a trimming limit, not a guarantee of how long records last. Redis
can still drop the whole stream, unread records included:

- under an `allkeys-*` eviction policy (`maxmemory-policy`), when memory runs
  short, even while the stream is below `max_len`. The stream shares Redis with
  the response cache, so cache growth can evict it.
- on a restart, if Redis persists nothing to disk.

Where the records must survive either, run Redis with a `noeviction` or
`volatile-*` policy and with persistence enabled.

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

With Pydantic AI, set `extra_headers` in `ModelSettings`, or add a request event
hook to the provider's HTTP client that injects them per run. Keep that client to
one event loop: its pooled connections belong to the loop that opened them.

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

- `request_id` is unique per record, so a consumer can write records
  idempotently.
- `model_group` is what the caller asked for, whether or not the proxy serves
  it: a name it does not serve is kept as asked, cut to 128 characters, and a
  request that named none records `null`. `endpoint_model` and
  `endpoint_id` are the configured deployment the request was sent to. They
  differ whenever a model group fans out across several deployments. Failures
  name the endpoint too: when every endpoint failed, or none was left to try
  after a failure, the record names the last one tried. They are `null` only
  when no upstream call was made.
- `attempts` counts the upstream requests made. Above 1 means the call was
  retried: on another endpoint after a failure, or, for a Responses follow-up,
  on the endpoint holding the state it continues, which is the only one that
  can serve it. `0` means no upstream call was made: the cache answered, the
  model was missing or not served, no endpoint was available, or a follow-up
  could not be routed.
- `status_code` is the status the caller received. Streams, which have already
  answered 200 when they end, record one of three instead when they do not
  finish cleanly: `499` when the client went away, `500` when the stream
  raised, and `502` when the upstream reported a failure inside the stream.
- `cache_hit` marks a replayed response. The `usage` on those rows is what the
  cached body reported, not tokens billed again — **price only rows where
  `cache_hit` is false**, and use the cache-hit rows to quantify what the cache
  saved.
- `usage` is `null` when the upstream never reported it: most failed requests,
  and chat streams from clients that did not set `stream_options.include_usage`.
- `reasoning_effort` is read from `reasoning.effort` (Responses) or
  `reasoning_effort` (Chat Completions). It is `null` when unset, and when the
  value is not a short lowercase word.
- `error` is present only on failures, as a short code rather than text: the
  provider's error code or type when it sends one (`content_filter`),
  `http_<status>` when it does not, the proxy's own, or an exception class
  name. The proxy's own are `model_required` and `model_not_configured` (the
  request named no model, or one the proxy does not serve),
  `no_available_endpoints` and `all_endpoints_failed`, and, for a Responses
  follow-up it cannot route,
  `affinity_expired` (the mapping to its endpoint has lapsed),
  `affinity_endpoint_unavailable` (that endpoint is no longer configured) or
  `affinity_conflict` (its items belong to different endpoints).

Prompt and response content is never recorded — only counts and labels. That
is why `error` is a code: upstream error bodies can quote the request back.

## Consuming the stream

Read the stream with `XRANGE` from a position you store alongside the records
you write, and advance it in the same database transaction:

```python
import json
import redis

client = redis.Redis(host="localhost", port=6379, decode_responses=True)


def next_id(entry_id: str) -> str:
    # XRANGE's exclusive "(" prefix needs Redis 6.2; this works on 5.0 and up.
    milliseconds, sequence = entry_id.split("-")
    return f"{milliseconds}-{int(sequence) + 1}"


with transaction():  # your database's
    position = load_position("llmproxy-telemetry:usage")  # "0-0" at first
    entries = client.xrange(
        "llmproxy-telemetry:usage", min=next_id(position), max="+", count=500
    )
    for _entry_id, fields in entries:
        store(json.loads(fields["payload"]))  # idempotent on request_id
    if entries:
        save_position("llmproxy-telemetry:usage", entries[-1][0])
```

Keeping the position with the data makes the consumer restartable without
losing or double-counting records, and it rewinds with the data: a database
reset or a restore from backup simply re-reads whatever the stream still holds.
Reading does not modify the stream, so several independent consumers can read
it side by side.

A consumer group also works, but its position lives in Redis rather than with
the data, so a database rewound past it cannot re-read what the group already
delivered. Acknowledging an entry does not delete it, and recovering entries a
crashed consumer left pending needs `XAUTOCLAIM` (Redis 6.2+) or a re-read of
the consumer's own pending list.
