# Finding 01 — Infinite Retry Loop In Failover

## Summary

`BaseRequestHandler._execute_with_failover` never stops retrying when the load balancer repeatedly returns an endpoint that is already in the `attempted_endpoints` set. Under high load this degenerates into an infinite loop that ultimately returns `None`, yielding a 500 to the caller and occupying workers indefinitely.

## Impact

- Proxy worker hangs on the request while holding client connection.
- Downstream logging reports `TypeError` from `_finalize_response` once the loop exits without a valid response.
- High load amplifies the issue because endpoint availability snapshots may be stale while Redis cool-down logic keeps returning the same endpoint ID.

## Reproduction Steps

1. Configure Redis so every endpoint is marked cooling down.
2. Send a chat completion request via `/chat/completions`.
3. Observe `_execute_with_failover` repeatedly calling `load_balancer.select_endpoint` without progress, effectively hanging.

## Evidence

```191:229:llmproxy/api/base_handler.py
for attempt in range(self.config.general_settings.num_retries):
    endpoint = None
    for _ in range(max_selection_attempts):
        candidate = await self.load_balancer.select_endpoint(
            model_group, exclude_ids=attempted_endpoints
        )
        if Candidate is None: return 503
        if candidate.id in attempted_endpoints:
            continue  # loop continues forever if the LB keeps returning same id
```

## Recommendation

- Track the count of unique endpoints attempted and break once all are exhausted.
- Consider adding a guard that caps the inner loop and raises a deterministic 503 rather than looping indefinitely.

## Fix Status

- `BaseRequestHandler._execute_with_failover` now stops before requesting another endpoint once every configured endpoint has been attempted and returns a deterministic 503 response when selection is exhausted without progress.
