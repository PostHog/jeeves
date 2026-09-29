# jeeves-sdk

A drop-in replacement for Jev's Python SDK (`typesafe-sdk`) that talks to a Jeeves server. Every class, method and error is the official SDK's; switching is a one-line import change:

```python
from jeeves_sdk import Choice, Noul, Score, TypeSafeClient

with TypeSafeClient() as client:
    result = client.system_one(
        state="I was charged twice. Please help.",
        questions={
            "billing": Noul(instructions="Is this about billing?"),
            "tone": Choice(instructions="What is the tone?", criteria={"calm": None, "angry": None}),
            "urgency": Score(instructions="How urgent is this?", criteria=["can wait", "this week", "today"]),
        },
    )
    print(result.nouls["billing"].noul, result.choices["tone"].choice, result.scores["urgency"].score)
```

## Install

```bash
pip install ./sdk
```

## What differs from `typesafe-sdk`

- **Server.** Requests go to `http://127.0.0.1:8009` by default; set `base_url=` or `JEEVES_BASE_URL` to point elsewhere. No API key is needed; one is accepted and ignored.
- **Timeout.** 120 s by default instead of 10 s, since thinking can take several seconds per request.
- **Reasoning options.** `system_one` accepts four optional keyword arguments, sent in the request's `options` object. Leave them out and the request is exactly what `typesafe-sdk` would send.

| argument | effect |
|---|---|
| `think=False` | answer from the prompt alone, without reasoning |
| `max_think=768` | truncate each reasoning chain at this many tokens |
| `nothink_threshold=0.9` | skip reasoning for questions whose no-think answer is already this confident |
| `return_reasoning=True` | include each question's reasoning in `result.reasoning` |

- **Response extras.** `result.usage.reasoning_tokens` counts generated reasoning tokens, `result.latency_ms` is the server time, and `result.reasoning` maps question names to `Reasoning(thought, closed, tokens, text)` when requested.

```python
result = client.system_one(state=..., questions=..., max_think=768, nothink_threshold=0.9, return_reasoning=True)
print(result.reasoning["billing"].text)
```

`JeevesClient` and `AsyncJeevesClient` are aliases of `TypeSafeClient` and `AsyncTypeSafeClient`.
