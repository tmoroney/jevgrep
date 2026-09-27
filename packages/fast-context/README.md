# fast-context

Answers a natural-language question about a repository with exact file line
ranges, so a slower, pricier coding model can skip the search. Everything lives
in [`src/fast-context.ts`](src/fast-context.ts); [`src/cli.ts`](src/cli.ts) is a
thin command-line wrapper.

```sh
export CEREBRAS_API_KEY=...
node --experimental-strip-types packages/fast-context/src/cli.ts \
  "Where does requests add the Content-Length header when preparing a request body?" ./requests
```

```text
src/requests/models.py:654-668
654:     def prepare_content_length(self, body: _t.BodyType) -> None:
655:         """Prepare Content-Length header based on request method and body"""
...
[3 turns, 3.1s, 4057 in / 330 out, ~$0.0045]
```

From code:

```ts
import { findContext } from "@repo/fast-context";

const result = await findContext({ root: ".", question: "Where is the retry policy applied?" });
result.snippets; // [{ path, startLine, endLine, code }], code with numbered lines
result.text; // the snippets rendered for the calling model
```

## How it works

1. The model sees the root listing and the question (plus optional `hints`,
   such as leads from a local keyword search).
2. Each turn it replies with one JSON object: a batch of up to 8 `grep`,
   `read` and `list` actions, or an answer. The harness runs the batch in
   parallel. Models without SWE-grep's RL training make one native tool call
   per turn, so the harness enforces the fan-out instead of the model.
3. After at most 4 turns it must answer with up to 6 ranges of at most 80
   lines each.
4. Code checks the answer: each path must exist inside the root, must have
   appeared in a tool result, and is clamped to the file and the size limit.

Every line the model reads and every line returned carries its line number,
so ranges are copied rather than counted. Paths outside the root, including
through symlinks, are refused. `grep` uses ripgrep when installed and falls back
to `grep -rE`.

## Defaults

| Setting                  | Default                                                                  |
| ------------------------ | ------------------------------------------------------------------------ |
| Endpoint                 | `https://api.cerebras.ai/v1`, any OpenAI-compatible chat completions API |
| Model                    | `qwen-3.8-27b`, reasoning effort `low`                                   |
| Price for cost estimates | $0.99 / $1.49 per million input / output tokens                          |
| Turns, actions per turn  | 4, 8                                                                     |
| Answer                   | at most 6 ranges of at most 80 lines                                     |

Override through `endpoint` and `limits`, or `--model`, `--effort` and
`--base-url` on the CLI. `complete` replaces the HTTP client (the tests use it).

## Measured

On the [search benchmark](../../evals/implementation/search-bench/README.md)
(34 SWE-bench tasks, Python only) this agent with Qwen at `low` found the edited
code in 24 of 34 tasks with 57% snippet precision, in a median of 1.8 seconds,
for about $0.009 per search, returning about 500 tokens.

On Node 22 behind an HTTPS proxy, set `NODE_USE_ENV_PROXY=1` so `fetch` uses it.
