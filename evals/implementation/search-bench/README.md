# Code-search benchmark

Does a search tool hand a coding agent the code it needs, with little else, for
less than the agent would spend finding it itself? Every tool gets the same
repository checkout and the same one-sentence question, returns snippets, and is
scored on what it returned. This is a diagnostic for search design, not product
acceptance evidence; the [cost and quality policy](../../cost-quality-policy.md)
still owns acceptance.

## Tasks

`build_tasks.py build` samples 44 SWE-bench Verified tasks across 12 Python
projects (moatless-tools packaging, pinned commit). A task is a repository at a
fixed commit plus an answer key: the declarations a real change edited
(`edit`, 55 in total) and the existing tests next to them (`test`, 52). Those are
code an agent working on that change had to find.

`make_queries.ts` writes the question an agent would ask a search tool, e.g.
"Where does requests add the 'Content-Length' header when preparing a GET
request?". GPT-OSS-120B writes it from the issue text alone and never sees the
answer key.

## Tools

| Tool | What it does |
| --- | --- |
| `bm25` | Local keyword ranking over every Python declaration; top 8. Free. |
| `rerank` | BM25 top 30, then one Jev Noul per (query, candidate) pair as in TypeSafe's re-ranking cookbook, using `candidateRequest` from `packages/core/src/literal-requests.ts`; keeps scores above 0.5, at most 8. |
| `jg` | The current jevgrep pipeline in process, with its stdout rendering. |
| `agent` | SWE-grep-style subagent without RL: each turn GPT-OSS-120B on Cerebras returns a JSON batch of up to 8 grep/read/list actions that the harness runs in parallel; it must answer with line ranges by turn 4. |
| `agent-bm25` | `agent` plus the top 15 BM25 declarations as starting hints. |
| `agent-native` | The same tools through native tool calling. The model made one call per turn, so the harness now enforces the fan-out instead. |

## Scores

- **found edit**: share of tasks where a returned snippet overlaps an edited declaration.
- **edit recall**: share of edited declarations covered. **test recall** likewise for tests.
- **precision**: share of returned snippets that overlap any answer-key declaration.
  The key is narrow (callers and helpers count as misses), so read it comparatively.
- **tokens to agent**: returned text at about 4 characters per token.
- **tool $/search** and **seconds**: measured per search.

## First results (2026-09-27)

Head to head on the 7 tasks every tool finished (jg stopped at the $0.45 Jev budget):

| Tool | found edit | precision | tokens to agent (median) | tool $/search | seconds (median) |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25 | 57% | 11% | 1,725 | $0 | 0.7 |
| rerank | 71% | 56% | 288 | $0.0008 | 1.3 |
| jg | 71% | 53% | 810 | $0.063 | 28.0 |
| agent | 57% | 40% | 655 | $0.0059 | 2.2 |
| agent-bm25 | 86% | 51% | 470 | $0.0052 | 2.2 |

Each tool on every task it finished:

| Tool | tasks | found edit | precision | tool $/search |
| --- | ---: | ---: | ---: | ---: |
| bm25 | 44 | 52% | 9% | $0 |
| rerank | 44 | 59% | 67% | $0.0008 |
| jg | 7 | 71% | 53% | $0.063 |
| agent | 34 | 62% | 45% | $0.0051 |
| agent-bm25 | 35 | 71% | 53% | $0.0042 |

- Jev as a re-ranker puts an edit site first in 28 of 44 tasks, against 11 of 44
  for BM25 alone. Its ceiling is BM25's shortlist, which contains an edit site in
  35 of 44 tasks.
- jg's mean output was 3,301 tokens (up to 11,536) and its searches cost
  $0.02 to $0.16 each.
- Pricing the `agent-bm25` search traces at Opus 5.5 rates (40k context
  already present, 250 output tokens per turn, 20 later turns re-reading what the
  search added) gives about $0.08 for Opus to search itself, against about $0.02
  for delegating to `rerank` or `agent-bm25` and $0.10 for `jg`.

### Qwen 3.8 27B on Cerebras (direct API)

On the 34 tasks every agent variant and `rerank` finished. `low` and `none` are
Cerebras `reasoning_effort` values; GPT-OSS ran at `low` through OpenRouter.
Qwen costs are estimated at $0.99/$1.49 per million tokens, since Cerebras
responses carry no price.

| Tool | found edit | precision | output tokens | seconds (median / p90) | $/search |
| --- | ---: | ---: | ---: | ---: | ---: |
| agent, GPT-OSS-120B low | 21/34 | 45% | 471 | 2.1 / 2.6 | $0.0051 |
| agent, Qwen low | 24/34 | 57% | 716 | 1.8 / 2.3 | $0.0092 |
| agent, Qwen none | 20/34 | 55% | 256 | 1.5 / 2.2 | $0.0080 |
| agent-bm25, GPT-OSS-120B low | 24/34 | 52% | 473 | 3.2 / 5.6 | $0.0042 |
| agent-bm25, Qwen low | 24/34 | 51% | 694 | 2.9 / 4.6 | $0.0085 |
| agent-bm25, Qwen none | 24/34 | 48% | 245 | 2.4 / 4.3 | $0.0084 |
| rerank | 20/34 | 64% | n/a | 1.9 / 4.1 | $0.0009 |

`agent-bm25` times include about a second of local BM25 in Python.

Samples are small: treat differences of one or two tasks as noise.

## Running

```sh
python3 evals/implementation/search-bench/build_tasks.py build
bun evals/implementation/search-bench/make_queries.ts                 # OPENROUTER_API_KEY
bun evals/implementation/search-bench/bench.ts --tools bm25,rerank    # TYPESAFE_API_KEY
bun evals/implementation/search-bench/bench.ts --tools jg --sample 12 --max-jev-usd 0.45
bun evals/implementation/search-bench/bench.ts --tools agent,agent-bm25   # OPENROUTER_API_KEY
bun evals/implementation/search-bench/bench.ts --report
```

`--agent-model cerebras/qwen-3.8-27b` calls Cerebras directly with `CEREBRAS_API_KEY`;
`--agent-effort low|none` sets its reasoning effort and `--label` stores the run as
`<tool>-<label>` beside the default results.
Results append to `evals/runs/search-bench/results/<tool>.jsonl` and reruns skip
finished tasks. Jev spend accumulates in `evals/runs/search-bench/jev-ledger.json`,
and `--max-jev-usd` caps it across runs.
