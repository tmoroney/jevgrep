# Retrieval-pair diagnostic

A cheap check of Jev relevance judgments on their own, without a coding agent.
It asks whether a request design ranks the declarations a real fix changed above
plausible distractors, and what that costs in Jev input tokens. It is a
diagnostic for request design, not product acceptance evidence; the
[cost and quality policy](../../cost-quality-policy.md) still owns acceptance.

## Pairs

`build_pairs.py` samples SWE-bench Verified tasks from the moatless-tools
packaging (pinned commit), whose `expected_spans` name the declarations each gold
patch edits. For each task it indexes every Python declaration at the base commit
and keeps a small candidate set:

| Label | Meaning |
| --- | --- |
| `edit` | A declaration the gold patch changes |
| `test` | An existing test declaration named in `test_file_spans` |
| `negative` / `bm25` | A top local BM25 match for the problem statement that the patch leaves alone |
| `negative` / `sibling` | Another declaration in an edited file |
| `negative` / `random` | Any other declaration |

BM25 negatives are the blocks a code-side prefilter would hand to Jev, so they
are the realistic hard case. Each record also keeps every candidate's BM25 rank,
which shows whether such a prefilter can surface the edit site at all.

```sh
python3 evals/implementation/retrieval-pairs/build_pairs.py   # writes evals/runs/retrieval-pairs/
```

## Request designs

| Variant | Builder | Shape |
| --- | --- | --- |
| `current` | `evidenceRequest` (`packages/core/src/requests.ts`) | jevgrep today: 8 declarations per request, whole file up to 16 KB, line-range questions |
| `literal` | `blockRequest` (`packages/core/src/literal-requests.ts`) | One block per request; separate `edit`, `behavior`, `test` Nouls with criteria |
| `facets` | `facetRequest` | As `literal`, but judged against a one-line behavior written blind by a generative model (`make_facets.ts`) |
| `shortlist` | `shortlistRequest` | One Choice over all of a task's candidates plus an absolute "any" Noul |

The literal builders follow TypeSafe's jev-1.13 guidance: small state, questions
that name the state field they read, one judgment per question with boundary
cases in `criteria`, no counting, and lexical facts computed in code.

## Running

Every run is a dry run unless `--live` is passed, and a live run stops before
its measured spend would exceed `--max-usd` (default $0.05). Answers append to
`evals/runs/retrieval-pairs/answers/<variant>.jsonl`, so a rerun resumes.

```sh
bun evals/implementation/retrieval-pairs/run.ts --variant literal          # estimate
JEV_PROVIDER=openrouter JEV_API_KEY=... \
  bun evals/implementation/retrieval-pairs/run.ts --variant literal --live --max-usd 0.03
OPENROUTER_API_KEY=... bun evals/implementation/retrieval-pairs/make_facets.ts --live
bun evals/implementation/retrieval-pairs/run.ts --report
```

`--mock` sends every request through the real SDK to a fake endpoint, which
checks request validity and the report without network or spend; use it with
`--dir` pointing at a scratch copy so mock answers never mix with real ones.

The report gives, per score: AUC of edit blocks and of test blocks against
negatives, true- and false-positive rates at 0.5 (split by negative kind), how
often the task's top-scored block is an edit site, precision of everything
above 0.5, and measured input tokens per task.
