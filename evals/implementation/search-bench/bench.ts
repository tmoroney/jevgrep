/**
 * Code-search benchmark: every tool gets the same repository checkout and the same
 * natural-language question, returns snippets, and is scored against the task's answer
 * key (declarations a real change edited, plus the tests next to them).
 *
 *   bun evals/implementation/search-bench/bench.ts --tools bm25,rerank --limit 44
 *   bun evals/implementation/search-bench/bench.ts --tools jg --sample 10 --max-jev-usd 0.45
 *   bun evals/implementation/search-bench/bench.ts --report
 *
 * Tools:
 *   bm25    local keyword ranking of every Python declaration; top 8. Free.
 *   rerank  BM25 top 30, then one Jev Noul per candidate (TypeSafe's re-ranking cookbook);
 *           keeps candidates above --threshold, at most 8.
 *   jg      the current jevgrep pipeline, in process, with its stdout rendering.
 *   agent   the fast-context agent from packages/fast-context (--agent-model, --agent-effort).
 *   agent-bm25    agent, plus the top 15 BM25 declarations as starting hints.
 *
 * Results append to evals/runs/search-bench/results/<tool>.jsonl; reruns skip finished tasks.
 * Jev spend is recorded in evals/runs/search-bench/jev-ledger.json and capped across runs.
 */
import { createTypeSafeAi } from "../../../packages/core/node_modules/@ai-sdk/typesafe-ai";
import { experimental_evaluate as evaluate } from "../../../packages/core/node_modules/ai";
import { retrieve, createEvaluator } from "../../../packages/core/src/index";
import { candidateRequest } from "../../../packages/core/src/literal-requests";
import { renderResult } from "../../../apps/cli/src/render";
import { findContext, type Endpoint } from "../../../packages/fast-context/src/fast-context";
import { jevLedger, readJsonl, runDir, type Task } from "./common";
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { appendFile, mkdir, readFile, readdir, rm } from "node:fs/promises";
import { join, resolve } from "node:path";
import { parseArgs } from "node:util";

const { values } = parseArgs({
  options: {
    tools: { type: "string", default: "bm25" },
    limit: { type: "string" },
    sample: { type: "string" },
    tasks: { type: "string" },
    "max-jev-usd": { type: "string", default: "0.45" },
    threshold: { type: "string", default: "0.5" },
    "agent-model": { type: "string", default: "openai/gpt-oss-120b" },
    "agent-effort": { type: "string", default: "low" },
    // Stored as <tool>-<label>, so runs with another model sit beside the default ones.
    label: { type: "string" },
    keep: { type: "boolean", default: false },
    report: { type: "boolean", default: false },
  },
});
type Snippet = { path: string; startLine: number; endLine: number };
type Result = {
  instance_id: string;
  tool: string;
  query: string;
  snippets: Snippet[];
  files?: string[];
  outputChars: number;
  toolUsd: number;
  jevTokens?: number;
  seconds: number;
  status?: string;
  trace?: Array<{ calls: number; toolOutputChars: number; inputTokens: number; outputTokens: number }>;
  detail?: unknown;
  error?: string;
};
const builder = resolve(import.meta.dir, "build_tasks.py");
const tasks = await readJsonl<Task>(`${runDir}/tasks.jsonl`);
const queries = new Map(
  (await readJsonl<{ instance_id: string; query: string }>(`${runDir}/queries.jsonl`)).map((q) => [
    q.instance_id,
    q.query,
  ]),
);

// ---------- shared helpers ----------

function render(root: string, snippets: Snippet[], maxLines = 80) {
  return Promise.all(
    snippets.map(async (s) => {
      const lines = (await readFile(join(root, s.path), "utf8")).split("\n");
      const end = Math.min(s.endLine, s.startLine + maxLines - 1, lines.length);
      return `${s.path}:${s.startLine}-${end}\n${lines.slice(s.startLine - 1, end).join("\n")}`;
    }),
  ).then((parts) => parts.join("\n\n"));
}

type Block = { path: string; name: string; start_line: number; end_line: number; source: string };
function bm25(root: string, query: string, k: number): Block[] {
  const out = spawnSync("python3", [builder, "bm25", root, query, "--k", String(k)], {
    maxBuffer: 1 << 28,
  });
  if (out.status !== 0) throw new Error(out.stderr.toString());
  return JSON.parse(out.stdout.toString());
}
const toSnippet = (b: Block): Snippet => ({ path: b.path, startLine: b.start_line, endLine: b.end_line });

// ---------- tools ----------

async function runBm25(root: string, query: string) {
  const blocks = bm25(root, query, 8);
  const snippets = blocks.map(toSnippet);
  return { snippets, output: await render(root, snippets), toolUsd: 0 };
}

const ledger = await jevLedger(Number(values["max-jev-usd"]));
const jevModel = () =>
  createTypeSafeAi({
    apiKey: process.env.TYPESAFE_API_KEY!,
    baseURL: "https://api.typesafe.ai/v1",
  }).evaluationModel("jev-1.13.0");

async function runRerank(root: string, query: string) {
  const shortlist = bm25(root, query, 30);
  const model = jevModel();
  let tokens = 0;
  const scores = await Promise.all(
    shortlist.map(async (block) => {
      const request = candidateRequest(query, {
        path: block.path,
        name: block.name,
        source: block.source.slice(0, 4000),
      });
      const result = await evaluate({
        model,
        ...request,
        questions: request.questions as never,
        maxRetries: 2,
      });
      tokens += result.usage.inputTokens ?? 0;
      return (result.answers as Record<string, { probability: number }>).match.probability;
    }),
  );
  await ledger.add(tokens);
  const threshold = Number(values.threshold);
  const ranked = shortlist
    .map((block, i) => ({ block, score: scores[i]! }))
    .sort((a, b) => b.score - a.score);
  const kept = ranked.filter((r) => r.score > threshold).slice(0, 8);
  const snippets = kept.map((r) => toSnippet(r.block));
  return {
    snippets,
    output: await render(root, snippets),
    toolUsd: tokens * 0.042e-6,
    jevTokens: tokens,
    detail: ranked.map((r) => ({ ...toSnippet(r.block), score: r.score })),
  };
}

async function runJg(root: string, query: string) {
  const controller = new AbortController();
  let tokens = 0;
  const evaluator = createEvaluator({
    provider: "typesafe",
    apiKey: process.env.TYPESAFE_API_KEY!,
    signal: controller.signal,
    fetch: async (input, init) => {
      const response = await fetch(input, init);
      response
        .clone()
        .json()
        .then((body: { usage?: { input_tokens?: number } }) => {
          tokens += body.usage?.input_tokens ?? 0;
          if (tokens * 0.042e-6 > ledger.remaining()) controller.abort();
        })
        .catch(() => {});
      return response;
    },
  });
  const result = await retrieve({ root, query, signal: controller.signal }, evaluator);
  await new Promise((done) => setTimeout(done, 200)); // let pending usage parses land
  await ledger.add(tokens);
  const snippets = result.files.flatMap((file) =>
    file.excerpts.map((e) => ({ path: file.path, startLine: e.range.startLine, endLine: e.range.endLine })),
  );
  return {
    snippets,
    files: result.files.map((f) => f.path),
    output: renderResult(result),
    toolUsd: tokens * 0.042e-6,
    jevTokens: tokens,
    status: controller.signal.aborted ? "budget-stopped" : result.status,
  };
}

/** Endpoint for the fast-context agent: `cerebras/<model>` direct, anything else via OpenRouter. */
function agentEndpoint(): Partial<Endpoint> {
  const model = values["agent-model"]!;
  const reasoningEffort = values["agent-effort"] as Endpoint["reasoningEffort"];
  if (model.startsWith("cerebras/"))
    return { model: model.slice("cerebras/".length), reasoningEffort };
  const key = process.env.OPENROUTER_API_KEY;
  if (!key) throw new Error("Set OPENROUTER_API_KEY");
  return {
    baseURL: "https://openrouter.ai/api/v1",
    apiKey: key,
    model,
    reasoningEffort,
    extraBody: {
      usage: { include: true },
      ...(model === "openai/gpt-oss-120b"
        ? { provider: { order: ["Cerebras"], allow_fallbacks: false } }
        : {}),
    },
  };
}

async function runFastContext(root: string, query: string, hints = false) {
  const result = await findContext({
    root,
    question: query,
    endpoint: agentEndpoint(),
    ...(hints
      ? {
          hints: bm25(root, query, 15).map(
            (b) => `${b.name}  ${b.path}:${b.start_line}-${b.end_line}`,
          ),
        }
      : {}),
  });
  return {
    snippets: result.snippets.map(({ path, startLine, endLine }) => ({ path, startLine, endLine })),
    output: result.text,
    toolUsd: result.usage.usd,
    trace: result.trace.map((t) => ({
      calls: t.actions,
      toolOutputChars: t.toolOutputChars,
      inputTokens: t.inputTokens,
      outputTokens: t.outputTokens,
    })),
    ...(result.status === "answered" ? {} : { status: result.status }),
  };
}

const tools: Record<string, (root: string, query: string) => Promise<Omit<Result, "instance_id" | "tool" | "query" | "seconds" | "outputChars"> & { output: string }>> = {
  bm25: runBm25,
  rerank: runRerank,
  jg: runJg,
  agent: (root, query) => runFastContext(root, query),
  "agent-bm25": (root, query) => runFastContext(root, query, true),
};

// ---------- runner ----------

const stored = (name: string) => (values.label ? `${name}-${values.label}` : name);

function chooseTasks(): Task[] {
  if (values.tasks) return values.tasks.split(",").map((id) => tasks.find((t) => t.instance_id === id)!);
  if (values.sample) {
    // One task per repository first, so a small sample still covers every project.
    const seen = new Set<string>();
    const first = tasks.filter((t) => !seen.has(t.repo) && seen.add(t.repo));
    return [...first, ...tasks.filter((t) => !first.includes(t))].slice(0, Number(values.sample));
  }
  return tasks.slice(0, Number(values.limit ?? tasks.length));
}

async function run() {
  const selected = chooseTasks();
  const names = values.tools!.split(",");
  for (const name of names) if (!tools[name]) throw new Error(`unknown tool ${name}`);
  await mkdir(`${runDir}/results`, { recursive: true });
  const done = new Map<string, Set<string>>();
  for (const name of names)
    done.set(name, new Set((await readJsonl<Result>(`${runDir}/results/${stored(name)}.jsonl`)).map((r) => r.instance_id)));
  for (const task of selected) {
    const pending = names.filter((name) => !done.get(name)!.has(task.instance_id));
    if (!pending.length) continue;
    const query = queries.get(task.instance_id);
    if (!query) throw new Error(`no query for ${task.instance_id}; run make_queries.ts`);
    const root = resolve(`${runDir}/work/${task.instance_id}`);
    if (!existsSync(root)) {
      const out = spawnSync("python3", [builder, "checkout", task.instance_id, root]);
      if (out.status !== 0) throw new Error(out.stderr.toString());
    }
    for (const name of pending) {
      if ((name === "jg" || name === "rerank") && ledger.remaining() <= 0) {
        console.log(`${name}: Jev budget exhausted ($${ledger.usd.toFixed(4)} spent); skipping`);
        continue;
      }
      const start = performance.now();
      let row: Result;
      try {
        const { output, ...rest } = await tools[name]!(root, query);
        row = {
          instance_id: task.instance_id,
          tool: stored(name),
          query,
          ...rest,
          outputChars: output.length,
          seconds: (performance.now() - start) / 1000,
        };
      } catch (error) {
        row = {
          instance_id: task.instance_id,
          tool: name,
          query,
          snippets: [],
          outputChars: 0,
          toolUsd: 0,
          seconds: (performance.now() - start) / 1000,
          error: error instanceof Error ? error.message.slice(0, 500) : String(error),
        };
      }
      await appendFile(`${runDir}/results/${stored(name)}.jsonl`, JSON.stringify(row) + "\n");
      const s = score(task, row);
      console.log(
        `${task.instance_id.padEnd(36)} ${stored(name).padEnd(16)} edit ${pct(s.editRecall)} test ${pct(s.testRecall)} prec ${pct(s.precision)} ` +
          `${String(row.snippets.length).padStart(2)} snip ${Math.round(row.outputChars / 4).toString().padStart(6)} tok $${row.toolUsd.toFixed(4)} ${row.seconds.toFixed(1)}s` +
          (row.status && row.status !== "complete" ? ` [${row.status}]` : "") +
          (row.error ? ` ERROR ${row.error.slice(0, 120)}` : ""),
      );
    }
    if (!values.keep) await rm(root, { recursive: true, force: true });
  }
  console.log(`Jev ledger: $${ledger.usd.toFixed(4)} of $${values["max-jev-usd"]}`);
}

// ---------- scoring ----------

const overlaps = (s: Snippet, g: Task["gold"][number]) =>
  s.path === g.path && s.startLine <= g.end_line && s.endLine >= g.start_line;
function score(task: Task, row: Result) {
  const edits = task.gold.filter((g) => g.label === "edit");
  const tests = task.gold.filter((g) => g.label === "test");
  const found = (gold: typeof edits) =>
    gold.length ? gold.filter((g) => row.snippets.some((s) => overlaps(s, g))).length / gold.length : NaN;
  const goldFiles = new Set(task.gold.map((g) => g.path));
  const n = row.snippets.length;
  return {
    editRecall: found(edits),
    testRecall: found(tests),
    anyEdit: edits.some((g) => row.snippets.some((s) => overlaps(s, g))) ? 1 : 0,
    editFile: edits.some((g) => (row.files ?? row.snippets.map((s) => s.path)).includes(g.path)) ? 1 : 0,
    precision: n ? row.snippets.filter((s) => task.gold.some((g) => overlaps(s, g))).length / n : NaN,
    inGoldFile: n ? row.snippets.filter((s) => goldFiles.has(s.path)).length / n : NaN,
  };
}
const pct = (x: number) => (Number.isNaN(x) ? "  - " : `${Math.round(100 * x).toString().padStart(3)}%`);
const mean = (xs: number[]) => {
  const v = xs.filter((x) => !Number.isNaN(x));
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN;
};
const median = (xs: number[]) => {
  const v = [...xs].sort((a, b) => a - b);
  return v.length ? v[Math.floor(v.length / 2)]! : NaN;
};

async function report() {
  const byTool = new Map<string, Result[]>();
  const files = (await readdir(`${runDir}/results`)).filter((f) => f.endsWith(".jsonl")).sort();
  for (const name of files.map((f) => f.slice(0, -".jsonl".length))) {
    const rows = await readJsonl<Result>(`${runDir}/results/${name}.jsonl`);
    if (rows.length) byTool.set(name, rows);
  }
  // Compare tools on the tasks every reported tool finished, then on each tool's full set.
  const common = [...byTool.values()].reduce<Set<string> | undefined>(
    (acc, rows) => {
      const ids = new Set(rows.filter((r) => !r.error).map((r) => r.instance_id));
      return acc ? new Set([...acc].filter((id) => ids.has(id))) : ids;
    },
    undefined,
  );
  const table = (title: string, filter: (r: Result) => boolean) => {
    console.log(`\n${title}`);
    console.log(
      "tool             tasks  found edit  edit recall  edit file  test recall  precision  in gold file  snippets  tokens to agent  tool $/search  seconds",
    );
    for (const [name, rows] of byTool) {
      const scored = rows.filter((r) => !r.error && filter(r)).map((r) => ({ r, s: score(tasks.find((t) => t.instance_id === r.instance_id)!, r) }));
      if (!scored.length) continue;
      const errors = rows.filter((r) => r.error && filter(r)).length;
      console.log(
        `${name.padEnd(7)} ${String(scored.length).padStart(5)}  ${pct(mean(scored.map((x) => x.s.anyEdit))).padStart(10)}  ${pct(mean(scored.map((x) => x.s.editRecall))).padStart(11)}  ${pct(mean(scored.map((x) => x.s.editFile))).padStart(9)}  ${pct(mean(scored.map((x) => x.s.testRecall))).padStart(11)}  ${pct(mean(scored.map((x) => x.s.precision))).padStart(9)}  ${pct(mean(scored.map((x) => x.s.inGoldFile))).padStart(12)}  ${median(scored.map((x) => x.r.snippets.length)).toString().padStart(8)}  ${Math.round(median(scored.map((x) => x.r.outputChars / 4))).toLocaleString().padStart(15)}  ${("$" + mean(scored.map((x) => x.r.toolUsd)).toFixed(4)).padStart(13)}  ${median(scored.map((x) => x.r.seconds)).toFixed(1).padStart(7)}` +
          (errors ? `  (${errors} errored)` : ""),
      );
    }
  };
  if (common && byTool.size > 1) table(`Head to head on the ${common.size} tasks every tool finished (medians for snippets, tokens, seconds; means otherwise)`, (r) => common.has(r.instance_id));
  table("Each tool on every task it ran", () => true);
}

if (values.report) await report();
else await run();
