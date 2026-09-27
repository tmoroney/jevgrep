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
 *   agent   SWE-grep-style subagent without RL: each turn a fast LLM returns a JSON batch of
 *           up to 8 grep/read/list actions that the harness runs in parallel; it must answer
 *           with line ranges by turn 4. The harness, not the model, enforces the fan-out.
 *   agent-bm25    agent, plus the top 15 BM25 declarations as starting hints.
 *   agent-native  the same tools through native tool calling, where the model decides how many
 *           calls to make per turn (kept to show that untrained models barely parallelize).
 *
 * Results append to evals/runs/search-bench/results/<tool>.jsonl; reruns skip finished tasks.
 * Jev spend is recorded in evals/runs/search-bench/jev-ledger.json and capped across runs.
 */
import { createTypeSafeAi } from "../../../packages/core/node_modules/@ai-sdk/typesafe-ai";
import { experimental_evaluate as evaluate } from "../../../packages/core/node_modules/ai";
import { retrieve, createEvaluator } from "../../../packages/core/src/index";
import { candidateRequest } from "../../../packages/core/src/literal-requests";
import { renderResult } from "../../../apps/cli/src/render";
import { chat, jevLedger, readJsonl, runDir, type ChatMessage, type Task } from "./common";
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { appendFile, mkdir, readFile, readdir, rm, stat } from "node:fs/promises";
import { join, normalize, resolve } from "node:path";
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

const agentTools = [
  {
    type: "function",
    function: {
      name: "grep",
      description:
        "Search file contents with a ripgrep regex. Returns up to 40 matching lines as path:line:text.",
      parameters: {
        type: "object",
        properties: {
          pattern: { type: "string", description: "Rust regex, e.g. 'def prepare_body|Content-Length'" },
          glob: { type: "string", description: "Optional file glob, e.g. '*.py' or 'src/**/models.py'" },
        },
        required: ["pattern"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "read",
      description: "Read up to 120 lines of a file, with line numbers.",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string" },
          start_line: { type: "integer" },
          end_line: { type: "integer" },
        },
        required: ["path", "start_line", "end_line"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "list",
      description: "List a directory's entries (up to 100).",
      parameters: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
    },
  },
] as const;
const answerTool = {
  type: "function",
  function: {
    name: "answer",
    description:
      "Return the code the question asks for: at most 6 line ranges, each at most 80 lines, most relevant first.",
    parameters: {
      type: "object",
      properties: {
        snippets: {
          type: "array",
          items: {
            type: "object",
            properties: {
              path: { type: "string" },
              start_line: { type: "integer" },
              end_line: { type: "integer" },
            },
            required: ["path", "start_line", "end_line"],
          },
        },
      },
      required: ["snippets"],
    },
  },
} as const;
const agentSystem = `You are a fast code-search subagent. Another coding agent asked you a question about this repository; find the code it needs and return exact line ranges.
Work in parallel: each turn, issue several tool calls at once (up to 8), e.g. multiple grep patterns for different names, then read the most promising hits.
You have at most 3 turns of searching. By your 4th turn you must call answer.
Return only code that directly answers the question: the definitions that implement the behavior and tests of it. Precision matters more than recall; every extra snippet costs the other agent context. Do not return imports, unrelated helpers, or whole files.`;

function inside(root: string, path: string) {
  const full = normalize(join(root, path));
  if (!full.startsWith(root)) throw new Error("path outside repository");
  return full;
}
async function runAgentTool(root: string, name: string, args: Record<string, unknown>) {
  try {
    if (name === "grep") {
      const rg = spawnSync(
        "rg",
        [
          "-n",
          "--no-heading",
          "-S",
          "--max-count",
          "5",
          "--max-columns",
          "200",
          ...(args.glob ? ["-g", String(args.glob)] : []),
          "-e",
          String(args.pattern),
          ".",
        ],
        { cwd: root, maxBuffer: 1 << 26 },
      );
      const lines = rg.stdout.toString().split("\n").filter(Boolean);
      return (
        lines
          .slice(0, 40)
          .map((l) => l.replace(/^\.\//, ""))
          .join("\n") +
        (lines.length > 40 ? `\n[${lines.length - 40} more matches]` : "") ||
        rg.stderr.toString().slice(0, 300) ||
        "no matches"
      );
    }
    if (name === "read") {
      const lines = (await readFile(inside(root, String(args.path)), "utf8")).split("\n");
      const start = Math.max(1, Number(args.start_line) || 1);
      const end = Math.min(lines.length, Number(args.end_line) || start + 119, start + 119);
      return lines
        .slice(start - 1, end)
        .map((l, i) => `${start + i}: ${l}`)
        .join("\n");
    }
    if (name === "list") {
      const dir = inside(root, String(args.path ?? "."));
      const entries = await readdir(dir, { withFileTypes: true });
      return entries
        .slice(0, 100)
        .map((e) => (e.isDirectory() ? `${e.name}/` : e.name))
        .join("\n");
    }
    return `unknown tool ${name}`;
  } catch (error) {
    return `error: ${error instanceof Error ? error.message : error}`;
  }
}

const planSystem = `You are a fast code-search subagent. Another coding agent asked you a question about this repository; find the code it needs and return exact line ranges.
You work in at most 4 turns. Every turn, reply with one JSON object and nothing else:
{"actions": [ ... ], "answer": null}   or   {"actions": [], "answer": [{"path": "...", "start_line": 1, "end_line": 40}]}
Actions run in parallel, so batch them. Each action is one of:
  {"tool": "grep", "pattern": "<ripgrep regex>", "glob": "<optional glob such as *.py>"}   -> up to 40 matching lines as path:line:text
  {"tool": "read", "path": "<file>", "start_line": N, "end_line": M}                      -> up to 120 numbered lines
  {"tool": "list", "path": "<directory>"}                                                -> directory entries
Turn 1: issue 6 to 8 greps at once, covering different names, spellings and phrases from the question (function names, class names, option strings, error text).
Turn 2: issue up to 8 actions, mostly reads of the most promising grep hits, plus greps for names you discovered.
Turn 3: answer if you have found the code, otherwise up to 8 more actions.
Turn 4: you must answer.
The answer lists at most 6 ranges, most relevant first, each at most 80 lines: the definitions that implement what the question asks about, and tests of it. Precision matters more than recall; every extra range costs the other agent context. Never answer with a range you have not seen in a read or grep result.`;

async function runPlanAgent(root: string, query: string, hints = false) {
  let intro = `Repository root listing:\n${await runAgentTool(root, "list", { path: "." })}`;
  if (hints)
    intro += `\n\nTop local keyword matches for the question (declaration, path:lines), which may or may not be relevant:\n${bm25(root, query, 15)
      .map((b) => `${b.name}  ${b.path}:${b.start_line}-${b.end_line}`)
      .join("\n")}`;
  const messages: ChatMessage[] = [
    { role: "system", content: planSystem },
    { role: "user", content: `${intro}\n\nQuestion: ${query}\n\nTurn 1 of 4.` },
  ];
  const trace: NonNullable<Result["trace"]> = [];
  let usd = 0;
  for (let turn = 0; turn < 4; turn++) {
    const response = await chat({
      model: values["agent-model"]!,
      messages,
      response_format: { type: "json_object" },
      reasoning: { effort: "low" },
      max_tokens: 4000,
    });
    usd += response.cost;
    messages.push({ role: "assistant", content: response.content ?? "" });
    let plan: {
      actions?: Array<Record<string, unknown>>;
      answer?: Array<{ path: string; start_line: number; end_line: number }> | null;
    } = {};
    try {
      plan = JSON.parse(response.content ?? "{}");
    } catch {}
    const actions = (plan.actions ?? []).slice(0, 8);
    if (plan.answer?.length || turn === 3) {
      trace.push({ calls: 0, toolOutputChars: 0, inputTokens: response.inputTokens, outputTokens: response.outputTokens });
      const snippets: Snippet[] = [];
      for (const s of (plan.answer ?? []).slice(0, 6)) {
        const path = String(s.path).replace(/^\.\//, "");
        try {
          if (!(await stat(inside(root, path))).isFile()) continue;
        } catch {
          continue;
        }
        const startLine = Math.max(1, Math.floor(s.start_line));
        snippets.push({ path, startLine, endLine: Math.max(startLine, Math.floor(s.end_line)) });
      }
      return { snippets, output: await render(root, snippets), toolUsd: usd, trace, ...(snippets.length ? {} : { status: "no-answer" }) };
    }
    const outputs = await Promise.all(actions.map((a) => runAgentTool(root, String(a.tool), a)));
    const toolOutputChars = outputs.reduce((n, o) => n + o.length, 0);
    trace.push({ calls: actions.length, toolOutputChars, inputTokens: response.inputTokens, outputTokens: response.outputTokens });
    messages.push({
      role: "user",
      content:
        (actions.length
          ? actions.map((a, i) => `### ${i + 1}. ${JSON.stringify(a)}\n${outputs[i]}`).join("\n\n")
          : "No actions were given.") + `\n\nTurn ${turn + 2} of 4.${turn + 2 === 4 ? " You must answer now." : ""}`,
    });
  }
  return { snippets: [], output: "", toolUsd: usd, trace, status: "no-answer" };
}

async function runAgent(root: string, query: string) {
  const messages: ChatMessage[] = [
    { role: "system", content: agentSystem },
    { role: "user", content: `Repository root listing:\n${await runAgentTool(root, "list", { path: "." })}\n\nQuestion: ${query}` },
  ];
  const trace: NonNullable<Result["trace"]> = [];
  let usd = 0;
  for (let turn = 0; turn < 4; turn++) {
    const final = turn === 3;
    const response = await chat({
      model: values["agent-model"]!,
      messages,
      tools: final ? [answerTool] : [...agentTools, answerTool],
      tool_choice: final ? "required" : "auto",
      parallel_tool_calls: true,
      reasoning: { effort: "low" },
      max_tokens: 4000,
    });
    usd += response.cost;
    const calls = response.message.tool_calls ?? [];
    messages.push({ role: "assistant", content: response.message.content ?? "", tool_calls: calls });
    const answer = calls.find((c) => c.function.name === "answer");
    let toolOutputChars = 0;
    if (!answer) {
      if (!calls.length)
        messages.push({ role: "user", content: "Call tools, or call answer if you have found the code." });
      const outputs = await Promise.all(
        calls.map((c) => {
          let args: Record<string, unknown> = {};
          try {
            args = JSON.parse(c.function.arguments || "{}");
          } catch {}
          return runAgentTool(root, c.function.name, args);
        }),
      );
      calls.forEach((c, i) => {
        toolOutputChars += outputs[i]!.length;
        messages.push({ role: "tool", tool_call_id: c.id, content: outputs[i]! });
      });
    }
    trace.push({
      calls: calls.length,
      toolOutputChars,
      inputTokens: response.inputTokens,
      outputTokens: response.outputTokens,
    });
    if (answer) {
      const parsed = JSON.parse(answer.function.arguments || "{}") as {
        snippets?: Array<{ path: string; start_line: number; end_line: number }>;
      };
      const snippets: Snippet[] = [];
      for (const s of (parsed.snippets ?? []).slice(0, 6)) {
        const path = s.path.replace(/^\.\//, "");
        try {
          if (!(await stat(inside(root, path))).isFile()) continue;
        } catch {
          continue;
        }
        const startLine = Math.max(1, Math.floor(s.start_line));
        snippets.push({ path, startLine, endLine: Math.max(startLine, Math.floor(s.end_line)) });
      }
      return { snippets, output: await render(root, snippets), toolUsd: usd, trace };
    }
  }
  return { snippets: [], output: "", toolUsd: usd, trace, status: "no-answer" };
}

const tools: Record<string, (root: string, query: string) => Promise<Omit<Result, "instance_id" | "tool" | "query" | "seconds" | "outputChars"> & { output: string }>> = {
  bm25: runBm25,
  rerank: runRerank,
  jg: runJg,
  "agent-native": runAgent,
  agent: (root, query) => runPlanAgent(root, query),
  "agent-bm25": (root, query) => runPlanAgent(root, query, true),
};

// ---------- runner ----------

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
    done.set(name, new Set((await readJsonl<Result>(`${runDir}/results/${name}.jsonl`)).map((r) => r.instance_id)));
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
          tool: name,
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
      await appendFile(`${runDir}/results/${name}.jsonl`, JSON.stringify(row) + "\n");
      const s = score(task, row);
      console.log(
        `${task.instance_id.padEnd(36)} ${name.padEnd(6)} edit ${pct(s.editRecall)} test ${pct(s.testRecall)} prec ${pct(s.precision)} ` +
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
  for (const name of Object.keys(tools)) {
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
      "tool    tasks  found edit  edit recall  edit file  test recall  precision  in gold file  snippets  tokens to agent  tool $/search  seconds",
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
