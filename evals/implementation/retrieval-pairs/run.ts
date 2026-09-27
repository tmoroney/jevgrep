/**
 * Score labeled (task, block) pairs with Jev using one request design per run,
 * then report ranking quality and measured cost. Dry run by default.
 *
 *   bun evals/implementation/retrieval-pairs/run.ts --variant literal            # estimate only
 *   JEV_PROVIDER=openrouter JEV_API_KEY=... \
 *   bun evals/implementation/retrieval-pairs/run.ts --variant literal --live --max-usd 0.03
 *   bun evals/implementation/retrieval-pairs/run.ts --report
 *
 * Variants:
 *   current    jevgrep's evidenceRequest with its whole-file/window context, 8 declarations per request
 *   literal    blockRequest: one block per request, separate edit/behavior/test Nouls
 *   facets     facetRequest from facets.jsonl ({instance_id, behavior, test?}), written blind to the patch
 *   shortlist  shortlistRequest: one Choice over every candidate block of a task, plus an "any" Noul
 *
 * Answers append to evals/runs/retrieval-pairs/answers/<variant>.jsonl; reruns skip finished requests.
 */
import { createTypeSafeAi } from "../../../packages/core/node_modules/@ai-sdk/typesafe-ai";
import { experimental_evaluate as evaluate } from "../../../packages/core/node_modules/ai";
import { providers, isProviderId } from "../../../packages/core/src/providers";
import { evidenceRequest } from "../../../packages/core/src/requests";
import {
  blockRequest,
  facetRequest,
  shortlistRequest,
  namesSymbol,
  type Facets,
} from "../../../packages/core/src/literal-requests";
import { appendFile, mkdir, readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { parseArgs } from "node:util";

type Block = {
  id: string;
  path: string;
  name: string;
  start_line: number;
  end_line: number;
  source: string;
  label: "edit" | "test" | "negative";
  negative_kind: "bm25" | "sibling" | "random" | null;
  bm25_rank: number;
};
type Task = { instance_id: string; repo: string; problem_statement: string; blocks: Block[] };
type Job = { key: string; task: Task; blocks: Block[]; request: { state: unknown; questions: any } };
type Answer = {
  instance_id: string;
  key: string;
  blocks: string[];
  answers: Record<string, any>;
  inputTokens?: number;
  estimatedTokens: number;
};

const variants = ["current", "literal", "facets", "shortlist"] as const;
type Variant = (typeof variants)[number];
const { values } = parseArgs({
  options: {
    variant: { type: "string" },
    live: { type: "boolean", default: false },
    // Offline pipeline check: the real SDK builds and validates requests; a fake endpoint answers.
    mock: { type: "boolean", default: false },
    report: { type: "boolean", default: false },
    "max-usd": { type: "string", default: "0.05" },
    "usd-per-mtok": { type: "string", default: "0.042" },
    "bytes-per-token": { type: "string", default: "3.6" },
    tasks: { type: "string" },
    dir: { type: "string", default: "evals/runs/retrieval-pairs" },
    concurrency: { type: "string", default: "8" },
  },
});
const dir = resolve(values.dir!);
const price = Number(values["usd-per-mtok"]) / 1e6;
const bytesPerToken = Number(values["bytes-per-token"]);
const readJsonl = async <T>(path: string): Promise<T[]> =>
  existsSync(path)
    ? (await readFile(path, "utf8"))
        .split("\n")
        .filter(Boolean)
        .map((line) => JSON.parse(line))
    : [];
let tasks = await readJsonl<Task>(join(dir, "pairs.jsonl"));
if (!tasks.length) throw new Error(`No pairs in ${dir}; run build_pairs.py first`);
if (values.tasks) tasks = tasks.slice(0, Number(values.tasks));

function currentJobs(files: Map<string, Record<string, string>>): Job[] {
  // Mirrors selection.ts: declarations grouped 8 at a time within a file; the whole
  // file is sent when it is at most 16000 bytes, otherwise 20 opening lines plus a window.
  const jobs: Job[] = [];
  for (const task of tasks) {
    const byPath = Map.groupBy(task.blocks, (block) => block.path);
    for (const [path, blocks] of byPath) {
      const source = files.get(task.instance_id)![path]!;
      const lines = source.split("\n");
      blocks.sort((a, b) => a.start_line - b.start_line);
      for (let i = 0; i < blocks.length; i += 8) {
        const group = blocks.slice(i, i + 8);
        const first = Math.max(1, group[0]!.start_line - 8);
        const last = Math.min(lines.length, group.at(-1)!.end_line + 8);
        const context =
          Buffer.byteLength(source) <= 16000
            ? source
            : `Opening context:\n${lines.slice(0, 20).join("\n")}\nSource lines ${first}-${last}:\n${lines.slice(first - 1, last).join("\n")}`;
        jobs.push({
          key: `${path}#${i}`,
          task,
          blocks: group,
          request: evidenceRequest(
            task.problem_statement,
            path,
            context,
            group.map((block) => ({ name: block.name, startLine: block.start_line, endLine: block.end_line })),
          ),
        });
      }
    }
  }
  return jobs;
}

async function jobsFor(variant: Variant): Promise<Job[]> {
  if (variant === "current") {
    const files = new Map(
      (await readJsonl<{ instance_id: string; files: Record<string, string> }>(join(dir, "files.jsonl"))).map(
        (row) => [row.instance_id, row.files],
      ),
    );
    return currentJobs(files);
  }
  if (variant === "shortlist")
    return tasks.map((task) => ({
      key: "shortlist",
      task,
      blocks: task.blocks,
      request: shortlistRequest(task.problem_statement, task.blocks),
    }));
  if (variant === "facets") {
    const facets = new Map(
      (await readJsonl<Facets & { instance_id: string }>(join(dir, "facets.jsonl"))).map((row) => [row.instance_id, row]),
    );
    const missing = tasks.filter((task) => !facets.has(task.instance_id)).length;
    if (missing) throw new Error(`facets.jsonl lacks ${missing} task(s)`);
    return tasks.flatMap((task) =>
      task.blocks.map((block) => ({
        key: block.id,
        task,
        blocks: [block],
        request: facetRequest({ behavior: facets.get(task.instance_id)!.behavior, test: facets.get(task.instance_id)!.test }, block),
      })),
    );
  }
  return tasks.flatMap((task) =>
    task.blocks.map((block) => ({ key: block.id, task, blocks: [block], request: blockRequest(task.problem_statement, block) })),
  );
}

const estimate = (job: Job) => Math.ceil(Buffer.byteLength(JSON.stringify(job.request)) / bytesPerToken);

async function execute(variant: Variant) {
  const jobs = await jobsFor(variant);
  const answersPath = join(dir, "answers", `${variant}.jsonl`);
  const done = new Set((await readJsonl<Answer>(answersPath)).map((a) => `${a.instance_id}/${a.key}`));
  const pending = jobs.filter((job) => !done.has(`${job.task.instance_id}/${job.key}`));
  const tokens = pending.reduce((sum, job) => sum + estimate(job), 0);
  console.log(
    `${variant}: ${jobs.length} requests (${pending.length} pending), ~${tokens.toLocaleString()} input tokens, ~$${(tokens * price).toFixed(4)} at $${values["usd-per-mtok"]}/M`,
  );
  if (!values.live && !values.mock) return console.log("Dry run. Add --live to call Jev.");
  const cap = values.mock ? Infinity : Number(values["max-usd"]);
  const provider = values.mock ? "typesafe" : process.env.JEV_PROVIDER;
  const apiKey = values.mock ? "mock" : process.env.JEV_API_KEY;
  if (!isProviderId(provider) || !apiKey)
    throw new Error(`Set JEV_PROVIDER (${Object.keys(providers).join("|")}) and JEV_API_KEY`);
  const preset = providers[provider];
  const model = createTypeSafeAi({
    apiKey,
    baseURL: preset.baseURL,
    ...(values.mock ? { fetch: mockFetch } : {}),
  }).evaluationModel(preset.model);
  await mkdir(join(dir, "answers"), { recursive: true });
  let spent = 0;
  let reserved = 0;
  let failures = 0;
  let capped = false;
  let next = 0;
  const worker = async () => {
    while (next < pending.length) {
      const job = pending[next++]!;
      const estimated = estimate(job);
      // Reserve the estimate before sending so concurrent workers cannot overshoot the cap.
      if ((spent + reserved + estimated) * price > cap) {
        capped = true;
        next = pending.length;
        break;
      }
      reserved += estimated;
      try {
        const result = await evaluate({
          model,
          state: job.request.state as any,
          questions: job.request.questions,
          maxRetries: 1,
          abortSignal: AbortSignal.timeout(30_000),
        });
        const used = result.usage.inputTokens ?? estimated;
        spent += used;
        const row: Answer = {
          instance_id: job.task.instance_id,
          key: job.key,
          blocks: job.blocks.map((block) => block.id),
          answers: result.answers,
          inputTokens: result.usage.inputTokens,
          estimatedTokens: estimated,
        };
        await appendFile(answersPath, JSON.stringify(row) + "\n");
      } catch (error) {
        failures++;
        console.error(`${job.task.instance_id}/${job.key}: ${error instanceof Error ? error.message : error}`);
      } finally {
        reserved -= estimated;
      }
    }
  };
  await Promise.all(Array.from({ length: Number(values.concurrency) }, worker));
  console.log(
    `${variant}: spent ${spent.toLocaleString()} input tokens (~$${(spent * price).toFixed(4)}), ${failures} failed` +
      (capped ? "; stopped before exceeding --max-usd (rerun resumes)" : ""),
  );
}

async function mockFetch(_input: unknown, init?: RequestInit) {
  const body = JSON.parse(String(init?.body));
  const answers = Object.fromEntries(
    Object.entries(body.questions as Record<string, any>).map(([id, question]) => {
      if (question.type === "choice") {
        const options = Object.keys(question.criteria);
        const weights = options.map(() => Math.random());
        const total = weights.reduce((a, b) => a + b, 0);
        const probabilities = Object.fromEntries(options.map((option, i) => [option, weights[i]! / total]));
        const choice = options[weights.indexOf(Math.max(...weights))]!;
        return [id, { type: "choice", choice, probabilities }];
      }
      return [id, { type: "noul", noul: Math.round(Math.random() * 100) / 100 }];
    }),
  );
  const inputTokens = Math.ceil(Buffer.byteLength(String(init?.body)) / bytesPerToken);
  return Response.json({ model: body.model, answers, usage: { input_tokens: inputTokens, output_tokens: 1 } });
}

// ---------- report ----------

type Scored = { task: Task; block: Block; scores: Record<string, number> };
function auc(pos: number[], neg: number[]) {
  if (!pos.length || !neg.length) return NaN;
  let wins = 0;
  for (const p of pos) for (const n of neg) wins += p > n ? 1 : p === n ? 0.5 : 0;
  return wins / (pos.length * neg.length);
}
const pct = (x: number) => (Number.isNaN(x) ? "  n/a" : `${(100 * x).toFixed(0).padStart(4)}%`);

async function scoredRows(variant: Variant): Promise<Scored[] | undefined> {
  const rows = await readJsonl<Answer>(join(dir, "answers", `${variant}.jsonl`));
  if (!rows.length) return;
  const taskById = new Map(tasks.map((task) => [task.instance_id, task]));
  const out: Scored[] = [];
  for (const row of rows) {
    const task = taskById.get(row.instance_id);
    if (!task) continue;
    const blocks = row.blocks.map((id) => task.blocks.find((block) => block.id === id)!);
    if (variant === "current")
      blocks.forEach((block, i) => out.push({ task, block, scores: { useful: row.answers[`q${i}`].probability } }));
    else if (variant === "shortlist") {
      const probabilities = row.answers.pick.probabilities ?? {};
      blocks.forEach((block, i) =>
        out.push({ task, block, scores: { pick: probabilities[`c${i}`] ?? (row.answers.pick.choice === `c${i}` ? 1 : 0) } }),
      );
    } else {
      const scores = Object.fromEntries(Object.entries(row.answers).map(([id, a]) => [id, a.probability as number]));
      if (scores.edit !== undefined && scores.behavior !== undefined)
        scores["max(edit,behavior)"] = Math.max(scores.edit, scores.behavior);
      out.push({ task, block: blocks[0]!, scores });
    }
  }
  return out;
}

async function report() {
  console.log(`tasks: ${tasks.length}, blocks: ${tasks.reduce((n, t) => n + t.blocks.length, 0)}`);
  const lexical = tasks.flatMap((task) => task.blocks.map((block) => ({ block, hit: namesSymbol(task.problem_statement, block) })));
  const rate = (label: string) => {
    const rows = lexical.filter((row) => row.block.label === label);
    return rows.filter((row) => row.hit).length / Math.max(1, rows.length);
  };
  console.log(
    `code-side "statement names the symbol": edit ${pct(rate("edit"))}, test ${pct(rate("test"))}, negative ${pct(rate("negative"))}`,
  );
  console.log(
    "\nvariant    score                  AUC edit  AUC test  TPR@.5  FPR@.5  FPR bm25/sib/rand   top1=edit  prec>.5  tokens/task",
  );
  for (const variant of variants) {
    const rows = await scoredRows(variant);
    if (!rows) continue;
    const answers = await readJsonl<Answer>(join(dir, "answers", `${variant}.jsonl`));
    const answeredTasks = new Set(rows.map((row) => row.task.instance_id));
    const tokensPerTask =
      answers.reduce((sum, a) => sum + (a.inputTokens ?? a.estimatedTokens), 0) / Math.max(1, answeredTasks.size);
    for (const key of Object.keys(rows[0]!.scores)) {
      const withKey = rows.filter((row) => row.scores[key] !== undefined);
      const of = (label: string) => withKey.filter((row) => row.block.label === label).map((row) => row.scores[key]!);
      const negatives = withKey.filter((row) => row.block.label === "negative");
      const fpr = (rows: Scored[]) => rows.filter((row) => row.scores[key]! > 0.5).length / Math.max(1, rows.length);
      const byKind = (["bm25", "sibling", "random"] as const).map((kind) =>
        pct(fpr(negatives.filter((row) => row.block.negative_kind === kind))),
      );
      const perTask = Map.groupBy(withKey, (row) => row.task.instance_id);
      let top1 = 0;
      let kept = 0;
      let keptRelevant = 0;
      for (const taskRows of perTask.values()) {
        const best = taskRows.reduce((a, b) => (b.scores[key]! > a.scores[key]! ? b : a));
        if (best.block.label === "edit") top1++;
        for (const row of taskRows)
          if (row.scores[key]! > 0.5) {
            kept++;
            if (row.block.label !== "negative") keptRelevant++;
          }
      }
      const edits = of("edit");
      console.log(
        `${variant.padEnd(10)} ${key.padEnd(22)} ${pct(auc(edits, of("negative"))).padStart(8)}  ${pct(auc(of("test"), of("negative"))).padStart(8)}  ${pct(edits.filter((s) => s > 0.5).length / Math.max(1, edits.length)).padStart(6)}  ${pct(fpr(negatives)).padStart(6)}  ${byKind.join("/")}  ${pct(top1 / perTask.size).padStart(9)}  ${pct(kept ? keptRelevant / kept : NaN).padStart(7)}  ${Math.round(tokensPerTask).toLocaleString().padStart(11)}`,
      );
    }
  }
}

if (values.report) await report();
else {
  if (!variants.includes(values.variant as Variant))
    throw new Error(`--variant must be one of ${variants.join(", ")} (or pass --report)`);
  await execute(values.variant as Variant);
}
