/**
 * Write facets.jsonl for the `facets` variant: a generative model turns each problem
 * statement into one literal behavior (and optional test) description. It sees only
 * the problem statement, never the gold patch or the candidate blocks.
 *
 *   OPENROUTER_API_KEY=... bun evals/implementation/retrieval-pairs/make_facets.ts --live
 */
import { appendFile, readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { parseArgs } from "node:util";

const { values } = parseArgs({
  options: {
    live: { type: "boolean", default: false },
    model: { type: "string", default: "openai/gpt-oss-120b" },
    dir: { type: "string", default: "evals/runs/retrieval-pairs" },
  },
});
const dir = resolve(values.dir!);
const lines = async (path: string) =>
  existsSync(path) ? (await readFile(path, "utf8")).split("\n").filter(Boolean).map((l) => JSON.parse(l)) : [];
const tasks: Array<{ instance_id: string; problem_statement: string }> = await lines(join(dir, "pairs.jsonl"));
const done = new Set((await lines(join(dir, "facets.jsonl"))).map((row) => row.instance_id));
const pending = tasks.filter((task) => !done.has(task.instance_id));
console.log(`${pending.length} of ${tasks.length} tasks need facets (model ${values.model})`);
if (!values.live) process.exit(console.log("Dry run. Add --live to call OpenRouter.") ?? 0);
const key = process.env.OPENROUTER_API_KEY;
if (!key) throw new Error("Set OPENROUTER_API_KEY");

const instructions = `You turn a bug report into search descriptions for a code classifier that reads one function at a time and interprets text literally.
Return JSON: {"behavior": string, "test": string}.
- behavior: one sentence naming the single concrete computation or action that is wrong, in the form "<verb phrase> <object>", e.g. "computes the Content-Length header for a prepared request". Name the function, class, or module if the report names it. No negations, no "or", no speculation about the fix.
- test: one sentence describing an existing test that would check that behavior, e.g. "a test that prepares a GET request and asserts its headers".`;

for (const task of pending) {
  const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      model: values.model,
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: instructions },
        { role: "user", content: task.problem_statement.slice(0, 8000) },
      ],
    }),
  });
  if (!response.ok) throw new Error(`${task.instance_id}: HTTP ${response.status} ${await response.text()}`);
  const body = (await response.json()) as { choices: Array<{ message: { content: string } }> };
  const facets = JSON.parse(body.choices[0]!.message.content) as { behavior?: unknown; test?: unknown };
  if (typeof facets.behavior !== "string") throw new Error(`${task.instance_id}: no behavior in response`);
  await appendFile(
    join(dir, "facets.jsonl"),
    JSON.stringify({
      instance_id: task.instance_id,
      behavior: facets.behavior,
      ...(typeof facets.test === "string" ? { test: facets.test } : {}),
    }) + "\n",
  );
  console.log(`${task.instance_id}: ${facets.behavior}`);
}
