/**
 * Write the search question a coding agent would hand a code-search tool for each task.
 * The model sees only the issue text and the repository name, never the answer key.
 *
 *   bun evals/implementation/search-bench/make_queries.ts
 */
import { appendFile } from "node:fs/promises";
import { parseArgs } from "node:util";
import { chat, readJsonl, runDir, type Task } from "./common";

const { values } = parseArgs({
  options: { model: { type: "string", default: "openai/gpt-oss-120b" } },
});
const tasks = await readJsonl<Task>(`${runDir}/tasks.jsonl`);
const done = new Set(
  (await readJsonl<{ instance_id: string }>(`${runDir}/queries.jsonl`)).map((q) => q.instance_id),
);
const system = `You are a coding agent about to work on an issue in a repository you have not opened yet.
Before reading any files, you ask a code-search tool one question to find the code you need to read.
Write that question: one sentence in plain English that asks where specific behavior is implemented.
Use names that appear in the issue (functions, classes, options, error messages). Do not guess file paths or names the issue does not mention.
Do not describe the bug or the fix; ask where the relevant code lives.
Examples:
- "Where does requests build the Content-Length header when preparing a request body?"
- "Where does Django's migration autodetector decide to generate AlterField operations?"
Return JSON: {"query": string}`;
let spent = 0;
for (const task of tasks.filter((task) => !done.has(task.instance_id))) {
  const { content, cost } = await chat({
    model: values.model!,
    messages: [
      { role: "system", content: system },
      {
        role: "user",
        content: `Repository: ${task.repo}\n\nIssue:\n${task.problem_statement.slice(0, 8000)}`,
      },
    ],
    response_format: { type: "json_object" },
  });
  spent += cost;
  const query = (JSON.parse(content ?? "{}") as { query?: unknown }).query;
  if (typeof query !== "string") throw new Error(`${task.instance_id}: no query in ${content}`);
  await appendFile(
    `${runDir}/queries.jsonl`,
    JSON.stringify({ instance_id: task.instance_id, query }) + "\n",
  );
  console.log(`${task.instance_id}: ${query}`);
}
console.log(`OpenRouter spend: $${spent.toFixed(4)}`);
