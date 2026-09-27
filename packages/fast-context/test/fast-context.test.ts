import { afterAll, beforeAll, expect, test } from "bun:test";
import { mkdtemp, mkdir, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { findContext, renderSnippets, type Completion, type Message } from "../src/fast-context";

let root: string;
let outside: string;
beforeAll(async () => {
  root = await mkdtemp(join(tmpdir(), "fast-context-"));
  outside = await mkdtemp(join(tmpdir(), "fast-context-outside-"));
  await mkdir(join(root, "pkg"));
  await writeFile(
    join(root, "pkg", "models.py"),
    Array.from({ length: 200 }, (_, i) =>
      i === 99 ? "def prepare_content_length(self, body):" : `# line ${i + 1}`,
    ).join("\n"),
  );
  await writeFile(join(root, "pkg", "other.py"), "def unrelated():\n    pass\n");
  await writeFile(join(outside, "secret.py"), "SECRET = 1\n");
  await symlink(join(outside, "secret.py"), join(root, "pkg", "link.py"));
});
afterAll(async () => {
  await rm(root, { recursive: true, force: true });
  await rm(outside, { recursive: true, force: true });
});

/** Replays one JSON reply per turn and records every conversation it was sent. */
function scripted(...replies: unknown[]) {
  const seen: Message[][] = [];
  const complete: Completion = async (messages) => {
    seen.push(structuredClone(messages));
    const reply = replies[seen.length - 1];
    return {
      content: typeof reply === "string" ? reply : JSON.stringify(reply),
      inputTokens: 100,
      outputTokens: 10,
      usd: 0.001,
    };
  };
  return { complete, seen };
}

test("runs a turn's actions in parallel and returns numbered snippets", async () => {
  const { complete, seen } = scripted(
    {
      actions: [
        { tool: "grep", pattern: "prepare_content_length" },
        { tool: "grep", pattern: "unrelated" },
        { tool: "list", path: "pkg" },
      ],
      answer: null,
    },
    {
      actions: [{ tool: "read", path: "pkg/models.py", start_line: 98, end_line: 102 }],
      answer: null,
    },
    { actions: [], answer: [{ path: "pkg/models.py", start_line: 100, end_line: 101 }] },
  );
  const result = await findContext({
    root,
    question: "Where is the content length set?",
    complete,
  });
  expect(result.status).toBe("answered");
  expect(result.trace.map((t) => t.actions)).toEqual([3, 1, 0]);
  const turn2 = seen[1]!.at(-1)!.content;
  expect(turn2).toContain("pkg/models.py:100:def prepare_content_length");
  expect(turn2).toContain("pkg/other.py:1:def unrelated");
  expect(turn2).toContain("models.py\nother.py");
  expect(seen[2]!.at(-1)!.content).toContain("100: def prepare_content_length(self, body):");
  expect(result.snippets).toEqual([
    {
      path: "pkg/models.py",
      startLine: 100,
      endLine: 101,
      code: "100: def prepare_content_length(self, body):\n101: # line 101",
    },
  ]);
  expect(result.text).toBe(renderSnippets(result.snippets));
  expect(result.text.startsWith("pkg/models.py:100-101\n100: ")).toBe(true);
  expect(result.usage).toEqual({ inputTokens: 300, outputTokens: 30, usd: 0.003 });
});

test("forces an answer on the last turn and drops ranges the model never saw", async () => {
  const { complete, seen } = scripted(
    { actions: [{ tool: "grep", pattern: "prepare" }], answer: null },
    { actions: [{ tool: "grep", pattern: "x" }], answer: null },
    { actions: [{ tool: "grep", pattern: "y" }], answer: null },
    {
      actions: [],
      answer: [
        { path: "pkg/other.py", start_line: 1, end_line: 2 },
        { path: "pkg/models.py", start_line: 100, end_line: 900 },
      ],
    },
  );
  const result = await findContext({ root, question: "q", complete });
  expect(seen).toHaveLength(4);
  expect(seen[3]!.at(-1)!.content).toContain("Turn 4 of 4. You must answer now.");
  // other.py never appeared in a result; models.py is clamped to the snippet limit and file end.
  expect(result.snippets.map((s) => [s.path, s.startLine, s.endLine])).toEqual([
    ["pkg/models.py", 100, 179],
  ]);
});

test("refuses paths outside the root, including through symlinks", async () => {
  const { complete, seen } = scripted(
    {
      actions: [
        { tool: "read", path: "../../etc/passwd", start_line: 1, end_line: 5 },
        { tool: "read", path: "pkg/link.py", start_line: 1, end_line: 5 },
      ],
      answer: null,
    },
    { actions: [], answer: [{ path: "pkg/link.py", start_line: 1, end_line: 1 }] },
  );
  const result = await findContext({ root, question: "q", complete });
  const results = seen[1]!.at(-1)!.content;
  expect(results.match(/error: /g)).toHaveLength(2);
  expect(results).not.toContain("SECRET");
  expect(result.status).toBe("no-answer");
});

test("tolerates fenced or invalid JSON replies", async () => {
  const { complete, seen } = scripted(
    "not json",
    '```json\n{"actions": [{"tool": "grep", "pattern": "unrelated"}], "answer": null}\n```',
    { actions: [], answer: [{ path: "pkg/other.py", start_line: 1, end_line: 2 }] },
  );
  const result = await findContext({ root, question: "q", complete });
  expect(seen[1]!.at(-1)!.content).toContain("not a single JSON object");
  expect(result.snippets[0]?.path).toBe("pkg/other.py");
});

test("shows hints to the model", async () => {
  const { complete, seen } = scripted(
    { actions: [], answer: [] },
    { actions: [], answer: [] },
    {},
    {},
  );
  await findContext({
    root,
    question: "q",
    hints: ["prepare_content_length  pkg/models.py:100-101"],
    complete,
  });
  expect(seen[0]![1]!.content).toContain("Leads from a local keyword search");
  expect(seen[0]![1]!.content).toContain("pkg/models.py:100-101");
});
