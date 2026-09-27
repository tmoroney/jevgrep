/**
 * Fast-context agent: answers a natural-language question about a repository with
 * exact file line ranges, so a slower, pricier coding model can skip the search.
 *
 * Design (measured in evals/implementation/search-bench):
 * - A fast model (default Qwen 3.8 27B on Cerebras, reasoning "low") plans each turn as
 *   one JSON batch of grep/read/list actions. The harness runs the batch in parallel.
 *   Untrained models barely parallelize through native tool calling, so the harness,
 *   not the model, enforces the fan-out that SWE-grep learned through RL.
 * - At most 4 turns: search, read, optional follow-up, then a forced answer.
 * - Every line the model sees and every line returned carries its line number, so
 *   ranges are copied, never counted.
 * - Answers are checked in code: paths must exist inside the root and have appeared
 *   in a tool result; ranges are clamped to the file and to the snippet size limit.
 *
 * Talks to any OpenAI-compatible chat completions endpoint. No dependencies beyond
 * Node; uses ripgrep when installed and falls back to grep.
 */
import { execFile } from "node:child_process";
import { readFile, readdir, realpath, stat } from "node:fs/promises";
import { isAbsolute, relative, resolve, sep } from "node:path";

export type Snippet = { path: string; startLine: number; endLine: number; code: string };
export type Message = { role: "system" | "user" | "assistant"; content: string };
export type Completion = (
  messages: Message[],
  signal?: AbortSignal,
) => Promise<{ content: string; inputTokens: number; outputTokens: number; usd?: number }>;
export type TurnTrace = {
  actions: number;
  toolOutputChars: number;
  inputTokens: number;
  outputTokens: number;
  seconds: number;
};
export type FastContextResult = {
  status: "answered" | "no-answer";
  snippets: Snippet[];
  /** Snippets rendered for the calling model: `path:start-end` headers, numbered lines. */
  text: string;
  usage: { inputTokens: number; outputTokens: number; usd: number };
  trace: TurnTrace[];
};
export type FastContextOptions = {
  root: string;
  question: string;
  /** Optional leads from a cheap local search, shown to the model as unverified hints. */
  hints?: string[];
  /** Replaces the HTTP client, e.g. for tests. */
  complete?: Completion;
  endpoint?: Partial<Endpoint>;
  limits?: Partial<typeof defaultLimits>;
  signal?: AbortSignal;
};
export type Endpoint = {
  baseURL: string;
  apiKey: string;
  model: string;
  reasoningEffort: "none" | "low" | "medium" | "high";
  /** USD per million input and output tokens, used when the response carries no cost. */
  price: { input: number; output: number };
  /** Extra request fields, e.g. OpenRouter provider routing. */
  extraBody?: Record<string, unknown>;
};

export const defaultEndpoint: Omit<Endpoint, "apiKey"> = {
  baseURL: "https://api.cerebras.ai/v1",
  model: "qwen-3.8-27b",
  reasoningEffort: "low",
  price: { input: 0.99, output: 1.49 },
};
export const defaultLimits = {
  turns: 4,
  actionsPerTurn: 8,
  grepLines: 40,
  grepMatchesPerFile: 5,
  grepColumns: 200,
  readLines: 120,
  listEntries: 100,
  snippets: 6,
  snippetLines: 80,
};

export const systemPrompt = (limits: typeof defaultLimits) =>
  `You are a fast code-search subagent. Another coding agent asked you a question about this repository; find the code it needs and return exact line ranges.
You work in at most ${limits.turns} turns. Every turn, reply with one JSON object and nothing else:
{"actions": [ ... ], "answer": null}   or   {"actions": [], "answer": [{"path": "...", "start_line": 1, "end_line": 40}]}
Actions run in parallel, so batch them. Each action is one of:
  {"tool": "grep", "pattern": "<regex>", "glob": "<optional glob such as *.py>"}   -> up to ${limits.grepLines} matches as path:line:text
  {"tool": "read", "path": "<file>", "start_line": N, "end_line": M}             -> up to ${limits.readLines} lines as line: text
  {"tool": "list", "path": "<directory>"}                                       -> directory entries
Turn 1: issue 6 to ${limits.actionsPerTurn} greps at once, covering different names, spellings and phrases from the question (function names, class names, option strings, error text).
Turn 2: issue up to ${limits.actionsPerTurn} actions, mostly reads of the most promising grep hits, plus greps for names you discovered.
Turn 3: answer if you have found the code, otherwise up to ${limits.actionsPerTurn} more actions.
Turn ${limits.turns}: you must answer.
Every result line starts with its line number; copy line numbers from results, never count lines yourself.
The answer lists at most ${limits.snippets} ranges, most relevant first, each at most ${limits.snippetLines} lines: the definitions that implement what the question asks about, and tests of it. Precision matters more than recall; every extra range costs the other agent context. Never answer with a range you have not seen in a read or grep result.`;

type Action = {
  tool?: unknown;
  pattern?: unknown;
  glob?: unknown;
  path?: unknown;
  start_line?: unknown;
  end_line?: unknown;
};
type Plan = {
  actions?: Action[];
  answer?: Array<{ path?: unknown; start_line?: unknown; end_line?: unknown }> | null;
};

export async function findContext(options: FastContextOptions): Promise<FastContextResult> {
  const limits = { ...defaultLimits, ...options.limits };
  const root = await realpath(resolve(options.root));
  const complete = options.complete ?? httpCompletion(options.endpoint);
  const tools = createTools(root, limits);
  const seen = new Set<string>();
  const usage = { inputTokens: 0, outputTokens: 0, usd: 0 };
  const trace: TurnTrace[] = [];

  let intro = `Repository root listing:\n${await tools.list(".")}`;
  if (options.hints?.length)
    intro += `\n\nLeads from a local keyword search, which may or may not be relevant:\n${options.hints.join("\n")}`;
  const messages: Message[] = [
    { role: "system", content: systemPrompt(limits) },
    {
      role: "user",
      content: `${intro}\n\nQuestion: ${options.question}\n\nTurn 1 of ${limits.turns}.`,
    },
  ];

  for (let turn = 1; turn <= limits.turns; turn++) {
    const started = performance.now();
    const reply = await complete(messages, options.signal);
    usage.inputTokens += reply.inputTokens;
    usage.outputTokens += reply.outputTokens;
    usage.usd += reply.usd ?? 0;
    messages.push({ role: "assistant", content: reply.content });
    const plan = parsePlan(reply.content);

    if (plan?.answer?.length || turn === limits.turns) {
      trace.push({
        actions: 0,
        toolOutputChars: 0,
        inputTokens: reply.inputTokens,
        outputTokens: reply.outputTokens,
        seconds: (performance.now() - started) / 1000,
      });
      const snippets = await tools.snippets(plan?.answer ?? [], seen);
      return {
        status: snippets.length ? "answered" : "no-answer",
        snippets,
        text: renderSnippets(snippets),
        usage,
        trace,
      };
    }

    const actions = (plan?.actions ?? []).slice(0, limits.actionsPerTurn);
    const outputs = await Promise.all(actions.map((action) => tools.run(action, seen)));
    trace.push({
      actions: actions.length,
      toolOutputChars: outputs.reduce((n, output) => n + output.length, 0),
      inputTokens: reply.inputTokens,
      outputTokens: reply.outputTokens,
      seconds: (performance.now() - started) / 1000,
    });
    const next = turn + 1;
    const results = plan
      ? actions.length
        ? actions
            .map((action, i) => `### ${i + 1}. ${JSON.stringify(action)}\n${outputs[i]}`)
            .join("\n\n")
        : "No actions were given."
      : "Your reply was not a single JSON object.";
    messages.push({
      role: "user",
      content: `${results}\n\nTurn ${next} of ${limits.turns}.${next === limits.turns ? " You must answer now." : ""}`,
    });
  }
  return { status: "no-answer", snippets: [], text: "", usage, trace };
}

/** Numbered lines under one `path:start-end` header per snippet. */
export function renderSnippets(snippets: Snippet[]) {
  return snippets.map((s) => `${s.path}:${s.startLine}-${s.endLine}\n${s.code}`).join("\n\n");
}

const numbered = (lines: string[], first: number) =>
  lines.map((line, i) => `${first + i}: ${line}`).join("\n");

function parsePlan(content: string): Plan | undefined {
  // Models sometimes wrap JSON in a code fence or add a sentence around it.
  const start = content.indexOf("{");
  const end = content.lastIndexOf("}");
  if (start < 0 || end < start) return;
  try {
    const value: unknown = JSON.parse(content.slice(start, end + 1));
    return value && typeof value === "object" ? (value as Plan) : undefined;
  } catch {
    return;
  }
}

function createTools(root: string, limits: typeof defaultLimits) {
  /** Resolves a repository-relative path, refusing anything outside the root (including via symlinks). */
  async function inside(path: string) {
    const full = resolve(root, path.replace(/^\.\//, ""));
    const real = await realpath(full);
    const rel = relative(root, real);
    if (rel === ".." || rel.startsWith(`..${sep}`) || isAbsolute(rel))
      throw new Error("path is outside the repository");
    return { full: real, rel: rel || "." };
  }

  async function grep(pattern: string, glob?: string) {
    const rg = [
      "-n",
      "--no-heading",
      "--color",
      "never",
      "-S",
      "--max-count",
      String(limits.grepMatchesPerFile),
      "--max-columns",
      String(limits.grepColumns),
      ...(glob ? ["-g", glob] : []),
      "-e",
      pattern,
      ".",
    ];
    let out = await exec("rg", rg, root);
    if (out.missing) {
      const include = glob && !glob.includes("/") ? [`--include=${glob}`] : [];
      out = await exec(
        "grep",
        [
          "-rnIE",
          "--exclude-dir=.git",
          "--exclude-dir=node_modules",
          ...include,
          "-m",
          String(limits.grepMatchesPerFile),
          "-e",
          pattern,
          ".",
        ],
        root,
      );
    }
    const lines = out.stdout
      .split("\n")
      .filter(Boolean)
      .map((line) => line.replace(/^\.\//, "").slice(0, limits.grepColumns + 200));
    if (!lines.length) return out.stderr.trim().slice(0, 300) || "no matches";
    const shown = lines.slice(0, limits.grepLines).join("\n");
    return lines.length > limits.grepLines
      ? `${shown}\n[${lines.length - limits.grepLines} more matches]`
      : shown;
  }

  async function read(path: string, startLine: number, endLine: number) {
    const lines = (await readFile((await inside(path)).full, "utf8")).split("\n");
    const start = Math.max(1, Math.floor(startLine) || 1);
    const end = Math.min(lines.length, Math.floor(endLine) || start, start + limits.readLines - 1);
    if (start > lines.length) return `file has ${lines.length} lines`;
    return numbered(lines.slice(start - 1, end), start);
  }

  async function list(path: string) {
    const entries = await readdir((await inside(path)).full, { withFileTypes: true });
    const names = entries
      .filter((e) => e.name !== ".git")
      .map((e) => (e.isDirectory() ? `${e.name}/` : e.name))
      .sort();
    return (
      names.slice(0, limits.listEntries).join("\n") +
      (names.length > limits.listEntries ? `\n[${names.length - limits.listEntries} more]` : "")
    );
  }

  return {
    list,
    async run(action: Action, seen: Set<string>) {
      try {
        if (action.tool === "grep" && typeof action.pattern === "string") {
          const output = await grep(
            action.pattern,
            typeof action.glob === "string" ? action.glob : undefined,
          );
          for (const line of output.split("\n")) {
            const match = /^(.+?):\d+:/.exec(line);
            if (match) seen.add(match[1]!);
          }
          return output;
        }
        if (action.tool === "read" && typeof action.path === "string") {
          const output = await read(
            action.path,
            Number(action.start_line),
            Number(action.end_line),
          );
          seen.add((await inside(action.path)).rel);
          return output;
        }
        if (action.tool === "list")
          return await list(typeof action.path === "string" ? action.path : ".");
        return `unsupported action ${JSON.stringify(action)}`;
      } catch (error) {
        return `error: ${error instanceof Error ? error.message : String(error)}`;
      }
    },
    /** Keeps answer ranges that name real files already seen in results; clamps sizes. */
    async snippets(answer: NonNullable<Plan["answer"]>, seen: Set<string>) {
      const snippets: Snippet[] = [];
      for (const item of answer) {
        if (snippets.length >= limits.snippets) break;
        if (typeof item.path !== "string") continue;
        try {
          const { full, rel } = await inside(item.path);
          if (!seen.has(rel) || !(await stat(full)).isFile()) continue;
          const lines = (await readFile(full, "utf8")).split("\n");
          const startLine = Math.min(
            lines.length,
            Math.max(1, Math.floor(Number(item.start_line)) || 1),
          );
          const requested = Math.floor(Number(item.end_line)) || startLine;
          const endLine = Math.min(
            lines.length,
            Math.max(startLine, requested),
            startLine + limits.snippetLines - 1,
          );
          snippets.push({
            path: rel,
            startLine,
            endLine,
            code: numbered(lines.slice(startLine - 1, endLine), startLine),
          });
        } catch {
          continue;
        }
      }
      return snippets;
    },
  };
}

function exec(command: string, args: string[], cwd: string) {
  return new Promise<{ stdout: string; stderr: string; missing: boolean }>((done) => {
    execFile(
      command,
      args,
      { cwd, maxBuffer: 1 << 26, timeout: 20_000 },
      (error, stdout, stderr) => {
        const missing = (error as NodeJS.ErrnoException | null)?.code === "ENOENT";
        done({ stdout: String(stdout ?? ""), stderr: String(stderr ?? ""), missing });
      },
    );
  });
}

/** OpenAI-compatible chat completions with JSON output and retries on 429/5xx. */
export function httpCompletion(overrides: Partial<Endpoint> = {}): Completion {
  const endpoint = { ...defaultEndpoint, ...overrides };
  const apiKey = overrides.apiKey ?? process.env.CEREBRAS_API_KEY;
  if (!apiKey) throw new Error("Set CEREBRAS_API_KEY or pass endpoint.apiKey");
  return async (messages, signal) => {
    for (let attempt = 0; ; attempt++) {
      const response = await fetch(`${endpoint.baseURL.replace(/\/+$/, "")}/chat/completions`, {
        method: "POST",
        headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
        body: JSON.stringify({
          model: endpoint.model,
          messages,
          response_format: { type: "json_object" },
          reasoning_effort: endpoint.reasoningEffort,
          max_tokens: 4000,
          ...endpoint.extraBody,
        }),
        signal: signal
          ? AbortSignal.any([signal, AbortSignal.timeout(60_000)])
          : AbortSignal.timeout(60_000),
      });
      if ((response.status === 429 || response.status >= 500) && attempt < 3) {
        await new Promise((wake) => setTimeout(wake, 500 * 2 ** attempt));
        continue;
      }
      if (!response.ok)
        throw new Error(`HTTP ${response.status}: ${(await response.text()).slice(0, 500)}`);
      const data = (await response.json()) as {
        choices?: Array<{ message?: { content?: string | null } }>;
        usage?: { prompt_tokens?: number; completion_tokens?: number; cost?: number };
      };
      const inputTokens = data.usage?.prompt_tokens ?? 0;
      const outputTokens = data.usage?.completion_tokens ?? 0;
      return {
        content: data.choices?.[0]?.message?.content ?? "",
        inputTokens,
        outputTokens,
        usd:
          data.usage?.cost ??
          (inputTokens * endpoint.price.input + outputTokens * endpoint.price.output) / 1e6,
      };
    }
  };
}
