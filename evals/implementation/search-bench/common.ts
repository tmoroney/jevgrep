import { existsSync } from "node:fs";
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

export const runDir = resolve(import.meta.dir, "../../runs/search-bench");
export type Gold = {
  path: string;
  name: string;
  start_line: number;
  end_line: number;
  label: "edit" | "test";
};
export type Task = {
  instance_id: string;
  repo: string;
  base_commit: string;
  problem_statement: string;
  gold: Gold[];
};

export async function readJsonl<T>(path: string): Promise<T[]> {
  if (!existsSync(path)) return [];
  return (await readFile(path, "utf8"))
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line) as T);
}

export type ChatMessage = {
  role: "system" | "user" | "assistant" | "tool";
  content: string | null;
  tool_calls?: Array<{ id: string; type: "function"; function: { name: string; arguments: string } }>;
  tool_call_id?: string;
};

/**
 * One OpenAI-compatible chat call. `cerebras/<model>` goes straight to Cerebras with
 * CEREBRAS_API_KEY; anything else goes to OpenRouter, pinned to Cerebras hardware
 * when OpenRouter offers it for that model.
 */
export async function chat(body: Record<string, unknown> & { model: string }) {
  const direct = body.model.startsWith("cerebras/");
  const url = direct
    ? "https://api.cerebras.ai/v1/chat/completions"
    : "https://openrouter.ai/api/v1/chat/completions";
  const key = direct ? process.env.CEREBRAS_API_KEY : process.env.OPENROUTER_API_KEY;
  if (!key) throw new Error(direct ? "Set CEREBRAS_API_KEY" : "Set OPENROUTER_API_KEY");
  const request = direct
    ? { ...body, model: body.model.slice("cerebras/".length) }
    : {
        ...body,
        usage: { include: true },
        ...(cerebrasOnOpenRouter.has(body.model)
          ? { provider: { order: ["Cerebras"], allow_fallbacks: false } }
          : {}),
      };
  for (let attempt = 0; ; attempt++) {
    const response = await fetch(url, {
      method: "POST",
      headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
      body: JSON.stringify(request),
      signal: AbortSignal.timeout(120_000),
    });
    if ((response.status === 429 || response.status >= 500) && attempt < 3) {
      await new Promise((done) => setTimeout(done, 1000 * 2 ** attempt));
      continue;
    }
    if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    const data = (await response.json()) as {
      choices: Array<{ message: ChatMessage }>;
      usage?: { prompt_tokens?: number; completion_tokens?: number; cost?: number };
    };
    const usage = data.usage ?? {};
    const cost =
      usage.cost ??
      ((usage.prompt_tokens ?? 0) * (directPrices[request.model as string]?.[0] ?? 0) +
        (usage.completion_tokens ?? 0) * (directPrices[request.model as string]?.[1] ?? 0)) /
        1e6;
    const message = data.choices[0]!.message;
    return {
      message,
      content: message.content,
      cost,
      inputTokens: usage.prompt_tokens ?? 0,
      outputTokens: usage.completion_tokens ?? 0,
    };
  }
}
const cerebrasOnOpenRouter = new Set(["openai/gpt-oss-120b"]);
/** USD per million input/output tokens for direct Cerebras calls, which report no cost. */
const directPrices: Record<string, [number, number]> = {
  "qwen-3.8-27b": [0.99, 1.49],
  "gpt-oss-120b": [0.35, 0.75],
};

/** Jev spend persists across runs so a budget holds for the whole study. */
export const jevUsdPerToken = 0.042 / 1e6;
export async function jevLedger(capUsd: number) {
  const path = `${runDir}/jev-ledger.json`;
  const state = existsSync(path)
    ? (JSON.parse(await readFile(path, "utf8")) as { tokens: number })
    : { tokens: 0 };
  return {
    get usd() {
      return state.tokens * jevUsdPerToken;
    },
    remaining() {
      return capUsd - state.tokens * jevUsdPerToken;
    },
    async add(tokens: number) {
      state.tokens += tokens;
      await writeFile(path, JSON.stringify(state));
    },
  };
}
