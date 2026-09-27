#!/usr/bin/env -S node --experimental-strip-types
/**
 *   fast-context "Where does requests add the Content-Length header?" [root]
 *
 * Prints numbered snippets for the calling agent, or the full result with --json.
 * Reads CEREBRAS_API_KEY; --base-url and --model point it at any OpenAI-compatible endpoint.
 */
import { parseArgs } from "node:util";
import { defaultEndpoint, findContext, type Endpoint } from "./fast-context.ts";

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    json: { type: "boolean", default: false },
    model: { type: "string" },
    "base-url": { type: "string" },
    effort: { type: "string" },
    help: { type: "boolean", short: "h", default: false },
  },
});
if (values.help || !positionals[0] || positionals.length > 2) {
  console.error(
    `Usage: fast-context [--json] [--model ${defaultEndpoint.model}] [--effort low|none] [--base-url URL] "question" [root]`,
  );
  process.exit(values.help ? 0 : 2);
}
const result = await findContext({
  question: positionals[0],
  root: positionals[1] ?? ".",
  endpoint: {
    ...(values.model ? { model: values.model } : {}),
    ...(values["base-url"] ? { baseURL: values["base-url"] } : {}),
    ...(values.effort ? { reasoningEffort: values.effort as Endpoint["reasoningEffort"] } : {}),
  },
});
if (values.json) console.log(JSON.stringify(result, null, 2));
else {
  console.log(result.text || "No relevant code found.");
  const seconds = result.trace.reduce((sum, turn) => sum + turn.seconds, 0);
  console.error(
    `[${result.trace.length} turns, ${seconds.toFixed(1)}s, ${result.usage.inputTokens} in / ${result.usage.outputTokens} out, ~$${result.usage.usd.toFixed(4)}]`,
  );
}
