/**
 * Experimental request builders that follow TypeSafe's jev-1.13 guidance
 * (docs.typesafe.ai/model-jaggedness/jev-1.13). Not wired into retrieve() yet;
 * evals/implementation/retrieval-pairs compares them with the current requests.
 *
 * - Small state: one task and the block(s) being judged, nothing else (#5).
 * - Name the state field each question reads, e.g. `block.source` (#4).
 * - One literal judgment per question, with boundary cases in criteria (#1).
 * - No line numbers or counting asked of the model; blocks arrive pre-cut (#2).
 * - Lexical facts code can compute (does the task name this symbol?) stay in code.
 * - Choice ranks a shortlist; separate Nouls decide absolute relevance (#8).
 */

export type Block = { path: string; name: string; source: string };
export type Facets = {
  /** One concrete behavior, e.g. "computes the Content-Length header for a request". */
  behavior: string;
  /** Optional description of a test that would exercise the behavior. */
  test?: string;
};
type Text = string;
export type LiteralQuestion =
  | {
      type: "boolean";
      instructions: Text;
      criteria?: { true?: Text; false?: Text };
    }
  | { type: "choice"; instructions: Text; criteria: Record<string, Text | null> };
export type LiteralRequest = {
  state: Record<string, unknown>;
  questions: Record<string, LiteralQuestion>;
};

const blockState = (block: Block) => ({ path: block.path, name: block.name, source: block.source });

/** Judgments available for one block against a free-text task description. */
export const taskJudgments = {
  edit: {
    type: "boolean",
    instructions:
      "Would a fix for the problem described in `task.description` change at least one line of `block.source`?",
    criteria: {
      true: "The fix edits, adds, or removes a line inside `block.source`.",
      false:
        "The fix is made in other code. `block.source` stays unchanged even if it calls, tests, or resembles the changed code.",
    },
  },
  behavior: {
    type: "boolean",
    instructions:
      "Does `block.source` contain the statements that produce the behavior described in `task.description`?",
    criteria: {
      true: "The statements in `block.source` compute the value or perform the action that `task.description` describes.",
      false:
        "`block.source` only calls that code, wraps it, tests it, or uses the same words for something else.",
    },
  },
  test: {
    type: "boolean",
    instructions:
      "Is `block.source` a test that exercises the behavior described in `task.description`?",
    criteria: {
      true: "`block.source` is a test function or test class, and its assertions check that behavior.",
      false: "`block.source` is not a test, or it tests a different behavior.",
    },
  },
} as const satisfies Record<string, LiteralQuestion>;
export type TaskJudgment = keyof typeof taskJudgments;

/** One block, one task, and only the judgments asked for. Output is free, so asking several shares the input cost. */
export function blockRequest(
  description: string,
  block: Block,
  judgments: readonly TaskJudgment[] = ["edit", "behavior", "test"],
): LiteralRequest {
  return {
    state: { task: { description }, block: blockState(block) },
    questions: Object.fromEntries(judgments.map((id) => [id, taskJudgments[id]])),
  };
}

/**
 * Facets move the indirection ("where would this bug be fixed?") to the caller,
 * which is a generative model. Jev only checks a literal description against a block.
 */
export function facetRequest(facets: Facets, block: Block): LiteralRequest {
  const questions: Record<string, LiteralQuestion> = {
    behavior: {
      type: "boolean",
      instructions: "Does `block.source` contain the statements that perform `facet.behavior`?",
      criteria: {
        true: "The statements in `block.source` carry out `facet.behavior` themselves.",
        false:
          "`block.source` only calls, wraps, or configures that code, or uses the same words for something else.",
      },
    },
  };
  if (facets.test !== undefined)
    questions.test = {
      type: "boolean",
      instructions: "Is `block.source` a test that matches `facet.test`?",
      criteria: {
        true: "`block.source` is a test function or test class, and its assertions check what `facet.test` describes.",
        false: "`block.source` is not a test, or it checks something else.",
      },
    };
  return { state: { facet: facets, block: blockState(block) }, questions };
}

/** Choice supports at most 255 options; one is reserved for "none". */
export const maxShortlist = 254;

/**
 * Relative ranking over a shortlist, plus an absolute gate in the same request.
 * The Choice picks which candidate is most likely; the Noul decides whether any is.
 * Their probabilities are not comparable to each other or to per-block Nouls.
 */
export function shortlistRequest(description: string, blocks: Block[]): LiteralRequest {
  if (!blocks.length || blocks.length > maxShortlist)
    throw new RangeError(`Shortlist size must be 1-${maxShortlist}`);
  const candidates = Object.fromEntries(blocks.map((block, i) => [`c${i}`, blockState(block)]));
  return {
    state: { task: { description }, candidates },
    questions: {
      pick: {
        type: "choice",
        instructions:
          "Which entry of `candidates` would a fix for the problem described in `task.description` change?",
        criteria: {
          ...Object.fromEntries(
            blocks.map((block, i) => [
              `c${i}`,
              `\`candidates.c${i}\`: ${block.name} in ${block.path}`,
            ]),
          ),
          none: "The fix changes none of the entries in `candidates`.",
        },
      },
      any: {
        type: "boolean",
        instructions:
          "Would a fix for the problem described in `task.description` change at least one entry of `candidates`?",
      },
    },
  };
}

/** Lexical evidence computed in code instead of asked of the model. */
export function namesSymbol(description: string, block: Block) {
  const leaf = block.name.split(".").at(-1)!;
  return (
    leaf.length > 2 &&
    new RegExp(`\\b${leaf.replace(/[$^\\.*+?()[\]{}|]/g, "\\$&")}\\b`).test(description)
  );
}
