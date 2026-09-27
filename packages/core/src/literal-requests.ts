/**
 * Experimental Jev request builders for re-ranking code candidates against a search
 * query. They follow TypeSafe's jev-1.13 guidance (docs.typesafe.ai/model-jaggedness/jev-1.13)
 * and its re-ranking and line-search cookbooks. Not wired into retrieve() yet;
 * evals/implementation/search-bench compares them with the current pipeline.
 *
 * - Small state: the query and the candidate(s) being judged, nothing else (#5).
 * - Questions name the state field they read, e.g. `candidate.source` (#4).
 * - One literal judgment per question, with boundary cases in criteria (#1).
 * - No line numbers or counting asked of the model; candidates arrive pre-cut (#2).
 * - Lexical facts code can compute (does the query name this symbol?) stay in code.
 * - A Choice ranks a shortlist; a separate Noul decides absolute relevance (#8).
 */

export type Candidate = { path: string; name: string; source: string };
type Text = string;
export type LiteralQuestion =
  | { type: "boolean"; instructions: Text; criteria?: { true?: Text; false?: Text } }
  | { type: "choice"; instructions: Text; criteria: Record<string, Text | null> };
export type LiteralRequest = {
  state: Record<string, unknown>;
  questions: Record<string, LiteralQuestion>;
};

const candidateState = (candidate: Candidate) => ({
  path: candidate.path,
  name: candidate.name,
  source: candidate.source,
});

export const matchQuestion = {
  type: "boolean",
  instructions: "Is `candidate.source` the code that `search.query` asks to find?",
  criteria: {
    true: "`candidate.source` defines, performs, or directly tests the behavior that `search.query` asks about.",
    false:
      "`candidate.source` only calls that code, shares words with `search.query`, or handles a different case.",
  },
} as const satisfies LiteralQuestion;

/** One query and one candidate: the re-ranking cookbook's pair request. Sort candidates by its noul. */
export function candidateRequest(query: string, candidate: Candidate): LiteralRequest {
  return {
    state: { search: { query }, candidate: candidateState(candidate) },
    questions: { match: matchQuestion },
  };
}

/** Choice supports at most 255 options; one is reserved for "none". */
export const maxShortlist = 254;

/**
 * Relative ranking over a shortlist plus an absolute gate in the same request, as in the
 * line-search cookbook. Choice probabilities sum to one, so the Noul decides whether any
 * candidate is relevant; neither is comparable to the per-candidate noul.
 */
export function shortlistRequest(query: string, candidates: Candidate[]): LiteralRequest {
  if (!candidates.length || candidates.length > maxShortlist)
    throw new RangeError(`Shortlist size must be 1-${maxShortlist}`);
  return {
    state: {
      search: { query },
      candidates: Object.fromEntries(candidates.map((c, i) => [`c${i}`, candidateState(c)])),
    },
    questions: {
      pick: {
        type: "choice",
        instructions: "Which entry of `candidates` is the code that `search.query` asks to find?",
        criteria: {
          ...Object.fromEntries(candidates.map((_, i) => [`c${i}`, null])),
          none: "No entry of `candidates` is the code that `search.query` asks to find.",
        },
      },
      any: {
        type: "boolean",
        instructions: "Is any entry of `candidates` the code that `search.query` asks to find?",
        criteria: {
          true: "At least one entry defines, performs, or directly tests what `search.query` asks about.",
          false: "Entries only call that code, share words with the query, or handle other cases.",
        },
      },
    },
  };
}

/** Lexical evidence computed in code instead of asked of the model. */
export function namesSymbol(query: string, candidate: Candidate) {
  const leaf = candidate.name.split(".").at(-1)!;
  return (
    leaf.length > 2 &&
    new RegExp(`\\b${leaf.replace(/[$^\\.*+?()[\]{}|]/g, "\\$&")}\\b`).test(query)
  );
}
