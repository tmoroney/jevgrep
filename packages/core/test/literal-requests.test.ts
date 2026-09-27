import { expect, test } from "bun:test";
import {
  candidateRequest,
  maxShortlist,
  namesSymbol,
  shortlistRequest,
  type Candidate,
} from "../src/literal-requests";

const candidate: Candidate = {
  path: "requests/models.py",
  name: "PreparedRequest.prepare_content_length",
  source: "def prepare_content_length(self, body):\n    self.headers['Content-Length'] = '0'",
};
const query = "Where does requests add the Content-Length header when preparing a GET request?";

function referencedFields(text: string) {
  return [...text.matchAll(/`([a-z]+(?:\.[a-z0-9]+)+)`/g)].map((match) => match[1]!);
}

function resolves(state: Record<string, unknown>, field: string) {
  let value: unknown = state;
  for (const part of field.split(".")) {
    if (!value || typeof value !== "object" || !(part in value)) return false;
    value = (value as Record<string, unknown>)[part];
  }
  return true;
}

test("candidate requests hold only the query and one candidate", () => {
  const request = candidateRequest(query, candidate);
  expect(request.state).toEqual({ search: { query }, candidate });
  expect(Object.keys(request.questions)).toEqual(["match"]);
});

test("every question and criterion names state fields that exist", () => {
  const requests = [
    candidateRequest(query, candidate),
    shortlistRequest(query, [candidate, { ...candidate, name: "PreparedRequest.prepare_body" }]),
  ];
  for (const request of requests)
    for (const question of Object.values(request.questions)) {
      const texts = [
        question.instructions,
        ...Object.values(question.criteria ?? {}).filter(
          (text): text is string => typeof text === "string",
        ),
      ];
      const fields = texts.flatMap(referencedFields);
      expect(fields.length).toBeGreaterThan(0);
      for (const field of fields) expect(resolves(request.state, field)).toBe(true);
    }
});

test("questions never ask the model to locate line numbers", () => {
  expect(JSON.stringify(candidateRequest(query, candidate).questions)).not.toMatch(/lines? \d/);
});

test("shortlist offers one option per candidate plus none, within the Choice limit", () => {
  const pick = shortlistRequest(query, [candidate, { ...candidate, name: "Session.send" }])
    .questions.pick!;
  expect(pick.type).toBe("choice");
  expect(Object.keys(pick.criteria!)).toEqual(["c0", "c1", "none"]);
  expect(() => shortlistRequest(query, [])).toThrow(RangeError);
  expect(() =>
    shortlistRequest(
      query,
      Array.from({ length: maxShortlist + 1 }, () => candidate),
    ),
  ).toThrow(RangeError);
});

test("symbol mentions are computed in code on whole identifiers", () => {
  expect(namesSymbol("where is prepare_content_length", candidate)).toBe(true);
  expect(namesSymbol("where is prepare_content_length_v2", candidate)).toBe(false);
  expect(namesSymbol(query, candidate)).toBe(false);
  expect(namesSymbol("uses a.b", { ...candidate, name: "X.a" })).toBe(false);
});
