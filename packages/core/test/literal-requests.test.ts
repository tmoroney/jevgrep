import { expect, test } from "bun:test";
import {
  blockRequest,
  facetRequest,
  maxShortlist,
  namesSymbol,
  shortlistRequest,
  type Block,
} from "../src/literal-requests";

const block: Block = {
  path: "requests/models.py",
  name: "PreparedRequest.prepare_content_length",
  source: "def prepare_content_length(self, body):\n    self.headers['Content-Length'] = '0'",
};
const description = "requests.get always sends a Content-Length header.";

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

test("block requests hold only the task and one block", () => {
  const request = blockRequest(description, block);
  expect(request.state).toEqual({ task: { description }, block });
  expect(Object.keys(request.questions)).toEqual(["edit", "behavior", "test"]);
  expect(Object.keys(blockRequest(description, block, ["edit"]).questions)).toEqual(["edit"]);
});

test("every question and criterion names state fields that exist", () => {
  const requests = [
    blockRequest(description, block),
    facetRequest(
      { behavior: "computes the Content-Length header", test: "a GET without a body" },
      block,
    ),
    shortlistRequest(description, [block, { ...block, name: "PreparedRequest.prepare_body" }]),
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

test("questions avoid asking the model to count lines", () => {
  const text = JSON.stringify(blockRequest(description, block).questions);
  expect(text).not.toMatch(/lines? \d/);
});

test("facet test question is optional", () => {
  expect(Object.keys(facetRequest({ behavior: "x" }, block).questions)).toEqual(["behavior"]);
});

test("shortlist offers one option per block plus none, within the Choice limit", () => {
  const blocks = [block, { ...block, name: "Session.send" }];
  const pick = shortlistRequest(description, blocks).questions.pick!;
  expect(pick.type).toBe("choice");
  expect(Object.keys(pick.criteria!)).toEqual(["c0", "c1", "none"]);
  expect(() => shortlistRequest(description, [])).toThrow(RangeError);
  expect(() =>
    shortlistRequest(
      description,
      Array.from({ length: maxShortlist + 1 }, () => block),
    ),
  ).toThrow(RangeError);
});

test("symbol mentions are computed in code on whole identifiers", () => {
  expect(namesSymbol("prepare_content_length is wrong", block)).toBe(true);
  expect(namesSymbol("prepare_content_length_v2 is wrong", block)).toBe(false);
  expect(namesSymbol(description, block)).toBe(false);
  expect(namesSymbol("uses a.b", { ...block, name: "X.a" })).toBe(false);
});
