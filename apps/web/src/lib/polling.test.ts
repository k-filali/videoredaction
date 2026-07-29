import { describe, expect, it } from "vitest";

import { processingPollDelay } from "./polling";

describe("processing polling cadence", () => {
  it("backs off from two to five seconds while status is unchanged", () => {
    expect([0, 1, 2, 3, 4, 20].map(processingPollDelay)).toEqual([
      2_000,
      2_750,
      3_500,
      4_250,
      5_000,
      5_000,
    ]);
  });
});
