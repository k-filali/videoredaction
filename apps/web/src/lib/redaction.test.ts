import { describe, expect, it } from "vitest";

import { defaultShapeFor, resolveTreatment } from "./redaction";

describe("redaction treatments", () => {
  it("defaults faces to ellipses and other classes to rectangles", () => {
    expect(defaultShapeFor("face")).toBe("ellipse");
    expect(defaultShapeFor("license_plate")).toBe("rectangle");
    expect(defaultShapeFor("scene_text")).toBe("rectangle");
  });

  it("falls back to the default style when no override exists", () => {
    expect(resolveTreatment("face", "pixelate", {})).toEqual({
      style: "pixelate",
      shape: "ellipse",
    });
  });

  it("prefers per-class overrides over defaults", () => {
    const treatments = {
      face: { style: "black_box" as const, shape: "rectangle" as const },
      license_plate: { shape: "ellipse" as const },
    };

    expect(resolveTreatment("face", "pixelate", treatments)).toEqual({
      style: "black_box",
      shape: "rectangle",
    });
    expect(resolveTreatment("license_plate", "gaussian_blur", treatments)).toEqual({
      style: "gaussian_blur",
      shape: "ellipse",
    });
  });
});
