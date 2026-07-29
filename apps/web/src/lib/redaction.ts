import type { ClassTreatment, RedactionShape, RedactionStyle } from "../types";

const defaultShapes: Record<string, RedactionShape> = {
  face: "ellipse",
};

export function defaultShapeFor(className: string): RedactionShape {
  return defaultShapes[className] ?? "rectangle";
}

export function resolveTreatment(
  className: string,
  defaultStyle: RedactionStyle,
  treatments: Record<string, ClassTreatment>,
): { style: RedactionStyle; shape: RedactionShape } {
  const treatment = treatments[className];
  return {
    style: treatment?.style ?? defaultStyle,
    shape: treatment?.shape ?? defaultShapeFor(className),
  };
}
