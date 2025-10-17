/**
 * Schema utilities for structured output generation
 * Uses Zod for runtime validation (TypeScript equivalent of Pydantic)
 *
 * Important Notes:
 * - OpenAI's strict mode requires all properties to be required (no optional fields)
 * - Optional fields in Zod schemas are converted to required fields for OpenAI
 * - Use default values or "unknown" strings for fields that might be missing
 * - Gemini and Anthropic handle optional fields more naturally
 */

import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";

/**
 * Type alias for any Zod schema
 */
export type Schema = z.ZodType<any, any, any>;

/**
 * Convert Zod schema to JSON Schema (for OpenAI strict mode)
 * OpenAI's strict mode requires all properties to be required
 */
export function schemaToJsonSchema(schema: Schema): any {
  const jsonSchema = zodToJsonSchema(schema, {
    target: "openApi3",
    $refStrategy: "none", // Inline all definitions
  });

  // Remove $schema property as OpenAI doesn't expect it
  const { $schema, ...rest } = jsonSchema as any;

  // Fix for OpenAI strict mode: make all properties required
  function fixStrictMode(obj: any): any {
    if (obj && typeof obj === "object") {
      // If it's an object schema, make all properties required
      if (obj.type === "object" && obj.properties) {
        obj.additionalProperties = false;
        // Make all properties required for OpenAI strict mode
        obj.required = Object.keys(obj.properties);

        // Recursively fix nested objects
        for (const key in obj.properties) {
          obj.properties[key] = fixStrictMode(obj.properties[key]);
        }
      }

      // Handle arrays
      if (obj.type === "array" && obj.items) {
        obj.items = fixStrictMode(obj.items);
      }

      // Handle anyOf (used for optional fields)
      if (obj.anyOf) {
        // Remove null type and use first non-null option
        const nonNullOption = obj.anyOf.find(
          (option: any) => option.type !== "null"
        );
        if (nonNullOption) {
          return fixStrictMode(nonNullOption);
        }
      }
    }
    return obj;
  }

  return fixStrictMode(rest);
}

/**
 * Convert Zod schema to Gemini response schema format
 * Gemini doesn't support 'additionalProperties' field
 */
export function schemaToGeminiFormat(schema: Schema): any {
  const jsonSchema = zodToJsonSchema(schema, {
    target: "openApi3",
    $refStrategy: "none",
  });

  const { $schema, ...rest } = jsonSchema as any;

  // Remove additionalProperties recursively (Gemini doesn't support it)
  function removeAdditionalProperties(obj: any): any {
    if (obj && typeof obj === "object") {
      // Remove additionalProperties if it exists
      if ("additionalProperties" in obj) {
        delete obj.additionalProperties;
      }

      // Recursively process nested objects
      if (obj.properties) {
        for (const key in obj.properties) {
          obj.properties[key] = removeAdditionalProperties(obj.properties[key]);
        }
      }

      // Handle arrays
      if (obj.items) {
        obj.items = removeAdditionalProperties(obj.items);
      }

      // Handle anyOf (for optional fields)
      if (obj.anyOf) {
        obj.anyOf = obj.anyOf.map((item: any) =>
          removeAdditionalProperties(item)
        );
      }
    }
    return obj;
  }

  return removeAdditionalProperties(rest);
}

/**
 * Parse and validate data against a Zod schema
 */
export function parseWithSchema<T>(
  schema: z.ZodType<T>,
  data: string | object
): T {
  const parsed = typeof data === "string" ? JSON.parse(data) : data;
  return schema.parse(parsed);
}

/**
 * Safely parse data, returning success/error result
 */
export function safeParseWithSchema<T>(
  schema: z.ZodType<T>,
  data: string | object
): { success: true; data: T } | { success: false; error: z.ZodError } {
  try {
    const parsed = typeof data === "string" ? JSON.parse(data) : data;
    const result = schema.safeParse(parsed);
    if (result.success) {
      return { success: true, data: result.data };
    }
    return { success: false, error: result.error };
  } catch (error) {
    // JSON parse error
    return {
      success: false,
      error: new z.ZodError([
        {
          code: "custom",
          path: [],
          message:
            error instanceof Error ? error.message : "Invalid JSON format",
        },
      ]),
    };
  }
}

/**
 * Re-export Zod for convenience
 */
export { z };
