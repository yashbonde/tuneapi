/**
 * Shared test runner utility for TuneAPI TypeScript tests
 *
 * This module provides functionality to run tests with multiple models
 * and display results in a concise format.
 */

import { OpenAIModel } from "../apis/model_openai";
import { GeminiModel } from "../apis/model_gemini";
import { AnthropicModel } from "../apis/model_anthropic";

export interface TestResult {
  name: string;
  time: number;
  status: "PASS" | "FAIL";
}

export interface TestConfig {
  testName: string;
  modelType: string;
  modelId?: string;
  verbose?: boolean;
}

/**
 * Creates a model instance based on the model type
 */
export function createModel(
  modelType: string,
  modelId?: string
): { model: any; modelName: string } {
  switch (modelType.toLowerCase()) {
    case "openai":
      return {
        model: modelId ? new OpenAIModel(modelId) : new OpenAIModel(),
        modelName: "OpenAI",
      };
    case "gemini":
      return {
        model: modelId ? new GeminiModel(modelId) : new GeminiModel(),
        modelName: "Gemini",
      };
    case "anthropic":
      return {
        model: new AnthropicModel(),
        modelName: "Anthropic",
      };
    default:
      throw new Error(`Unknown model type: ${modelType}`);
  }
}

/**
 * Wraps a function to log its execution
 */
export function logFunction<T extends (...args: any[]) => Promise<any>>(
  fn: T,
  name: string
): T {
  return (async (...args: any[]) => {
    console.log(`  → Running ${name}...`);
    try {
      const result = await fn(...args);
      console.log(`  ✓ ${name} completed`);
      return result;
    } catch (error: any) {
      console.log(`  ✗ ${name} failed: ${error.message}`);
      throw error;
    }
  }) as T;
}

/**
 * Runs a test function with the specified configuration
 */
export async function runTest(
  config: TestConfig,
  testFn: (model: any, modelName: string) => Promise<void>
): Promise<TestResult> {
  const startTime = Date.now();
  let status: "PASS" | "FAIL" = "PASS";

  try {
    const { model, modelName } = createModel(config.modelType, config.modelId);

    if (config.verbose) {
      await testFn(model, modelName);
    } else {
      // Suppress console output but keep function logging
      const originalLog = console.log;
      const originalError = console.error;
      const originalWrite = process.stdout.write;

      // Create a filter that only allows our logging messages
      console.log = (message: any, ...args: any[]) => {
        const msg = String(message);
        if (
          msg.includes("→ Running") ||
          msg.includes("✓") ||
          msg.includes("✗")
        ) {
          originalLog(message, ...args);
        }
      };
      console.error = () => {};
      process.stdout.write = () => true;

      try {
        await testFn(model, modelName);
      } finally {
        console.log = originalLog;
        console.error = originalError;
        process.stdout.write = originalWrite;
      }
    }
  } catch (error: any) {
    status = "FAIL";
  }

  const endTime = Date.now();
  const timeTaken = (endTime - startTime) / 1000;

  return {
    name: `${config.testName} (${config.modelType})`,
    time: timeTaken,
    status,
  };
}

/**
 * Runs tests with all models and displays results
 */
export async function runAllModels(
  testName: string,
  testFn: (model: any, modelName: string) => Promise<void>,
  modelId?: string
): Promise<void> {
  console.log("Running all tests with all models...\n");
  const models = ["openai", "gemini", "anthropic"];
  const results: TestResult[] = [];

  for (const modelType of models) {
    // Print test start (pytest style)
    const testFullName = `${testName} (${modelType})`;
    process.stdout.write(`${testFullName.padEnd(35)} ... `);

    const result = await runTest(
      {
        testName,
        modelType,
        modelId,
        verbose: false,
      },
      testFn
    );
    results.push(result);

    // Print result inline (pytest style)
    const timeStr = `${result.time.toFixed(2)}s`;
    const statusSymbol = result.status === "PASS" ? "✓" : "✗";
    console.log(`${timeStr.padEnd(8)} ${statusSymbol} ${result.status}`);
  }

  // Print summary
  console.log("\n" + "-".repeat(60));
  const passed = results.filter((r) => r.status === "PASS").length;
  const failed = results.filter((r) => r.status === "FAIL").length;
  const totalTime = results.reduce((sum, r) => sum + r.time, 0);

  console.log(
    `Total: ${results.length} tests | ` +
      `Passed: ${passed} | ` +
      `Failed: ${failed} | ` +
      `Time: ${totalTime.toFixed(2)}s`
  );

  if (failed > 0) {
    process.exit(1);
  }
}

/**
 * Parses command line arguments
 */
export interface ParsedArgs {
  modelType: string;
  modelId?: string;
  runAll: boolean;
}

export function parseArgs(args: string[]): ParsedArgs {
  let modelType: string = "openai"; // default
  let modelId: string | undefined;
  let runAll: boolean = false;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--model" && i + 1 < args.length) {
      const nextArg = args[i + 1].toLowerCase();
      if (nextArg === "all") {
        runAll = true;
      } else {
        modelType = nextArg;
      }
    }
    if (args[i] === "--model-id" && i + 1 < args.length) {
      modelId = args[i + 1];
    }
    if (args[i] === "--all") {
      runAll = true;
    }
  }

  return { modelType, modelId, runAll };
}

/**
 * Validates model type
 */
export function validateModelType(modelType: string, runAll: boolean): void {
  if (!runAll && !["openai", "gemini", "anthropic"].includes(modelType)) {
    console.error(
      `Invalid model: ${modelType}. Must be one of: openai, gemini, anthropic`
    );
    process.exit(1);
  }
}
