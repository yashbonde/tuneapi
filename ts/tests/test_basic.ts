/**
 * Example usage of TuneAPI TypeScript SDK
 *
 * This demonstrates how to use OpenAI, Gemini, and Anthropic models
 * with the TuneAPI interface.
 */

import "dotenv/config"; // Add this line at the very top

import { createThread, human, system } from "../index";
import {
  parseArgs,
  validateModelType,
  runAllModels,
  createModel,
  logFunction,
} from "./test_runner";

// file consts
const block_chat = "What is the capital of France?";
const streaming_thread = createThread(
  system("You are a helpful assistant that speaks like a pirate."),
  human("Tell me about TypeScript in 1 paragraph.")
);

async function example(model: any, modelName: string) {
  console.log(`=== ${modelName} Example ===`);

  // Simple string chat
  const response1 = await model.chat(block_chat);
  console.log("Simple chat:", response1);

  // Thread-based chat with system message
  // Streaming example
  console.log("=============================== Streaming:");
  for await (const chunk of model.streamChat(streaming_thread)) {
    process.stdout.write(chunk);
  }
  console.log("\n");
}

async function main() {
  const { modelType, runAll } = parseArgs(process.argv.slice(2));
  validateModelType(modelType, runAll);

  try {
    // Make sure to set your API keys in environment variables:
    // - OPENAI_API_KEY or OPENAI_TOKEN
    // - GEMINI_API_KEY or GEMINI_TOKEN
    // - ANTHROPIC_API_KEY or ANTHROPIC_TOKEN

    if (runAll) {
      await runAllModels("test_basic", (model, modelName) =>
        logFunction(example, "example")(model, modelName)
      );
    } else {
      console.log(`Running tests with: ${modelType.toUpperCase()}`);

      const { model, modelName } = createModel(modelType);
      await example(model, modelName);

      console.log("\n" + "=".repeat(70));
      console.log("All examples completed!");
      console.log("=".repeat(70));
    }
  } catch (error: any) {
    console.error("Error:", error);
  }
}

// Run the tests
main();
