/**
 * Test file for thinking/reasoning models
 *
 * This file demonstrates how to use thinking models like OpenAI's o1 series
 * or other models with extended reasoning capabilities. It uses streaming to
 * show the model's reasoning process in real-time.
 *
 * Usage:
 *   tsx tests/test_thinking.ts --model openai --model-id o1-mini
 *   tsx tests/test_thinking.ts --model gemini --model-id gemini-2.0-flash-thinking-exp
 */

import { createThread, human } from "../types/chats";
import {
  parseArgs,
  validateModelType,
  runAllModels,
  createModel,
  logFunction,
} from "./test_runner";

// ============================================================================
// Thinking Model Example
// ============================================================================

/**
 * Complex problem requiring multi-step reasoning
 */
async function exampleComplexProblem(model: any, modelName: string) {
  console.log(`\n=== Complex Reasoning Problem (${modelName}) ===\n`);

  const problem = `A farmer needs to transport a fox, a chicken, and a sack of grain across a river. 
The boat is small and can only carry the farmer and one item at a time. 
If left alone together, the fox will eat the chicken, and the chicken will eat the grain.

How can the farmer get everything across the river safely? 
Think before answering, give answer in just one line.`;

  const thread = createThread(human(problem));

  console.log("Problem:");
  console.log(problem);
  console.log("\n" + "-".repeat(70));
  console.log("Model Response (streaming with reasoning):");
  console.log("-".repeat(70) + "\n");

  // Determine options based on model type
  let options: any = {};

  options.thinking = { include_thoughts: true };

  // Stream the response to see reasoning in real-time
  let isInThinkingMode = false;
  let thinkingBuffer = "";
  let outputBuffer = "";

  for await (const chunk of model.streamChat(thread, options)) {
    // Check if this chunk contains thinking marker
    if (chunk.startsWith("__THINKING__")) {
      // Extract thinking content
      const thinkingContent = chunk.slice(12); // Remove __THINKING__ prefix
      thinkingBuffer += thinkingContent;

      if (!isInThinkingMode) {
        console.log("\n" + "=".repeat(70));
        console.log("💭 THINKING/REASONING:");
        console.log("=".repeat(70));
        isInThinkingMode = true;
      }
      process.stdout.write(thinkingContent);
    } else {
      // Regular output
      if (isInThinkingMode) {
        // Transition from thinking to output
        console.log("\n" + "=".repeat(70));
        console.log("📝 ANSWER:");
        console.log("=".repeat(70));
        isInThinkingMode = false;
      }
      outputBuffer += chunk;
      process.stdout.write(chunk);
    }
  }

  console.log("\n");

  // Summary
  if (thinkingBuffer) {
    console.log("\n" + "=".repeat(70));
    console.log("📊 SUMMARY:");
    console.log(`Thinking tokens: ${thinkingBuffer.length} characters`);
    console.log(`Output tokens: ${outputBuffer.length} characters`);
    console.log("=".repeat(70));
  }
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  const { modelType, modelId, runAll } = parseArgs(process.argv.slice(2));
  validateModelType(modelType, runAll);

  try {
    // Make sure to set your API keys:
    // export OPENAI_API_KEY="your-key"
    // export GEMINI_API_KEY="your-key"
    // export ANTHROPIC_API_KEY="your-key"

    if (runAll) {
      await runAllModels(
        "test_thinking",
        (model, modelName) =>
          logFunction(exampleComplexProblem, "exampleComplexProblem")(
            model,
            modelName
          ),
        modelId
      );
    } else {
      const { model, modelName } = createModel(
        modelType,
        modelId ||
          (modelType === "openai"
            ? "o3-mini"
            : modelType === "gemini"
            ? "gemini-2.0-flash-thinking-exp"
            : undefined)
      );

      console.log("=".repeat(70));
      console.log("TuneAPI Thinking/Reasoning Model Test");
      console.log(`Running with: ${modelName}`);
      console.log("=".repeat(70));

      await exampleComplexProblem(model, modelName);

      console.log("=".repeat(70));
      console.log("Test completed!");
      console.log("=".repeat(70));
    }
  } catch (error: any) {
    console.error("\nError:", error.message);
    if (error.stack) {
      console.error(error.stack);
    }
  }
}

// Run the tests
main();
