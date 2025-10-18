/**
 * Example: Tool/Function Calling with TuneAPI
 *
 * This demonstrates how to use tools (function calling) with OpenAI, Gemini,
 * and Anthropic models. Tools allow the LLM to invoke functions to gather
 * information or perform actions.
 *
 * IMPORTANT: Manual Tool Calling Pattern
 * ========================================
 * In this implementation, tool calls are NOT executed automatically. Instead:
 *
 * 1. When you call `model.chat(thread)`, the response can be:
 *    - A string (final answer)
 *    - An array of ToolCall objects (tools that need to be executed)
 *
 * 2. If you receive ToolCall[], you must:
 *    - Execute each tool manually (using tool.toolFn())
 *    - Update the thread with functionResp() (tool_call is set in the response)
 *    - Call model.chat(thread) again to continue the conversation
 *
 * 3. Repeat until you receive a string response (final answer)
 *
 * This pattern gives you full control over:
 * - Whether to execute a tool (you can review/reject tool calls)
 * - When to execute tools (async, parallel, batched, etc.)
 * - How to handle errors during tool execution
 * - The agentic loop (you control the iteration)
 *
 * Example:
 * ```typescript
 * const response = await model.chat(thread);
 * if (Array.isArray(response)) {
 *   // Tool calls detected - handle them manually
 *   for (const toolCall of response) {
 *     const result = await tool.toolFn(toolCall);
 *     thread.chats.push(functionResp(result));
 *   }
 *   // Continue conversation
 *   const finalResponse = await model.chat(thread);
 * }
 * ```
 */
import "dotenv/config";

import {
  createThread,
  createTool,
  createProp,
  human,
  system,
  functionResp,
  ToolResponse,
  ToolCall,
} from "../index";
import {
  parseArgs,
  validateModelType,
  runAllModels,
  createModel,
  logFunction,
} from "./test_runner";

// ============================================================================
// Define Tools
// ============================================================================

/**
 * Simple tool: Get current weather for a location
 */
const getWeatherTool = createTool(
  "get_weather",
  "Get the current weather for a specific location",
  [
    createProp(
      "location",
      "string",
      true,
      "The city name, e.g. 'Paris' or 'London'"
    ),
    createProp(
      "units",
      "string",
      false,
      "Temperature units (celsius or fahrenheit)",
      undefined,
      ["celsius", "fahrenheit"]
    ),
  ],
  async (tool_call: ToolCall): Promise<ToolResponse> => {
    // Simulate API call
    const location = tool_call.arguments.location;
    const units = tool_call.arguments.units || "celsius";
    const temp = units === "celsius" ? 22 : 72;

    console.log(
      `  [Tool Called] get_weather(location="${location}", units="${units}")`
    );

    return {
      tool_call: tool_call,
      result: {
        location,
        temperature: temp,
        conditions: "sunny",
        humidity: 45,
        units,
      },
    };
  },
  "You can get weather information for any location.",
  { units: "celsius" }
);

/**
 * Calculator tool: Perform basic arithmetic
 */
const calculatorTool = createTool(
  "calculate",
  "Perform arithmetic calculations",
  [
    createProp(
      "operation",
      "string",
      true,
      "The operation to perform",
      undefined,
      ["add", "subtract", "multiply", "divide"]
    ),
    createProp("a", "number", true, "First number"),
    createProp("b", "number", true, "Second number"),
  ],
  async (tool_call: ToolCall): Promise<ToolResponse> => {
    const { operation, a, b } = tool_call.arguments;

    console.log(
      `  [Tool Called] calculate(operation="${operation}", a=${a}, b=${b})`
    );

    let result: number;
    switch (operation) {
      case "add":
        result = a + b;
        break;
      case "subtract":
        result = a - b;
        break;
      case "multiply":
        result = a * b;
        break;
      case "divide":
        result = b !== 0 ? a / b : NaN;
        break;
      default:
        return {
          tool_call: tool_call,
          result: { error: "Unknown operation" },
        };
    }

    return {
      tool_call: tool_call,
      result: result,
    };
  },
  "You can perform basic arithmetic operations.",
  {}
);

/**
 * Database query tool: Simulate database lookups
 */
const queryDatabaseTool = createTool(
  "query_database",
  "Query a database for user information",
  [createProp("user_id", "string", true, "The user ID to look up")],
  async (tool_call: ToolCall): Promise<ToolResponse> => {
    const userId = tool_call.arguments.user_id;

    console.log(`  [Tool Called] query_database(user_id="${userId}")`);

    // Simulate database lookup
    const mockData: Record<string, any> = {
      user_123: {
        name: "Alice Johnson",
        email: "alice@example.com",
        subscription: "premium",
        joined: "2023-01-15",
      },
      user_456: {
        name: "Bob Smith",
        email: "bob@example.com",
        subscription: "free",
        joined: "2024-05-20",
      },
    };

    const userData = mockData[userId] || { error: "User not found" };
    return {
      tool_call: tool_call,
      result: userData,
    };
  },
  "You can query user information from the database.",
  {}
);

// ============================================================================
// Helper Function
// ============================================================================

/**
 * Helper function to handle tool calls manually with parallel execution
 * Executes tools in parallel and updates the thread until a final text response is received
 */
async function handleToolCalls(
  model: any,
  thread: any,
  initialResponse: any
): Promise<string> {
  let response = initialResponse;

  while (Array.isArray(response)) {
    console.log("\nTool calls requested:", response);

    // Execute all tools in parallel
    const toolExecutions = response.map(async (toolCall: any) => {
      const tool = thread.tools.find((t: any) => t.name === toolCall.name);
      if (!tool) {
        throw new Error(`Tool '${toolCall.name}' not found in thread`);
      }

      // Execute the tool (all tools are async)
      const result = await tool.toolFn(toolCall);

      return result;
    });

    // Wait for all tools to complete
    const results = await Promise.all(toolExecutions);

    // Add all function responses to the thread
    for (const result of results) {
      thread.chats.push(functionResp(result));
    }

    // Continue conversation
    response = await model.chat(thread);
  }

  return response;
}

// ============================================================================
// Examples
// ============================================================================

async function exampleSingleTool(model: any, modelName: string) {
  console.log(`\n=== Example 1: Single Tool - Weather (${modelName}) ===\n`);

  const thread = createThread(human("What's the weather like in Tokyo?"));
  thread.tools = [getWeatherTool];

  console.log("User query:", thread.chats[0].value);

  const initialResponse = await model.chat(thread);
  const finalResponse = await handleToolCalls(model, thread, initialResponse);

  console.log("\nFinal assistant response:", finalResponse);
}

async function exampleMultipleTools(model: any, modelName: string) {
  console.log(`\n=== Example 2: Multiple Tools (${modelName}) ===\n`);

  const thread = createThread(
    system(
      "You are a helpful assistant with access to weather and calculator tools."
    ),
    human("What's the weather in Paris? Also, what's 15 multiplied by 8?")
  );
  thread.tools = [getWeatherTool, calculatorTool];

  console.log("User query:", thread.chats[1].value);

  const initialResponse = await model.chat(thread);
  const finalResponse = await handleToolCalls(model, thread, initialResponse);

  console.log("\nFinal assistant response:", finalResponse);
}

async function exampleComplexToolUse(model: any, modelName: string) {
  console.log(`\n=== Example 3: Complex Tool Usage (${modelName}) ===\n`);

  const thread = createThread(
    system("You are a customer support assistant with access to user data."),
    human(
      "Can you look up information for user_123 and tell me when they joined?"
    )
  );
  thread.tools = [queryDatabaseTool];

  console.log("User query:", thread.chats[1].value);

  const initialResponse = await model.chat(thread);
  const finalResponse = await handleToolCalls(model, thread, initialResponse);

  console.log("\nFinal assistant response:", finalResponse);
}

async function exampleCalculatorChain(model: any, modelName: string) {
  console.log(`\n=== Example 4: Calculator Chain (${modelName}) ===\n`);

  const thread = createThread(
    human("Calculate (25 + 17) and then multiply the result by 3")
  );
  thread.tools = [calculatorTool];

  console.log("User query:", thread.chats[0].value);

  const initialResponse = await model.chat(thread);
  const finalResponse = await handleToolCalls(model, thread, initialResponse);

  console.log("\nFinal assistant response:", finalResponse);
}

async function exampleWithOptions(model: any, modelName: string) {
  console.log(`\n=== Example 5: Tools with Chat Options (${modelName}) ===\n`);

  const thread = createThread(
    human("What's the temperature in London in fahrenheit?")
  );
  thread.tools = [getWeatherTool];

  const initialResponse = await model.chat(thread, {
    temperature: 0.3,
    max_tokens: 500,
  });
  const finalResponse = await handleToolCalls(model, thread, initialResponse);

  console.log("\nFinal assistant response:", finalResponse);
}

async function exampleDivision(model: any, modelName: string) {
  console.log(`\n=== Example 6: Division Tool Call (${modelName}) ===\n`);

  const thread = createThread(human("What's 144 divided by 12?"));
  thread.tools = [calculatorTool];

  console.log("User query:", thread.chats[0].value);

  const initialResponse = await model.chat(thread);
  const finalResponse = await handleToolCalls(model, thread, initialResponse);

  console.log("\nFinal assistant response:", finalResponse);
}

// ============================================================================
// Main
// ============================================================================

async function runAllExamples(model: any, modelName: string) {
  await logFunction(exampleSingleTool, "exampleSingleTool")(model, modelName);
  await logFunction(exampleMultipleTools, "exampleMultipleTools")(
    model,
    modelName
  );
  await logFunction(exampleComplexToolUse, "exampleComplexToolUse")(
    model,
    modelName
  );
  await logFunction(exampleCalculatorChain, "exampleCalculatorChain")(
    model,
    modelName
  );
  await logFunction(exampleWithOptions, "exampleWithOptions")(model, modelName);
  await logFunction(exampleDivision, "exampleDivision")(model, modelName);
}

async function main() {
  const { modelType, runAll } = parseArgs(process.argv.slice(2));
  validateModelType(modelType, runAll);

  try {
    // Make sure to set your API keys:
    // export OPENAI_API_KEY="your-key"
    // export GEMINI_API_KEY="your-key"
    // export ANTHROPIC_API_KEY="your-key"

    if (runAll) {
      await runAllModels("test_toolcall", runAllExamples);
    } else {
      const { model, modelName } = createModel(
        modelType,
        modelType === "openai"
          ? "gpt-4o-mini"
          : modelType === "gemini"
          ? "gemini-2.0-flash-exp"
          : undefined
      );

      console.log("=".repeat(70));
      console.log("TuneAPI Tool/Function Calling Examples");
      console.log(`Running tests with: ${modelName}`);
      console.log("=".repeat(70));

      await runAllExamples(model, modelName);

      console.log("\n" + "=".repeat(70));
      console.log("All examples completed!");
      console.log("=".repeat(70));
    }
  } catch (error: any) {
    console.error("\nError:", error.message);
    if (error.stack) {
      console.error("\nStack trace:", error.stack);
    }
  }
}

// Run the tests
main();
