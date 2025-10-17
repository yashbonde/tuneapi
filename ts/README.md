# TuneAPI - TypeScript

A general purpose LLM application building toolkit. Ported from [python-tuneapi](https://github.com/yashbonde/tuneapi).

## Installation

```bash
npm install tuneapi
```

## Quick Start

```typescript
import { OpenAIModel, human, createThread } from 'tuneapi';

// Initialize a model (requires OPENAI_API_KEY environment variable)
const model = new OpenAIModel("gpt-4o-mini");

// Simple chat
const response = await model.chat("What is TypeScript?");
console.log(response);

// Streaming chat
for await (const chunk of model.streamChat("Count from 1 to 5")) {
  process.stdout.write(chunk);
}
```

## Supported Models

TuneAPI provides official implementations for:

- **OpenAI** - GPT-4, GPT-4o, GPT-3.5-turbo, etc.
- **Google Gemini** - Gemini 2.0 Flash, Gemini 1.5 Pro, etc.
- **Anthropic** - Claude 3.5 Sonnet, Claude 3 Opus, etc.

### OpenAI

```typescript
import { OpenAIModel } from 'tuneapi';

const openai = new OpenAIModel(
  "gpt-4o",           // model ID (optional, default: "gpt-4o")
  "your-api-key",    // API token (optional, uses ENV.OPENAI_TOKEN)
  "https://..."      // base URL (optional)
);

// Simple chat
const response = await openai.chat("Hello!");

// With options
const response2 = await openai.chat("Hello!", {
  model: "gpt-4o-mini",
  max_tokens: 100,
  temperature: 0.7,
});

// Streaming
for await (const chunk of openai.streamChat("Tell me a story")) {
  process.stdout.write(chunk);
}
```

### Google Gemini

```typescript
import { GeminiModel } from 'tuneapi';

const gemini = new GeminiModel(
  "gemini-2.0-flash-exp",  // model ID (optional, default shown)
  "your-api-key"           // API token (optional, uses ENV.GEMINI_TOKEN)
);

// Usage is identical to OpenAI
const response = await gemini.chat("What is quantum computing?");
```

### Anthropic Claude

```typescript
import { AnthropicModel } from 'tuneapi';

const anthropic = new AnthropicModel(
  "claude-3-5-sonnet-20241022",  // model ID (optional, default shown)
  "your-api-key"                  // API token (optional, uses ENV.ANTHROPIC_TOKEN)
);

// Usage is identical to OpenAI
const response = await anthropic.chat("Explain neural networks");
```

## Environment Variables

TuneAPI provides a convenient `ENV` utility for accessing API tokens:

```typescript
import { ENV } from 'tuneapi';

// These check for multiple environment variable names
const openaiToken = ENV.OPENAI_TOKEN();      // Checks OPENAI_API_KEY, OPENAI_TOKEN
const geminiToken = ENV.GEMINI_TOKEN();      // Checks GEMINI_API_KEY, GEMINI_TOKEN
const anthropicToken = ENV.ANTHROPIC_TOKEN(); // Checks ANTHROPIC_API_KEY, ANTHROPIC_TOKEN

// With default values
const token = ENV.OPENAI_TOKEN("default-value");
```

## Usage

### Using Threads with Models

All models support both simple string inputs and structured Thread objects:

```typescript
import { OpenAIModel, createThread, human, system } from 'tuneapi';

const model = new OpenAIModel();

// Create a thread with system and user messages
const thread = createThread(
  system("You are a helpful coding assistant."),
  human("How do I reverse a string in TypeScript?")
);

// Chat with the thread
const response = await model.chat(thread);
console.log(response);

// Continue the conversation by adding messages to the thread
thread.chats.push(
  // Add assistant response
  // Add another human message
);
```

### Working with Tools (Function Calling)

```typescript
import { OpenAIModel, createThread, createTool, createProp, human } from 'tuneapi';

// Define a tool
const weatherTool = createTool(
  "get_weather",
  "Get the current weather for a location",
  [
    createProp("location", "string", true, "The city name"),
    createProp("units", "string", false, "Temperature units", undefined, ["celsius", "fahrenheit"])
  ],
  async (args) => {
    // Your tool implementation
    return { temperature: 22, conditions: "sunny" };
  }
);

// Create thread with tools
const thread = createThread(human("What's the weather in Paris?"));
thread.tools = [weatherTool];

// The model can now use the tool
const response = await model.chat(thread);
```

### Image Support

All models support sending images (base64 encoded):

```typescript
import { human } from 'tuneapi';

// Create a message with an image
const msg = human(
  "What's in this image?",
  ["base64-encoded-image-data"]  // Array of base64 strings
);

const response = await model.chat(msg);
```

### Structured Output (JSON Schema)

TuneAPI supports structured output using Zod schemas (TypeScript equivalent of Pydantic):

```typescript
import { z, OpenAIModel, createThread, human } from 'tuneapi';

// Define a schema
const PersonSchema = z.object({
  name: z.string().describe("Person's full name"),
  age: z.number().describe("Person's age"),
  occupation: z.string().describe("Person's job"),
});

// Infer TypeScript type from schema
type Person = z.infer<typeof PersonSchema>;

// Use schema with thread
const thread = createThread(
  human("John Smith is a 35-year-old software engineer")
);
thread.schema = PersonSchema;

// Get typed, validated response
const person: Person = await model.chat(thread);
console.log(person.name);  // TypeScript knows this exists!
```

**Supported by all providers:**
- **OpenAI**: Uses native strict JSON schema mode
- **Gemini**: Uses responseSchema in generation config
- **Anthropic**: Schema injected as prompt instructions

**Important Notes:**
- OpenAI's strict mode requires all fields to be present (converts optional to required)
- Use descriptions with `.describe()` for better results
- Complex nested schemas and arrays are supported
- Runtime validation ensures type safety

See `tests/test_structured.ts` for more examples!

### Creating Messages

```typescript
import { human, assistant, system, createMessage } from 'tuneapi';

// Using convenience functions (recommended)
const userMsg = human("Hello, how are you?");
const systemMsg = system("You are a helpful assistant");
const aiMsg = assistant("I'm doing great!");

// Using factory function
const msg = createMessage("Hello!", "user");
```

### Creating Threads

```typescript
import { createThread, human, assistant } from 'tuneapi';

const thread = createThread(
  human("What is 2+2?"),
  assistant("2+2 equals 4")
);

console.log(thread.chats.length); // 2
```

### Working with Tools

```typescript
import { createTool, createProp } from 'tuneapi';

const tool = createTool(
  "get_weather",
  "Get the current weather for a location",
  [
    createProp("location", "string", true, "The city name"),
    createProp("units", "string", false, "Temperature units", undefined, ["celsius", "fahrenheit"])
  ],
  (args) => {
    // Tool implementation
    return `Weather in ${args.location}`;
  }
);
```

### Usage Tracking

```typescript
import { createUsage } from 'tuneapi';

const usage = createUsage(100, 50, 10, "gpt-4");
console.log(usage.total_tokens); // 150
```

### Thinking/Reasoning Models

TuneAPI supports advanced reasoning capabilities for models that offer extended thinking:

#### OpenAI o1 Series

Use the `thinking.reasoning_effort` parameter to control reasoning depth. The library automatically handles o1-specific parameters:

```typescript
import { OpenAIModel, human, createThread } from 'tuneapi';

const model = new OpenAIModel("o1-mini");
const thread = createThread(
  human("Solve this complex logic puzzle: ...")
);

// Control reasoning effort: "low", "medium", or "high"
// Library automatically uses max_completion_tokens for o1 models
// and filters out unsupported parameters (temperature, tools, etc.)
const response = await model.chat(thread, {
  thinking: {
    reasoning_effort: "high"
  },
  max_tokens: 4096 // Automatically converted to max_completion_tokens for o1
});
```

**Note**: OpenAI o1 models have specific limitations:
- Don't expose reasoning tokens in the response (internal only)
- Don't support `temperature`, `tools`, or streaming function calls
- Use `max_completion_tokens` instead of `max_tokens` (handled automatically)

#### Anthropic Claude Extended Thinking

Use the `thinking` parameter to allocate tokens for internal reasoning. Thinking content is streamed separately with `__THINKING__` prefix:

```typescript
import { AnthropicModel, human, createThread } from 'tuneapi';

const model = new AnthropicModel("claude-3-7-sonnet");
const thread = createThread(
  human("Design a distributed system for ...")
);

// Allocate budget for thinking tokens
// Note: max_tokens must be greater than budget_tokens
// The library automatically adjusts if not specified
const response = await model.chat(thread, {
  thinking: {
    type: "enabled",
    budget_tokens: 2000
  },
  max_tokens: 4096 // Optional: auto-adjusted if less than budget_tokens
});

// Streaming response - thinking content has __THINKING__ prefix
// Anthropic streams thinking as thinking_delta events, separate from text_delta
for await (const chunk of model.streamChat(thread, { 
  thinking: { type: "enabled", budget_tokens: 2000 },
  max_tokens: 4096 
})) {
  if (chunk.startsWith("__THINKING__")) {
    const thinkingContent = chunk.slice(12); // Remove __THINKING__ prefix
    process.stdout.write(`[THINKING] ${thinkingContent}`);
  } else {
    process.stdout.write(`[ANSWER] ${chunk}`);
  }
}
```

**How it works:**
- Anthropic streams `thinking_delta` events containing reasoning
- Library converts these to chunks with `__THINKING__` prefix
- Regular response comes as `text_delta` events (no prefix)

#### Gemini Thinking Models

Gemini thinking models include their reasoning process when enabled via `thinking.include_thoughts`:

```typescript
import { GeminiModel, human, createThread } from 'tuneapi';

const model = new GeminiModel("gemini-2.0-flash-thinking-exp");
const thread = createThread(
  human("Explain the solution to this math problem...")
);

// Enable thinking with budget
const response = await model.chat(thread, {
  thinking: {
    include_thoughts: true,
    max_tokens: 2000  // Budget for thinking (-1 for unlimited)
  }
});

// Streaming - thoughts prefixed with __THINKING__
for await (const chunk of model.streamChat(thread, {
  thinking: { include_thoughts: true, max_tokens: 2000 }
})) {
  if (chunk.startsWith("__THINKING__")) {
    const thought = chunk.slice(12);
    process.stdout.write(`[THINKING] ${thought}`);
  } else {
    process.stdout.write(`[ANSWER] ${chunk}`);
  }
}
```

**How it works:**
- Gemini streams `thought` parts containing reasoning
- Library converts these to chunks with `__THINKING__` prefix
- Regular text comes as `text` parts (no prefix)

**Unified Thinking Configuration**: All models use the `thinking` parameter with provider-specific options:

```typescript
// OpenAI o1 series
{ thinking: { reasoning_effort: "low" | "medium" | "high" } }

// Anthropic Claude
{ thinking: { include_thoughts: true, budget_tokens: 2000 } }

// Gemini
{ thinking: { include_thoughts: true, max_tokens: 2000 } }
```

**Usage Tracking**: Models that use reasoning tokens will include `reasoning_tokens` in the `Usage` object:

```typescript
interface Usage {
  input_tokens: number;
  output_tokens: number;
  reasoning_tokens?: number; // For o1, Claude extended thinking
  total_tokens: number;
  // ...
}
```

## Type Definitions

All types are exported and can be used for type annotations:

```typescript
import type { Message, Thread, Tool, Prop, Usage, ModelInterface } from 'tuneapi';

function processThread(thread: Thread): void {
  // Your logic here
}
```

### Implementing a Model Provider

To implement a custom model provider, implement the `ModelInterface`:

```typescript
import type { ModelInterface, Thread, ChatOptions } from 'tuneapi';

class MyCustomModel implements ModelInterface {
  model_id: string;
  api_token: string;
  extra_headers: Record<string, any>;
  base_url: string;
  client: any;
  async_client: any;

  constructor(model_id: string, api_token: string) {
    this.model_id = model_id;
    this.api_token = api_token;
    this.extra_headers = {};
    this.base_url = "https://api.example.com";
    this.client = null;
    this.async_client = null;
  }

  setApiToken(token: string): void {
    this.api_token = token;
  }

  setClient(client?: any): void {
    this.client = client;
  }

  setAsyncClient(client?: any): void {
    this.async_client = client;
  }

  async *streamChat(chats: Thread | string, options?: any): AsyncGenerator<string, void, unknown> {
    // Implementation
    yield "Response chunk";
  }

  async chat(chats: Thread | string, options?: ChatOptions): Promise<string> {
    // Implementation
    return "Response";
  }

  // ... implement all other methods
}
```

## API Reference

### Types

- `Message` - A single message in a conversation
- `Thread` - A collection of messages
- `Tool` - A tool definition for function calling
- `Prop` - A property definition for tools
- `Usage` - Token usage tracking

### Factory Functions

- `createMessage(value, role, images?, id?, metadata?)` - Create a message
- `createThread(...messages)` - Create a thread
- `createTool(name, description, parameters, wrapper, system?, default_values?)` - Create a tool (wrapper must be async)
- `createProp(name, type?, required?, description?, items?, enumValues?, _value?)` - Create a property
- `createUsage(input_tokens, output_tokens, cached_tokens?, model?, extra?)` - Create usage object

### Convenience Functions

- `human(value, images?, id?)` - Create a user message
- `system(value, images?, id?)` - Create a system message
- `assistant(value, images?, id?)` - Create an assistant message
- `functionCall(value, id?)` - Create a function call message (value is a ToolCall)
- `functionResp(toolResponse, id?)` - Create a function response message (toolResponse is a ToolResponse)
- `thinking(value, id?)` - Create a thinking/reasoning message (for extended thinking models)

### Constants

- `MESSAGE_ROLES` - Standard message role constants
- `KNOWN_ROLES` - Mapping of role aliases to standard roles

## Testing & Examples

The `tests/` directory contains comprehensive examples:

### Basic Usage
```bash
tsx tests/test_basic.ts --model openai
tsx tests/test_basic.ts --model gemini
tsx tests/test_basic.ts --model anthropic
```

### Structured Output (Zod Schemas)
```bash
tsx tests/test_structured.ts --model openai
tsx tests/test_structured.ts --model gemini
tsx tests/test_structured.ts --model anthropic
```

### Tool/Function Calling
```bash
tsx tests/test_toolcall.ts --model openai
tsx tests/test_toolcall.ts --model gemini
tsx tests/test_toolcall.ts --model anthropic
```

### Thinking/Reasoning Models
Test advanced reasoning capabilities with streaming output:

```bash
# OpenAI o1 models (best for complex reasoning)
tsx tests/test_thinking.ts --model openai --model-id o1-mini
tsx tests/test_thinking.ts --model openai --model-id o1-preview

# Gemini thinking models
tsx tests/test_thinking.ts --model gemini --model-id gemini-2.0-flash-thinking-exp

# Regular models
tsx tests/test_thinking.ts --model anthropic
```

This test demonstrates:
- Complex multi-step reasoning problems
- Real-time streaming of the model's thought process
- Step-by-step problem solving

## License

MIT

