/**
 * TuneAPI - A general purpose LLM application building toolkit
 * Ported from python-tuneapi
 */

// Export all types
export type {
  Prop,
  Tool,
  ToolCall,
  ToolResponse,
  Message,
  Thread,
  Usage,
  ThinkingConfig,
  ModelInterface,
  ChatOptions,
  StreamChatOptions,
  DistributedChatOptions,
} from "./types/chats";

// Export constants
export { MESSAGE_ROLES, KNOWN_ROLES } from "./types/chats";

// Export factory functions
export {
  createProp,
  createTool,
  createMessage,
  createThread,
  createUsage,
} from "./types/chats";

// Export convenience aliases
export {
  human,
  system,
  assistant,
  functionResp,
  thinking,
} from "./types/chats";

// Export model implementations
export { OpenAIModel } from "./apis/model_openai";
export { GeminiModel } from "./apis/model_gemini";
export { AnthropicModel } from "./apis/model_anthropic";

// Export utilities
export { ENV } from "./utils/env";

// Export schema utilities (Zod)
export {
  z,
  type Schema,
  schemaToJsonSchema,
  schemaToGeminiFormat,
  parseWithSchema,
  safeParseWithSchema,
} from "./types/schema";
