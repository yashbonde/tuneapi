/**
 * This file contains all the datatypes relevant for a chat conversation.
 */

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * An individual property is called a prop.
 */
export interface Prop {
  name: string;
  type: string;
  required: boolean;
  description: string;
  items?: Record<string, any>;
  enumValues?: string[];
  _value?: any;
}

/**
 * A tool is a container for telling the LLM what it can do.
 * All tool wrappers are expected to be async functions.
 */
export interface Tool {
  name: string;
  description: string;
  parameters: Prop[];
  wrapper: (args: any) => Promise<ToolResponse>;
  system: string;
  default_values: Record<string, any>;
}

export interface ToolCall {
  id: string;
  name: string;
  arguments: Record<string, any>;
  raw_body: any;
}

export interface ToolResponse {
  tool_call: ToolCall;
  result: any;
}

/**
 * A message is the unit element of information in a thread.
 */
export interface Message {
  role: string;
  value: string | ToolResponse;
  id: string;
  metadata: Record<string, any>;
  images: string[];
}

/**
 * This is a container for a list of chat messages.
 */
export interface Thread {
  chats: Message[];
  evals?: Record<string, any>;
  model?: string;
  id: string;
  title: string;
  tools: Tool[];
  schema?: any; // Zod schema for structured output
  meta: Record<string, any>;
  keys: string[];
  values: any[];
  value_hash: number;
}

/**
 * Usage tracking for token consumption.
 */
export interface Usage {
  input_tokens: number;
  output_tokens: number;
  cached_tokens: number;
  reasoning_tokens?: number; // For models with thinking/reasoning tokens (o1, Claude extended thinking)
  total_tokens: number;
  model: string;
  extra: Record<string, any>;
}

/**
 * Thinking/reasoning configuration for models that support extended reasoning
 */
export interface ThinkingConfig {
  budget_tokens?: number;
  // For OpenAI o1 series: { reasoning_effort: "low" | "medium" | "high" }
  reasoning_effort?: "low" | "medium" | "high";
  include_thoughts?: boolean;
}

/**
 * Options for chat methods
 */
export interface ChatOptions {
  model?: string;
  max_tokens?: number;
  temperature?: number;
  // Thinking/reasoning configuration (supports OpenAI, Anthropic, Gemini)
  thinking?: ThinkingConfig;
  [key: string]: any; // For additional kwargs
}

/**
 * Options for streaming chat methods
 */
export interface StreamChatOptions extends ChatOptions {
  // Gemini thinking models automatically include thoughts in response
  include_thoughts?: boolean; // Whether to include thinking process in output
}

/**
 * Options for distributed chat methods
 */
export interface DistributedChatOptions {
  [key: string]: any;
}

/**
 * This is the generic abstract interface implemented by all the model APIs.
 * Any class implementing this interface must provide all these methods.
 */
export interface ModelInterface {
  /** This is the model ID for the model */
  model_id: string;

  /** This is the API token for the model */
  api_token: string;

  // Chat methods

  /** This is the blocking function to stream chat with the model where each token is iteratively generated */
  streamChat(
    chats: Thread | string,
    options?: StreamChatOptions
  ): AsyncGenerator<string, void, unknown>;

  /** This is the blocking function to chat with the model */
  chat(
    chats: Thread | string,
    options?: ChatOptions
  ): Promise<string | ToolCall[]>;

  /** This is the blocking function to chat with the model in a distributed manner */
  distributedChat(
    prompts: Thread[],
    options?: DistributedChatOptions
  ): Promise<any[]>;
}

// ============================================================================
// Constants
// ============================================================================

export const MESSAGE_ROLES = {
  SYSTEM: "system",
  HUMAN: "human",
  GPT: "gpt",
  FUNCTION_RESP: "function_resp",
  THINKING: "thinking", // For model reasoning/thinking process
} as const;

export const KNOWN_ROLES: Record<string, string> = {
  // system
  system: MESSAGE_ROLES.SYSTEM,
  sys: MESSAGE_ROLES.SYSTEM,
  // user
  user: MESSAGE_ROLES.HUMAN,
  human: MESSAGE_ROLES.HUMAN,
  // assistants
  gpt: MESSAGE_ROLES.GPT,
  assistant: MESSAGE_ROLES.GPT,
  machine: MESSAGE_ROLES.GPT,
  // function response
  function_resp: MESSAGE_ROLES.FUNCTION_RESP,
  tool: MESSAGE_ROLES.FUNCTION_RESP,
  // thinking/reasoning
  thinking: MESSAGE_ROLES.THINKING,
  reasoning: MESSAGE_ROLES.THINKING,
};

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create a new Prop (property definition)
 */
export function createProp(
  name: string,
  type: string = "string",
  required: boolean = false,
  description: string = "",
  items?: Record<string, any>,
  enumValues?: string[],
  _value?: any
): Prop {
  return {
    name,
    type,
    required,
    description,
    items,
    enumValues,
    _value,
  };
}

/**
 * Create a new Tool
 * All tool wrappers must be async functions that return ToolResponse
 */
export function createTool(
  name: string,
  description: string,
  parameters: Prop[],
  wrapper: (args: any) => Promise<ToolResponse>,
  system: string = "",
  default_values: Record<string, any> = {}
): Tool {
  return {
    name,
    description,
    parameters,
    wrapper,
    system,
    default_values,
  };
}

/**
 * Create a new Message with validation
 */
export function createMessage(
  value: string | ToolResponse,
  role: string,
  images: string[] = [],
  id?: string,
  metadata: Record<string, any> = {}
): Message {
  if (!(role in KNOWN_ROLES)) {
    throw new Error(`Unknown role: ${role}. Update dictionary KNOWN_ROLES`);
  }
  if (value === null || value === undefined) {
    throw new Error("value cannot be null or undefined");
  }

  const normalizedRole = KNOWN_ROLES[role];
  const messageId =
    id || `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

  return {
    role: normalizedRole,
    value,
    id: messageId,
    metadata,
    images,
  };
}

/**
 * Create a new Thread
 */
export function createThread(...messages: Message[]): Thread {
  return {
    chats: messages,
    evals: undefined,
    model: undefined,
    id: `thread_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    title: "",
    tools: [],
    schema: undefined,
    meta: {},
    keys: [],
    values: [],
    value_hash: 0,
  };
}

/**
 * Create a new Usage object
 */
export function createUsage(
  input_tokens: number,
  output_tokens: number,
  cached_tokens: number = 0,
  model: string = "",
  extra: Record<string, any> = {}
): Usage {
  return {
    input_tokens,
    output_tokens,
    cached_tokens,
    total_tokens: input_tokens + output_tokens,
    model,
    extra,
  };
}

// ============================================================================
// Convenience Aliases (matching Python API)
// ============================================================================

/**
 * Convenience function for creating a human message
 */
export function human(
  value: string,
  images: string[] = [],
  id?: string
): Message {
  return createMessage(value, MESSAGE_ROLES.HUMAN, images, id);
}

/**
 * Convenience function for creating a system message
 */
export function system(
  value: string,
  images: string[] = [],
  id?: string
): Message {
  return createMessage(value, MESSAGE_ROLES.SYSTEM, images, id);
}

/**
 * Convenience function for creating an assistant message
 */
export function assistant(
  value: string,
  images: string[] = [],
  id?: string
): Message {
  return createMessage(value, MESSAGE_ROLES.GPT, images, id);
}

/**
 * Convenience function for creating a function response message
 */
export function functionResp(toolResponse: ToolResponse, id?: string): Message {
  return createMessage(toolResponse, MESSAGE_ROLES.FUNCTION_RESP, [], id);
}

/**
 * Convenience function for creating a thinking/reasoning message
 */
export function thinking(value: string, id?: string): Message {
  return createMessage(value, MESSAGE_ROLES.THINKING, [], id);
}
