/**
 * Anthropic Claude API implementation for TuneAPI
 */

import Anthropic from "@anthropic-ai/sdk";
import type {
  Thread,
  Message,
  ModelInterface,
  ChatOptions,
  StreamChatOptions,
  DistributedChatOptions,
  Tool,
  ToolCall,
} from "../types/chats";
import {
  MESSAGE_ROLES,
  createThread,
  human,
  functionResp,
  ToolResponse,
} from "../types/chats";
import { ENV } from "../utils/env";
import { schemaToJsonSchema, parseWithSchema } from "../types/schema";

/**
 * Anthropic model implementation using official Anthropic SDK
 */
export class AnthropicModel implements ModelInterface {
  model_id: string;
  api_token: string;
  private client: Anthropic;

  constructor(id: string = "claude-sonnet-4-5-20250929", api_token?: string) {
    this.model_id = id;
    this.api_token = api_token || ENV.ANTHROPIC_TOKEN("");

    if (!this.api_token) {
      throw new Error(
        "Anthropic API key not found. Please set ANTHROPIC_API_KEY or ANTHROPIC_TOKEN environment variable or pass through constructor"
      );
    }

    this.client = new Anthropic({
      apiKey: this.api_token,
    });
  }

  /**
   * Set API token
   */
  setApiToken(token: string): void {
    this.api_token = token;
    this.client = new Anthropic({
      apiKey: token,
    });
  }

  /**
   * Translate Thread format to Anthropic messages format
   * Returns [systemInstruction, messages]
   */
  private translateThread(thread: Thread): [string, any[]] {
    let systemInstruction = "";
    const claudeMessages: any[] = [];

    // Check if first message is system
    if (
      thread.chats.length > 0 &&
      thread.chats[0].role === MESSAGE_ROLES.SYSTEM
    ) {
      systemInstruction = thread.chats[0].value as string;
    }

    // Process remaining messages
    const startIdx = systemInstruction ? 1 : 0;
    for (let i = startIdx; i < thread.chats.length; i++) {
      const m = thread.chats[i];

      if (m.role === MESSAGE_ROLES.HUMAN) {
        if (typeof m.value !== "string") {
          throw new Error(
            `HUMAN message value must be a string. Got: '${typeof m.value}'`
          );
        }

        const content: any[] = [{ type: "text", text: m.value }];

        // Add images if present
        if (m.images.length > 0) {
          for (const img of m.images) {
            content.push({
              type: "image",
              source: {
                type: "base64",
                media_type: "image/png",
                data: img,
              },
            });
          }
        }

        claudeMessages.push({ role: "user", content });
      } else if (m.role === MESSAGE_ROLES.GPT) {
        if (typeof m.value !== "string") {
          throw new Error(
            `GPT message value must be a string. Got: '${typeof m.value}'`
          );
        }

        const trimmed = (m.value as string).trim();
        if (!trimmed) {
          continue;
        }

        const content: any[] = [{ type: "text", text: trimmed }];

        // Add images if present
        if (m.images.length > 0) {
          for (const img of m.images) {
            content.push({
              type: "image",
              source: {
                type: "base64",
                media_type: "image/png",
                data: img,
              },
            });
          }
        }

        claudeMessages.push({ role: "assistant", content });
      } else if (m.role === MESSAGE_ROLES.FUNCTION_RESP) {
        // Function responses - add assistant message with tool_use and user message with tool_result
        const toolResp = m.value as ToolResponse;

        // Add assistant message with tool_use using raw_body
        claudeMessages.push({
          role: "assistant",
          content: [toolResp.tool_call.raw_body],
        });

        // Add user message with tool_result
        claudeMessages.push({
          role: "user",
          content: [
            {
              type: "tool_result",
              tool_use_id: toolResp.tool_call.id,
              content: JSON.stringify(toolResp.result),
            },
          ],
        });
      } else if (m.role === MESSAGE_ROLES.THINKING) {
        // Thinking content - add as assistant message with thinking block
        if (typeof m.value !== "string") {
          throw new Error(
            `THINKING message value must be a string. Got: '${typeof m.value}'`
          );
        }

        const trimmed = (m.value as string).trim();
        if (!trimmed) {
          continue;
        }

        claudeMessages.push({
          role: "assistant",
          content: [
            {
              type: "thinking",
              thinking: trimmed,
            },
          ],
        });
      } else {
        throw new Error(`Unknown role: ${m.role}`);
      }
    }

    // Handle schema if present (Anthropic doesn't have native schema support like OpenAI)
    if (thread.schema && claudeMessages.length > 0) {
      const lastMessage = claudeMessages[claudeMessages.length - 1];
      if (lastMessage.role === "user") {
        const jsonSchema = schemaToJsonSchema(thread.schema);
        lastMessage.content.push({
          type: "text",
          text:
            "You are given this JSON schema. You will generate only the output filled in this schema and " +
            "nothing else. No thinking or extra tokens. Only the JSON. It is crucial you do this because " +
            "the response text will be parsed from JSON. If it fails, everything has gone to waste. Here " +
            "is the schema:\n\n" +
            JSON.stringify(jsonSchema, null, 2),
        });
      }
    }

    return [systemInstruction, claudeMessages];
  }

  /**
   * Convert Tool to Anthropic tool format
   */
  private convertToolsToAnthropicFormat(tools: Tool[]): any[] {
    return tools.map((tool) => ({
      name: tool.name,
      description: tool.description,
      input_schema: {
        type: "object",
        properties: tool.parameters.reduce((acc, param) => {
          acc[param.name] = {
            type: param.type,
            description: param.description,
            ...(param.enumValues && { enum: param.enumValues }),
            ...(param.items && { items: param.items }),
          };
          return acc;
        }, {} as Record<string, any>),
        required: tool.parameters.filter((p) => p.required).map((p) => p.name),
      },
    }));
  }

  /**
   * Build request body for Anthropic API
   */
  private buildRequestBody(
    messages: any[],
    systemInstruction: string,
    thread: Thread,
    options?: ChatOptions | StreamChatOptions,
    stream: boolean = false
  ): any {
    const requestBody: any = {
      model: options?.model || this.model_id,
      max_tokens: options?.max_tokens || 8192,
      messages,
    };

    if (stream) {
      requestBody.stream = true;
    }

    if (systemInstruction) {
      requestBody.system = systemInstruction;
    }

    // Add extended thinking for Claude models that support it
    if (options?.thinking) {
      requestBody.thinking = {
        type: "enabled",
        budget_tokens: options.thinking.budget_tokens || 8192,
      };
      // Ensure max_tokens is at least as large as thinking budget
      requestBody.max_tokens =
        Math.max(requestBody.max_tokens, requestBody.thinking.budget_tokens) +
        1;
    }

    // Add tools if present
    if (thread.tools && thread.tools.length > 0) {
      requestBody.tools = this.convertToolsToAnthropicFormat(thread.tools);
    }

    return requestBody;
  }

  /**
   * Stream chat with Anthropic - yields tool calls as JSON without executing them
   * The caller is responsible for executing tools and updating the thread
   */
  async *streamChat(
    chats: Thread | string,
    options?: StreamChatOptions
  ): AsyncGenerator<string, void, unknown> {
    const thread =
      typeof chats === "string" ? createThread(human(chats)) : chats;
    const [systemInstruction, messages] = this.translateThread(thread);

    const requestBody = this.buildRequestBody(
      messages,
      systemInstruction,
      thread,
      options,
      true
    );

    try {
      const stream = await this.client.messages.stream(requestBody);
      const toolUseBlocks: any[] = [];
      let currentToolUse: any = null;
      let hasToolCalls = false;
      let isInThinkingBlock = false;

      for await (const event of stream) {
        if (event.type === "content_block_start") {
          const contentBlock = event.content_block;

          if (contentBlock.type === "tool_use") {
            hasToolCalls = true;
            currentToolUse = {
              id: contentBlock.id,
              name: contentBlock.name,
              input: "",
            };
          } else if (contentBlock.type === "thinking") {
            isInThinkingBlock = true;
          }
        } else if (event.type === "content_block_delta") {
          const delta = event.delta;

          if (delta.type === "thinking_delta") {
            // Thinking/reasoning content
            yield `__THINKING__${delta.thinking}`;
          } else if (delta.type === "text_delta") {
            // Regular text content
            yield delta.text;
          } else if (delta.type === "input_json_delta") {
            // Accumulate tool call JSON input
            if (currentToolUse) {
              currentToolUse.input += delta.partial_json;
            }
          }
        } else if (event.type === "content_block_stop") {
          if (currentToolUse) {
            toolUseBlocks.push(currentToolUse);
            currentToolUse = null;
          }
          // Reset thinking flag when block stops
          isInThinkingBlock = false;
        }
      }

      // If tool calls were made, yield them as JSON with raw_body
      if (hasToolCalls && toolUseBlocks.length > 0) {
        const toolCalls: any[] = [];

        for (const toolUse of toolUseBlocks) {
          try {
            // Try to parse the JSON input
            const parsedInput = JSON.parse(toolUse.input);
            toolCalls.push({
              id: toolUse.id,
              name: toolUse.name,
              arguments: parsedInput,
              raw_body: {
                type: "tool_use",
                id: toolUse.id,
                name: toolUse.name,
                input: parsedInput,
              },
            });
          } catch (jsonError: any) {
            // Handle malformed JSON gracefully
            console.warn(
              `Failed to parse tool call JSON for ${toolUse.name}: ${jsonError.message}. Input: ${toolUse.input}`
            );

            // Create a tool call with empty arguments and raw input
            toolCalls.push({
              id: toolUse.id,
              name: toolUse.name,
              arguments: {}, // Empty arguments for malformed JSON
              raw_body: {
                type: "tool_use",
                id: toolUse.id,
                name: toolUse.name,
                input: toolUse.input, // Keep raw input string
              },
              _error: `JSON parse error: ${jsonError.message}`,
            });
          }
        }

        // Yield tool calls as JSON with a special prefix (only if we have any)
        if (toolCalls.length > 0) {
          yield `__TOOL_CALLS__${JSON.stringify(toolCalls)}`;
        }
      }
    } catch (error: any) {
      // Provide more context for streaming errors
      const errorMessage = error.message || "Unknown error";
      const errorType = error.type || error.name || "Error";

      console.error(`Anthropic streaming error [${errorType}]:`, errorMessage);

      // Check for common issues
      if (
        errorMessage.includes("JSON") ||
        errorMessage.includes("Unexpected end of JSON input")
      ) {
        throw new Error(
          `Anthropic API Error: Failed to parse streaming response. This may be due to incomplete JSON in tool calls or streaming interruption. Original error: ${errorMessage}`
        );
      }

      throw new Error(`Anthropic API Error: ${errorMessage}`);
    }
  }

  /**
   * Chat with Anthropic (non-streaming) - returns tool calls without executing them
   * The caller is responsible for executing tools and updating the thread
   */
  async chat<T = string | ToolCall[]>(
    chats: Thread | string,
    options?: ChatOptions
  ): Promise<T> {
    const thread =
      typeof chats === "string" ? createThread(human(chats)) : chats;
    const [systemInstruction, messages] = this.translateThread(thread);

    const requestBody = this.buildRequestBody(
      messages,
      systemInstruction,
      thread,
      options,
      false
    );

    try {
      const response = await this.client.messages.create(requestBody);

      // Check if there are tool calls - return them without executing
      const toolUseContent = response.content.filter(
        (block: any) => block.type === "tool_use"
      );

      if (toolUseContent.length > 0) {
        const toolCalls: ToolCall[] = toolUseContent.map((toolUse: any) => ({
          id: toolUse.id,
          name: toolUse.name,
          arguments: toolUse.input,
          raw_body: toolUse,
        }));

        return toolCalls as T;
      }

      // No tool calls, extract the text content and thinking (if present)
      let thinkingText = "";
      const thinkingContent = response.content.filter(
        (block: any) => block.type === "thinking"
      );
      if (thinkingContent.length > 0) {
        thinkingText = thinkingContent
          .map((block: any) => block.thinking)
          .join("");
      }

      const textContent = response.content.filter(
        (block: any) => block.type === "text"
      );
      const content = textContent.map((block: any) => block.text).join("");

      // If there's thinking content, prepend it with marker
      const fullContent = thinkingText
        ? `__THINKING__${thinkingText}\n\n${content}`
        : content;

      // Parse with schema if present (use content without thinking for schema parsing)
      if (thread.schema && content) {
        try {
          return parseWithSchema(thread.schema, content) as T;
        } catch (error: any) {
          throw new Error(`Schema validation failed: ${error.message}`);
        }
      }

      return fullContent as T;
    } catch (error: any) {
      throw new Error(`Anthropic API Error: ${error.message}`);
    }
  }

  /**
   * Distributed chat (placeholder for future implementation)
   */
  async distributedChat(
    prompts: Thread[],
    options?: DistributedChatOptions
  ): Promise<any[]> {
    throw new Error("Distributed chat not yet implemented");
  }
}
