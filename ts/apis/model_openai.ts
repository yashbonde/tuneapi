/**
 * OpenAI API implementation for TuneAPI
 */

import OpenAI from "openai";
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
 * OpenAI model implementation using official OpenAI SDK
 */
export class OpenAIModel implements ModelInterface {
  model_id: string;
  api_token: string;
  private client: OpenAI;
  private base_url?: string;

  constructor(id: string = "gpt-4o", api_token?: string, base_url?: string) {
    this.model_id = id;
    this.api_token = api_token || ENV.OPENAI_TOKEN("");
    this.base_url = base_url;

    if (!this.api_token) {
      throw new Error(
        "OpenAI API key not found. Please set OPENAI_API_KEY or OPENAI_TOKEN environment variable or pass through constructor"
      );
    }

    this.client = new OpenAI({
      apiKey: this.api_token,
      baseURL: this.base_url,
    });
  }

  /**
   * Set API token
   */
  setApiToken(token: string): void {
    this.api_token = token;
    this.client = new OpenAI({
      apiKey: token,
      baseURL: this.base_url,
    });
  }

  /**
   * Translate Thread format to OpenAI Responses API messages format
   */
  private translateThread(thread: Thread): any[] {
    const finalMessages: any[] = [];

    // Extract system message first if present
    let systemContent: string | null = null;
    if (
      thread.chats.length > 0 &&
      thread.chats[0].role === MESSAGE_ROLES.SYSTEM
    ) {
      systemContent = thread.chats[0].value as string;

      // Add tool usage instructions to system message if tools are present
      if (thread.tools.length > 0) {
        let toolPrompt = "\n\n# Tool Usage Instructions\n\n";
        for (const tool of thread.tools) {
          if (tool.system) {
            toolPrompt += `${tool.system}\n`;
          }
        }
        systemContent += toolPrompt;
      }

      finalMessages.push({
        role: "system",
        content: systemContent,
      });
    }

    // Process remaining messages (skip first if it was system)
    const startIndex = systemContent !== null ? 1 : 0;
    for (let i = startIndex; i < thread.chats.length; i++) {
      const m = thread.chats[i];

      if (m.role === MESSAGE_ROLES.SYSTEM) {
        throw new Error(
          "Only the first message in thread can be the system message."
        );
      } else if (m.role === MESSAGE_ROLES.HUMAN) {
        if (typeof m.value !== "string") {
          throw new Error(
            `HUMAN message value must be a string. Got: '${typeof m.value}'`
          );
        }

        const content: any[] = [{ type: "input_text", text: m.value }];

        // Add images if present
        for (const img of m.images) {
          content.push({
            type: "input_image",
            image_url: `data:image/png;base64,${img}`,
            detail: "auto",
          });
        }

        finalMessages.push({ role: "user", content });
      } else if (m.role === MESSAGE_ROLES.GPT) {
        if (typeof m.value !== "string") {
          throw new Error(
            `GPT message value must be a string. Got: '${typeof m.value}'`
          );
        }
        finalMessages.push({
          role: "assistant",
          content: [{ type: "output_text", text: m.value }],
        });
      } else if (m.role === MESSAGE_ROLES.FUNCTION_RESP) {
        // Function responses - represent as user message with function result
        // Note: Responses API doesn't allow function_call in input, so we describe it in text
        const toolResp = m.value as ToolResponse;

        finalMessages.push({
          role: "user",
          content: [
            {
              type: "input_text",
              text: `Function ${
                toolResp.tool_call.name
              } returned: ${JSON.stringify(toolResp.result)}`,
            },
          ],
        });
      } else if (m.role === MESSAGE_ROLES.THINKING) {
        if (typeof m.value !== "string") {
          throw new Error(
            `THINKING message value must be a string. Got: '${typeof m.value}'`
          );
        }

        finalMessages.push({
          role: "assistant",
          content: [{ type: "output_text", text: m.value }],
        });
      } else {
        throw new Error(`Invalid message role: ${m.role}`);
      }
    }

    return finalMessages;
  }

  /**
   * Convert Tool to OpenAI Responses API tool format
   * Note: Responses API uses a flatter structure compared to Chat Completions API
   */
  private convertToolsToOpenAIFormat(tools: Tool[]): any[] {
    return tools.map((tool) => ({
      type: "function",
      name: tool.name,
      description: tool.description,
      parameters: {
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
   * Check if the model is an o1 series model
   */
  private isSmartModel(modelId: string): boolean {
    return (
      modelId.toLowerCase().includes("gpt-5") ||
      modelId.toLowerCase().includes("o4-") ||
      modelId.toLowerCase().includes("o3-") ||
      modelId.toLowerCase().includes("o1-")
    );
  }

  /**
   * Create request body for OpenAI Responses API
   */
  private createRequestBody(
    thread: Thread,
    options: ChatOptions | StreamChatOptions | undefined
  ): any {
    const messages = this.translateThread(thread);
    const modelId = options?.model || this.model_id;
    const isSmart = this.isSmartModel(modelId);

    const requestBody: any = {
      model: modelId,
      input: messages,
      store: false,
    };

    // Responses API doesn't use max_tokens parameter
    // It uses max_output_tokens if needed, but we'll leave it to defaults for now

    // Smart models don't support temperature
    if (options?.temperature !== undefined && !isSmart) {
      requestBody.temperature = options.temperature;
    }

    // Add reasoning for smart models
    if (
      (options?.thinking?.include_thoughts ||
        options?.thinking?.reasoning_effort) &&
      isSmart
    ) {
      requestBody.reasoning = {
        effort: options.thinking.reasoning_effort || "medium",
        summary: options.thinking.include_thoughts ? "detailed" : "auto",
      };
    }

    // Add tools if present (smart models don't support tools)
    if (thread.tools && thread.tools.length > 0 && !isSmart) {
      requestBody.tools = this.convertToolsToOpenAIFormat(thread.tools);
    }

    // Add schema if present (structured output)
    if (thread.schema) {
      const jsonSchema = schemaToJsonSchema(thread.schema);
      requestBody.text = {
        format: {
          type: "json_schema",
          name: "response",
          strict: true,
          schema: jsonSchema,
        },
      };
    }

    return requestBody;
  }

  /**
   * Stream chat with OpenAI - accumulates and yields tool calls as JSON strings
   * The caller is responsible for executing tools and updating the thread
   */
  async *streamChat(
    chats: Thread | string,
    options?: StreamChatOptions
  ): AsyncGenerator<string, void, unknown> {
    const thread =
      typeof chats === "string" ? createThread(human(chats)) : chats;
    const requestBody = this.createRequestBody(thread, options);
    requestBody.stream = true;

    try {
      const stream = (await this.client.responses.create(requestBody)) as any;

      // Accumulate tool calls if they appear
      const toolCallsMap: Record<number, any> = {};
      let hasToolCalls = false;

      for await (const chunk of stream) {
        // Handle different event types from Responses API
        if (chunk.type === "response.output_text.delta") {
          // Text content delta - yield the delta field directly
          if (chunk.delta) {
            yield chunk.delta;
          }
        } else if (chunk.type === "response.reasoning_content.delta") {
          // Reasoning/thinking content delta - yield with special prefix
          if (chunk.delta) {
            yield `__THINKING__${chunk.delta}`;
          }
        } else if (
          chunk.type === "response.output_item.added" &&
          chunk.item?.type === "reasoning"
        ) {
          // Reasoning block started - the content will come in subsequent deltas
        } else if (
          chunk.type === "response.output_item.done" &&
          chunk.item?.type === "reasoning"
        ) {
          // Reasoning block completed - check for summary
          // Note: As of now, OpenAI o3 models use reasoning tokens internally but don't
          // expose the actual reasoning content in the API response (summary array is empty)
          if (
            chunk.item.summary &&
            Array.isArray(chunk.item.summary) &&
            chunk.item.summary.length > 0
          ) {
            for (const part of chunk.item.summary) {
              if (part.type === "text" && part.text) {
                yield `__THINKING__${part.text}`;
              }
            }
          }
        } else if (chunk.type === "response.function_call_arguments.delta") {
          // Tool call delta - ID is provided by the API
          hasToolCalls = true;
          const index = chunk.output_index || 0;

          if (!toolCallsMap[index]) {
            toolCallsMap[index] = {
              id: chunk.call_id,
              type: "function",
              function: {
                name: chunk.name || "",
                arguments: chunk.delta || "",
              },
            };
          } else {
            toolCallsMap[index].function.arguments += chunk.delta || "";
          }
        } else if (chunk.choices?.[0]?.delta) {
          // Fallback for Chat Completions API format
          const delta = chunk.choices[0].delta;

          if (delta.content) {
            yield delta.content;
          } else if (delta.tool_calls) {
            hasToolCalls = true;

            for (const tcDelta of delta.tool_calls) {
              const index = tcDelta.index;

              if (!toolCallsMap[index]) {
                toolCallsMap[index] = {
                  id: tcDelta.id || "",
                  type: "function",
                  function: {
                    name: tcDelta.function?.name || "",
                    arguments: tcDelta.function?.arguments || "",
                  },
                };
              } else {
                if (tcDelta.function?.name) {
                  toolCallsMap[index].function.name += tcDelta.function.name;
                }
                if (tcDelta.function?.arguments) {
                  toolCallsMap[index].function.arguments +=
                    tcDelta.function.arguments;
                }
              }
            }
          }
        }
      }

      // If tool calls were made, yield them as a special marker
      if (hasToolCalls && Object.keys(toolCallsMap).length > 0) {
        const toolCalls = Object.values(toolCallsMap).map((tc) => ({
          id: tc.id,
          name: tc.function.name,
          arguments: JSON.parse(tc.function.arguments),
        }));

        // Yield tool calls as JSON with a special prefix, including raw_body
        const toolCallsWithRaw = Object.values(toolCallsMap).map((tc) => ({
          id: tc.id,
          name: tc.function.name,
          arguments: JSON.parse(tc.function.arguments),
          raw_body: tc,
        }));
        yield `__TOOL_CALLS__${JSON.stringify(toolCallsWithRaw)}`;
      }
    } catch (error: any) {
      throw new Error(`OpenAI API Error: ${error.message}`);
    }
  }

  /**
   * Chat with OpenAI (non-streaming) - returns tool calls without executing them
   * The caller is responsible for executing tools and updating the thread
   */
  async chat(
    chats: Thread | string,
    options?: ChatOptions
  ): Promise<string | ToolCall[]> {
    const thread =
      typeof chats === "string" ? createThread(human(chats)) : chats;
    const requestBody = this.createRequestBody(thread, options);
    requestBody.stream = false;

    try {
      const response = (await this.client.responses.create(requestBody)) as any;

      // Handle Responses API format
      if (response.output) {
        const toolCalls: ToolCall[] = [];
        let textContent = "";

        // Process output items
        for (const item of response.output) {
          if (item.type === "message") {
            // Extract text content
            if (item.content) {
              for (const contentItem of item.content) {
                if (
                  (contentItem.type === "output_text" ||
                    contentItem.type === "text") &&
                  contentItem.text
                ) {
                  textContent += contentItem.text;
                }
              }
            }
          } else if (item.type === "function_call") {
            // Extract function call with ID generated by the API, store raw_body
            toolCalls.push({
              id: item.id,
              name: item.name,
              arguments: JSON.parse(item.arguments),
              raw_body: item,
            });
          }
        }

        // Return tool calls if present
        if (toolCalls.length > 0) {
          return toolCalls;
        }

        // Parse with schema if present
        if (thread.schema && textContent) {
          try {
            return parseWithSchema(thread.schema, textContent);
          } catch (error: any) {
            throw new Error(`Schema validation failed: ${error.message}`);
          }
        }

        return textContent;
      }

      // Fallback to Chat Completions API format
      const message = response.choices?.[0]?.message;

      if (!message) {
        throw new Error("No response from OpenAI");
      }

      // Check if there are tool calls - return them without executing, store raw_body
      if (message.tool_calls && message.tool_calls.length > 0) {
        const toolCalls: ToolCall[] = [];

        for (const toolCall of message.tool_calls) {
          if (toolCall.type === "function") {
            toolCalls.push({
              id: toolCall.id,
              name: toolCall.function.name,
              arguments: JSON.parse(toolCall.function.arguments),
              raw_body: toolCall,
            });
          }
        }

        return toolCalls;
      }

      // No tool calls, return the content
      const content = message.content || "";

      // Parse with schema if present
      if (thread.schema && content) {
        try {
          return parseWithSchema(thread.schema, content);
        } catch (error: any) {
          throw new Error(`Schema validation failed: ${error.message}`);
        }
      }

      return content;
    } catch (error: any) {
      throw new Error(`OpenAI API Error: ${error.message}`);
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
