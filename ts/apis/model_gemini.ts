/**
 * Google Gemini API implementation for TuneAPI
 */

import { GoogleGenerativeAI } from "@google/generative-ai";
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
import { schemaToGeminiFormat, parseWithSchema } from "../types/schema";

/**
 * Google Gemini model implementation using official Google Generative AI SDK
 */
export class GeminiModel implements ModelInterface {
  model_id: string;
  api_token: string;
  private client: GoogleGenerativeAI;

  constructor(id: string = "gemini-2.5-flash", api_token?: string) {
    this.model_id = id;
    this.api_token = api_token || ENV.GEMINI_TOKEN("");

    if (!this.api_token) {
      throw new Error(
        "Gemini API key not found. Please set GEMINI_API_KEY or GEMINI_TOKEN environment variable or pass through constructor"
      );
    }

    this.client = new GoogleGenerativeAI(this.api_token);
  }

  /**
   * Set API token
   */
  setApiToken(token: string): void {
    this.api_token = token;
    this.client = new GoogleGenerativeAI(token);
  }

  /**
   * Translate Thread format to Gemini messages format
   * Returns [systemInstruction, messages]
   */
  private translateThread(thread: Thread): [string, any[]] {
    let systemInstruction = "";
    const messages: any[] = [];

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
        const parts: any[] = [{ text: m.value as string }];

        // Add images if present
        for (const img of m.images) {
          parts.push({
            inlineData: {
              mimeType: "image/png",
              data: img,
            },
          });
        }

        messages.push({ role: "user", parts });
      } else if (m.role === MESSAGE_ROLES.GPT) {
        const parts: any[] = [{ text: m.value as string }];

        // Add images if present
        for (const img of m.images) {
          parts.push({
            inlineData: {
              mimeType: "image/png",
              data: img,
            },
          });
        }

        messages.push({ role: "model", parts });
      } else if (m.role === MESSAGE_ROLES.FUNCTION_RESP) {
        // Function responses - add model message with functionCall and user message with functionResponse
        const toolResp = m.value as ToolResponse;

        // Add model message with functionCall using raw_body
        messages.push({
          role: "model",
          parts: [toolResp.tool_call.raw_body],
        });

        // Gemini requires response to be an object (Struct), not a primitive
        // Wrap primitives in an object
        let responseContent = toolResp.result;
        if (typeof responseContent !== "object" || responseContent === null) {
          responseContent = { result: responseContent };
        }

        // Add user message with functionResponse
        messages.push({
          role: "user",
          parts: [
            {
              functionResponse: {
                name: toolResp.tool_call.name,
                response: responseContent,
              },
            },
          ],
        });
      } else if (m.role === MESSAGE_ROLES.THINKING) {
        // Thinking content - add as model message
        if (typeof m.value !== "string") {
          throw new Error(
            `THINKING message value must be a string. Got: '${typeof m.value}'`
          );
        }

        messages.push({
          role: "model",
          parts: [{ text: `[THINKING]\n${m.value}` }],
        });
      } else {
        throw new Error(`Unknown role: ${m.role}`);
      }
    }

    return [systemInstruction, messages];
  }

  /**
   * Convert Tool to Gemini function declaration format
   */
  private convertToolsToGeminiFormat(tools: Tool[]): any[] {
    return tools.map((tool) => {
      const functionDecl: any = {
        name: tool.name,
        description: tool.description,
      };

      // Only add parameters if there are any
      if (tool.parameters.length > 0) {
        functionDecl.parameters = {
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
          required: tool.parameters
            .filter((p) => p.required)
            .map((p) => p.name),
        };
      }

      return functionDecl;
    });
  }

  /**
   * Build request configuration for Gemini API
   */
  private buildRequestConfig(
    messages: any[],
    thread: Thread,
    options?: ChatOptions | StreamChatOptions
  ): any {
    const generationConfig: any = {};
    if (options?.max_tokens) {
      generationConfig.maxOutputTokens = options.max_tokens;
    }
    if (options?.temperature !== undefined) {
      generationConfig.temperature = options.temperature;
    }

    // Add schema if present (structured output)
    if (thread.schema) {
      generationConfig.responseMimeType = "application/json";
      generationConfig.responseSchema = schemaToGeminiFormat(thread.schema);
    }

    if (options?.thinking?.include_thoughts) {
      generationConfig.thinkingConfig = {
        includeThoughts: options.thinking.include_thoughts,
      };
      if (options.thinking.budget_tokens) {
        generationConfig.thinkingConfig.thinkingBudget =
          options.thinking.budget_tokens || 8192; // default to 8192 if not provided
      } else {
        generationConfig.thinkingConfig.thinkingBudget = -1;
      }
    } else if (options?.thinking) {
      generationConfig.thinkingConfig = { thinkingBudget: 0 };
    }

    const requestConfig: any = {
      contents: messages,
      generationConfig,
    };

    // Add tools if present
    if (thread.tools && thread.tools.length > 0) {
      const functionDeclarations = this.convertToolsToGeminiFormat(
        thread.tools
      );
      requestConfig.tools = [{ functionDeclarations }];
      requestConfig.toolConfig = {
        functionCallingConfig: {
          mode: "AUTO",
        },
      };
    }

    return requestConfig;
  }

  /**
   * Stream chat with Gemini - yields tool calls as JSON without executing them
   * The caller is responsible for executing tools and updating the thread
   *
   * Note: Gemini thinking models (e.g., gemini-2.0-flash-thinking-exp) automatically
   * include thinking/reasoning process in the response stream. The thoughts appear
   * naturally in the output before the final answer.
   */
  async *streamChat(
    chats: Thread | string,
    options?: StreamChatOptions
  ): AsyncGenerator<string, void, unknown> {
    const thread =
      typeof chats === "string" ? createThread(human(chats)) : chats;
    const [systemInstruction, messages] = this.translateThread(thread);

    const model = this.client.getGenerativeModel({
      model: options?.model || this.model_id,
      systemInstruction: systemInstruction || undefined,
    });

    const requestConfig = this.buildRequestConfig(messages, thread, options);

    try {
      const result = await model.generateContentStream(requestConfig);
      const collectedFunctionCalls: any[] = [];
      let hasToolCalls = false;

      for await (const chunk of result.stream) {
        // Check for thinking content in candidates
        if (chunk.candidates && chunk.candidates[0]) {
          const candidate = chunk.candidates[0];
          if (candidate.content && candidate.content.parts) {
            for (const part of candidate.content.parts) {
              // Check for thought part (Gemini thinking models)
              const partAny = part as any;
              if (partAny.thought) {
                yield `__THINKING__${part.text}`;
              } else if (part.text) {
                yield part.text;
              }
            }
          }
        }

        // Collect function calls
        const functionCalls = chunk.functionCalls();
        if (functionCalls && functionCalls.length > 0) {
          hasToolCalls = true;
          collectedFunctionCalls.push(...functionCalls);
        }
      }

      // If tool calls were made, yield them as JSON with raw_body
      if (hasToolCalls && collectedFunctionCalls.length > 0) {
        const toolCalls = collectedFunctionCalls.map((fc) => ({
          id: fc.name, // Gemini doesn't use IDs, use function name as ID
          name: fc.name,
          arguments: fc.args,
          raw_body: {
            functionCall: {
              name: fc.name,
              args: fc.args,
            },
          },
        }));

        // Yield tool calls as JSON with a special prefix
        yield `__TOOL_CALLS__${JSON.stringify(toolCalls)}`;
      }
    } catch (error: any) {
      throw new Error(`Gemini API Error: ${error.message}`);
    }
  }

  /**
   * Chat with Gemini (non-streaming) - returns tool calls without executing them
   * The caller is responsible for executing tools and updating the thread
   */
  async chat(
    chats: Thread | string,
    options?: ChatOptions
  ): Promise<string | ToolCall[]> {
    const thread =
      typeof chats === "string" ? createThread(human(chats)) : chats;
    const [systemInstruction, messages] = this.translateThread(thread);

    const model = this.client.getGenerativeModel({
      model: options?.model || this.model_id,
      systemInstruction: systemInstruction || undefined,
    });

    const requestConfig = this.buildRequestConfig(messages, thread, options);

    try {
      const result = await model.generateContent(requestConfig);
      const response = result.response;

      // Check if there are function calls - return them without executing, store raw_body
      const functionCalls = response.functionCalls();
      if (functionCalls && functionCalls.length > 0) {
        const toolCalls: ToolCall[] = functionCalls.map((fc) => ({
          id: fc.name, // Gemini doesn't use IDs, use function name as ID
          name: fc.name,
          arguments: fc.args,
          raw_body: {
            functionCall: {
              name: fc.name,
              args: fc.args,
            },
          },
        }));

        return toolCalls;
      }

      // Extract thinking content if present
      let thinkingText = "";
      if (response.candidates && response.candidates[0]) {
        const candidate = response.candidates[0];
        if (candidate.content && candidate.content.parts) {
          for (const part of candidate.content.parts) {
            const partAny = part as any;
            if (partAny.thought) {
              thinkingText += part.text;
            }
          }
        }
      }

      // No tool calls, return the content
      const content = response.text() || "";

      // Combine thinking and content if both present
      const fullContent = thinkingText
        ? `__THINKING__${thinkingText}\n\n${content}`
        : content;

      // Parse with schema if present (use content without thinking)
      if (thread.schema && content) {
        try {
          return parseWithSchema(thread.schema, content);
        } catch (error: any) {
          throw new Error(`Schema validation failed: ${error.message}`);
        }
      }

      return fullContent;
    } catch (error: any) {
      throw new Error(`Gemini API Error: ${error.message}`);
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
