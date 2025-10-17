/**
 * Environment variable utilities for TuneAPI
 * Provides a consistent interface for accessing API tokens
 */

/**
 * Environment variable accessor with fallback support
 */
export const ENV = {
  /**
   * Get OpenAI API token from environment
   * @param defaultValue - Optional default value if env var not found
   * @returns The API token or default value
   */
  OPENAI_TOKEN: (defaultValue: string = ""): string => {
    return process.env.OPENAI_TOKEN || defaultValue;
  },

  /**
   * Get Gemini API token from environment
   * @param defaultValue - Optional default value if env var not found
   * @returns The API token or default value
   */
  GEMINI_TOKEN: (defaultValue: string = ""): string => {
    return process.env.GEMINI_TOKEN || defaultValue;
  },

  /**
   * Get Anthropic API token from environment
   * @param defaultValue - Optional default value if env var not found
   * @returns The API token or default value
   */
  ANTHROPIC_TOKEN: (defaultValue: string = ""): string => {
    return process.env.ANTHROPIC_TOKEN || defaultValue;
  },

  /**
   * Get Mistral API token from environment
   * @param defaultValue - Optional default value if env var not found
   * @returns The API token or default value
   */
  MISTRAL_TOKEN: (defaultValue: string = ""): string => {
    return process.env.MISTRAL_TOKEN || defaultValue;
  },

  /**
   * Get Groq API token from environment
   * @param defaultValue - Optional default value if env var not found
   * @returns The API token or default value
   */
  GROQ_TOKEN: (defaultValue: string = ""): string => {
    return process.env.GROQ_TOKEN || defaultValue;
  },
};
