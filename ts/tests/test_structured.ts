/**
 * Example: Structured Output Generation with TuneAPI
 *
 * This demonstrates how to use Zod schemas with OpenAI, Gemini, and Anthropic
 * to get type-safe, validated responses from LLMs.
 */
import "dotenv/config"; // Add this line at the very top

import { createThread, human, system, z } from "../index";
import {
  parseArgs,
  validateModelType,
  runAllModels,
  createModel,
  logFunction,
} from "./test_runner";

// ============================================================================
// Define Schemas using Zod (similar to Pydantic in Python)
// ============================================================================

/**
 * Simple schema: Extract person information
 * Note: OpenAI's strict mode requires all fields to be present in the response.
 * Optional fields will still be required by OpenAI, so use empty string/null as default.
 */
const PersonSchema = z.object({
  name: z.string().describe("The person's full name"),
  age: z.number().describe("The person's age in years"),
  occupation: z.string().describe("The person's job or profession"),
  location: z
    .string()
    .describe("Where the person lives, or 'unknown' if not mentioned"),
});

/**
 * Complex nested schema: Movie recommendation with details
 */
const MovieSchema = z.object({
  title: z.string().describe("The movie title"),
  year: z.number().describe("Release year"),
  genre: z.array(z.string()).describe("List of genres"),
  director: z.string().describe("Director's name"),
  rating: z.number().min(0).max(10).describe("Rating out of 10"),
  summary: z.string().describe("Brief plot summary"),
  cast: z
    .array(
      z.object({
        actor: z.string(),
        character: z.string(),
      })
    )
    .describe("Main cast members"),
});

/**
 * Array schema: Extract multiple entities
 */
const ShoppingListSchema = z.object({
  items: z
    .array(
      z.object({
        name: z.string(),
        quantity: z.number(),
        category: z.enum([
          "produce",
          "dairy",
          "meat",
          "bakery",
          "pantry",
          "other",
        ]),
        urgent: z.boolean(),
      })
    )
    .describe("List of items to buy"),
  estimatedTotal: z.number().describe("Estimated total cost in dollars"),
});

// ============================================================================
// Examples
// ============================================================================

async function examplePersonExtraction(model: any, modelName: string) {
  console.log(`\n=== Example 1: Person Extraction (${modelName}) ===\n`);
  const thread = createThread(
    human(
      "John Smith is a 35-year-old software engineer living in San Francisco."
    )
  );
  thread.schema = PersonSchema;

  const result = await model.chat(thread);

  // TypeScript knows the exact type!
  console.log("Extracted person:", result);
  console.log(`Name: ${result.name}`);
  console.log(`Age: ${result.age}`);
  console.log(`Occupation: ${result.occupation}`);
  console.log(`Location: ${result.location}`);
}

async function exampleMovieRecommendation(model: any, modelName: string) {
  console.log(`\n=== Example 2: Complex Nested Schema (${modelName}) ===\n`);

  const thread = createThread(
    system(
      "You are a movie expert. Recommend movies based on user preferences."
    ),
    human("Recommend a sci-fi movie from the 2010s with good ratings")
  );
  thread.schema = MovieSchema;

  const result = await model.chat(thread);

  console.log("Movie Recommendation:");
  console.log(`Title: ${result.title} (${result.year})`);
  console.log(`Genre: ${result.genre.join(", ")}`);
  console.log(`Director: ${result.director}`);
  console.log(`Rating: ${result.rating}/10`);
  console.log(`Summary: ${result.summary}`);
  console.log("\nMain Cast:");
  result.cast.forEach((member: any) => {
    console.log(`  - ${member.actor} as ${member.character}`);
  });
}

async function exampleShoppingList(model: any, modelName: string) {
  console.log(`\n=== Example 3: Array Schema (${modelName}) ===\n`);

  const thread = createThread(
    human(
      "I need to make a pasta dinner and a salad. What should I buy? " +
        "I need 2 pounds of pasta, some tomatoes, basil, garlic, " +
        "lettuce, olive oil, and parmesan cheese. The tomatoes are urgent."
    )
  );
  thread.schema = ShoppingListSchema;

  const result = await model.chat(thread);

  console.log("Shopping List:");
  console.log(`Estimated Total: $${result.estimatedTotal.toFixed(2)}\n`);
  console.log("Items:");
  result.items.forEach((item: any, idx: number) => {
    const urgentTag = item.urgent ? " [URGENT]" : "";
    console.log(
      `  ${idx + 1}. ${item.name} - ${item.quantity} (${
        item.category
      })${urgentTag}`
    );
  });
}

async function exampleWithOptions(model: any, modelName: string) {
  console.log(`\n=== Example 4: Schema with Chat Options (${modelName}) ===\n`);

  const thread = createThread(
    human("Extract info: Sarah Johnson, 28, marine biologist in Miami")
  );
  thread.schema = PersonSchema;

  // You can still use all the normal chat options
  const result = await model.chat(thread, {
    temperature: 0.3,
    max_tokens: 200,
  });

  console.log("Extracted with low temperature:", result);
}

async function exampleErrorHandling(model: any, modelName: string) {
  console.log(`\n=== Example 5: Error Handling (${modelName}) ===\n`);

  // Schema with strict validation
  const StrictAgeSchema = z.object({
    name: z.string(),
    age: z.number().min(0).max(120), // Age must be between 0-120
  });

  const thread = createThread(human("John is 25 years old"));
  thread.schema = StrictAgeSchema;

  try {
    const result = await model.chat(thread);
    console.log("Valid result:", result);
  } catch (error: any) {
    console.error("Validation error:", error.message);
  }
}

async function exampleInferredTypes(model: any, modelName: string) {
  console.log(`\n=== Example 6: Type Inference (${modelName}) ===\n`);

  // Define inline schema
  const UserProfileSchema = z.object({
    username: z.string(),
    email: z.string().email(),
    bio: z.string(),
    age: z.number(),
    interests: z.array(z.string()),
  });

  // TypeScript automatically infers the type
  type UserProfile = z.infer<typeof UserProfileSchema>;

  const thread = createThread(
    human(
      "Create a profile: username is 'johndoe', email john@example.com, " +
        "bio is 'Love hiking and photography', age 30, interests are hiking, photography, coding"
    )
  );
  thread.schema = UserProfileSchema;

  const profile: UserProfile = await model.chat(thread);

  // TypeScript knows all these properties exist and their types!
  console.log(`Username: ${profile.username}`);
  console.log(`Email: ${profile.email}`);
  console.log(`Bio: ${profile.bio}`);
  console.log(`Age: ${profile.age}`);
  console.log(`Interests: ${profile.interests.join(", ")}`);
}

// ============================================================================
// Main
// ============================================================================

async function runAllExamples(model: any, modelName: string) {
  await logFunction(examplePersonExtraction, "examplePersonExtraction")(
    model,
    modelName
  );
  await logFunction(exampleMovieRecommendation, "exampleMovieRecommendation")(
    model,
    modelName
  );
  await logFunction(exampleShoppingList, "exampleShoppingList")(
    model,
    modelName
  );
  await logFunction(exampleWithOptions, "exampleWithOptions")(model, modelName);
  await logFunction(exampleErrorHandling, "exampleErrorHandling")(
    model,
    modelName
  );
  await logFunction(exampleInferredTypes, "exampleInferredTypes")(
    model,
    modelName
  );
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
      await runAllModels("test_structured", runAllExamples);
    } else {
      console.log("=".repeat(70));
      console.log("TuneAPI Structured Output Examples");
      console.log(`Running tests with: ${modelType.toUpperCase()}`);
      console.log("=".repeat(70));

      const { model, modelName } = createModel(
        modelType,
        modelType === "openai"
          ? "gpt-4o-mini"
          : modelType === "gemini"
          ? "gemini-2.0-flash-exp"
          : undefined
      );

      await runAllExamples(model, modelName);

      console.log("\n" + "=".repeat(70));
      console.log("All examples completed!");
      console.log("=".repeat(70));
    }
  } catch (error: any) {
    console.error("\nError:", error.message);
  }
}

// Run the tests
main();
