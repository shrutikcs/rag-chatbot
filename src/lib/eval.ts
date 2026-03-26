import { generateText } from "ai";
import { groq } from "@ai-sdk/groq";

interface EvalResult {
  score: number;
  reasoning: string;
}

/**
 * Evaluates the faithfulness of an answer given the retrieved context.
 * Faithfulness: Is every claim in the answer supported by the context?
 */
export async function evaluateFaithfulness(
  query: string,
  answer: string,
  context: string
): Promise<EvalResult> {
  const { text } = await generateText({
    model: groq("llama-3.3-70b-versatile"),
    system: "You are an expert evaluator for RAG systems. Your task is to judge the FAITHFULNESS of an answer based ONLY on the provided context.",
    prompt: `
    QUERY: ${query}
    CONTEXT: ${context}
    ANSWER: ${answer}

    Is the answer faithful to the context? Does it make any claims not supported by the context?
    Provide a score between 0 and 1 (where 1 is perfectly faithful) and a brief reasoning.
    
    Output format:
    Score: [score]
    Reasoning: [brief explanation]
    `,
  });

  const scoreMatch = text.match(/Score:\s*([\d.]+)/);
  const reasoningMatch = text.match(/Reasoning:\s*([\s\S]*)/);

  return {
    score: scoreMatch ? parseFloat(scoreMatch[1]) : 0,
    reasoning: reasoningMatch ? reasoningMatch[1].trim() : "No reasoning provided",
  };
}

/**
 * Evaluates the relevance of an answer to the original query.
 */
export async function evaluateAnswerRelevance(
  query: string,
  answer: string,
  context: string
): Promise<EvalResult> {
  const { text } = await generateText({
    model: groq("llama-3.3-70b-versatile"),
    system: "You are an expert evaluator for RAG systems. Your task is to judge the RELEVANCE of an answer to the user's query.",
    prompt: `
    QUERY: ${query}
    CONTEXT: ${context}
    ANSWER: ${answer}

    How relevant is the answer to the query? Does it address the user's intent?
    Provide a score between 0 and 1 (where 1 is perfectly relevant) and a brief reasoning.
    
    Output format:
    Score: [score]
    Reasoning: [brief explanation]
    `,
  });

  const scoreMatch = text.match(/Score:\s*([\d.]+)/);
  const reasoningMatch = text.match(/Reasoning:\s*([\s\S]*)/);

  return {
    score: scoreMatch ? parseFloat(scoreMatch[1]) : 0,
    reasoning: reasoningMatch ? reasoningMatch[1].trim() : "No reasoning provided",
  };
}

/**
 * Runs a full RAG assessment.
 */
export async function runRagAssessment(
  query: string,
  context: string,
  answer: string
) {
  const [faithfulness, relevance] = await Promise.all([
    evaluateFaithfulness(query, answer, context),
    evaluateAnswerRelevance(query, answer, context),
  ]);

  return {
    faithfulness: {
      score: faithfulness.score,
      metadata: { reasoning: faithfulness.reasoning }
    },
    relevance: {
      score: relevance.score,
      metadata: { reasoning: relevance.reasoning }
    },
    averageScore: (faithfulness.score + relevance.score) / 2,
  };
}
