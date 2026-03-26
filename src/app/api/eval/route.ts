import { NextResponse } from "next/server";
import { runRagAssessment } from "@/lib/eval";

export async function POST(req: Request) {
  try {
    const { query, context, answer } = await req.json();

    console.log("Evaluation request received:", { query, answer, contextLength: context?.length });

    if (!query || !context || !answer) {
      console.error("Evaluation failed: Missing required fields");
      return NextResponse.json(
        { error: "Missing required fields: query, context, or answer" },
        { status: 400 }
      );
    }

    const assessment = await runRagAssessment(query, context, answer);
    console.log("Assessment completed:", assessment);

    return NextResponse.json(assessment);
  } catch (error) {
    console.error("CRITICAL error in evaluation route:", error);
    if (error instanceof Error) {
      console.error("Error stack:", error.stack);
    }
    return NextResponse.json(
      { error: "Failed to run RAG assessment", details: error instanceof Error ? error.message : String(error) },
      { status: 500 }
    );
  }
}
