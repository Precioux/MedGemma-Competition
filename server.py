from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import datetime

# --- Import Agents ---
from agents.interview_simulator import InterviewSimulator
from agents.quantitative_assessor_f import QuantitativeAssessor as QuantitativeAssessorF
from agents.qualitative_assessor_f import QualitativeAssessor
from agents.meta_reviewer import MetaReviewerAgent
from agents.qualitive_evaluator import QualitativeEvaluatorAgent
from agents.quantitative_assessor_z import QuantitativeAssessorZ

app = FastAPI(title="AI Psychiatrist Pipeline", version="1.2.0")


def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# --- Initialize Shared Agents ---
interview_loader = InterviewSimulator()
qualitative_assessor = QualitativeAssessor()
qualitative_evaluator = QualitativeEvaluatorAgent()
meta_reviewer = MetaReviewerAgent()
quantitative_assessor_F = QuantitativeAssessorF()
quantitative_assessor_Z = QuantitativeAssessorZ()


class InterviewRequest(BaseModel):
    mode: int = Field(0, ge=0, le=1, description="0=zero-shot (Z), 1=few-shot (F)")


@app.post("/full_pipeline")
def run_full_pipeline(request: InterviewRequest):
    pipeline_start = datetime.datetime.now()

    print(f"\n" + "=" * 70)
    print(f"[{get_timestamp()}] 🚀 PIPELINE EXECUTION STARTED")
    print(f"[{get_timestamp()}] Mode: {'Few-Shot (F)' if request.mode == 1 else 'Zero-Shot (Z)'}")
    print("=" * 70)

    # --- STEP 0: INTERVIEW LOADING ---
    print(f"[{get_timestamp()}] [STEP 0] Invoking: InterviewSimulator Agent...")
    try:
        conversation = interview_loader.load()
        print(f"[{get_timestamp()}] [SUCCESS] Transcript loaded successfully.")
    except Exception as e:
        print(f"[{get_timestamp()}] [FAILED] Step 0 error: {e}")
        raise HTTPException(status_code=500, detail=f"Load error: {e}")

    # --- STEP 1: QUANTITATIVE ASSESSMENT ---
    q_variant = "F" if request.mode == 1 else "Z"
    q_agent = quantitative_assessor_F if request.mode == 1 else quantitative_assessor_Z

    print(f"[{get_timestamp()}] [STEP 1] Invoking: QuantitativeAssessor (Variant: {q_variant})...")
    try:
        quantitative_result = q_agent.assess(conversation)
        print(f"[{get_timestamp()}] [SUCCESS] Quantitative PHQ-8 scores generated.")
    except Exception as e:
        print(f"[{get_timestamp()}] [CRITICAL ERROR] Step 1 failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # --- STEP 2: QUALITATIVE ASSESSMENT ---
    print(f"[{get_timestamp()}] [STEP 2] Invoking: QualitativeAssessor Agent...")
    qualitative_result = qualitative_assessor.assess(conversation)
    print(f"[{get_timestamp()}] [SUCCESS] Qualitative analysis complete.")

    # --- STEP 3: QUALITATIVE EVALUATION ---
    print(f"[{get_timestamp()}] [STEP 3] Invoking: QualitativeEvaluator Agent...")
    qualitative_evaluation = qualitative_evaluator.assess(conversation, qualitative_result)
    print(f"[{get_timestamp()}] [SUCCESS] Qualitative evaluation complete.")

    # --- STEP 4: META REVIEW ---
    print(f"[{get_timestamp()}] [STEP 4] Invoking: MetaReviewer Agent (Final Synthesis)...")
    final_review = meta_reviewer.review(
        interview=conversation,
        quantitative=quantitative_result,
        qualitative=qualitative_result
    )
    print(f"[{get_timestamp()}] [SUCCESS] Meta-review generated.")

    # --- PIPELINE SUMMARY ---
    duration = datetime.datetime.now() - pipeline_start
    print("=" * 70)
    print(f"[{get_timestamp()}] ✅ PIPELINE COMPLETE | Total Duration: {duration}")
    print("=" * 70 + "\n")

    return {
        "mode": request.mode,
        "execution_time": str(duration),
        "qualitative_result": qualitative_result,
        "quantitative_score": quantitative_result,
        "meta_review": final_review
    }