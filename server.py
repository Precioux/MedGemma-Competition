from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
import datetime

# --- Import Agents ---
from agents.interview_simulator import InterviewSimulator
from agents.quantitative_assessor_f import QuantitativeAssessor as QuantitativeAssessorF
from agents.qualitative_assessor_f import QualitativeAssessor as QualitativeAssessorF_cls
from agents.qualitative_assessor_z import QualitativeAssessor as QualitativeAssessorZ_cls
from agents.meta_reviewer import MetaReviewerAgent
from agents.qualitative_evaluator import QualitativeEvaluatorAgent  # fixed spelling
from agents.quantitative_assessor_z import QuantitativeAssessorZ

app = FastAPI(title="AI Psychiatrist Pipeline", version="1.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Serve the webpage at http://localhost:8000/ ---
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    html_path = Path("index.html")
    if not html_path.exists():
        return HTMLResponse(content="<h2>index.html not found. Place it in the project root.</h2>", status_code=404)
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# --- File Upload Endpoint ---
@app.post("/upload_transcript")
async def upload_transcript(file: UploadFile = File(...)):
    content = await file.read()
    save_path = Path("data/transcripts/uploaded_transcript.txt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(content)
    line_count = content.decode("utf-8", errors="replace").count("\n")
    return {
        "filename": file.filename,
        "saved_to": str(save_path),
        "size_bytes": len(content),
        "line_count": line_count,
    }


def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# --- Initialize Shared Agents ---
interview_loader = InterviewSimulator()
qualitative_assessor_F = QualitativeAssessorF_cls()
qualitative_assessor_Z = QualitativeAssessorZ_cls()
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
        # Always use uploaded transcript if it exists
        uploaded_path = Path("data/transcripts/uploaded_transcript.txt")
        loader = InterviewSimulator(default_path=str(uploaded_path)) if uploaded_path.exists() else interview_loader
        conversation = loader.load()
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
        print(f"[{get_timestamp()}] [DEBUG] Quantitative raw output:\n{quantitative_result}\n")
    except Exception as e:
        print(f"[{get_timestamp()}] [CRITICAL ERROR] Step 1 failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # --- STEP 2: QUALITATIVE ASSESSMENT ---
    print(f"[{get_timestamp()}] [STEP 2] Invoking: QualitativeAssessor Agent...")
    ql_agent = qualitative_assessor_F if request.mode == 1 else qualitative_assessor_Z
    qualitative_result = ql_agent.assess(conversation)
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
        "qualitative_evaluation": qualitative_evaluation,
        "meta_review": final_review,
    }
