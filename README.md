# 🧠 MedGemma AI Psychiatrist Assistant

> An AI-powered depression screening assistant built on [MedGemma](https://huggingface.co/google/medgemma), Google's medical-domain fine-tuned LLM. Designed to **support — not replace** — clinical psychiatrists by providing structured, evidence-backed first-pass analysis of psychiatric interview transcripts.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![Ollama](https://img.shields.io/badge/Ollama-MedGemma-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Demo

> 🎬 **[Watch the full demo video on YouTube](#)** ← replace with your YouTube link

![Input and pipeline progress](data/1_new.png)

![PHQ-8 quantitative scores](data/2.png)

![Qualitative risk factor analysis and final diagnosis](data/3.png)

---

## Overview

MedGemma AI Psychiatrist Assistant is a multi-agent clinical pipeline that analyzes psychiatric interview transcripts to screen for depression. It uses the **PHQ-8** (Patient Health Questionnaire-8) as its clinical framework and produces both quantitative scores and qualitative risk assessments, evaluated and synthesized by a chain of specialized AI agents.

The system is intended as a **clinical decision support tool** — providing psychiatrists with a structured, evidence-backed pre-analysis so they can focus their expertise on diagnosis and treatment rather than documentation.

The system supports two assessment modes:
- **Zero-Shot (Z):** MedGemma reasons directly from the transcript with no reference examples — fast and concise
- **Few-Shot (F):** Uses **RAG (Retrieval-Augmented Generation)** — an embedding model retrieves the most clinically similar cases from the DAIC-WOZ database and provides them as examples to guide MedGemma's reasoning, improving output depth and clinical alignment. Prompts were co-designed with practicing psychiatrists.

---

## Pipeline Architecture

```
Transcript Input
      │
      ▼
┌─────────────────────┐
│  InterviewSimulator │  Step 0 — Load & validate transcript
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ QuantitativeAssessor│  Step 1 — PHQ-8 scoring (0–3 per symptom)
│   (Z or F variant)  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ QualitativeAssessor │  Step 2 — Risk factor analysis
│   (Z or F variant)  │           (social, biological, overall)
│   Few-Shot uses RAG │           RAG retrieves similar DAIC-WOZ cases
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ QualitativeEvaluator│  Step 3 — Assessment quality scoring
│                     │           (coherence, completeness,
│                     │            accuracy, specificity)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│    MetaReviewer     │  Step 4 — Final diagnosis & severity
│                     │           (PHQ-8 total, binary label,
│                     │            severity 0–4)
└─────────────────────┘
```

---

## Agents

| # | Agent | Role |
|---|-------|------|
| 0 | **InterviewSimulator** | Loads and validates E-DAIC format transcripts |
| 1 | **QuantitativeAssessor** | Scores all 8 PHQ-8 items (0–3 or N/A) with transcript evidence |
| 2 | **QualitativeAssessor** | Analyzes social, biological, and overall mental health factors. Few-Shot variant uses RAG over DAIC-WOZ |
| 3 | **QualitativeEvaluator** | Scores the qualitative assessment on 4 metrics (1–5 each) |
| 4 | **MetaReviewer** | Synthesizes all outputs into a final diagnosis and severity label |

---

## Assessment Modes

### Zero-Shot (Z)
MedGemma receives the transcript and reasons directly from its medical knowledge. Fast, lightweight, and effective for straightforward cases.

### Few-Shot (F) with RAG
An embedding model encodes the input transcript and retrieves the most clinically similar interview cases from the **DAIC-WOZ** (Distress Analysis Interview Corpus) database. These retrieved examples are provided alongside the transcript as few-shot context, guiding MedGemma to produce more structured, clinically grounded outputs. The qualitative prompts used in this mode were co-designed with a team of practicing psychiatrists.

---

## PHQ-8 Severity Scale

| Score | Level |
|-------|-------|
| 0–4 | No significant depressive symptoms |
| 5–9 | Mild |
| 10–14 | Moderate |
| 15–19 | Moderately Severe |
| 20–24 | Severe |

---

## Web Interface

The project includes a browser-based UI served at `http://localhost:8000`:

- Upload `.txt` or `.csv` transcript files (E-DAIC format, UTF-8)
- Select Zero-Shot or Few-Shot (RAG) assessment mode
- Step-by-step pipeline progress tracking
- PHQ-8 score grid with color-coded severity bar
- Structured qualitative risk factor display with exact transcript quotes
- Assessment quality scores (coherence, completeness, accuracy, specificity)
- Final diagnosis badge with severity level

---

## Design Philosophy

This system was built in close collaboration with practicing psychiatrists. Three core principles guided every decision:

- **Simple** — no technical knowledge required from the clinician
- **Lightweight** — runs entirely on an ordinary laptop using MedGemma's 4B parameter model
- **Private** — no cloud connection, no external API calls; patient data never leaves the device

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) running locally
- MedGemma model pulled via Ollama:

```bash
ollama pull alibayram/medgemma
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Precioux/MedGemma-Competition.git
cd MedGemma-Competition

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Make sure Ollama is running and MedGemma is loaded
ollama serve
ollama pull alibayram/medgemma
```

---

## Usage

### Launch the Web Interface

```bash
python main.py
```

Open your browser at `http://localhost:8000`.

1. Upload a transcript file (`.txt` or `.csv`, E-DAIC format)
2. Select **Zero-Shot** or **Few-Shot** mode
3. Click **Run Pipeline**
4. Results appear after ~2–5 minutes depending on transcript length

### API

```bash
# Run full pipeline (mode 0 = Zero-Shot, mode 1 = Few-Shot with RAG)
curl -X POST http://localhost:8000/full_pipeline \
  -H "Content-Type: application/json" \
  -d '{"mode": 0}'

# Upload a transcript
curl -X POST http://localhost:8000/upload_transcript \
  -F "file=@your_transcript.txt"
```

---

## Project Structure

```
MedGemma-Competition/
├── agents/
│   ├── interview_simulator.py       # Step 0: Transcript loader
│   ├── quantitative_assessor_z.py   # Step 1: PHQ-8 scoring (Zero-Shot)
│   ├── quantitative_assessor_f.py   # Step 1: PHQ-8 scoring (Few-Shot + RAG)
│   ├── qualitative_assessor_z.py    # Step 2: Risk analysis (Zero-Shot)
│   ├── qualitative_assessor_f.py    # Step 2: Risk analysis (Few-Shot + RAG)
│   ├── qualitative_evaluator.py     # Step 3: Assessment quality scoring
│   └── meta_reviewer.py             # Step 4: Final diagnosis synthesis
├── data/
│   ├── transcripts/                 # Input transcript files
│   ├── source/                      # Raw DAIC-WOZ CSV source files
│   ├── embeddings/                  # Precomputed DAIC-WOZ embeddings for RAG
│   ├── 1.png                        # Screenshot: input & pipeline
│   ├── 2.png                        # Screenshot: PHQ-8 scores
│   └── 3.png                        # Screenshot: qualitative analysis
├── index.html                       # Web UI
├── server.py                        # FastAPI server
├── main.py                          # Application launcher
└── requirements.txt
```

---

## Data Format

Transcripts should follow the **E-DAIC** (Extended Distress Analysis Interview Corpus) format — plain text with alternating speaker turns:

```
Ellie: How are you feeling today?
Participant: I've been feeling pretty down lately.
Ellie: Can you tell me more about that?
...
```

---

## Model

This project uses **[alibayram/medgemma](https://ollama.com/alibayram/medgemma)** via Ollama — a fine-tuned variant of Google's MedGemma optimized for medical dialogue and psychiatric assessment tasks.

**Hardware recommendations:**
- Apple Silicon (M1/M2/M3) — runs fully on GPU via Metal
- NVIDIA GPU with 8GB+ VRAM recommended for best performance
- CPU inference supported but significantly slower

---

## Assessment Quality Metrics

The `QualitativeEvaluator` scores the qualitative assessment on four axes:

| Metric | Description |
|--------|-------------|
| **Coherence** | Logical consistency, no contradictions |
| **Completeness** | Coverage of all relevant PHQ-8 symptoms |
| **Accuracy** | Alignment with DSM-5 / PHQ-8 criteria |
| **Specificity** | Evidence-backed claims, no vague generalities |

Each metric is scored 1–5 where 5 = no errors.

---

## Reference

> **AI Psychiatrist Assistant: An LLM-based Multi-Agent System for Depression Assessment from Clinical Interviews**
> [https://openreview.net/forum?id=mV0xJpO7A0](https://openreview.net/forum?id=mV0xJpO7A0)

---

## Acknowledgements

- [Google MedGemma](https://developers.google.com/health-ai-developer-foundations/medgemma) — base medical LLM
- [DAIC-WOZ Dataset](https://dcapswoz.ict.usc.edu/) — depression interview corpus used for RAG retrieval
- [PHQ-8](https://www.phqscreeners.com/) — Patient Health Questionnaire

---

## Author

**Precioux** — [@Precioux](https://github.com/Precioux)

---

> ⚠️ **Disclaimer:** This tool is intended to **assist** qualified psychiatrists and is not a substitute for professional clinical evaluation. All outputs must be reviewed by a licensed clinician before informing any diagnostic or treatment decision.
