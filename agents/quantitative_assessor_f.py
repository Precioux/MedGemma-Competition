#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import math
import time
import pickle
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------- Verbose switch -----------------------------
VERBOSE = True  # use --quiet to disable

# ----------------------------- Keys -----------------------------
PHQ8_KEYS = [
    "PHQ8_NoInterest","PHQ8_Depressed","PHQ8_Sleep","PHQ8_Tired",
    "PHQ8_Appetite","PHQ8_Failure","PHQ8_Concentrating","PHQ8_Moving"
]
def _empty_item(reason: str) -> Dict[str, Any]:
    return {"evidence": "No relevant evidence found", "reason": reason, "score": "N/A"}

def _phq8_skeleton(reason: str) -> Dict[str, Any]:
    return {k: _empty_item(reason) for k in PHQ8_KEYS}

def _extract_json_between_answer(raw: str) -> Optional[str]:
    m = re.search(r"<answer>\s*(\{.*?\})\s*</answer>", raw, flags=re.S)
    if not m:
        return None
    block = m.group(1)
    # normalize curly/smart quotes and strip trailing commas before } or ]
    block = (block.replace("“", '"').replace("”", '"')
                  .replace("’", "'").replace("‘", "'")
                  .replace("\u200b", ""))
    block = re.sub(r",\s*([}\]])", r"\1", block)
    return block

def _llm_json_repair(ollama_host: str, model: str, broken: str, timeout: int = 120) -> Optional[Dict[str, Any]]:
    """Ask the model to repair malformed JSON to EXACT keys; return dict or None."""
    repair_system = ""
    repair_user = (
        "You will be given malformed JSON for a PHQ-8 result. "
        "Output ONLY a valid JSON object with these EXACT keys:\n"
        f"{', '.join(PHQ8_KEYS)}\n"
        'Each value must be an object: {"evidence": <string>, "reason": <string>, "score": <int 0-3 or "N/A">}.\n'
        "If something is missing or unclear, fill with "
        '{"evidence":"No relevant evidence found","reason":"Auto-repaired","score":"N/A"}.\n\n'
        "Malformed JSON:\n"
        f"{broken}\n\n"
        "Return only the fixed JSON. No prose, no markdown, no tags."
    )
    try:
        fixed = ollama_chat(ollama_host, model, repair_system, repair_user, timeout=timeout)
        fixed = _strip_json_block(fixed)  # in case it adds fences
        fixed = _tolerant_fixups(fixed)
        return json.loads(fixed)
    except Exception:
        return None

def _sentences(txt: str) -> List[str]:
    parts = re.split(r'(?<=[\.\?\!])\s+|\n+', txt.strip())
    return [p.strip(" \t-") for p in parts if p and len(p.strip()) > 0]

# ----------------------------- Utils -----------------------------
def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

# def _log(msg: str):
#     if VERBOSE:
#         print(f"[{_now()}] {msg}")

def _log(msg: str):
    if VERBOSE:
        print(f"[{_now()}] {msg}")

def _log_header(title: str):
    print("\n" + "="*60)
    print(f" >>> {title.upper()} <<< ")
    print("="*60 + "\n")

def _log_substep(name: str, detail: str = ""):
    print(f"  [➔] {name}: {detail}")
def _l2norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def _strip_json_block(s: str) -> str:
    t = s.strip()
    if "<answer>" in t and "</answer>" in t:
        t = t.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    if t.startswith("```json"):
        t = t[len("```json"):].strip()
        if t.endswith("```"):
            t = t[:-3].strip()
    elif t.startswith("```"):
        t = t[len("```"):].strip()
        if t.endswith("```"):
            t = t[:-3].strip()
    if "{" in t and "}" in t:
        t = t[t.find("{"): t.rfind("}") + 1]
    t = t.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    t = re.sub(r",\s*([}\]])", r"\1", t)
    return t


def _tolerant_fixups(s: str) -> str:
    # 1. Standardize quotes
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")

    # 2. Fix missing commas between JSON fields (Crucial for MedGemma)
    s = re.sub(r'(\d|true|false|"N/A")\s*\n?\s*"', r'\1, "', s)

    # 3. Fix missing commas between objects
    s = re.sub(r'\}\s*\n?\s*\{', r'}, {', s)

    # 4. Remove trailing commas before closing braces
    s = re.sub(r",\s*([}\]])", r"\1", s)

    # 5. Remove any invisible characters
    s = s.strip().replace('\u200b', '')
    return s

def _normalize_item(v: Any) -> Dict[str, Any]:
    if not isinstance(v, dict):
        v = {}
    ev = v.get("evidence", "No relevant evidence found")
    if isinstance(ev, list):
        ev = [str(x).strip() for x in ev if str(x).strip()]
        ev_str = ev[0] if ev else "No relevant evidence found"
    else:
        ev_str = str(ev).strip() or "No relevant evidence found"
    reason = v.get("reason", "")
    if not isinstance(reason, str):
        reason = str(reason)
    score = v.get("score", "N/A")
    if isinstance(score, str):
        s = score.strip().upper()
        if s == "N/A":
            score = "N/A"
        else:
            try:
                score = int(s)
            except Exception:
                score = "N/A"
    if isinstance(score, int):
        score = max(0, min(3, score))
    elif score != "N/A":
        score = "N/A"
    return {"evidence": ev_str, "reason": reason, "score": score}

def _validate_and_normalize(obj: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in PHQ8_KEYS:
        out[k] = _normalize_item(obj.get(k, {}))
    return out

def _compute_total_and_severity(res: Dict[str, Any]) -> Tuple[int, str]:
    total = 0
    for k in PHQ8_KEYS:
        sc = res[k].get("score", "N/A")
        if isinstance(sc, int):
            total += sc
    if total <= 4:
        sev = "minimal"
    elif total <= 9:
        sev = "mild"
    elif total <= 14:
        sev = "moderate"
    elif total <= 19:
        sev = "mod_severe"
    else:
        sev = "severe"
    return total, sev

# ----------------------------- Ollama clients -----------------------------
def ollama_chat(host: str, model: str, system_prompt: str, user_prompt: str, timeout=600) -> str:
    url = f"http://{host}:11434/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 8192,     # Essential for long transcripts
            "num_predict": 2048   # Enough space for the full PHQ-8 JSON
        }
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]["content"]


def ollama_embed(host: str, model: str, text: str, dim: Optional[int] = None, timeout=120) -> List[float]:
    import requests, math

    # Trying both common endpoints for maximum compatibility
    endpoints = ["/api/embed", "/api/embeddings"]
    data = None

    for endpoint in endpoints:
        url = f"http://{host}:11434{endpoint}"
        try:
            # Standard payload for modern Ollama versions
            payload = {"model": model, "input": text}
            r = requests.post(url, json=payload, timeout=timeout)

            if r.status_code == 200:
                data = r.json()
                break
        except Exception:
            continue

    if not data or "error" in data:
        error_msg = data.get("error") if data else "No response"
        raise RuntimeError(f"Ollama embeddings error: {error_msg} (model={model})")

    # Correctly extracting the vector based on the endpoint's response format
    emb = None
    if isinstance(data.get("embedding"), list):
        emb = data["embedding"]
    elif isinstance(data.get("embeddings"), list) and data["embeddings"]:
        # /api/embed returns a list of lists, take the first one
        emb = data["embeddings"][0]

    if not emb:
        raise RuntimeError(f"Empty or invalid embedding response: {data}")

    # Keep your original logic for dimensions and normalization
    if dim is not None:
        emb = emb[:dim]

    # L2 Normalization (keep this for consistent similarity scoring in PHQ-8)
    n = math.sqrt(sum(x * x for x in emb))
    if n > 0:
        emb = [x / n for x in emb]

    _log(f"[EMB] host={host} model={model} dim={len(emb)} text_len={len(text)}")
    return emb
#
# ----------------------------- Prompts (filled) -----------------------------
quantitative_prompt = f"""You are a AI psychiatrist assessment tool specialized in analyzing interview transcripts to evaluate PHQ-8 (Patient Health Questionnaire-8) scores. The PHQ-8 is a validated depression screening instrument that assesses eight of the nine DSM-IV criteria for depressive disorders over the past 2 weeks.

PHQ-8 Scoring System:
- 0 = Not at all (0-1 days in the past 2 weeks)
- 1 = Several days (2-6 days)
- 2 = More than half the days (7-11 days)
- 3 = Nearly every day (12-14 days)

Key Principles:
1. Base all assessments strictly on evidence from the transcript
2. When no relevant evidence exists for a symptom, output "N/A" rather than assuming absence (score of 0)
3. Consider both explicit statements and behavioral indicators in the interview
4. Be conservative - only assign scores when there is clear evidence of frequency
5. Remember that absence of mention does not necessarily mean absence of symptom
6. When reference examples are provided, use them for cross-analysis to inform scoring decisions
7. Weight reference examples based on their similarity to the current evidence - highly similar references should be considered more heavily
8. Follow the format specified exactly
Analyze the following interview transcript and predict PHQ-8 scores for each symptom domain.

Interview transcript:
<transcript>
{{patient_transcript}}
</transcript>

{{reference_evidence}}

Analyze each symptom using the following approach in <thinking> tags:
1. Search for direct quotes or behavioral evidence related to each PHQ-8 symptom
2. When reference examples are provided, compare the current evidence with similar reference cases
3. Evaluate the frequency/severity based on available evidence and reference comparisons
4. Consider how similar the reference examples are to the current evidence - if highly similar, give more weight to the reference scores; if less similar, rely more on direct analysis
5. If no relevant evidence exists, mark as "N/A" rather than assuming absence
6. Only assign numeric scores (0-3) when evidence clearly indicates frequency

After your analysis, provide your final assessment in <answer> tags as a JSON object.

For each symptom, provide:
1. "evidence": exact quotes from transcript (use "No relevant evidence found" if not discussed)
2. "reason": explanation of scoring decision, including cross-reference analysis when applicable and why N/A if applicable
3. "score": integer 0-3 based on evidence, or "N/A" if no relevant evidence

Return ONLY a JSON object in <answer> tags with these exact keys:
- "PHQ8_NoInterest": {{evidence, reason, score}} for little interest or pleasure in doing things (anhedonia)
- "PHQ8_Depressed": {{evidence, reason, score}} for feeling down, depressed, or hopeless (depressed mood)
- "PHQ8_Sleep": {{evidence, reason, score}} for trouble falling or staying asleep, or sleeping too much (sleep problems)
- "PHQ8_Tired": {{evidence, reason, score}} for feeling tired or having little energy (fatigue)
- "PHQ8_Appetite": {{evidence, reason, score}} for poor appetite or overeating (appetite/weight changes)
- "PHQ8_Failure": {{evidence, reason, score}} for feeling bad about yourself or that you are a failure (negative self-perception)
- "PHQ8_Concentrating": {{evidence, reason, score}} for trouble concentrating on things like reading or watching TV (concentration problems)
- "PHQ8_Moving": {{evidence, reason, score}} for moving or speaking slowly, or being fidgety/restless (psychomotor changes)"""

EVIDENCE_EXTRACT_PROMPT = """Analyze the following therapy transcript and extract specific text chunks that provide evidence for each PHQ-8 domain. Based on the evidence found, determine the appropriate PHQ-8 score for each domain.

PHQ-8 scoring reference:
- 0 = not at all (0-1 days)
- 1 = several days (2-6 days)
- 2 = more than half the days (7-11 days)
- 3 = nearly every day (12-14 days)

For each domain below, extract direct quotes from the transcript that indicate the presence, absence, frequency, or severity of symptoms. Then determine the most appropriate score based on the evidence.

PHQ-8 domains:
- nointerest: little interest or pleasure in activities
- depressed: feeling down, depressed, or hopeless
- sleep: sleep problems (trouble falling/staying asleep or sleeping too much)
- tired: feeling tired or having little energy
- appetite: appetite changes (poor appetite or overeating)
- failure: negative self-perception or feeling like a failure
- concentrating: trouble concentrating on tasks
- moving: psychomotor changes (moving/speaking slowly or restlessness)

Return a JSON object with arrays of relevant transcript quotes for each domain. Each quote should be a direct excerpt from the transcript that supports your scoring decision. Include as many relevant quotes as you find - this could range from 0 quotes (if no evidence) to multiple quotes per domain.

Therapy transcript:
{transcript}

Respond with valid JSON matching this structure:
{
    "PHQ8_NoInterest": ["evidence_1", "evidence_2"],
    "PHQ8_Depressed": ["evidence_1"],
    "PHQ8_Sleep": ["evidence_1", "evidence_2"],
    "PHQ8_Tired": ["evidence_1"],
    "PHQ8_Appetite": [],
    "PHQ8_Failure": ["evidence_1"],
    "PHQ8_Concentrating": ["evidence_1"],
    "PHQ8_Moving": ["evidence_1"]
}

Important: Extract UNIQUE quotes only and do not reformat them.
"""

SYSTEM_PROMPT = """You are a AI psychiatrist assessment tool specialized in analyzing interview transcripts to evaluate PHQ-8 (Patient Health Questionnaire-8) scores. The PHQ-8 is a validated depression screening instrument that assesses eight of the nine DSM-IV criteria for depressive disorders over the past 2 weeks.

PHQ-8 Scoring System:
- 0 = Not at all (0-1 days in the past 2 weeks)
- 1 = Several days (2-6 days)
- 2 = More than half the days (7-11 days)
- 3 = Nearly every day (12-14 days)

Key Principles:
1. Base all assessments strictly on evidence from the transcript
2. When no relevant evidence exists for a symptom, output "N/A" rather than assuming absence (score of 0)
3. Consider both explicit statements and behavioral indicators in the interview
4. Be conservative - only assign scores when there is clear evidence of frequency
5. Remember that absence of mention does not necessarily mean absence of symptom
6. When reference examples are provided, use them for cross-analysis to inform scoring decisions
7. Weight reference examples based on their similarity to the current evidence - highly similar references should be considered more heavily
8. Follow the format specified exactly
"""

def make_scoring_user_prompt(transcript: str, reference_bundle: str) -> str:
    return f"""Analyze the following interview transcript and predict PHQ-8 scores for each symptom domain.

Interview transcript:
<transcript>
{transcript}
</transcript>

{reference_bundle}

Analyze each symptom using the following approach in <thinking> tags:
1. Search for direct quotes or behavioral evidence related to each PHQ-8 symptom
2. When reference examples are provided, compare the current evidence with similar reference cases
3. Evaluate the frequency/severity based on available evidence and reference comparisons
4. Consider how similar the reference examples are to the current evidence - if highly similar, give more weight to the reference scores; if less similar, rely more on direct analysis
5. If no relevant evidence exists, mark as "N/A" rather than assuming absence
6. Only assign numeric scores (0-3) when evidence clearly indicates frequency

After your analysis, provide your final assessment in <answer> tags as a JSON object.

For each symptom, provide:
1. "evidence": exact quotes from transcript (use "No relevant evidence found" if not discussed)
2. "reason": explanation of scoring decision, including cross-reference analysis when applicable and why N/A if applicable
3. "score": integer 0-3 based on evidence, or "N/A" if no relevant evidence

Return ONLY a JSON object in <answer> tags with these exact keys:
- "PHQ8_NoInterest": {{evidence, reason, score}}
- "PHQ8_Depressed": {{evidence, reason, score}}
- "PHQ8_Sleep": {{evidence, reason, score}}
- "PHQ8_Tired": {{evidence, reason, score}}
- "PHQ8_Appetite": {{evidence, reason, score}}
- "PHQ8_Failure": {{evidence, reason, score}}
- "PHQ8_Concentrating": {{evidence, reason, score}}
- "PHQ8_Moving": {{evidence, reason, score}}"""

# ----------------------------- Retrieval -----------------------------
def _find_similar_chunks(q_emb: List[float], pet: Dict[int, List[Tuple[str, List[float]]]], top_k: int) -> List[Dict[str, Any]]:
    sims = []
    q = [q_emb]
    for pid, pairs in pet.items():
        for raw, emb in pairs:
            sim = cosine_similarity(q, [emb])[0][0]
            sims.append({"participant_id": int(pid), "raw_text": raw, "similarity": float(sim)})
    sims.sort(key=lambda x: x["similarity"], reverse=True)
    return sims[:top_k]

def _build_reference_for_item(
    evidence_quotes: List[str],
    item_key: str,
    ollama_host: str,
    emb_model: str,
    top_k: int,
    pet: Dict[int, List[Tuple[str, List[float]]]],
    phq_df: pd.DataFrame,
    dim: Optional[int] = None,
    min_chars: int = 8
) -> Tuple[str, List[str]]:
    text = "\n".join(evidence_quotes or [])
    if len(text) < min_chars:
        return "<Reference Examples>\nNo valid evidence found\n<Reference Examples>", []
    emb = ollama_embed(ollama_host, emb_model, text, dim=dim)
    hits = _find_similar_chunks(emb, pet, top_k)
    lines, sims = [], []
    for h in hits:
        pid = h["participant_id"]
        sims.append(f"{h['similarity']:.4f}")
        row = phq_df.loc[phq_df["Participant_ID"] == pid]
        if row.empty:
            continue
        val = row[item_key].values[0]
        try:
            val = int(val)
        except Exception:
            val = "N/A"
        lines.append(f"({item_key} Score: {val})\n{h['raw_text']}")
    if not lines:
        return "<Reference Examples>\nNo valid evidence found\n<Reference Examples>", sims
    block = "<Reference Examples>\n\n" + "\n\n".join(lines) + "\n\n<Reference Examples>"
    return block, sims

# ----------------------------- Main class -----------------------------
class QuantitativeAssessor:
    def __init__(
        self,
        ollama_host: str = "127.0.0.1",
        chat_model: str = "alibayram/medgemma:latest",
        emb_model: str  = "dengcao/Qwen3-Embedding-8B:Q4_K_M",
        pickle_path: str = "utils/chunk_8_step_2_participant_embedded_transcripts.pkl",
        gt_train_csv: str = "utils/train_split_Depression_AVEC2017.csv",
        gt_dev_csv: str   = "utils/dev_split_Depression_AVEC2017.csv",
        top_k: int = 3,
        dim: Optional[int] = None
    ):
        self.ollama_host = ollama_host
        self.chat_model  = chat_model
        self.emb_model   = emb_model
        self.top_k       = top_k
        self.dim         = dim

        df_train = pd.read_csv(gt_train_csv)
        df_dev   = pd.read_csv(gt_dev_csv)
        df = pd.concat([df_train, df_dev], ignore_index=True)
        df["Participant_ID"] = df["Participant_ID"].astype(int)
        self.phq_df = df.sort_values("Participant_ID").reset_index(drop=True)

        with open(pickle_path, "rb") as f:
            pet_raw = pickle.load(f)

        pet: Dict[int, List[Tuple[str, List[float]]]] = {}
        for pid, pairs in pet_raw.items():
            pid_int = int(pid)
            norm_pairs = []
            for raw, emb in pairs:
                v = np.asarray(emb, dtype=np.float32)
                if self.dim is not None:
                    v = v[: self.dim]
                v = _l2norm(v)
                norm_pairs.append((raw, v.tolist()))
            pet[pid_int] = norm_pairs
        self.participant_embedded_transcripts = pet

    def extract_evidence(self, transcript: str) -> Dict[str, List[str]]:
        _log_header("Step 1: Semantic Evidence Extraction")

        sentences = _sentences(transcript)
        chunks = [" ".join(sentences[i: i + 5]) for i in range(0, len(sentences), 5)]
        all_extracted_evidence = {k: [] for k in PHQ8_KEYS}
        symptoms_str = ", ".join(PHQ8_KEYS)

        # Template string without double curly braces to avoid .format() issues
        # Updated base_prompt with standard clinical definitions
        base_prompt = """
                ACT AS A CLINICAL PSYCHOLOGIST. Extract EXACT quotes for these 8 symptoms based on DSM-5 standards:

                1. PHQ8_NoInterest: Loss of interest in USUAL activities (Hobbies/Social). 
                2. PHQ8_Depressed: Persistent sadness or "feeling down".
                3. PHQ8_Sleep: Insomnia (hard to sleep) or hypersomnia.
                4. PHQ8_Tired: PHYSICAL exhaustion and lack of energy ONLY.
                5. PHQ8_Appetite: Changes in eating habits or weight.
                6. PHQ8_Failure: Guilt or feeling like a failure/burden. (e.g., 'not providing for family').
                7. PHQ8_Concentrating: Cognitive struggle. (e.g., 'brain not switching off', 'can't focus').
                8. PHQ8_Moving: Restlessness (fidgety) or physical slowness.

                STRICT RULES:
                - DO NOT extract 'not traveling' as NoInterest. 
                - DO NOT put mood-related quotes into Tired or Appetite categories.

                Segment: "SEGMENT_TEXT"
                Return ONLY valid JSON. If no match, return {}.
                """

        for i, segment in enumerate(chunks):
            # Safe string replacement instead of .format() to avoid KeyError
            user_prompt = base_prompt.replace("SYMPTOMS_LIST", symptoms_str).replace("SEGMENT_TEXT", segment)

            try:
                raw_response = ollama_chat(
                    self.ollama_host,
                    self.chat_model,
                    "You are a clinical evidence extractor. Be extremely category-specific.",
                    user_prompt
                )

                cleaned_json = _strip_json_block(raw_response)
                segment_data = json.loads(cleaned_json)

                if isinstance(segment_data, list): segment_data = {}

                for key in PHQ8_KEYS:
                    quotes = segment_data.get(key, [])
                    if isinstance(quotes, list) and quotes:
                        for q in quotes:
                            if q.strip() and q not in all_extracted_evidence[key]:
                                all_extracted_evidence[key].append(q.strip())
                                _log(f"      [MATCH] Found for {key}: '{q[:30]}...'")

            except Exception as e:
                _log(f"      [WARN] Segment {i + 1} failed: {str(e)}")
                continue

        _log_header("Extraction Summary")
        for k, v in all_extracted_evidence.items():
            _log(f" - {k}: {len(v)} quotes")

        return all_extracted_evidence


    def build_reference_bundle(self, evidence_dict: Dict[str, List[str]]) -> Tuple[str, Dict[str, List[str]]]:
        blocks = []
        sim_scores: Dict[str, List[str]] = {}
        for item in PHQ8_KEYS:
            block, sims = _build_reference_for_item(
                evidence_quotes=evidence_dict.get(item, []),
                item_key=item,
                ollama_host=self.ollama_host,
                emb_model=self.emb_model,
                top_k=self.top_k,
                pet=self.participant_embedded_transcripts,
                phq_df=self.phq_df,
                dim=self.dim
            )
            blocks.append(f"[{item}]\n{block}")
            sim_scores[item] = sims
        return "\n\n".join(blocks), sim_scores

    def score_with_references(self, transcript: str, reference_bundle: str) -> Dict[str, Any]:
        user_prompt = make_scoring_user_prompt(transcript, reference_bundle)
        raw = ollama_chat(self.ollama_host, self.chat_model, SYSTEM_PROMPT, user_prompt)

        try:
            # Attempt 1: Standard JSON parse
            txt = _extract_json_between_answer(raw)
            if not txt:
                start, end = raw.find('{'), raw.rfind('}') + 1
                if start != -1 and end != -1: txt = raw[start:end]

            if txt:
                return _validate_and_normalize(json.loads(_tolerant_fixups(txt)))
        except Exception:
            _log("[WARN] JSON parse failed, switching to Regex recovery...")

        # Attempt 2: Emergency Regex Recovery (The Bulletproof Way)
        recovered_obj = {}
        for key in PHQ8_KEYS:
            # Look for "Key": { ... "score": X } or "score": "N/A"
            pattern = rf'"{key}"\s*:\s*\{{[^}}]*?"score"\s*:\s*(\d|"N/A")'
            match = re.search(pattern, raw, re.DOTALL)
            if match:
                score_val = match.group(1).replace('"', '')
                recovered_obj[key] = {
                    "evidence": "Extracted via recovery",
                    "reason": "JSON was malformed, score recovered via regex",
                    "score": int(score_val) if score_val.isdigit() else "N/A"
                }
            else:
                recovered_obj[key] = _empty_item("Recovery failed")

        return _validate_and_normalize(recovered_obj)



    def assess(self, interview_text: str) -> Dict[str, Any]:
        _log("[STEP] Interview transcript:")
        _log(interview_text.strip())
        evidence = self.extract_evidence(interview_text)
        ref_bundle, _ = self.build_reference_bundle(evidence)
        result = self.score_with_references(interview_text, ref_bundle)
        total, sev = _compute_total_and_severity(result)
        result["_total"] = total
        result["_severity"] = sev
        _log("[STEP] Final result:")
        _log(json.dumps(result, ensure_ascii=False, indent=2))
        return result

# ----------------------------- CLI -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama_host", default="127.0.0.1")
    parser.add_argument("--chat_model", default="alibayram/medgemma:latest")
    parser.add_argument("--emb_model",  default="dengcao/Qwen3-Embedding-8B:Q4_K_M")
    parser.add_argument("--pickle_path", default="chunk_8_step_2_participant_embedded_transcripts.pkl")
    parser.add_argument("--gt_train_csv", default="train_split_Depression_AVEC2017.csv")
    parser.add_argument("--gt_dev_csv",   default="dev_split_Depression_AVEC2017.csv")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.quiet:
        VERBOSE = False

    qa = QuantitativeAssessor(
        ollama_host=args.ollama_host,
        chat_model=args.chat_model,
        emb_model=args.emb_model,
        pickle_path=args.pickle_path,
        gt_train_csv=args.gt_train_csv,
        gt_dev_csv=args.gt_dev_csv,
        top_k=args.top_k,
        dim=args.dim
    )

