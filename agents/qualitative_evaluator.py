import requests
import json
import re
import time


def parse_score_and_explanation(response_text):
    """
    Extract score and explanation from model response
    """
    score_patterns = [
        r'score[:\s]*(\d+)',                          # Score: 4
        r'(?:coherence|completeness|accuracy|specificity)[:\s]*(\d+)',  # Coherence: 4
        r'rating[:\s]*(\d+)',                          # Rating: 4
        r'^\s*(\d+)\s*$',                              # Lone digit on its own line
        r'(\d+)[/\s]*(?:out of\s*)?5',                # 4/5 or 4 out of 5
        r'^(\d+)',                                     # Number at start of line
    ]

    score = None
    for pattern in score_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
        if match:
            potential_score = int(match.group(1))
            if 1 <= potential_score <= 5:
                score = potential_score
                break

    return score, response_text.strip()


class QualitativeEvaluatorAgent:
    def __init__(self, model="alibayram/medgemma:latest", endpoint="http://localhost:11434/api/chat"):
        self.model = model
        self.endpoint = endpoint

    def assess(self, transcript, qualitative_output: str):
        result = {}

        # metric prompts
        coherence_prompt = f"""Rate COHERENCE of this psychiatric assessment (1-5).
Coherence: Is the assessment logically consistent with no contradictions?
5=0 errors, 4=1-2, 3=3-4, 2=5-6, 1=7+

Transcript: {transcript}
Assessment: {qualitative_output}

Format:
Explanation: [brief explanation]
Score: [1-5]"""

        completeness_prompt = f"""Rate COMPLETENESS of this psychiatric assessment (1-5).
Completeness: Does it cover all relevant PHQ-8 symptoms with frequency/severity?
5=0 omissions, 4=1-2, 3=3-4, 2=5-6, 1=7+

Transcript: {transcript}
Assessment: {qualitative_output}

Format:
Explanation: [brief explanation]
Score: [1-5]"""

        specificity_prompt = f"""Rate SPECIFICITY of this psychiatric assessment (1-5).
Specificity: Are claims specific with evidence, not vague generalities?
5=0 vague statements, 4=1-2, 3=3-4, 2=5-6, 1=7+

Transcript: {transcript}
Assessment: {qualitative_output}

Format:
Explanation: [brief explanation]
Score: [1-5]"""

        accuracy_prompt = f"""Rate ACCURACY of this psychiatric assessment (1-5).
Accuracy: Are symptoms aligned with DSM-5/PHQ-8 criteria?
5=0 errors, 4=1-2, 3=3-4, 2=5-6, 1=7+

Transcript: {transcript}
Assessment: {qualitative_output}

Format:
Explanation: [brief explanation]
Score: [1-5]"""

        labels = ['coherence', 'completeness', 'accuracy', 'specificity']
        prompts = {
            "specificity": specificity_prompt,
            "accuracy": accuracy_prompt,
            "coherence": coherence_prompt,
            "completeness": completeness_prompt
        }

        # Build request
        reqs = {}
        print('creating requests....')
        for label in labels:
            print(prompts[label])
            request = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompts[label]}],
                "stream": False,
                "options": {"temperature": 0, "top_k": 20, "top_p": 0.9}
            }
            reqs[label] = request

        responses = []
        contents = []
        result = {}
        for label, request in reqs.items():
            print(f"  Getting {label} response...")
            try:
                print(request)
                response = requests.post(self.endpoint, json=request, timeout=180)
                if response.status_code == 200:
                    responses.append(response)
                    content = response.json()['message']['content']
                    print('content: ')
                    print(content)
                    contents.append(content)
                    score, explanation = parse_score_and_explanation(content)
                    result[label] = score
                    print(f"  {label} score: {score}")
                else:
                    result[label] = None
                    print(f"  {label} request failed with status:", response.status_code)
            except Exception as e:
                print(f"  Error during {label} request:", e)
                result[label] = None

        return result
