import os
import datetime
from pathlib import Path


class InterviewSimulator:
    """
    Agent responsible for Step 0: Loading and validating clinical transcripts.
    Ensures data is ready for MedGemma analysis.

    Priority order for transcript path:
      1. Explicit default_path argument
      2. TRANSCRIPT_PATH environment variable
      3. Uploaded file from the web UI (data/transcripts/uploaded_transcript.txt)
      4. Fallback to the original sample transcript (302_Dialogue.txt)
    """

    def __init__(self, default_path: str = None, encoding: str = "utf-8"):
        self.encoding = encoding
        base_dir = Path(__file__).resolve().parent.parent

        uploaded_path = base_dir / "data" / "transcripts" / "uploaded_transcript.txt"
        fallback_path = base_dir / "data" / "transcripts" / "302_Dialogue.txt"

        if default_path:
            self.transcript_path = default_path
        elif os.getenv("TRANSCRIPT_PATH"):
            self.transcript_path = os.getenv("TRANSCRIPT_PATH")
        elif uploaded_path.exists():
            # Use the file uploaded through the web UI
            self.transcript_path = str(uploaded_path)
        else:
            # Fall back to the bundled sample
            self.transcript_path = str(fallback_path)

    def _now(self):
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def load(self) -> str:
        """
        Loads the reconstructed dialogue text.
        Follows E-DAIC clinical data standards.
        """
        print(f"[{self._now()}] [STEP 0] Invoking: InterviewSimulator...")

        p = Path(self.transcript_path)

        # 1. Path Validation
        if not p.exists():
            error_msg = f"Missing file: {self.transcript_path}. Upload a transcript via the web UI or ensure preprocessor was run."
            print(f"[{self._now()}] [CRITICAL] {error_msg}")
            raise FileNotFoundError(error_msg)

        # 2. Reading with robustness
        try:
            text = p.read_text(encoding=self.encoding).strip()
            # Normalize line endings for cross-platform compatibility
            text = text.replace("\r\n", "\n").replace("\r", "\n")

            if not text:
                print(f"[{self._now()}] [ERROR] Validation failed: Transcript is empty.")
                raise ValueError(f"Empty content at: {self.transcript_path}")

            # 3. Success Logging
            word_count = len(text.split())
            print(f"[{self._now()}] [SUCCESS] Loaded: {p.name}")
            print(f"[{self._now()}] [INFO] Data Stats: {word_count} words | {len(text)} chars")

            return text

        except Exception as e:
            print(f"[{self._now()}] [ERROR] Unexpected failure during load: {str(e)}")
            raise e
