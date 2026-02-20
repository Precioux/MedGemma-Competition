import os
import datetime
from pathlib import Path


class InterviewSimulator:
    """
    Agent responsible for Step 0: Loading and validating clinical transcripts.
    Ensures data is ready for MedGemma analysis.
    """

    def __init__(self, default_path: str = None, encoding: str = "utf-8"):
        self.encoding = encoding
        # Use dynamic absolute pathing to prevent 'File Not Found' on judge's machine
        base_dir = Path(__file__).resolve().parent.parent

        self.transcript_path = (
                default_path
                or os.getenv("TRANSCRIPT_PATH")
                or str(base_dir / "data" / "transcripts" / "302_Dialogue.txt")
        )

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
            error_msg = f"Missing file: {self.transcript_path}. Ensure preprocessor was run."
            print(f"[{self._now()}] [CRITICAL] {error_msg}")
            raise FileNotFoundError(error_msg)

        # 2. Reading with robustness
        try:
            text = p.read_text(encoding=self.encoding).strip()
            # Normalize line endings for cross-platform compatibility (Mac/Linux/Windows)
            text = text.replace("\r\n", "\n").replace("\r", "\n")

            if not text:
                print(f"[{self._now()}] [ERROR] Validation failed: Transcript is empty.")
                raise ValueError(f"Empty content at: {self.transcript_path}")

            # 3. Success Logging (Summary only, no full text dump)
            word_count = len(text.split())
            print(f"[{self._now()}] [SUCCESS] Loaded: {p.name}")
            print(f"[{self._now()}] [INFO] Data Stats: {word_count} words | {len(text)} chars")

            return text

        except Exception as e:
            print(f"[{self._now()}] [ERROR] Unexpected failure during load: {str(e)}")
            raise e