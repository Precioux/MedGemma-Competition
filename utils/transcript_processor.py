import pandas as pd
import os
import glob


def process_single_transcript(participant_id, base_dir):
    """
    Reads from: data/source/XXX_Transcript.csv
    Saves to: data/transcripts/XXX_Dialogue.txt
    """
    input_file = os.path.join(base_dir, "data", "source", f"{participant_id}_Transcript.csv")
    output_dir = os.path.join(base_dir, "data", "transcripts")
    output_file = os.path.join(output_dir, f"{participant_id}_Dialogue.txt")

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_file):
        return None

    try:
        # E-DAIC transcripts usually contain Start_Time, End_Time, and Text [cite: 13]
        df = pd.read_csv(input_file, sep=None, engine='python')
        df.columns = [c.strip().lower() for c in df.columns]

        # Mapping columns based on E-DAIC Manual [cite: 13, 27]
        text_col = next((c for c in df.columns if c in ['value', 'text']), None)
        speaker_col = next((c for c in df.columns if c in ['speaker', 'role']), None)

        if not text_col:
            return f"[SKIP] No text column in {participant_id}"

        formatted_dialogue = []
        for idx, row in df.iterrows():
            content = str(row[text_col]).strip()
            if not content or content.lower() == 'nan':
                continue

            # Identifying 'Ellie' (Virtual Interviewer) vs 'Participant' [cite: 4, 10]
            if speaker_col:
                speaker_val = str(row[speaker_col]).lower()
                role = "Ellie (AI)" if "ellie" in speaker_val else "Patient"
            else:
                role = "Patient" if idx % 2 != 0 else "Ellie (AI)"

            formatted_dialogue.append(f"{role}: {content}")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(formatted_dialogue))

        return participant_id

    except Exception as e:
        print(f"[ERROR] Failed {participant_id}: {e}")
        return None


def main():
    # Dynamic pathing from utils/ directory to project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_path = os.path.join(base_dir, "data", "source", "*_Transcript.csv")

    print("=" * 60)
    print("🚀 BATCH TRANSCRIPT PROCESSOR: STARTING")
    print(f"📍 Scanning: {os.path.join(base_dir, 'data', 'source')}")
    print("=" * 60)

    files = glob.glob(source_path)
    if not files:
        print("[!] No files found. Check if CSVs are in data/source/")
        return

    success_count = 0
    for file_path in files:
        # Extract ID (e.g., '308' from '308_Transcript.csv')
        filename = os.path.basename(file_path)
        p_id = filename.split('_')[0]

        result = process_single_transcript(p_id, base_dir)
        if result:
            print(f"[SUCCESS] Processed Participant: {result}")
            success_count += 1

    print("\n" + "=" * 60)
    print(f"✅ COMPLETED: {success_count} files processed.")
    print(f"📂 Transcripts saved in: {os.path.join(base_dir, 'data', 'transcripts')}")
    print("=" * 60)


if __name__ == "__main__":
    main()