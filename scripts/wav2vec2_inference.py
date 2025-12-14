from pathlib import Path
import csv
import torch
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# ---- Model ----
MODEL_NAME = "m3hrdadfi/wav2vec2-large-xlsr-turkish"

processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
model.eval()

# ---- Paths ----
DATA_DIR = Path("Dataset")
AUDIO_DIR = DATA_DIR / "processed_audio"
META_IN = DATA_DIR / "metadata" / "metadata_processed.csv"
META_OUT = DATA_DIR / "metadata" / "wav2vec2_predictions.csv"


def main():
    rows_out = []

    with META_IN.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames + ["wav2vec2_pred"]

        for i, row in enumerate(reader, start=1):
            audio_path = AUDIO_DIR / row["processed_file_name"]

            if not audio_path.exists():
                print(f"[WARN] Missing audio: {audio_path}")
                continue

            # ---- Load audio ----
            speech, sr = sf.read(audio_path)
            if sr != 16000:
                raise ValueError("Sample rate must be 16kHz")

            inputs = processor(
                speech,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )

            with torch.no_grad():
                logits = model(
                    inputs.input_values.to(device)
                ).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]

            row_out = dict(row)
            row_out["wav2vec2_pred"] = transcription
            rows_out.append(row_out)

            print(f"[{i}] {audio_path.name} → {transcription}")

    with META_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\n[INFO] wav2vec2 predictions saved to {META_OUT}")


if __name__ == "__main__":
    main()
