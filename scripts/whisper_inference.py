from pathlib import Path
import csv
from transformers import pipeline

# ---- Ayarlar ----
MODEL_NAME = "openai/whisper-small"
DATA_DIR = Path("Dataset")
PROC_DIR = DATA_DIR / "processed_audio"
META_PROC = DATA_DIR / "metadata" / "metadata_processed.csv"
OUT_CSV = DATA_DIR / "metadata" / "whisper_predictions.csv"

# ---- Model ----
asr = pipeline(
    "automatic-speech-recognition",
    model=MODEL_NAME,
    device="mps",        
    generate_kwargs={"language": "turkish", "task": "transcribe"},
)

def main():
    rows_out = []

    with META_PROC.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames if reader.fieldnames is not None else []

        # Çıktı dosyasına eklenecek yeni sütun
        if "whisper_pred" not in fieldnames:
            fieldnames.append("whisper_pred")

        for i, row in enumerate(reader, start=1):
            proc_name = row["processed_file_name"]
            audio_path = PROC_DIR / proc_name

            if not audio_path.exists():
                print(f"[WARN] Audio missing: {audio_path}")
                continue

            print(f"[{i}] Transcribing {audio_path} ...")
            result = asr(str(audio_path))
            pred_text = result["text"]

            row_out = dict(row)
            row_out["whisper_pred"] = pred_text
            rows_out.append(row_out)

    # Tüm sonuçları yeni CSV'ye yaz
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\n[INFO] Whisper tahminleri kaydedildi → {OUT_CSV}")

if __name__ == "__main__":
    main()
