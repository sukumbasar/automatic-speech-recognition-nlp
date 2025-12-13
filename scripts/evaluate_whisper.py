from pathlib import Path
import csv
import jiwer

DATA_DIR = Path("Dataset")
META_PRED = DATA_DIR / "metadata" / "whisper_predictions.csv"


def main():
    # --- Normalization pipeline ---
    transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])

    wers = []
    cers = []
    speaker_stats = {}
    common_wers = []
    personal_wers = []

    with META_PRED.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            raw_truth = row["text"]
            raw_pred = row["whisper_pred"]
            speaker = row["speaker_id"]
            is_common = int(row["is_common"])

            # --- Metinleri normalize et ---
            truth = transform(raw_truth)
            pred = transform(raw_pred)

            # --- NORMALIZED WER/CER ---
            wer = jiwer.wer(truth, pred)
            cer = jiwer.cer(truth, pred)

            wers.append(wer)
            cers.append(cer)

            if speaker not in speaker_stats:
                speaker_stats[speaker] = []
            speaker_stats[speaker].append(wer)

            if is_common == 1:
                common_wers.append(wer)
            else:
                personal_wers.append(wer)

    # ---- Sonuçlar ----
    print("\n=== WHISPER EVALUATION RESULTS (Normalized) ===")
    print(f"Genel Ortalama WER  : {sum(wers)/len(wers):.4f}")
    print(f"Genel Ortalama CER  : {sum(cers)/len(cers):.4f}")

    print("\n-- Speaker Bazlı WER --")
    for speaker, vals in speaker_stats.items():
        print(f"Speaker {speaker}: {sum(vals)/len(vals):.4f}")

    print("\n-- Ortak Cümle WER --")
    print(f"Ortak:   {sum(common_wers)/len(common_wers):.4f}")
    print(f"Kişisel: {sum(personal_wers)/len(personal_wers):.4f}")


if __name__ == "__main__":
    main()


