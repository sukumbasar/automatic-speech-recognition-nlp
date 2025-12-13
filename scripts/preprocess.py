import csv
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

# ---- Ayarlar ----
DATA_DIR = Path("Dataset")

RAW_DIR = DATA_DIR / "raw_audio"
PROC_DIR = DATA_DIR / "processed_audio"

METADATA_IN = DATA_DIR / "metadata" / "metadata.csv"
METADATA_OUT = DATA_DIR / "metadata" / "metadata_processed.csv"

TARGET_SR = 16000  # hedef sample rate (16 kHz)
TOP_DB = 30        # trim için eşik (sessizlik kesme)

PROC_DIR.mkdir(parents=True, exist_ok=True)


def process_one_file(row: dict) -> dict | None:
    """
    Tek bir satırdaki (bir ses kaydı) bilgiyi kullanarak:
      - dosyayı okur
      - 16kHz, mono olarak yükler
      - baş/son sessizliği trimler
      - sesi normalize eder
      - processed_audio altına .wav olarak kaydeder
      - metadata satırına processed_file_name alanını ekler
    """
    file_name = row["file_name"]
    in_path = RAW_DIR / file_name

    if not in_path.exists():
        print(f"[WARN] dosya bulunamadı: {in_path}")
        return None

    # 1) Ses dosyasını yükle (mono, 16kHz)
    y, sr = librosa.load(in_path, sr=TARGET_SR, mono=True)

    # 2) Baş ve son sessizliği kes (trim)
    y_trimmed, index = librosa.effects.trim(y, top_db=TOP_DB)

    if len(y_trimmed) == 0:
        print(f"[WARN] trim sonrası boş kalan dosya: {in_path}")
        return None

    # 3) Normalize (ses seviyesini eşitle)
    peak = np.max(np.abs(y_trimmed))
    if peak > 0:
        y_norm = 0.95 * (y_trimmed / peak)
    else:
        y_norm = y_trimmed

    # 4) Çıktı dosya adını oluştur (wav uzantılı)
    out_name = Path(file_name).with_suffix(".wav").name
    out_path = PROC_DIR / out_name

    # 5) WAV olarak kaydet
    sf.write(out_path, y_norm, TARGET_SR)
    print(f"[OK] işlendi → {out_path}")

    # 6) Metadata satırını güncelle
    row_out = dict(row)
    row_out["processed_file_name"] = out_name
    row_out["sample_rate"] = TARGET_SR
    return row_out


def main():
    rows_out = []

    with METADATA_IN.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames if reader.fieldnames is not None else []

        for row in reader:
            processed_row = process_one_file(row)
            if processed_row is not None:
                rows_out.append(processed_row)

    extra_fields = ["processed_file_name", "sample_rate"]
    for ef in extra_fields:
        if ef not in fieldnames:
            fieldnames.append(ef)

    with METADATA_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\n[INFO] İşlenmiş metadata yazıldı → {METADATA_OUT}")


if __name__ == "__main__":
    main()
