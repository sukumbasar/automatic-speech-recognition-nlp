# Automatic Speech Recognition (ASR) for Turkish

**Project Type:** Comparative Analysis of ASR Architectures  
**Dataset:** Custom Turkish Speech Corpus (ASR_EchoBase)

---

## 📌 Project Overview

This project presents a comparative analysis of six distinct Automatic Speech Recognition (ASR) architectures applied to a custom Turkish speech dataset. The primary objective was to evaluate the efficacy of various modeling approaches—ranging from baseline CNNs to state-of-the-art End-to-End Transformers—in handling the agglutinative and phonetic complexities of the Turkish language under data-constrained conditions.

---

## 📂 Repository Structure

```
.
├── notebooks/        # Jupyter notebooks containing implementation for each model
├── scripts/          # Python scripts for preprocessing, inference, and evaluation
├── metadata/         # CSV files containing ground truth and model predictions
├── README.md         # Project documentation
└── requirements.txt  # Dependencies
```

---

## 🗣️ The Dataset: ASR_EchoBase

A custom Turkish speech dataset, **ASR_EchoBase**, was curated specifically for this study.

* **Dataset Link:** [huggingface.co/datasets/sukumbasar/ASR_EchoBase_Raw](https://huggingface.co/datasets/sukumbasar/ASR_EchoBase_Raw)
* **Audio Specification:**

  * 16 kHz sampling rate
  * Mono channel

### Scale

* **Version 1 (Baseline):** 50 recordings
* **Version 2 (Extended):** 100 recordings

### Preprocessing Pipeline

* **Resampling:** Conversion to 16 kHz
* **Channel Normalization:** Stereo to Mono conversion
* **Silence Trimming:** Removal of non-speech segments at boundaries
* **Amplitude Normalization:** Standardization of audio levels

---

## 🧪 Methodologies & Architectures

Six distinct ASR architectures were implemented and evaluated:

### Method 0: CNN-CTC Baseline

A baseline Convolutional Neural Network (CNN) trained with Connectionist Temporal Classification (CTC) loss. Included to demonstrate the necessity of advanced temporal modeling.

### Method 1: Rule-Based CTC

Utilizes a pre-trained **Wav2Vec2** acoustic model with beam search decoding. No external language model was employed.

### Method 2: VOSK

An offline recognition system based on the **Kaldi** toolkit, utilizing pre-trained Turkish acoustic and language models.

### Method 3: HuBERT (Transfer Learning Experiment)

An experimental attempt to adapt the English-only `hubert-large` model to Turkish via tokenizer fine-tuning. This method yielded a negative result, highlighting the high data threshold required for cross-lingual transfer in monolingual models.

### Method 4: Wav2Vec2-XLSR

Implementation of `wav2vec2-large-xlsr-turkish`, a cross-lingual model pre-trained on 53 languages and fine-tuned for Turkish.

### Method 5: OpenAI Whisper (Small)

An end-to-end Transformer model with an internal language model and weak supervision, tested for robustness in low-resource settings.

---

## 📊 Evaluation Results

Performance was measured using **Word Error Rate (WER)** and **Character Error Rate (CER)**. Results below reflect normalized evaluation on the extended **V2 dataset (100 samples)**.

| Model               | WER        | CER        | Key Observation                                     |
| ------------------- | ---------- | ---------- | --------------------------------------------------- |
| **Whisper (V2)**    | **0.1004** | **0.0221** | State-of-the-art performance                        |
| **VOSK**            | 0.1349     | 0.0271     | Strong consistency due to integrated Language Model |
| **Rule-Based CTC**  | 0.2037     | 0.0413     | Reliable baseline; errors stem from lack of context |
| **Wav2Vec2 (XLSR)** | 0.3207     | 0.0551     | High sensitivity to speaker variability             |
| **CNN-CTC**         | 0.9900     | –          | Failed to converge (CTC posterior collapse)         |
| **HuBERT**          | 1.0000     | 0.4904     | Failed to overcome English bias (Silence Trap)      |

---

## Conclusion

* **End-to-End Superiority:** **Whisper** achieved the lowest WER (0.10), demonstrating that architectures with internal language models are significantly more robust for low-resource languages than pure CTC approaches.
* **Data Scaling:** Expanding the dataset from 50 to 100 recordings resulted in measurable performance improvements for Whisper (WER 0.125 → 0.100) and Wav2Vec2.
* **Architecture Limitations:** The failure of the English HuBERT model highlights the significant phonetic and linguistic gap between English and Turkish, emphasizing the importance of multilingual pre-training (e.g., XLSR) or substantially larger datasets.
