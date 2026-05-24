# 🎙️ ASR ROVER — Multilingual Meeting Transcription

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)
![Whisper](https://img.shields.io/badge/Whisper-Large%20V3-9cf.svg)
![NVIDIA Canary](https://img.shields.io/badge/NVIDIA%20Canary-1B-76B900.svg)
![Pyannote](https://img.shields.io/badge/Pyannote-Community--1-orange.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900.svg)

![License](https://img.shields.io/badge/license-MIT-green.svg)
![Contributions](https://img.shields.io/badge/contributions-welcome-orange.svg)

<p align="center">
  <img src="asset\signe-message-vocal.avif" alt="ASR ROVER" width="600">
</p>

---

## 📝 Project Description

A high-accuracy **multilingual speech-to-text** pipeline for long meetings, combining several state-of-the-art ASR systems with **ROVER fusion** for best-in-class WER. The system performs **speaker diarization** with Pyannote Community-1, then transcribes with both **Whisper Large V3** (robustness, 99 languages) and **NVIDIA Canary** (high accuracy on EN/FR/ES/DE), and finally fuses the outputs through **confidence-weighted voting**.

This project was created to explore **mixture-of-experts** ideas applied to ASR — instead of relying on a single model, the pipeline lets each system do what it does best and arbitrates word-by-word at the end. It also gave me a reason to dig into the **ROVER algorithm (NIST)** and word-level alignment.

---

## ⚙️ Features
  🤖 **Multi-system ASR** combining Whisper Large V3 and NVIDIA Canary Qwen 2.5B

  🗳️ **ROVER fusion** with confidence-weighted voting at word level (~4.5–5.5% WER target)

  👥 **Speaker diarization** through Pyannote Community-1 (auto-detect or fixed speaker count)

  🌍 **Multilingual** — optimized for English and French, 99+ languages via Whisper

  🧩 **Modular architecture** — Whisper-only mode if NeMo dependencies cause conflicts

  📄 **Multiple output formats** — JSON, TXT, and SRT subtitles with timestamps

  ⚡ **CPU & GPU support** — CUDA-ready for fast inference, fallback CPU config provided

---

## Example Outputs

### JSON
```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 5.2,
      "speaker": "SPEAKER_00",
      "text": "Hello, welcome to today's meeting.",
      "confidence": 0.95
    }
  ],
  "speakers": ["SPEAKER_00", "SPEAKER_01"],
  "duration": 3600.5,
  "language": "en"
}
```

### TXT
```
Meeting Transcription
Duration: 3600.50s
Language: en
Speakers: SPEAKER_00, SPEAKER_01
================================================================================

SPEAKER_00 [0.00s - 5.20s]:
Hello, welcome to today's meeting.

SPEAKER_01 [5.50s - 12.30s]:
Thank you for having me.
```

### SRT
```
1
00:00:00,000 --> 00:00:05,200
[SPEAKER_00] Hello, welcome to today's meeting.

2
00:00:05,500 --> 00:00:12,300
[SPEAKER_01] Thank you for having me.
```

### 📝 Notes & Observations

| Component | Model | WER | Speed (RTFx) |
|-----------|-------|-----|--------------|
| ASR 1 | Whisper Large V3 | ~7–8% | 68× |
| ASR 2 | NVIDIA Canary Qwen 2.5B | ~5.6% | 418× |
| **ROVER Fusion** | **Combined** | **~4.5–5.5%** | **~240×** |
| Diarization | Pyannote Community-1 | ~10% DER | 2.5% RTFx |

---

## ⚙️ How it works
  🎧 The audio file is loaded and resampled to 16 kHz mono through the audio utils.

  👁️ **Pyannote Community-1** segments the file into speaker turns (who-spoke-when).

  🧠 **Whisper Large V3** transcribes each segment with multilingual robustness.

  🧠 **NVIDIA Canary** transcribes the same segments with higher accuracy on EN/FR/ES/DE.

  🗳️ The **ROVER fusion** module aligns words across the two hypotheses and votes per word, weighted by each system's confidence.

  🧾 Final segments are reassembled with their speaker labels, timestamps and confidence scores.

  💾 Results are exported to JSON, TXT and SRT — pick whichever fits your downstream tool.

---

## 🗺️ Architecture Diagram

The pipeline is a **mixture-of-experts** style architecture: diarization first, then parallel ASR systems, then word-level ROVER fusion.

![Architecture Diagram](img/architecture.svg)

**Key components:**
- Diarizer: `pyannote/speaker-diarization-community-1`
- ASR 1: `openai/whisper-large-v3` (beam_size = 5, float16)
- ASR 2: `nvidia/canary-1b` (greedy decoding)
- Fusion: confidence-weighted ROVER (weights: whisper = 1.0, canary = 1.2)

---

## 📂 Repository structure
```bash
├── configs/
│   ├── config.yaml                 # Default config (GPU)
│   ├── config-cpu.yaml              # CPU-only config
│   └── config-windows-cpu.yaml      # Windows CPU config
│
├── src/
│   ├── asr/
│   │   ├── base_asr.py              # Common ASR interface
│   │   ├── whisper_asr.py           # Whisper Large V3 wrapper
│   │   └── canary_asr.py            # NVIDIA Canary wrapper
│   ├── diarization/
│   │   └── pyannote_diarizer.py     # Pyannote speaker diarization
│   ├── rover/
│   │   └── rover_fusion.py          # ROVER word-level fusion
│   ├── utils/
│   │   ├── audio_utils.py           # Audio loading & resampling
│   │   └── config_loader.py         # YAML config loader
│   └── pipeline.py                  # Main MeetingTranscriptionPipeline
│
├── examples/
│   ├── basic_usage.py               # Interactive examples menu
│   ├── basic_transcription.py
│   ├── custom_config.py
│   ├── whisper_only.py
│   └── cli_transcribe.py            # Command-line entry point
│
├── data/
│   ├── input/                       # Place your audio files here
│   └── output/                      # Generated transcriptions
│
├── tests/
├── setup_token.py                   # HuggingFace token setup
├── diagnostic_hf.py                 # HF access diagnostic
├── generate_test_audio.py           # Generate test WAV files
├── install_dependencies.sh
├── requirements.txt
├── requirements-whisper-only.txt
├── requirements-full.txt
│
├── LICENSE
└── README.md
```

---

## 💻 Run it on Your PC
Clone the repository and install dependencies:
```bash
git clone https://github.com/Thibault-GAREL/ASR-Mixture_of_expert-ROVER.git
cd ASR-Mixture_of_expert-ROVER

python -m venv .venv # if you don't have a virtual environment
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows

pip install -r requirements-whisper-only.txt
```

⚠️ For **maximum speed**, a **CUDA-compatible GPU** (8 GB+ VRAM) is strongly recommended. CPU mode works but is significantly slower on long meetings.

### HuggingFace setup

You must accept the licenses for these Pyannote models on HuggingFace, then configure your token:
```bash
python setup_token.py    # interactive token setup
python diagnostic_hf.py  # verify access to all required models
```

Required model licenses:
- `pyannote/speaker-diarization-3.1`
- `pyannote/segmentation-3.0`
- `pyannote/speaker-diarization-community-1`
- `pyannote/wespeaker-voxceleb-resnet34-LM`
- `pyannote/VoiceActivityDetection-PyanNet-ONNX`

### Run the pipeline

```bash
# Interactive examples menu
python examples/basic_usage.py

# Command-line transcription with auto-detection
python examples/cli_transcribe.py meeting.wav

# Force language and number of speakers
python examples/cli_transcribe.py meeting.wav --language fr --num-speakers 3 --output results/

# Whisper-only mode (no Canary, no NeMo dependencies)
python examples/cli_transcribe.py meeting.wav --whisper-only
```

### Full install with Canary (advanced)

⚠️ NeMo has strict dependency requirements and may conflict with other packages.
```bash
pip install -r requirements-full.txt
```

For Windows / CUDA / cache troubleshooting, see [WINDOWS_CUDA_GUIDE.md](WINDOWS_CUDA_GUIDE.md), [WINDOWS-GUIDE.md](WINDOWS-GUIDE.md), [INSTALL_ROVER.md](INSTALL_ROVER.md) and [FIX_CACHE_PYTHON.md](FIX_CACHE_PYTHON.md).

---

## 📖 Inspiration / Sources
This project is based on:
- 📄 [ROVER Algorithm (NIST)](https://ieeexplore.ieee.org/document/659110/) — Fiscus, 1997
- 🤗 [Whisper (OpenAI)](https://github.com/openai/whisper) and [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- 🟢 [NVIDIA Canary](https://developer.nvidia.com/blog/new-standard-for-speech-recognition-and-translation-from-the-nvidia-nemo-canary-model/)
- 🗣️ [Pyannote.audio](https://github.com/pyannote/pyannote-audio)
- 🏆 [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)

Code created by me 😎, Thibault GAREL - [Github](https://github.com/Thibault-GAREL)
