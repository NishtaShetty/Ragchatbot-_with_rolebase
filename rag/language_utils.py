# rag/language_utils.py

import torch
import numpy as np
import logging
from transformers import AutoModel
from deep_translator import GoogleTranslator

logger = logging.getLogger(__name__)

# ---------------- SUPPORTED LANGUAGES ----------------
SUPPORTED_LANGUAGES = {
    "en": "English",
    "kn": "Kannada",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam"
}

# ---------------- ASR CONFIG ----------------
INDIC_ASR_MODEL = "ai4bharat/indic-conformer-600m-multilingual"
DECODE_TYPE = "ctc"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- LOAD ASR MODEL ----------------
indic_asr_model = AutoModel.from_pretrained(
    INDIC_ASR_MODEL,
    trust_remote_code=True
).to(device)
indic_asr_model.eval()

# ---------------- TRANSLATION ----------------
def translate(text: str, src: str, tgt: str) -> str:
    if not text.strip() or src == tgt:
        return text
    return GoogleTranslator(source=src, target=tgt).translate(text)

# ---------------- ASR ----------------
def transcribe_audio(audio_np: np.ndarray, language: str) -> str:
    wav = torch.from_numpy(audio_np).float().unsqueeze(0).to(device)
    with torch.no_grad():
        return indic_asr_model(wav, language, DECODE_TYPE).strip()

# ---------------- PROCESS AUDIO ----------------
def process_audio(audio_np: np.ndarray, language: str) -> dict:
    transcription = transcribe_audio(audio_np, language)
    translated = translate(transcription, language, "en")

    return {
        "original_text": transcription,
        "translated_text": translated,
        "language": language
    }

# ---------------- PROCESS TEXT ----------------
def process_text(text: str, language: str) -> dict:
    translated = translate(text, language, "en")
    return {
        "original_text": text,
        "translated_text": translated,
        "language": language,
        "was_translated": language != "en"
    }

# ---------------- OUTPUT TRANSLATION ----------------
def translate_answer(answer: str, user_lang: str) -> str:
    return translate(answer, "en", user_lang)

# ---------------- HEALTH ----------------
def check_models_status():
    return {
        "asr_loaded": indic_asr_model is not None,
        "supported_languages": list(SUPPORTED_LANGUAGES.keys()),
        "device": device
    }
