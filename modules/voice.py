
import io
import os
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import config

# ── Result dataclass 

@dataclass
class TranscriptionResult:
    text: str
    language: str          
    confidence: float = 0.0
    method: str = ""       
    error: Optional[str] = None


# ── Language mappings 

# Google Cloud STT language codes
GCP_LANG_CODES = {
    "en": "en-IN",
    "hi": "hi-IN",
    "bn": "bn-IN",
    "ta": "ta-IN",
    "te": "te-IN",
}

# gTTS language codes
GTTS_LANG_CODES = {
    "en": "en",
    "hi": "hi",
    "bn": "bn",
    "ta": "ta",
    "te": "te",
}

# Whisper language codes
WHISPER_LANG_CODES = {
    "en": "en",
    "hi": "hi",
    "bn": "bn",
    "ta": "ta",
    "te": "te",
}


# ── Audio format helpers 
def convert_to_wav(audio_bytes: bytes, source_format: str = "webm") -> bytes:
    
    try:
        from pydub import AudioSegment

        audio_io = io.BytesIO(audio_bytes)
        segment  = AudioSegment.from_file(audio_io, format=source_format)

        # Convert to mono 16kHz (optimal for STT)
        segment = segment.set_channels(1).set_frame_rate(16000)

        wav_io = io.BytesIO()
        segment.export(wav_io, format="wav")
        return wav_io.getvalue()

    except Exception as e:
        print(f"  ⚠️  Audio conversion failed: {e}. Using raw bytes.")
        return audio_bytes


def get_audio_format(audio_bytes: bytes) -> str:
    """Detect audio format from magic bytes."""
    if audio_bytes[:4] == b"RIFF":
        return "wav"
    if audio_bytes[:3] == b"ID3" or audio_bytes[:2] == b"\xff\xfb":
        return "mp3"
    if audio_bytes[:4] == b"fLaC":
        return "flac"
    return "webm"   # default for browser recordings


# ── Speech-to-Text 

def transcribe_audio(
    audio_bytes: bytes,
    language: str = "en",
    auto_detect: bool = True,
) -> TranscriptionResult:
    
    if not audio_bytes or len(audio_bytes) < 100:
        return TranscriptionResult(
            text="", language=language, method="failed",
            error="No audio data received"
        )

    # Try Google Cloud first (better quality, supports Hindi)
    if _gcp_credentials_available():
        result = _transcribe_google_cloud(audio_bytes, language, auto_detect)
        if result.text:
            return result

    # Fallback to local Whisper
    result = _transcribe_whisper(audio_bytes, language)
    return result


def _gcp_credentials_available() -> bool:
    
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    return bool(creds_path and Path(creds_path).exists())


def _transcribe_google_cloud(
    audio_bytes: bytes,
    language: str,
    auto_detect: bool,
) -> TranscriptionResult:
    
    try:
        from google.cloud import speech

        client = speech.SpeechClient()

        # Convert to WAV if needed
        fmt = get_audio_format(audio_bytes)
        if fmt != "wav":
            audio_bytes = convert_to_wav(audio_bytes, source_format=fmt)

        audio = speech.RecognitionAudio(content=audio_bytes)

        lang_code = GCP_LANG_CODES.get(language, "en-IN")

        config_kwargs = dict(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code=lang_code,
            enable_automatic_punctuation=True,
            model="medical_dictation" if language == "en" else "default",
        )

        # Add alternative languages for auto-detect
        if auto_detect and language == "en":
            config_kwargs["alternative_language_codes"] = ["hi-IN"]
        elif auto_detect and language == "hi":
            config_kwargs["alternative_language_codes"] = ["en-IN"]

        gcp_config = speech.RecognitionConfig(**config_kwargs)
        response   = client.recognize(config=gcp_config, audio=audio)

        if not response.results:
            return TranscriptionResult(
                text="", language=language, method="google_cloud",
                error="No speech detected"
            )

        best    = response.results[0].alternatives[0]
        text    = best.transcript
        conf    = best.confidence
        det_lang = (
            response.results[0].language_code
            if hasattr(response.results[0], "language_code")
            else lang_code
        )

        # Map GCP code back to our short code
        short_lang = next(
            (k for k, v in GCP_LANG_CODES.items() if v == det_lang),
            language
        )

        return TranscriptionResult(
            text=text,
            language=short_lang,
            confidence=conf,
            method="google_cloud",
        )

    except Exception as e:
        return TranscriptionResult(
            text="", language=language, method="google_cloud",
            error=str(e)
        )


def _transcribe_whisper(
    audio_bytes: bytes,
    language: str,
) -> TranscriptionResult:
    
    try:
        import whisper

        # Save to temp file (Whisper needs a file path)
        fmt = get_audio_format(audio_bytes)
        suffix = f".{fmt}"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Load small model for speed (downloads ~150MB once)
        model = whisper.load_model("small")

        whisper_lang = WHISPER_LANG_CODES.get(language)
        result = model.transcribe(
            tmp_path,
            language=whisper_lang,
            task="transcribe",
            fp16=False,         # CPU-safe
        )

        Path(tmp_path).unlink(missing_ok=True)

        text = result.get("text", "").strip()
        det_lang = result.get("language", language)

        # Map Whisper lang back to our short code
        short_lang = next(
            (k for k, v in WHISPER_LANG_CODES.items() if v == det_lang),
            language
        )

        return TranscriptionResult(
            text=text,
            language=short_lang,
            confidence=0.85,   # Whisper doesn't return per-segment confidence
            method="whisper",
        )

    except Exception as e:
        return TranscriptionResult(
            text="", language=language, method="failed",
            error=f"Whisper error: {str(e)}"
        )


# ── Text-to-Speech 

def text_to_speech(
    text: str,
    language: str = "en",
    slow: bool = False,
) -> Optional[bytes]:
    
    try:
        from gtts import gTTS

        lang_code = GTTS_LANG_CODES.get(language, "en")

        # Truncate very long responses for TTS (keep first 500 chars)
        speak_text = text[:800] if len(text) > 800 else text

        tts    = gTTS(text=speak_text, lang=lang_code, slow=slow)
        mp3_io = io.BytesIO()
        tts.write_to_fp(mp3_io)
        mp3_io.seek(0)
        return mp3_io.read()

    except Exception as e:
        print(f"  ⚠️  TTS failed: {e}")
        return None


# ── Language Detection 
def detect_language(text: str) -> str:
    
    if not text:
        return "en"

    devanagari_count = sum(
        1 for ch in text if "\u0900" <= ch <= "\u097F"
    )
    ratio = devanagari_count / max(len(text.replace(" ", "")), 1)

    if ratio > 0.3:
        return "hi"

    # Try langdetect if available
    try:
        from langdetect import detect
        detected = detect(text)
        if detected in ("hi", "bn", "ta", "te"):
            return detected
    except Exception:
        pass

    return "en"


# ── Quick test 
if __name__ == "__main__":
    print("Voice module loaded successfully.")
    print("\nLanguage detection test:")
    test_texts = [
        ("Hello, I have a fever.", "en"),
        ("मुझे बुखार है और सिर दर्द है।", "hi"),
        ("Main theek nahi feel kar raha hoon.", "en"),
    ]
    for text, expected in test_texts:
        detected = detect_language(text)
        status = "✅" if detected == expected else "⚠️"
        print(f"  {status} '{text[:40]}' → detected: {detected}")

    print("\nTTS test (English):")
    audio = text_to_speech("Hello, I am your MediAssist AI health assistant.", "en")
    if audio:
        print(f"  ✅ TTS generated {len(audio):,} bytes of audio")
    else:
        print("  ❌ TTS failed")

    print("\nGCP credentials:", "✅ Available" if _gcp_credentials_available()
          else "⚠️  Not configured (Whisper fallback will be used)")
    print("\nVoice module ready.")