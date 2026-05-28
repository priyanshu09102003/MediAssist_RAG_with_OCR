import sys
import io
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont




def print_header(title: str):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")

def ok(msg):  print(f"  ✅  {msg}")
def warn(msg): print(f"  ⚠️   {msg}")
def fail(msg): print(f"  ❌  {msg}")


def make_test_image(text: str = "Red rash on arm") -> bytes:
    
    img = Image.new("RGB", (400, 300), color=(210, 180, 140))  # skin tone bg
    draw = ImageDraw.Draw(img)

    # Draw a red patch (simulating rash)
    draw.ellipse([120, 80, 280, 200], fill=(200, 50, 50), outline=(150, 20, 20), width=2)
    draw.ellipse([150, 110, 230, 170], fill=(220, 80, 80))

    # Add label text
    draw.text((10, 10), text, fill=(50, 50, 50))
    draw.text((10, 270), "Test image — not a real patient", fill=(100, 100, 100))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# ── Vision tests 

def test_vision():
    print_header("VISION MODULE TESTS (modules/vision.py)")

    from modules.vision import (
        analyze_patient_image,
        analyze_lab_report,
        image_result_to_text,
        validate_and_preprocess,
    )

    # ── Test 1: Image preprocessing
    print("\n[Test 1] Image preprocessing & validation")
    try:
        raw = make_test_image("Skin rash test")
        processed, mime = validate_and_preprocess(raw, "image/jpeg")
        ok(f"Image preprocessed: {len(processed):,} bytes, mime={mime}")
    except Exception as e:
        fail(f"Preprocessing failed: {e}")
        return

    # ── Test 2: Gemini Vision — patient image analysis
    print("\n[Test 2] Gemini Vision — patient image analysis")
    print("  Sending synthetic rash image to Gemini Vision API...")
    try:
        result = analyze_patient_image(
            image_bytes=processed,
            mime_type="image/jpeg",
            user_query="I have a red rash on my arm that appeared yesterday",
        )

        if result.description and len(result.description) > 20:
            ok(f"Description received ({len(result.description)} chars)")
            ok(f"Affected area    : {result.affected_area or '(not extracted)'}")
            ok(f"Visible symptoms : {result.visible_symptoms or '(none parsed)'}")
            ok(f"Severity hint    : {result.severity_hint}")
            ok(f"Urgent care      : {result.needs_urgent_care}")
            if result.possible_conditions:
                ok(f"Possible conds   : {', '.join(result.possible_conditions[:3])}")
            print(f"\n  --- Description Preview ---")
            print(f"  {result.description[:300]}")
        else:
            warn(f"Response was short or empty: '{result.description[:100]}'")
            if result.raw_response:
                print(f"  Raw response: {result.raw_response[:200]}")

    except Exception as e:
        fail(f"Vision analysis failed: {e}")

    # ── Test 3: image_result_to_text
    print("\n[Test 3] image_result_to_text() conversion")
    try:
        text_summary = image_result_to_text(result)
        ok(f"Text summary generated ({len(text_summary)} chars)")
        print(f"  {text_summary[:200]}...")
    except Exception as e:
        fail(f"Text conversion failed: {e}")

    # ── Test 4: Lab report analysis (synthetic) 
    print("\n[Test 4] Lab report analysis (synthetic image)")
    print("  Creating synthetic CBC report image...")
    try:
        lab_img = _make_synthetic_lab_report()
        lab_result = analyze_lab_report(lab_img, ".jpg")

        if lab_result.ai_summary and len(lab_result.ai_summary) > 20:
            ok(f"Report type  : {lab_result.report_type or 'detected'}")
            ok(f"Parameters   : {len(lab_result.parameters)} extracted")
            ok(f"Abnormal     : {len(lab_result.abnormal_flags)} flags")
            ok(f"Summary      : {lab_result.ai_summary[:150]}...")
        else:
            warn(f"Lab analysis returned minimal data — check Gemini response")
            if lab_result.extracted_text:
                print(f"  OCR text: {lab_result.extracted_text[:100]}")
    except Exception as e:
        fail(f"Lab report analysis failed: {e}")

    print("\n✅ Vision module tests complete.")


def _make_synthetic_lab_report() -> bytes:
   
    img = Image.new("RGB", (500, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    lines = [
        "PATHOLOGY LAB REPORT",
        "Patient: Test Patient  Age: 35  Gender: M",
        "Test: Complete Blood Count (CBC)",
        "Date: 2025-05-28",
        "",
        "PARAMETER      VALUE    UNIT    REFERENCE    STATUS",
        "Hemoglobin     9.2      g/dL    13.0-17.0    LOW *",
        "WBC            11500    /uL     4000-11000   HIGH *",
        "Platelets      180000   /uL     150000-400000 NORMAL",
        "Hematocrit     28.5     %       40-52        LOW *",
        "MCV            72       fL      80-100       LOW *",
        "MCH            22       pg      27-32        LOW *",
        "",
        "* = Abnormal value",
    ]

    y = 20
    for line in lines:
        color = (200, 0, 0) if "LOW *" in line or "HIGH *" in line else (0, 0, 0)
        if "PARAMETER" in line:
            color = (0, 0, 150)
        draw.text((20, y), line, fill=color)
        y += 24

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ── Voice tests 
def test_voice():
    print_header("VOICE MODULE TESTS (modules/voice.py)")

    from modules.voice import (
        detect_language,
        text_to_speech,
        _gcp_credentials_available,
        transcribe_audio,
        get_audio_format,
    )

    # ── Test 1: Language detection
    print("\n[Test 1] Language detection (no API needed)")
    test_cases = [
        ("I have a fever and bad headache",          "en"),
        ("मुझे बुखार है और सिर में दर्द है",          "hi"),
        ("Mujhe bukhaar hai",                         "en"),  # Hinglish → en
        ("मेरे पेट में दर्द हो रहा है",               "hi"),
        ("My blood sugar is high",                    "en"),
    ]
    lang_pass = 0
    for text, expected in test_cases:
        detected = detect_language(text)
        if detected == expected:
            ok(f"'{text[:40]}' → {detected}")
            lang_pass += 1
        else:
            warn(f"'{text[:40]}' → got '{detected}', expected '{expected}'")

    print(f"\n  Language detection: {lang_pass}/{len(test_cases)} correct")

    # ── Test 2: Audio format detection
    print("\n[Test 2] Audio format detection")
    try:
        wav_magic  = b"RIFF" + b"\x00" * 40
        mp3_magic  = b"ID3"  + b"\x00" * 40
        webm_magic = b"\x1a\x45\xdf\xa3" + b"\x00" * 40

        assert get_audio_format(wav_magic)  == "wav"
        assert get_audio_format(mp3_magic)  == "mp3"
        assert get_audio_format(webm_magic) == "webm"
        ok("WAV, MP3, WebM format detection working")
    except Exception as e:
        fail(f"Format detection failed: {e}")

    # ── Test 3: Text-to-Speech
    print("\n[Test 3] Text-to-Speech — English")
    try:
        audio_en = text_to_speech(
            "Hello! I am MediAssist, your AI health assistant. How can I help you today?",
            language="en",
        )
        if audio_en and len(audio_en) > 1000:
            ok(f"English TTS: {len(audio_en):,} bytes generated")
            # Save to file for manual listening
            Path("data").mkdir(exist_ok=True)
            Path("data/test_tts_en.mp3").write_bytes(audio_en)
            ok("Saved to data/test_tts_en.mp3 — open and listen to verify")
        else:
            fail("English TTS returned empty audio")
    except Exception as e:
        fail(f"English TTS failed: {e}")

    print("\n[Test 4] Text-to-Speech — Hindi")
    try:
        audio_hi = text_to_speech(
            "नमस्ते! मैं आपका AI स्वास्थ्य सहायक हूँ। आपकी क्या समस्या है?",
            language="hi",
        )
        if audio_hi and len(audio_hi) > 1000:
            ok(f"Hindi TTS: {len(audio_hi):,} bytes generated")
            Path("data/test_tts_hi.mp3").write_bytes(audio_hi)
            ok("Saved to data/test_tts_hi.mp3 — open and listen to verify")
        else:
            fail("Hindi TTS returned empty audio")
    except Exception as e:
        fail(f"Hindi TTS failed: {e}")

    # ── Test 5: STT availability check
    print("\n[Test 5] STT (Speech-to-Text) availability")
    gcp_ok = _gcp_credentials_available()
    if gcp_ok:
        ok("Google Cloud credentials found — GCP STT available")
    else:
        warn("GCP credentials NOT configured — Whisper fallback will be used")
        warn("To use GCP STT: set GOOGLE_APPLICATION_CREDENTIALS in .env")

    # Check Whisper
    try:
        import whisper
        ok("OpenAI Whisper installed — offline STT fallback available")
    except ImportError:
        warn("Whisper not importable — run: pip install openai-whisper")

    # ── Test 6: STT with synthetic WAV (silence test) 
    print("\n[Test 6] STT pipeline test (near-silence WAV)")
    print("  Note: transcription of silence/noise will return empty — that's expected.")
    try:
        silent_wav = _make_silent_wav(duration_ms=500)
        result = transcribe_audio(silent_wav, language="en")
        ok(f"STT pipeline ran — method: {result.method}")
        if result.error:
            warn(f"STT note: {result.error}")
        else:
            ok(f"Transcription: '{result.text or '(empty — expected for silence)'}'")
    except Exception as e:
        fail(f"STT pipeline error: {e}")

    print("\n✅ Voice module tests complete.")
    print("\n📁 Check these output files:")
    print("   data/test_tts_en.mp3  — play to hear English TTS")
    print("   data/test_tts_hi.mp3  — play to hear Hindi TTS")


def _make_silent_wav(duration_ms: int = 500, sample_rate: int = 16000) -> bytes:
    
    import struct, wave
    num_samples = int(sample_rate * duration_ms / 1000)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{num_samples}h", *([0] * num_samples)))
    return buf.getvalue()


# ── CLI_Test_Entry point

if __name__ == "__main__":
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    if arg in ("vision", "all"):
        test_vision()

    if arg in ("voice", "all"):
        test_voice()

    if arg not in ("vision", "voice", "all"):
        print(f"Usage: python test_modules.py [vision|voice|all]")