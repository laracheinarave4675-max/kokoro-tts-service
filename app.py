import io
import os

from flask import Flask, request, send_file, jsonify
from kokoro_onnx import Kokoro
import soundfile as sf

app = Flask(__name__)

MODEL_PATH = os.environ.get("KOKORO_MODEL_PATH", "kokoro-v1.0.int8.onnx")
VOICES_PATH = os.environ.get("KOKORO_VOICES_PATH", "voices-v1.0.bin")

# Loaded once at startup and reused for every request (keeps memory flat,
# no accumulation between calls).
kokoro = Kokoro(MODEL_PATH, VOICES_PATH)


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "kokoro-tts"})


@app.route("/tts", methods=["POST"])
def tts():
    """
    Body (JSON):
      text:  string, required. Any length — no artificial character cap.
      voice: string, optional. Default "af_heart".
      speed: float,  optional. Default 1.0 (0.5 - 2.0 range recommended).
      lang:  string, optional. Default "en-us".
    Returns: audio/wav binary.
    """
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    voice = data.get("voice", "af_heart")
    speed = float(data.get("speed", 1.0))
    lang = data.get("lang", "en-us")

    if not text:
        return jsonify({"error": "text is required"}), 400

    try:
        samples, sample_rate = kokoro.create(text, voice=voice, speed=speed, lang=lang)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV")
    buf.seek(0)

    # Nothing is written to disk — the WAV lives only in memory for the
    # duration of this request, so there is nothing to clean up afterwards.
    return send_file(
        buf,
        mimetype="audio/wav",
        as_attachment=True,
        download_name="voice.wav",
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
  
