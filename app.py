import io
import os
import time
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["ORT_NUM_THREADS"] = "1"

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from kokoro_onnx import Kokoro
import soundfile as sf

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.environ.get("KOKORO_MODEL_PATH", "kokoro-v1.0.int8.onnx")
VOICES_PATH = os.environ.get("KOKORO_VOICES_PATH", "voices-v1.0.bin")

_model_load_start = time.time()
kokoro = Kokoro(MODEL_PATH, VOICES_PATH)
_model_load_seconds = time.time() - _model_load_start
print(f"[TIMING] Model loaded at startup in {_model_load_seconds:.2f}s", flush=True)


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "kokoro-tts", "model_load_seconds": round(_model_load_seconds, 2)})


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
    t_request_start = time.time()

    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    voice = data.get("voice", "af_heart")
    speed = float(data.get("speed", 1.0))
    lang = data.get("lang", "en-us")

    if not text:
        return jsonify({"error": "text is required"}), 400

    try:
        t_inference_start = time.time()
        samples, sample_rate = kokoro.create(text, voice=voice, speed=speed, lang=lang)
        t_inference_end = time.time()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    t_encode_start = time.time()
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV")
    buf.seek(0)
    t_encode_end = time.time()

    inference_seconds = t_inference_end - t_inference_start
    encode_seconds = t_encode_end - t_encode_start
    total_seconds = time.time() - t_request_start

    print(
        f"[TIMING] text_len={len(text)} "
        f"inference={inference_seconds:.2f}s "
        f"encode={encode_seconds:.2f}s "
        f"total={total_seconds:.2f}s",
        flush=True,
    )

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
