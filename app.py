import base64
import io
import os
from functools import lru_cache

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "vae_decoder.keras")
Z_LIMIT = 3.0       # the latent map covers -3..3 on both axes
Z_CLAMP = 5.0       # hard limit for any z value sent by the browser

if not os.path.exists(MODEL_PATH):
    raise SystemExit(
        "Decoder not found at models/vae_decoder.keras. "
        "Unzip vae_models.zip from Colab and copy vae_decoder.keras into the models/ folder."
    )

decoder = keras.models.load_model(MODEL_PATH, compile=False)
LATENT_DIM = int(decoder.inputs[0].shape[-1])

app = Flask(__name__)


def decode(z_batch):
    """Latent vectors (n, LATENT_DIM) -> images (n, 28, 28) in the range 0-1."""
    z = np.asarray(z_batch, dtype="float32").reshape(-1, LATENT_DIM)
    return decoder(z, training=False).numpy()[..., 0]


def to_data_uri(img, size):
    arr = (np.clip(img, 0.0, 1.0) * 255).astype("uint8")
    pil = Image.fromarray(arr).resize((size, size), Image.Resampling.BICUBIC)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def parse_z(values):
    z = [float(v) for v in values][:LATENT_DIM]
    z += [0.0] * (LATENT_DIM - len(z))
    z = np.array(z, dtype="float32")
    if not np.all(np.isfinite(z)):
        raise ValueError("z values must be finite numbers.")
    return np.clip(z, -Z_CLAMP, Z_CLAMP)


@lru_cache(maxsize=4)
def latent_map(steps, limit):
    """Decode a steps x steps grid across the 2-D latent space into one image."""
    centers = (np.arange(steps) + 0.5) / steps * 2 * limit - limit
    xs = centers
    ys = centers[::-1]                      # top row = highest z2
    grid = np.array([[x, y] for y in ys for x in xs], dtype="float32")
    imgs = decode(grid).reshape(steps, steps, 28, 28)
    canvas = imgs.transpose(0, 2, 1, 3).reshape(steps * 28, steps * 28)
    return to_data_uri(canvas, steps * 28)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/info")
def info():
    return jsonify({"latent_dim": LATENT_DIM, "limit": Z_LIMIT})


@app.route("/api/generate", methods=["POST"])
def generate():
    body = request.get_json(silent=True) or {}
    try:
        z = parse_z(body.get("z", []))
    except (TypeError, ValueError) as err:
        return jsonify({"error": f"Invalid latent vector: {err}"}), 400
    image = decode(z)[0]
    return jsonify({"z": [round(float(v), 3) for v in z], "image": to_data_uri(image, 280)})


@app.route("/api/random")
def random_samples():
    n = request.args.get("n", default=16, type=int)
    n = max(1, min(n, 36))
    z = np.random.default_rng().standard_normal((n, LATENT_DIM)).astype("float32")
    images = decode(z)
    items = [
        {"z": [round(float(v), 3) for v in z[i]], "image": to_data_uri(images[i], 112)}
        for i in range(n)
    ]
    return jsonify({"items": items})


@app.route("/api/map")
def map_image():
    if LATENT_DIM != 2:
        return jsonify({"error": "The latent map needs a 2-D latent space."}), 400
    return jsonify({"image": latent_map(20, Z_LIMIT), "limit": Z_LIMIT})


if __name__ == "__main__":
    # debug=False so TensorFlow loads the model once
    app.run(host="127.0.0.1", port=5000, debug=False)
