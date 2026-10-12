import asyncio
import io
import logging
import os
from typing import Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, ImageOps, UnidentifiedImageError

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("roadwise")

app = FastAPI(title="RoadWise AI Service", version="1.1.0")


# --------------------------------------------------------------------------- #
# Configuration (override via environment variables)
# --------------------------------------------------------------------------- #
def _env_float(name: str, default: Optional[float]) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("Invalid value for %s=%r, using default %r", name, raw, default)
        return default


MODEL_PATH = os.getenv("MODEL_PATH", "rdd2022_d4_resnet38.onnx")
DEFAULT_INPUT_SIZE = int(os.getenv("INPUT_SIZE", "224"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(12 * 1024 * 1024)))  # 12 MB
Image.MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", str(60_000_000)))  # decompression-bomb guard

# Must match how the model was trained. Your original code used plain /255,
# so that is the default. Set USE_IMAGENET_NORM=1 if training used ImageNet mean/std.
USE_IMAGENET_NORM = os.getenv("USE_IMAGENET_NORM", "0") == "1"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
LETTERBOX_FILL = int(os.getenv("LETTERBOX_FILL", "114"))  # neutral grey padding

# Output-distribution gates. An image is "weak/ambiguous" if ANY of these trip.
CONF_FLOOR = _env_float("CONF_FLOOR", 0.55)        # top-1 probability must be >= this
MARGIN_MIN = _env_float("MARGIN_MIN", 0.15)        # top-1 minus top-2 must be >= this
ENTROPY_MAX = _env_float("ENTROPY_MAX", 0.85)      # normalized entropy (0..1) must be <= this
MAX_LOGIT_FLOOR = _env_float("MAX_LOGIT_FLOOR", None)  # optional; disabled unless set (see notes)

# Texture gate (gradient statistics on a 0..1 grayscale image)
TEXTURE_MEAN_MIN = _env_float("TEXTURE_MEAN_MIN", 0.012)      # mean gradient magnitude
TEXTURE_STRONG_MIN = _env_float("TEXTURE_STRONG_MIN", 0.010)  # fraction of strong-edge pixels
STRONG_EDGE_LEVEL = _env_float("STRONG_EDGE_LEVEL", 0.12)
# If the surface is flat, the model must be *very* sure to still be believed.
FLAT_OVERRIDE_CONF = _env_float("FLAT_OVERRIDE_CONF", 0.90)

# Run at most N inferences at once (shared CPU on Render).
MAX_CONCURRENT_INFERENCES = int(os.getenv("MAX_CONCURRENT_INFERENCES", "1"))
_inference_gate: Optional[asyncio.Semaphore] = None


# --------------------------------------------------------------------------- #
# Class / severity mapping
# --------------------------------------------------------------------------- #
# 4-class RDD2022 D4 convention
RDD_D4_CLASSES = [
    ("D00", "Longitudinal Crack", "Low"),
    ("D10", "Transverse Crack", "Low"),
    ("D20", "Alligator Crack", "Medium"),
    ("D40", "Pothole", "High"),
]
SEVERITY_3_CLASS = ["Low", "Medium", "High"]


def map_class(idx: int, num_classes: int):
    """
    Returns (severity, detected_damage_type) or (None, None) if the output size
    is not one we know how to interpret. Never raises IndexError.
    """
    if num_classes == 4 and 0 <= idx < len(RDD_D4_CLASSES):
        code, name, severity = RDD_D4_CLASSES[idx]
        return severity, f"{name} ({code})"
    if num_classes == 3 and 0 <= idx < len(SEVERITY_3_CLASS):
        severity = SEVERITY_3_CLASS[idx]
        return severity, f"Pavement distress ({severity} severity class)"
    return None, None


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
session: Optional[ort.InferenceSession] = None
input_name: Optional[str] = None
input_h = input_w = DEFAULT_INPUT_SIZE


def _load_model():
    global session, input_name, input_h, input_w
    try:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(
            MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        inp = session.get_inputs()[0]
        input_name = inp.name

        # Respect static input dims if the model declares them (NCHW).
        shape = inp.shape
        if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            input_h, input_w = shape[2], shape[3]
        log.info(">>> [AI READY] Model '%s' loaded (input %s, %dx%d).", MODEL_PATH, shape, input_h, input_w)

        # Warm-up so the first real request isn't slow.
        dummy = np.zeros((1, 3, input_h, input_w), dtype=np.float32)
        session.run(None, {input_name: dummy})
    except Exception as e:  # noqa: BLE001
        log.error(">>> [AI ERROR] Failed to load model: %s", e)
        session = None


_load_model()


@app.on_event("startup")
async def _startup():
    global _inference_gate
    _inference_gate = asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def letterbox(image: Image.Image, target_h: int, target_w: int):
    """
    Scale to fit inside (target_h, target_w) without distortion, centre on a grey canvas.
    Returns (canvas_image, content_image) where content_image is the un-padded resized photo.
    """
    w, h = image.size
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    content = image.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (target_w, target_h), (LETTERBOX_FILL,) * 3)
    canvas.paste(content, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return canvas, content


def to_tensor(canvas: Image.Image) -> np.ndarray:
    arr = np.asarray(canvas, dtype=np.float32) / 255.0   # HWC, [0,1]
    arr = np.transpose(arr, (2, 0, 1))                   # CHW
    if USE_IMAGENET_NORM:
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(arr[np.newaxis, ...], dtype=np.float32)  # NCHW


def load_image(contents: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(contents))
    try:
        # Fast JPEG downscale-on-decode: big phone photos decode much cheaper.
        img.draft("RGB", (input_w * 2, input_h * 2))
    except Exception:  # noqa: BLE001
        pass
    img = ImageOps.exif_transpose(img)  # honour phone orientation tag
    return img.convert("RGB")


# --------------------------------------------------------------------------- #
# Analysis helpers
# --------------------------------------------------------------------------- #
def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / np.sum(e)


def looks_like_probabilities(v: np.ndarray) -> bool:
    return bool(np.all(v >= 0.0) and np.all(v <= 1.0) and abs(float(v.sum()) - 1.0) < 1e-3)


def distribution_stats(probs: np.ndarray) -> dict:
    order = np.argsort(probs)[::-1]
    top1 = float(probs[order[0]])
    top2 = float(probs[order[1]]) if len(order) > 1 else 0.0
    n = len(probs)
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    entropy_norm = entropy / float(np.log(n)) if n > 1 else 0.0
    return {
        "top_idx": int(order[0]),
        "top1": top1,
        "top2": top2,
        "margin": top1 - top2,
        "entropy_norm": entropy_norm,
    }


def texture_stats(content: Image.Image) -> dict:
    """
    Pure NumPy surface-activity check on the un-padded photo.
    Smooth/uniform scenes have almost no gradient energy; cracked or broken
    pavement has thin, strong edges. This is a *supporting* signal only.
    """
    gray = np.asarray(content.convert("L"), dtype=np.float32) / 255.0
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return {"mean_gradient": 0.0, "strong_edge_ratio": 0.0, "flat": True}
    gx = np.abs(np.diff(gray, axis=1))[:-1, :]
    gy = np.abs(np.diff(gray, axis=0))[:, :-1]
    grad = gx + gy
    mean_grad = float(grad.mean())
    strong_ratio = float((grad > STRONG_EDGE_LEVEL).mean())
    flat = mean_grad < TEXTURE_MEAN_MIN and strong_ratio < TEXTURE_STRONG_MIN
    return {"mean_gradient": mean_grad, "strong_edge_ratio": strong_ratio, "flat": flat}


def build_response(severity, confidence, status, damage_type, message):
    return {
        "severity": severity,
        "confidence": confidence,
        "status": status,
        "detected_damage_type": damage_type,
        "message": message,
    }


def analyze(contents: bytes, debug: bool = False) -> dict:
    # ---- decode + preprocess ----
    try:
        image = load_image(contents)
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid or unsupported image: {e}")

    canvas, content = letterbox(image, input_h, input_w)
    tensor = to_tensor(canvas)

    # ---- inference ----
    outputs = session.run(None, {input_name: tensor})
    raw = np.asarray(outputs[0], dtype=np.float32).reshape(-1)  # (1,N) or (N,) -> (N,)
    num_classes = int(raw.shape[0])

    if num_classes < 2:
        return build_response(
            "Unassessed", 0.0, "LOW_CONFIDENCE", None,
            f"Model returned {num_classes} output(s); cannot interpret result."
        )

    probs = raw.astype(np.float64) if looks_like_probabilities(raw) else softmax(raw.astype(np.float64))
    stats = distribution_stats(probs)
    tex = texture_stats(content)
    max_logit = float(raw.max())

    # ---- ambiguity gates (OOD / undamaged road) ----
    reasons = []
    if stats["top1"] < CONF_FLOOR:
        reasons.append(f"top confidence {stats['top1']*100:.1f}% below floor {CONF_FLOOR*100:.0f}%")
    if stats["margin"] < MARGIN_MIN:
        reasons.append(f"top-2 margin {stats['margin']*100:.1f}% too narrow")
    if stats["entropy_norm"] > ENTROPY_MAX:
        reasons.append(f"prediction entropy {stats['entropy_norm']:.2f} too high")
    if MAX_LOGIT_FLOOR is not None and max_logit < MAX_LOGIT_FLOOR:
        reasons.append(f"max logit {max_logit:.2f} below floor {MAX_LOGIT_FLOOR}")
    weak = bool(reasons)

    flat_but_unsure = tex["flat"] and stats["top1"] < FLAT_OVERRIDE_CONF
    if tex["flat"] and not weak and flat_but_unsure:
        reasons.append("surface texture is smooth/uniform")

    diagnostics = {
        "num_classes": num_classes,
        "top1": round(stats["top1"], 4),
        "top2": round(stats["top2"], 4),
        "margin": round(stats["margin"], 4),
        "entropy_norm": round(stats["entropy_norm"], 4),
        "max_logit": round(max_logit, 3),
        "texture": {k: (round(v, 5) if isinstance(v, float) else v) for k, v in tex.items()},
        "reasons": reasons,
    }

    severity, damage_type = map_class(stats["top_idx"], num_classes)

    # ---- decision ----
    if weak or flat_but_unsure:
        if tex["flat"]:
            result = build_response(
                "Undamaged", 0.0, "UNDAMAGED", None,
                "No significant pavement distress detected. Surface appears smooth and the model "
                "found no clear damage pattern."
            )
        else:
            result = build_response(
                "Unassessed", 0.0, "LOW_CONFIDENCE", None,
                "The AI could not make a reliable assessment (image may not show a road surface, "
                "or the damage is unclear). Please retake the photo or review manually."
            )
    elif severity is None:
        result = build_response(
            "Unassessed", 0.0, "LOW_CONFIDENCE", None,
            f"Model produced {num_classes} classes, which has no configured severity mapping."
        )
    else:
        result = build_response(
            severity,
            round(stats["top1"] * 100.0, 1),
            "CONFIRMED",
            damage_type,
            f"{damage_type} detected; assessed severity: {severity}.",
        )

    if debug:
        result["diagnostics"] = diagnostics
    log.info("predict -> %s | %s", result["status"], diagnostics)
    return result


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/")
def health_check():
    """Health check endpoint to verify the service is awake."""
    return {
        "status": "online",
        "service": "RoadWise AI Inference",
        "model_loaded": session is not None,
    }


@app.post("/predict")
async def predict_severity(
    file: UploadFile = File(...),
    debug: bool = Query(False, description="Include raw diagnostics in the response"),
):
    """Receives an image, returns severity, confidence, status and damage type."""
    if session is None:
        raise HTTPException(status_code=503, detail="AI Model session is not initialized.")

    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )

    try:
        gate = _inference_gate or asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)
        async with gate:
            return await run_in_threadpool(analyze, contents, debug)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        log.exception("Inference failure")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
