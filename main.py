import asyncio
import gc
import io
import json
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

app = FastAPI(title="RoadWise AI Service", version="1.2.0")


# --------------------------------------------------------------------------- #
# Configuration
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


# 1. Switched to DINO default
MODEL_PATH = os.getenv("MODEL_PATH", "rdd2022_dino_d4.onnx")
DEFAULT_INPUT_SIZE = int(os.getenv("INPUT_SIZE", "224"))

# 2. Cleaned up syntax errors & capped upload at 8MB to protect 512MB RAM
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))
Image.MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", str(60_000_000)))

# Severity model preprocessing (plain /255 verified for DINO & ResNet-38)
USE_IMAGENET_NORM = os.getenv("USE_IMAGENET_NORM", "0") == "1"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
LETTERBOX_FILL = int(os.getenv("LETTERBOX_FILL", "114"))

# Gate Configuration
GATE_EMBEDDER_PATH = os.getenv("GATE_EMBEDDER_PATH", "gate_embedder.onnx")
GATE_HEAD_PATH = os.getenv("GATE_HEAD_PATH", "gate_head.json")
GATE_MIN_PROB = _env_float("GATE_MIN_PROB", 0.50)
SEVERITY_VERIFY_BELOW = _env_float("SEVERITY_VERIFY_BELOW", 0.50)

# Fallback gates (used if gate files are absent)
CONF_FLOOR = _env_float("CONF_FLOOR", 0.55)
MARGIN_MIN = _env_float("MARGIN_MIN", 0.15)
ENTROPY_MAX = _env_float("ENTROPY_MAX", 0.85)
MAX_LOGIT_FLOOR = _env_float("MAX_LOGIT_FLOOR", None)
ENERGY_FLOOR = _env_float("ENERGY_FLOOR", None)
TEXTURE_MEAN_MIN = _env_float("TEXTURE_MEAN_MIN", 0.012)
TEXTURE_STRONG_MIN = _env_float("TEXTURE_STRONG_MIN", 0.010)
STRONG_EDGE_LEVEL = _env_float("STRONG_EDGE_LEVEL", 0.12)
FLAT_OVERRIDE_CONF = _env_float("FLAT_OVERRIDE_CONF", 0.90)

MAX_CONCURRENT_INFERENCES = int(os.getenv("MAX_CONCURRENT_INFERENCES", "1"))
_inference_gate: Optional[asyncio.Semaphore] = None


# --------------------------------------------------------------------------- #
# Class / severity mapping
# --------------------------------------------------------------------------- #
RDD_D4_CLASSES = [
    ("D00", "Longitudinal Crack", "Low"),
    ("D10", "Transverse Crack", "Low"),
    ("D20", "Alligator Crack", "Medium"),
    ("D40", "Pothole", "High"),
]
SEVERITY_ORDER = ["Low", "Medium", "High"]


def class_table(num_classes: int):
    """[(label, severity)] per output index, or None if the size is unknown."""
    if num_classes == 4:
        return [(f"{n} ({c})", s) for c, n, s in RDD_D4_CLASSES]
    if num_classes == 3:
        return [(f"Pavement distress ({s} severity class)", s) for s in SEVERITY_ORDER]
    return None


def severity_from_probs(probs: np.ndarray):
    table = class_table(len(probs))
    if table is None:
        return None
    mass = {s: 0.0 for s in SEVERITY_ORDER}
    for p, (_, s) in zip(probs, table):
        mass[s] += float(p)
    severity = max(SEVERITY_ORDER, key=lambda s: mass[s])
    idxs = [i for i, (_, s) in enumerate(table) if s == severity]
    best = max(idxs, key=lambda i: probs[i])
    return severity, table[best][0], mass[severity]


# --------------------------------------------------------------------------- #
# Model loading & Memory Guards
# --------------------------------------------------------------------------- #
def _session_options():
    opts = ort.SessionOptions()
    # Critical for Render 512MB RAM: Releases memory back to the OS immediately
    opts.enable_cpu_mem_arena = False
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return opts


session: Optional[ort.InferenceSession] = None
input_name: Optional[str] = None
input_h = input_w = DEFAULT_INPUT_SIZE

gate_session: Optional[ort.InferenceSession] = None
gate_input_name: Optional[str] = None
gate_h = gate_w = 224
gate_head: Optional[dict] = None


def _static_hw(shape, default):
    if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
        return shape[2], shape[3]
    return default, default


def _load_model():
    global session, input_name, input_h, input_w
    try:
        session = ort.InferenceSession(MODEL_PATH, sess_options=_session_options(),
                                       providers=["CPUExecutionProvider"])
        inp = session.get_inputs()[0]
        input_name = inp.name
        input_h, input_w = _static_hw(inp.shape, DEFAULT_INPUT_SIZE)
        log.info(">>> [AI READY] Severity model '%s' loaded (%dx%d).", MODEL_PATH, input_h, input_w)
        session.run(None, {input_name: np.zeros((1, 3, input_h, input_w), dtype=np.float32)})
    except Exception as e:
        log.error(">>> [AI ERROR] Failed to load model: %s", e)
        session = None


def _load_gate():
    global gate_session, gate_input_name, gate_h, gate_w, gate_head
    if not (os.path.exists(GATE_EMBEDDER_PATH) and os.path.exists(GATE_HEAD_PATH)):
        log.info(">>> [GATE] files not found (%s, %s): using fallback gates.", GATE_EMBEDDER_PATH, GATE_HEAD_PATH)
        return
    try:
        with open(GATE_HEAD_PATH, "r", encoding="utf-8") as f:
            head = json.load(f)
        head["mean"] = np.asarray(head["mean"], dtype=np.float64)
        head["scale"] = np.asarray(head["scale"], dtype=np.float64)
        head["coef"] = np.asarray(head["coef"], dtype=np.float64)
        head["intercept"] = np.asarray(head["intercept"], dtype=np.float64)
        if head["coef"].shape[0] != len(head["classes"]):
            raise ValueError("coef rows do not match classes")

        gs = ort.InferenceSession(GATE_EMBEDDER_PATH, sess_options=_session_options(),
                                  providers=["CPUExecutionProvider"])
        inp = gs.get_inputs()[0]
        gh, gw = _static_hw(inp.shape, int(head.get("input_size", 224)))
        out = gs.run(None, {inp.name: np.zeros((1, 3, gh, gw), dtype=np.float32)})[0]
        dim = int(np.asarray(out).reshape(-1).shape[0])
        if dim != head["mean"].shape[0] or dim != head["coef"].shape[1]:
            raise ValueError(f"embedding size {dim} does not match head ({head['coef'].shape[1]})")

        gate_session, gate_input_name, gate_h, gate_w, gate_head = gs, inp.name, gh, gw, head
        log.info(">>> [GATE READY] classes=%s embedding=%d input=%dx%d", head["classes"], dim, gh, gw)
    except Exception as e:
        log.error(">>> [GATE ERROR] %s (falling back to older gates)", e)
        gate_session = None
        gate_head = None


_load_model()
_load_gate()


@app.on_event("startup")
async def _startup():
    global _inference_gate
    _inference_gate = asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def letterbox(image: Image.Image, target_h: int, target_w: int):
    w, h = image.size
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    content = image.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (target_w, target_h), (LETTERBOX_FILL,) * 3)
    canvas.paste(content, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return canvas, content


def to_tensor(canvas: Image.Image, imagenet_norm: Optional[bool] = None) -> np.ndarray:
    if imagenet_norm is None:
        imagenet_norm = USE_IMAGENET_NORM
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    if imagenet_norm:
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(arr[np.newaxis, ...], dtype=np.float32)


def load_image(contents: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(contents))
    try:
        img.draft("RGB", (max(input_w, gate_w) * 2, max(input_h, gate_h) * 2))
    except Exception:
        pass
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


# --------------------------------------------------------------------------- #
# Analysis helpers
# --------------------------------------------------------------------------- #
def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / np.sum(e)


def energy_score(logits: np.ndarray) -> float:
    m = float(np.max(logits))
    return m + float(np.log(np.sum(np.exp(logits - m))))


def looks_like_probabilities(v: np.ndarray) -> bool:
    return bool(np.all(v >= 0.0) and np.all(v <= 1.0) and abs(float(v.sum()) - 1.0) < 1e-3)


def distribution_stats(probs: np.ndarray) -> dict:
    order = np.argsort(probs)[::-1]
    top1 = float(probs[order[0]])
    top2 = float(probs[order[1]]) if len(order) > 1 else 0.0
    n = len(probs)
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    return {
        "top_idx": int(order[0]),
        "top1": top1,
        "top2": top2,
        "margin": top1 - top2,
        "entropy_norm": entropy / float(np.log(n)) if n > 1 else 0.0,
    }


def texture_stats(content: Image.Image) -> dict:
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


def gate_predict(image: Image.Image) -> dict:
    canvas, _ = letterbox(image, gate_h, gate_w)
    emb = gate_session.run(None, {gate_input_name: to_tensor(canvas, imagenet_norm=True)})[0]
    emb = np.asarray(emb, dtype=np.float64).reshape(-1)
    z = (emb - gate_head["mean"]) / np.where(gate_head["scale"] == 0, 1.0, gate_head["scale"])
    logits = gate_head["coef"] @ z + gate_head["intercept"]
    probs = softmax(logits)
    return {c: float(p) for c, p in zip(gate_head["classes"], probs)}


def run_severity_model(image: Image.Image):
    canvas, content = letterbox(image, input_h, input_w)
    outputs = session.run(None, {input_name: to_tensor(canvas)})
    raw = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
    is_prob = looks_like_probabilities(raw)
    probs = raw.astype(np.float64) if is_prob else softmax(raw.astype(np.float64))
    energy = None if is_prob else energy_score(raw.astype(np.float64))
    return raw, probs, energy, content


def build_response(severity, confidence, status, damage_type, message):
    return {
        "severity": severity,
        "confidence": confidence,
        "status": status,
        "detected_damage_type": damage_type,
        "message": message,
    }


# --------------------------------------------------------------------------- #
# Decision logic
# --------------------------------------------------------------------------- #
def _analyze_with_gate(image: Image.Image, debug: bool) -> dict:
    gate = gate_predict(image)
    top_class = max(gate, key=gate.get)
    top_p = gate[top_class]
    diag = {"mode": "gate", "gate": {k: round(v, 4) for k, v in gate.items()}}

    if top_p < GATE_MIN_PROB:
        result = build_response(
            "Unassessed", 0.0, "LOW_CONFIDENCE", None,
            "The AI could not tell whether this photo shows road damage. "
            "Please retake the photo or review manually.")
    elif top_class == "nonroad":
        result = build_response(
            "Unassessed", round(top_p * 100.0, 1), "LOW_CONFIDENCE", None,
            "This photo does not appear to show a road surface. Please upload a photo of the road.")
    elif top_class == "undamaged":
        result = build_response(
            "Undamaged", round(top_p * 100.0, 1), "UNDAMAGED", None,
            "No significant pavement distress detected.")
    else:
        raw, probs, energy, _content = run_severity_model(image)
        sev = severity_from_probs(probs)
        diag.update({"num_classes": int(raw.shape[0]),
                     "probs": [round(float(p), 4) for p in probs]})
        if sev is None:
            result = build_response(
                "Unassessed", 0.0, "LOW_CONFIDENCE", None,
                f"Damage detected, but the severity model returned {raw.shape[0]} classes, "
                "which has no configured severity mapping.")
        else:
            severity, damage_type, sev_p = sev
            msg = f"{damage_type} detected; assessed severity: {severity}."
            if sev_p < SEVERITY_VERIFY_BELOW:
                msg += " The AI is less certain about the exact damage type; please verify the severity."
            result = build_response(severity, round(sev_p * 100.0, 1), "CONFIRMED", damage_type, msg)
    if debug:
        result["diagnostics"] = diag
    log.info("predict(gate) -> %s | %s", result["status"], diag)
    return result


def _analyze_fallback(image: Image.Image, debug: bool) -> dict:
    raw, probs, energy, content = run_severity_model(image)
    num_classes = int(raw.shape[0])
    if num_classes < 2:
        return build_response("Unassessed", 0.0, "LOW_CONFIDENCE", None,
                              f"Model returned {num_classes} output(s); cannot interpret result.")

    stats = distribution_stats(probs)
    tex = texture_stats(content)
    reasons = []
    if stats["top1"] < CONF_FLOOR:
        reasons.append(f"top confidence {stats['top1']*100:.1f}% below floor {CONF_FLOOR*100:.0f}%")
    if stats["margin"] < MARGIN_MIN:
        reasons.append(f"top-2 margin {stats['margin']*100:.1f}% too narrow")
    if stats["entropy_norm"] > ENTROPY_MAX:
        reasons.append(f"prediction entropy {stats['entropy_norm']:.2f} too high")
    if MAX_LOGIT_FLOOR is not None and float(raw.max()) < MAX_LOGIT_FLOOR:
        reasons.append("max logit below floor")
    if ENERGY_FLOOR is not None and energy is not None and energy < ENERGY_FLOOR:
        reasons.append(f"energy {energy:.2f} below floor {ENERGY_FLOOR}")
    weak = bool(reasons)
    flat_but_unsure = tex["flat"] and stats["top1"] < FLAT_OVERRIDE_CONF

    diag = {
        "mode": "fallback", "num_classes": num_classes,
        "top1": round(stats["top1"], 4), "margin": round(stats["margin"], 4),
        "entropy_norm": round(stats["entropy_norm"], 4),
        "max_logit": round(float(raw.max()), 3),
        "energy": None if energy is None else round(energy, 3),
        "texture": {k: (round(v, 5) if isinstance(v, float) else v) for k, v in tex.items()},
        "reasons": reasons,
    }

    sev = severity_from_probs(probs)
    if weak or flat_but_unsure:
        if tex["flat"]:
            result = build_response("Undamaged", 0.0, "UNDAMAGED", None,
                                    "No significant pavement distress detected.")
        else:
            result = build_response("Unassessed", 0.0, "LOW_CONFIDENCE", None,
                                    "The AI could not make a reliable assessment. "
                                    "Please retake the photo or review manually.")
    elif sev is None:
        result = build_response("Unassessed", 0.0, "LOW_CONFIDENCE", None,
                                f"Model produced {num_classes} classes, which has no configured severity mapping.")
    else:
        severity, damage_type, sev_p = sev
        result = build_response(severity, round(sev_p * 100.0, 1), "CONFIRMED", damage_type,
                                f"{damage_type} detected; assessed severity: {severity}.")
    if debug:
        result["diagnostics"] = diag
    log.info("predict(fallback) -> %s | %s", result["status"], diag)
    return result


def analyze(contents: bytes, debug: bool = False) -> dict:
    try:
        image = load_image(contents)
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid or unsupported image: {e}")

    if gate_session is not None and gate_head is not None:
        return _analyze_with_gate(image, debug)
    return _analyze_fallback(image, debug)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "RoadWise AI Inference",
        "model_loaded": session is not None,
        "gate_loaded": gate_session is not None,
        "active_model": MODEL_PATH,
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
            res = await run_in_threadpool(analyze, contents, debug)
            # Clean up memory immediately to prevent memory leaks on Render 512MB
            del contents
            gc.collect()
            return res
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Inference failure")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")