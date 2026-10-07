import io
import numpy as np
import onnxruntime as ort
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException

app = FastAPI(title="RoadWise AI Service", version="1.0.0")

# 1. Load ONNX Model on Startup
MODEL_PATH = "rdd2022_d4_resnet38.onnx"
try:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    print(f">>> [AI READY] Model '{MODEL_PATH}' loaded successfully.")
except Exception as e:
    print(f">>> [AI ERROR] Failed to load model: {e}")
    session = None

SEVERITY_LEVELS = ["Low", "Medium", "High"]

# Confidence floor: random guessing across 3 classes is 33.3%.
# Threshold is set at 40% so challenging field photos (wet/dusty) are not discarded.
CONFIDENCE_FLOOR = 0.40


def is_non_road_upload(img: Image.Image) -> tuple[bool, str]:
    """
    Lightweight, targeted filter that ONLY intercepts non-road items
    (like printed paper, school documents, or digital clip art).
    Allows all real asphalt, concrete, rain puddles, and gravel to pass through.
    """
    rgb = np.array(img.convert("RGB"), dtype=np.float32)
    gray = np.array(img.convert("L"), dtype=np.float32)
    total_pixels = gray.size

    # 1. Pure White Document Trap (Catches bond paper / worksheets)
    # Documents are dominated by light background paper (> 200).
    # Real roads (even concrete under sunlight) rarely exceed 35% pure white pixels.
    white_ratio = float(np.sum(gray > 200.0) / total_pixels)
    if white_ratio > 0.45:
        return True, "Surface rejected: Document or paper surface detected."

    # 2. Digital Graphic / Pure Color Saturation Trap (Catches vector art / UI graphics)
    # We only check the CENTER 60% of the image to ignore roadside greenery/sky at the borders.
    h, w = gray.shape
    center_rgb = rgb[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
    r, g, b = center_rgb[:, :, 0], center_rgb[:, :, 1], center_rgb[:, :, 2]
    center_saturation = float(np.mean(np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])))

    # Road centers (even with yellow paint or cones) stay under 35 saturation.
    # Digital vector graphics easily exceed 45-55.
    if center_saturation > 42.0:
        return True, "Surface rejected: High-saturation digital graphic detected."

    # 3. Blank / Blackout Check
    mean_brightness = float(np.mean(gray))
    if mean_brightness < 20.0:
        return True, "Image is too dark to inspect."

    return False, "Passed"


@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "RoadWise AI Inference",
        "model_loaded": session is not None
    }


@app.post("/predict")
async def predict_severity(file: UploadFile = File(...)):
    if session is None:
        raise HTTPException(status_code=503, detail="AI Model session is not initialized.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        raw_image = Image.open(io.BytesIO(contents))

        # ------------------------------------------------------------------
        # GUARD: Filter ONLY obvious non-road files (Documents / Vector Art)
        # ------------------------------------------------------------------
        is_invalid, reason = is_non_road_upload(raw_image)
        if is_invalid:
            print(f">>> [REJECTED NON-ROAD] {reason}")
            return {
                "severity": "Unassessed",
                "confidence": 0.0,
                "status": "INVALID_SURFACE",
                "message": reason
            }

        # ------------------------------------------------------------------
        # Model Preprocessing & Inference (ResNet-38)
        # ------------------------------------------------------------------
        image_resized = raw_image.convert("RGB").resize((224, 224))
        img_np = np.array(image_resized, dtype=np.float32) / 255.0
        img_np = np.transpose(img_np, (2, 0, 1))
        img_np = np.expand_dims(img_np, axis=0)

        outputs = session.run(None, {input_name: img_np})
        logits = outputs[0][0]

        # Numerically stable Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        max_idx = int(np.argmax(probs))
        top_confidence = float(probs[max_idx])
        predicted_severity = SEVERITY_LEVELS[max_idx]

        # ------------------------------------------------------------------
        # Statistical Confidence Floor Check
        # ------------------------------------------------------------------
        if top_confidence < CONFIDENCE_FLOOR:
            print(f">>> [LOW CONFIDENCE] Score {top_confidence:.2f} below {CONFIDENCE_FLOOR}")
            return {
                "severity": "Unassessed",
                "confidence": round(top_confidence * 100.0, 1),
                "status": "UNCERTAIN_PREDICTION",
                "message": "Model confidence too low to determine distress severity."
            }

        return {
            "severity": predicted_severity,
            "confidence": round(top_confidence * 100.0, 1),
            "status": "CONFIRMED"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")