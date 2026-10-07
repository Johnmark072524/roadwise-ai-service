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

# 🛑 Rejection threshold: In a 3-class model (random guess = 33.3%),
# anything below 48% means the model is guessing and cannot be trusted.
CONFIDENCE_FLOOR = 0.48


def validate_road_surface(img: Image.Image) -> tuple[bool, str]:
    """
    Validates whether the image resembles real pavement (asphalt/concrete)
    before feeding it into the neural network.
    """
    rgb = np.array(img, dtype=np.float32)
    gray = np.array(img.convert("L"), dtype=np.float32)

    # 1. Brightness Check (catches white printer paper, bright sheets, or pitch-black photos)
    mean_brightness = float(np.mean(gray))
    if mean_brightness > 220.0:
        return False, "Overexposed or blank white surface (e.g., paper)."
    if mean_brightness < 25.0:
        return False, "Image is too dark to inspect."

    # 2. Texture Gradient Variance Check
    # Asphalt/concrete has granular surface variations with high gradient variance (> 80).
    # Blank paper, smooth table surfaces, or walls have near-zero gradient variance (< 30).
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    texture_variance = float(np.var(gx) + np.var(gy))
    if texture_variance < 35.0:
        return False, "Surface is too smooth; no pavement texture detected."

    # 3. Color Saturation Check
    # Road asphalt and concrete are dominated by neutral gray tones.
    # High color distance indicates non-road objects (e.g., clothes, vivid toys, bright plastics).
    color_difference = (
        np.abs(rgb[:, :, 0] - rgb[:, :, 1]) +
        np.abs(rgb[:, :, 1] - rgb[:, :, 2]) +
        np.abs(rgb[:, :, 2] - rgb[:, :, 0])
    ) / 3.0
    avg_colorfulness = float(np.mean(color_difference))
    if avg_colorfulness > 45.0:
        return False, "Image contains vivid non-pavement colors."

    return True, "Valid road surface"


@app.get("/")
def health_check():
    """Health check endpoint to verify the service is awake."""
    return {
        "status": "online",
        "service": "RoadWise AI Inference",
        "model_loaded": session is not None
    }


@app.post("/predict")
async def predict_severity(file: UploadFile = File(...)):
    """Receives image file, validates surface texture, runs inference, returns severity + confidence."""
    if session is None:
        raise HTTPException(status_code=503, detail="AI Model session is not initialized.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        # Load original image in memory
        raw_image = Image.open(io.BytesIO(contents)).convert("RGB")

        # ------------------------------------------------------------------
        # GUARD 1: Physical Surface Sanity Check
        # ------------------------------------------------------------------
        is_road, reason = validate_road_surface(raw_image)
        if not is_road:
            print(f">>> [REJECTED] {reason}")
            return {
                "severity": "Unassessed",
                "confidence": 0.0,
                "status": "INVALID_SURFACE",
                "message": reason
            }

        # ------------------------------------------------------------------
        # GUARD 2: Preprocess for ResNet-38 (224x224)
        # ------------------------------------------------------------------
        image_resized = raw_image.resize((224, 224))
        img_np = np.array(image_resized, dtype=np.float32) / 255.0
        img_np = np.transpose(img_np, (2, 0, 1))  # HWC -> CHW
        img_np = np.expand_dims(img_np, axis=0)    # Add batch -> (1, 3, 224, 224)

        # ------------------------------------------------------------------
        # Inference & Probability Distribution
        # ------------------------------------------------------------------
        outputs = session.run(None, {input_name: img_np})
        logits = outputs[0][0]

        # Numerically stable Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        max_idx = int(np.argmax(probs))
        top_confidence = float(probs[max_idx])

        # ------------------------------------------------------------------
        # GUARD 3: Statistical Confidence Floor Check
        # ------------------------------------------------------------------
        if top_confidence < CONFIDENCE_FLOOR:
            print(f">>> [LOW CONFIDENCE] Top score {top_confidence:.2f} below {CONFIDENCE_FLOOR}")
            return {
                "severity": "Unassessed",
                "confidence": round(top_confidence * 100.0, 1),
                "status": "UNCERTAIN_PREDICTION",
                "message": "Model confidence too low to determine distress severity."
            }

        return {
            "severity": SEVERITY_LEVELS[max_idx],
            "confidence": round(top_confidence * 100.0, 1),
            "status": "CONFIRMED"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")