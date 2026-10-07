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

# Confidence floor for statistical uncertainty
CONFIDENCE_FLOOR = 0.48


def validate_road_surface(img: Image.Image) -> tuple[bool, str]:
    """
    Validates whether the image resembles real outdoor pavement (asphalt/concrete)
    before feeding it into the ResNet classifier.
    """
    rgb = np.array(img.convert("RGB"), dtype=np.float32)
    gray = np.array(img.convert("L"), dtype=np.float32)

    # -------------------------------------------------------------
    # 1. Vibrant Color & Graphic Trap (Catches icons, charts, infographics)
    # -------------------------------------------------------------
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    color_saturation = float(np.mean(np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])))
    if color_saturation > 28.0:
        return False, "Surface rejected: Non-road vibrant color or graphic detected."

    # -------------------------------------------------------------
    # 2. Document & Paper Whitespace Detection (Catches worksheets & paper)
    # -------------------------------------------------------------
    gray_224 = np.array(img.convert("L").resize((224, 224)), dtype=np.float32)
    blocks = gray_224.reshape(14, 16, 14, 16).swapaxes(1, 2).reshape(196, 16, 16)
    block_means = np.mean(blocks, axis=(1, 2))
    block_stds = np.std(blocks, axis=(1, 2))

    # Margins & Whitespace: Paper has blocks that are light (mean > 125) AND flat (std < 9.0)
    flat_light_blocks = np.sum((block_means > 125.0) & (block_stds < 9.0))
    flat_light_ratio = float(flat_light_blocks / 196.0)

    if flat_light_ratio > 0.12:
        return False, f"Surface rejected: Document or paper surface detected (whitespace ratio: {round(flat_light_ratio * 100, 1)}%)."

    smoothest_15th_pct = float(np.percentile(block_stds, 15))
    if smoothest_15th_pct < 5.5 and np.mean(gray_224) > 110.0:
        return False, "Surface rejected: Surface is too smooth to be outdoor asphalt or concrete."

    # -------------------------------------------------------------
    # 3. Overall Exposure Sanity Check
    # -------------------------------------------------------------
    mean_brightness = float(np.mean(gray))
    if mean_brightness > 220.0:
        return False, "Overexposed or blank white image."
    if mean_brightness < 25.0:
        return False, "Image is too dark to inspect."

    # -------------------------------------------------------------
    # 4. Global Texture Variance Check
    # -------------------------------------------------------------
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    texture_variance = float(np.var(gx) + np.var(gy))
    if texture_variance < 30.0:
        return False, "Surface is too smooth; no pavement texture detected."

    return True, "Valid road surface"


def has_structural_distress(img: Image.Image) -> tuple[bool, str]:
    """
    Checks if the asphalt area contains genuine structural distress (cavities or cracks).
    Masks out traffic cones and road markers so clean, repaired roads are not falsely flagged.
    """
    rgb = np.array(img.convert("RGB"), dtype=np.float32)
    gray = np.array(img.convert("L"), dtype=np.float32)
    h, w = gray.shape

    # Focus on the pavement surface (lower 80% of frame)
    roi_rgb = rgb[int(h * 0.2):, :]
    roi_gray = gray[int(h * 0.2):, :]

    # 1. Mask out high-saturation objects (orange traffic cones, bollards, signs)
    color_delta = (
        np.abs(roi_rgb[:, :, 0] - roi_rgb[:, :, 1]) +
        np.abs(roi_rgb[:, :, 1] - roi_rgb[:, :, 2]) +
        np.abs(roi_rgb[:, :, 2] - roi_rgb[:, :, 0])
    ) / 3.0
    pavement_mask = color_delta < 25.0  # True only for neutral asphalt/concrete

    if np.sum(pavement_mask) < (roi_gray.size * 0.25):
        return True, "Sufficient pavement area not isolated; skipping filter."

    asphalt_pixels = roi_gray[pavement_mask]
    asphalt_mean = float(np.mean(asphalt_pixels))

    # 2. Dark Cavity / Void Check (Potholes produce deep shadow regions)
    deep_void_ratio = float(np.sum(asphalt_pixels < (asphalt_mean - 42.0)) / asphalt_pixels.size)

    # 3. Fracture Edge Gradient Check on pavement area
    gx = np.diff(roi_gray, axis=1)
    gy = np.diff(roi_gray, axis=0)
    grad = np.abs(gx[:-1, :]) + np.abs(gy[:, :-1])
    grad_mask = pavement_mask[:-1, :-1]

    pavement_grads = grad[grad_mask]
    crack_edge_ratio = float(np.sum(pavement_grads > 38.0) / pavement_grads.size)

    # If the asphalt contains neither cavity shadows nor fracture crack edges
    if deep_void_ratio < 0.015 and crack_edge_ratio < 0.012:
        return False, "Pavement surface appears smooth or repaired. No cavity or crack distress detected."

    return True, "Distress verified"


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
    """Receives image file, validates surface and distress features, runs inference."""
    if session is None:
        raise HTTPException(status_code=503, detail="AI Model session is not initialized.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        raw_image = Image.open(io.BytesIO(contents))

        # ------------------------------------------------------------------
        # GUARD 1: Physical Surface Sanity Check (Rejects paper, icons, darkness)
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
        image_resized = raw_image.convert("RGB").resize((224, 224))
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
        predicted_severity = SEVERITY_LEVELS[max_idx]

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

        # ------------------------------------------------------------------
        # GUARD 4: Distress Verification (Prevents clean asphalt + cones from triggering High)
        # ------------------------------------------------------------------
        if predicted_severity in ["Medium", "High"]:
            has_distress, distress_reason = has_structural_distress(raw_image)
            if not has_distress:
                print(f">>> [REPAIRED/INTACT DETECTED] {distress_reason}")
                return {
                    "severity": "Unassessed",
                    "confidence": 0.0,
                    "status": "NO_DISTRESS_DETECTED",
                    "message": distress_reason
                }

        return {
            "severity": predicted_severity,
            "confidence": round(top_confidence * 100.0, 1),
            "status": "CONFIRMED"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")