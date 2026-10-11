import io
import numpy as np
import onnxruntime as ort
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException

app = FastAPI(title="RoadWise AI Service", version="1.0.0")

# 1. Load ONNX Model on Startup
MODEL_PATH = "rdd2022_d4_resnet38.onnx"
try:
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    print(f">>> [AI READY] Model '{MODEL_PATH}' loaded successfully.")
except Exception as e:
    print(f">>> [AI ERROR] Failed to load model: {e}")
    session = None

SEVERITY_LEVELS = ["Low", "Medium", "High"]

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
    """Receives image file, runs ResNet-38 inference, returns severity + confidence."""
    if session is None:
        raise HTTPException(status_code=503, detail="AI Model session is not initialized.")

    # Read uploaded file bytes
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        # Load and convert image to RGB, resize to ResNet input dimensions (224x224)
        image = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))

        # Preprocessing: Normalize [0.0, 1.0] and rearrange to (1, 3, 224, 224)
        img_np = np.array(image, dtype=np.float32) / 255.0
        img_np = np.transpose(img_np, (2, 0, 1))  # Convert HWC to CHW
        img_np = np.expand_dims(img_np, axis=0)   # Add batch dimension -> (1, 3, 224, 224)

        # Execute native ONNX inference
        outputs = session.run(None, {input_name: img_np})
        logits = outputs[0][0]

        # Stable Softmax computation for true probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        max_idx = int(np.argmax(probs))

        return {
            "severity": SEVERITY_LEVELS[max_idx],
            "confidence": round(float(probs[max_idx]) * 100.0, 1)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")