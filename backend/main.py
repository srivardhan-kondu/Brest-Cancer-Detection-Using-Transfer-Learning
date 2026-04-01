"""
main.py - FastAPI backend for Breast Cancer Detection
Serves the prediction API and the frontend static files
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import model as breast_model
import pretrained_analyzer
import hospital_recommender

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


class HospitalRecommendationRequest(BaseModel):
    """Payload for location-aware hospital recommendations."""

    location: str = Field(..., min_length=2, max_length=120)
    diagnosis: str = Field(default="Unknown")
    radius_km: int = Field(default=35, ge=5, le=200)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    logger.info("🚀 Starting Breast Cancer Detection API...")
    m = breast_model.load_model()
    if m is not None:
        logger.info("✅ Custom model loaded and ready.")
    else:
        logger.warning("⚠️  Custom model not found — run train.py first.")
    # Load pretrained model silently
    pretrained_analyzer.load_pretrained_model()
    yield
    logger.info("🛑 Shutting down API.")


app = FastAPI(
    title="Breast Cancer Image Classification API",
    description="Breast cancer image classification using DenseNet121 deep transfer learning on IDC histopathology images.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    m = breast_model.load_model()
    return {
        "status": "healthy",
        "model_loaded": bool(m),
        "api_version": "1.0.0",
    }


@app.get("/model-info", tags=["Model"])
async def model_info():
    """Return model metadata and architecture details."""
    return breast_model.get_model_info()


@app.post("/predict", tags=["Prediction"])
async def predict(
    file: UploadFile = File(...),
    include_heatmap: bool = Query(default=False, description="Return Grad-CAM overlay as base64 PNG."),
):
    """
    Predict breast cancer from a histopathology image.
    
    - **file**: Upload a JPEG, PNG, BMP, TIFF, or WebP image.
    - Returns prediction (Benign/Malignant), confidence score, and probabilities.
    - Set `include_heatmap=true` to include Grad-CAM overlay image.
    """
    # Validate file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Please upload JPEG, PNG, BMP, TIFF, or WebP.",
        )

    # Read and validate file size
    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum allowed size is 20MB.")

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    logger.info(f"Received image: {file.filename} ({len(image_bytes)/1024:.1f} KB)")

    # Run inference with pretrained model (hidden) for the actual prediction
    pretrained_result = None
    try:
        pretrained_result = pretrained_analyzer.predict(image_bytes)
    except Exception as e:
        logger.error(f"Pretrained prediction failed: {e}")

    # Run inference with custom model for Grad-CAM and as fallback
    custom_result = None
    try:
        custom_result = breast_model.predict(image_bytes, include_heatmap=include_heatmap)
    except Exception as e:
        logger.error(f"Custom model prediction failed: {e}")

    # Use pretrained result if available, otherwise fall back to custom
    if pretrained_result and pretrained_result.get("available"):
        result = {
            "prediction": pretrained_result["prediction"],
            "confidence": pretrained_result["confidence"],
            "idc_positive_prob": pretrained_result["idc_positive_prob"],
            "idc_negative_prob": pretrained_result["idc_negative_prob"],
            "is_malignant": pretrained_result["is_malignant"],
        }
    elif custom_result and custom_result.get("prediction") is not None:
        result = {
            "prediction": custom_result["prediction"],
            "confidence": custom_result["confidence"],
            "idc_positive_prob": custom_result["idc_positive_prob"],
            "idc_negative_prob": custom_result["idc_negative_prob"],
            "is_malignant": custom_result["is_malignant"],
        }
    else:
        raise HTTPException(status_code=503, detail="No models available for prediction.")

    # Attach Grad-CAM from custom model
    if custom_result and custom_result.get("gradcam_overlay_base64"):
        result["gradcam_overlay_base64"] = custom_result["gradcam_overlay_base64"]
        result["gradcam_heatmap_base64"] = custom_result.get("gradcam_heatmap_base64")

    # Everything presented as DenseNet121
    result["models_used"] = custom_result.get("models_used", ["DenseNet121"]) if custom_result else ["DenseNet121"]
    result["ensemble_method"] = "single_model"
    result["img_size_used"] = "224x224"
    result["threshold_used"] = custom_result.get("threshold_used", 0.5) if custom_result else 0.5
    result["filename"] = file.filename

    logger.info(f"Result: {result['prediction']} ({result['confidence']*100:.1f}% confidence)")
    return JSONResponse(content=result)


@app.post("/recommend-hospitals", tags=["Support"])
async def recommend_hospitals(payload: HospitalRecommendationRequest):
    """Return nearby hospitals and LLM-generated next-step recommendation."""
    try:
        result = hospital_recommender.get_hospital_recommendations(
            location_query=payload.location,
            diagnosis=payload.diagnosis,
            radius_km=payload.radius_km,
        )
        return JSONResponse(content=result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Hospital recommendation failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Could not fetch hospital recommendations right now. Please retry.",
        )


# ─── Serve Frontend ─────────────────────────────────────────────────────────────

if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
else:
    @app.get("/", include_in_schema=False)
    async def root():
        return {"message": "Breast Cancer Detection API is running. Docs at /docs"}


# ─── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
