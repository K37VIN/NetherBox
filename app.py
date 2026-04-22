import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from src.logger import logger
from src.pipline.prediction_pipeline import PredictionPipeline
from src.pipline.training_pipeline import TrainingPipeline
from src.utils.main_utils import load_json

app = FastAPI(title="NetherBox API", version="1.0.0")

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://netherbox-fe.vercel.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Directories ────────────────────────────────────────────────────────────────
TEMPLATES_DIR = Path("templates")
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)


# ── Page routes ────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
async def home():
    path = TEMPLATES_DIR / "index.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="index.html not found in templates/")
    return HTMLResponse(content=path.read_text(encoding="utf-8"))


@app.get("/about", response_class=HTMLResponse)
@app.get("/about.html", response_class=HTMLResponse)
async def about():
    path = TEMPLATES_DIR / "about.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="about.html not found in templates/")
    return HTMLResponse(content=path.read_text(encoding="utf-8"))


@app.get("/app", response_class=HTMLResponse)
@app.get("/app.html", response_class=HTMLResponse)
async def app_page():
    path = TEMPLATES_DIR / "app.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="app.html not found in templates/")
    return HTMLResponse(content=path.read_text(encoding="utf-8"))


# ── API routes ─────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    return {"status": "ok", "service": "NetherBox API"}


@app.post("/upload-dataset")
async def upload_dataset(
    file: UploadFile = File(...),
    target_column: str = None,
    problem_type: str = None,
):
    """
    Accepts a CSV upload, saves it to /uploads/, returns the saved path.
    The frontend uses this path to trigger /train.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported")

    save_path = UPLOADS_DIR / file.filename
    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    logger.info(f"Dataset uploaded: {save_path} ({len(contents)} bytes)")
    return {
        "saved_path": str(save_path),
        "filename": file.filename,
        "size_bytes": len(contents),
    }


@app.post("/train")
async def train(request: Request):
    """
    Trigger the AutoML training pipeline.

    JSON body:
    {
        "dataset_path": "uploads/your_file.csv",
        "target_column": "target",
        "problem_type": "classification"   // optional
    }
    """
    body = await request.json()
    dataset_path = body.get("dataset_path")
    target_column = body.get("target_column")
    problem_type = body.get("problem_type", None)

    if not dataset_path or not target_column:
        raise HTTPException(status_code=400, detail="dataset_path and target_column are required")

    if not os.path.exists(dataset_path):
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {dataset_path}. Upload via /upload-dataset first.",
        )

    try:
        pipeline = TrainingPipeline(
            dataset_path=dataset_path,
            target_column=target_column,
            problem_type=problem_type,
        )
        result = pipeline.run()
        logger.info(f"Training complete: {result['best_model']} score={result['best_score']}")
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(None)):
    """
    Run predictions.
    - JSON body:   single row as dict  {"col1": val, "col2": val, ...}
    - File upload: multipart/form-data field "file" (CSV)
    """
    predictor = PredictionPipeline()

    try:
        if file is not None:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                contents = await file.read()
                tmp.write(contents)
                tmp.flush()
                predictions = predictor.predict(tmp.name)
            os.unlink(tmp.name)
        else:
            data = await request.json()
            predictions = predictor.predict(data)

        return {"predictions": predictions}

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics():
    """Return the latest training run metrics."""
    metrics_path = "artifacts/reports/metrics.json"
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="No metrics found. Run /train first.")
    return load_json(metrics_path)


@app.get("/download-model")
def download_model():
    """Download the trained AutoML estimator .pkl file."""
    model_path = "artifacts/models/automl_estimator.pkl"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="No trained model found. Run /train first.")
    return FileResponse(
        path=model_path,
        media_type="application/octet-stream",
        filename="automl_estimator.pkl",
    )


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    # Hugging Face Spaces requires port 7860
    # Locally defaults to 8000 unless PORT env var is set
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting NetherBox API on port {port}")
    logger.info(f"  Home  → http://localhost:{port}/")
    logger.info(f"  About → http://localhost:{port}/about")
    logger.info(f"  App   → http://localhost:{port}/app")
    logger.info(f"  Docs  → http://localhost:{port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=port)
