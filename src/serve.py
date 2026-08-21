from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

app = FastAPI()

BUCKET_NAME = os.environ.get("CLOUD_BUCKET") or os.environ.get("S3_BUCKET") or os.environ.get("GCS_BUCKET", "my-mlops-bucket")
MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")


def download_model():
    """
    Tai file model.pkl tu Cloud Storage (AWS S3 hoac GCS) ve may khi server khoi dong.
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Kiem tra neu su dung AWS S3 (boto3)
    try:
        import boto3
        s3 = boto3.client("s3")
        s3.download_file(BUCKET_NAME, MODEL_KEY, MODEL_PATH)
        print(f"Model da duoc tai xuong tu AWS S3 bucket: {BUCKET_NAME}")
        return
    except Exception as s3_err:
        print(f"Khong the tai tu S3 ({s3_err}), thu voi GCS...")

    # Fallback voi GCP GCS
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(MODEL_KEY)
        blob.download_to_filename(MODEL_PATH)
        print(f"Model da duoc tai xuong tu GCS bucket: {BUCKET_NAME}")
        return
    except Exception as gcs_err:
        print(f"Khong the tai tu GCS ({gcs_err})")

    # Neu da ton tai model local
    if os.path.exists("models/model.pkl") and not os.path.exists(MODEL_PATH):
        import shutil
        shutil.copy("models/model.pkl", MODEL_PATH)
        print("Su dung local models/model.pkl")


# Goi ham download khi server khoi dong
download_model()
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
elif os.path.exists("models/model.pkl"):
    model = joblib.load("models/model.pkl")
else:
    model = None


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    """
    Endpoint kiem tra suc khoe server.
    GitHub Actions goi endpoint nay sau khi deploy de xac nhan server dang chay.
    """
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint suy luan chinh.

    Dau vao : JSON {"features": [f1, f2, ..., f12]}
    Dau ra  : JSON {"prediction": <0|1|2>, "label": <"thap"|"trung_binh"|"cao">}

    Thu tu 12 dac trung (khop voi thu tu trong FEATURE_NAMES cua test):
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
        pH, sulphates, alcohol, wine_type
    """
    if len(req.features) != 12:
        raise HTTPException(
            status_code=400,
            detail="Expected 12 features (wine quality)",
        )

    if model is None:
        raise HTTPException(status_code=500, detail="Model chua duoc tai thanh cong")

    pred = int(model.predict([req.features])[0])
    label_map = {0: "thap", 1: "trung_binh", 2: "cao"}
    label = label_map.get(pred, "unknown")

    return {"prediction": pred, "label": label}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
