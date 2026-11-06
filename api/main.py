from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import joblib
import logging
import time

from prometheus_fastapi_instrumentator import Instrumentator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

logger = logging.getLogger(__name__)

MODEL_NAME = "random_forest"
MODEL_VERSION = "1.0.0"

app = FastAPI(
    title="Churn Prediction API",
    version="1.0.0"
)

model = joblib.load("models/model.pkl")
logger.info(
    "Model loaded: model=%s model_version=%s",
    MODEL_NAME, MODEL_VERSION
)


class ClientData(BaseModel):
    age: int
    anciennete: int
    revenu: float
    nb_reclamations: int
    montant_mensuel: float
    retards_paiement: int


@app.get("/")
def root():
    return {
        "application": "Churn Prediction API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "UP"
    }


@app.get("/model/info")
def model_info():
    return {
        "model": "Random Forest",
        "version": "1.0.0"
    }


@app.post("/predict")
def predict(data: ClientData, request: Request):

    start = time.time()
    client = request.client.host if request.client else "unknown"

    try:
        values = [[
            data.age,
            data.anciennete,
            data.revenu,
            data.nb_reclamations,
            data.montant_mensuel,
            data.retards_paiement
        ]]

        prediction = int(model.predict(values)[0])
        probability = float(model.predict_proba(values)[0][1])

    except Exception as exc:
        duration_ms = (time.time() - start) * 1000
        logger.error(
            "operation=predict client=%s model=%s model_version=%s "
            "status=error duration_ms=%.2f error=%s",
            client, MODEL_NAME, MODEL_VERSION, duration_ms, exc,
            exc_info=True
        )
        raise HTTPException(status_code=500, detail="Prediction error")

    duration_ms = (time.time() - start) * 1000
    label = "CHURN" if prediction == 1 else "NO_CHURN"

    logger.info(
        "operation=predict client=%s model=%s model_version=%s "
        "status=success prediction=%s probability=%.4f duration_ms=%.2f",
        client, MODEL_NAME, MODEL_VERSION, label, probability, duration_ms
    )

    return {
        "prediction": label,
        "probability": round(probability, 4),
        "model_version": MODEL_VERSION,
        "duration_ms": round(duration_ms, 2)
    }


Instrumentator().instrument(app).expose(app)
