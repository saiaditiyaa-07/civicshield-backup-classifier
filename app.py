
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import requests
from io import BytesIO
from PIL import Image
import numpy as np

app = FastAPI(title="CivicShield Backup Classifier")

MODEL_PATH = "civicshield_mobilenetv3.keras"

model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = [
    "drainage",
    "garbage",
    "not_applicable",
    "road"
]


class ImageRequest(BaseModel):
    image_url: str


@app.get("/")
def root():
    return {
        "status": "CivicShield Backup Classifier is running"
    }


@app.post("/predict")
def predict(request: ImageRequest):

    try:
        response = requests.get(
            request.image_url,
            timeout=10
        )
        response.raise_for_status()

        image = Image.open(
            BytesIO(response.content)
        ).convert("RGB")

        image = image.resize((224, 224))

        image_array = np.array(
            image
        ).astype(np.float32)

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        probabilities = model.predict(
            image_array,
            verbose=0
        )[0]

        predicted_index = int(
            np.argmax(probabilities)
        )

        category = CLASS_NAMES[predicted_index]

        confidence = float(
            probabilities[predicted_index]
        )

        # Severity is currently a heuristic.
        # The training dataset did not contain severity labels.
        severity = (
            0.0
            if category == "not_applicable"
            else round(confidence, 4)
        )

        notes = {
            "garbage":
                "The image appears to show garbage or litter.",

            "drainage":
                "The image appears to show a drainage-related issue.",

            "road":
                "The image appears to show road damage.",

            "not_applicable":
                "The image does not appear to show a supported civic issue."
        }

        return {
            "category": category,
            "confidence": round(confidence, 4),
            "severity": severity,
            "note": notes[category]
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Unable to process image: {str(e)}"
        )
