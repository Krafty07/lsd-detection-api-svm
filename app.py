from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
from PIL import Image
import tensorflow as tf
import joblib
import io

# ---------------- APP ----------------
app = FastAPI(title="LSD Detection API")

# ---------------- LOAD MODELS ----------------
print("Loading models...")

feature_extractor = tf.keras.models.load_model(
    "models/efficientnet_feature_extractor.h5"
)

svm_model = joblib.load("models/svm_lsd_model.pkl")

IMG_SIZE = (224, 224)

print("Models loaded successfully!")

# ---------------- PREPROCESS FUNCTION ----------------
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# ---------------- ROUTES ----------------

@app.get("/")
def home():
    return {"message": "LSD Detection API Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        # ✅ Allow only images
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Only image files (jpg, jpeg, png) are allowed"
            )

        image_bytes = await file.read()

        # preprocess image
        img = preprocess_image(image_bytes)

        # feature extraction
        features = feature_extractor.predict(img)

        # probability prediction
        proba = svm_model.predict_proba(features)[0]

        confidence = float(max(proba))
        prediction = int(np.argmax(proba))

        label = "Lumpy" if prediction == 1 else "Healthy"

        return {
            "success": True,
            "data": {
                "prediction": label,
                "confidence": round(confidence * 100, 2)
            }
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        print(f"Error during prediction: {e}")

        raise HTTPException(
            status_code=500,
            detail="Invalid or corrupted image"
        )