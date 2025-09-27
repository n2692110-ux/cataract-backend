from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import uvicorn
import os

# -------------------------------
# Model path (adjust if needed)
# -------------------------------
MODEL_PATH = "../final_model.keras"  # points to your trained model in project root
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

# Load model
print("🔹 Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# -------------------------------
# FastAPI app setup
# -------------------------------
app = FastAPI(
    title="Cataract Detection API",
    description="Upload an eye image and get prediction: Normal or Cataract",
    version="1.0"
)

# Allow frontend / other clients to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for testing; in production restrict origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Prediction endpoint
# -------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    contents = await file.read()
    temp_path = "temp.jpg"
    with open(temp_path, "wb") as f:
        f.write(contents)

    try:
        # Preprocess image
        img = image.load_img(temp_path, target_size=(224, 224))  # match training size
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        pred = model.predict(img_array)[0][0]
        label = "Cataract" if pred > 0.5 else "Normal"

        return {"prediction": label}

    except Exception as e:
        return {"error": str(e)}

    finally:
        # Remove temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

# -------------------------------
# Run server
# -------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")
