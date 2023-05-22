from fastapi import FastAPI, UploadFile, File
import uvicorn
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
import requests

app = FastAPI()

# MODEL = tf.keras.models.load_model("../saved_models/1")
endpoint = "http://localhost:8050/v1/models/potatoes_model:predict"
CLASS_NAMES =["Apple___Apple_scab", "Apple___Cedar_apple_rust", "Apple___healthy",
              "Cherry_(including_sour)___healthy","Cherry_(including_sour)___Powdery_mildew",
              "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot","Corn_(maize)___Common_rust_",
              "Corn_(maize)___healthy","Corn_(maize)___Northern_Leaf_Blight","Grape___Black_rot",
              "Grape___Esca_(Black_Measles)","Grape___healthy","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
              "Orange___Haunglongbing_(Citrus_greening)","Peach___Bacterial_spot","Peach___healthy",
              "Pepper,_bell___Bacterial_spot","Pepper,_bell___healthy","Potato___Early_blight",
              "Potato___healthy","Potato___Late_blight","Squash___Powdery_mildew",
              "Strawberry___healthy","Strawberry___Leaf_scorch","Tomato___Bacterial_spot",
              "Tomato___Early_blight","Tomato___healthy","Tomato___Late_blight","Tomato___Late_blight",
              "Tomato___Septoria_leaf_spot"]
@app.get("/ping")
async def ping():
    return "Hello My frnd"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    json_data = {
        "instances": img_batch.tolist()
    }
    response = requests.post(endpoint, json=json_data)
    # pass
    prediction = np.array(response.json()["predictions"][0])
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)
    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }
if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8082)
