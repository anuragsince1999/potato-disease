from fastapi import FastAPI, UploadFile, File
import uvicorn
from io import BytesIO
from PIL import Image
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("../saved_models/1")
# prod_model = tf.keras.models.load_model("../saved_models/1")
# beta_model = tf.keras.models.load_model("../saved_models/2")
CLASS_NAMES =["Apple scab Medicine: Infuse Spray", "Cedar apple rust Medicine: Neem oil", "Apple healthy",
              "Cherry healthy","Cherry Powdery mildew Medicine: Luna Sensation",
              "Corn Cercospora leaf spot Medicine:Mancozeb Spray","Corn Common rust Medicine:Hexaconazole",
              "Corn healthy","Corn Northern Leaf Blight Medicine:Propiconazole","Grape Black rot Medicine:Orchard Spray",
              "Grape Esca Medicine:Trichoderma","Grape healthy","Grape Leaf blight Medicine:Erwinia herbicola",
              "Orange (Citrus_greening) Medicine:PestOil","Peach Bacterial spot Medicine:Copper Fungicide","Peach healthy",
              "Pepper Bacterial spot Medicine:Copper Fungicide","Pepper healthy","Potato Early blight Medicine:Buonos",
              "Potato healthy","Potato Late blight Medicine:Acrobat","Squash Powdery mildew Medicine:Daconil",
              "Strawberry healthy","Strawberry Leaf scorch Medicine:RidomilGold","Tomato Bacterial spot Medicine:Chlorothalonil",
              "Tomato Early blight Medicine:Antracol","Tomato healthy","Tomato Late blight Medicine:Cuprofix","Tomato Late blight Medicine:Cuprofix",
              "Tomato Septoria leaf spot Medicine:Chlorothalonil"]
@app.get("/ping")
async def ping():
    return "Hello My frnd"

def read_file_as_image(data) -> np.ndarray:
    image=np.array(Image.open(BytesIO(data)))
    return image
@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch=np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    print(predicted_class, confidence)
    return {"class": predicted_class,
            "confidence": float(confidence) }




if __name__ == "__main__":
    # uvicorn.run(app, host="127.0.0.1", port=5049)
    # uvicorn.exe test: app --host = 127.0.0.1 --port = 5049
    uvicorn.run(app, host='localhost', port=8082 )