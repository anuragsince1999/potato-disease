from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

BUCKET_NAME='Archies-tf-models'
class_name=["Apple___Apple_scab", "Apple___Cedar_apple_rust", "Apple___healthy",
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
model=None

def download_cloc(bucket_name,source_blob_name,destination_blob_name)
    storage_client=storage.Client()
    bucket=storage_client.get_bucket(bucket_name)
    blob=bucket.blob(source_blob_name)

    blob.download_to_filename(destination_blob_name)

    def predict(request):
        global model
        if model is None:
            download_blob(
                BUCKET_NAME,
                "models/"
                "/tmp/potatoes.h5"
        )
        model=tf.keras.load_model("/tmp/pt")

        image=image/255
        img_array=tf.expand_dims(img,0)

        predictions=model.predict(img_array)
        print("Predictions:",predictions)

        predicted_class=class_name[np.argmax(predictions[0])]
        confidence=round(100*(np.max(predictions[0])),2)

        return {"class":predicted_class,"confidence":confidence}
