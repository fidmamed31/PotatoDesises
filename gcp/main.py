from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np
import tempfile

model = None
interpreter = None
input_index = None
output_index = None

class_names = ["Early Blight", "Late Blight", "Healthy"]

BUCKET_NAME = "fidmy-potatodesisease_classifier" # Here you need to put the name of your GCP bucket


def download_blob(bucket_name, source_blob_name, destination_file_name):
   
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")


def predict(request):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = f"{temp_dir}/saved_model2"
    global model
    if model is None:
        download_blob(
            BUCKET_NAME,
            "1/",
            temp_file,
        )
        model = tf.keras.models.load_model(temp_file)

    image = request.files["file"]

    image = np.array(
        Image.open(image).convert("RGB").resize((256, 256)) # image resizing
    )

    image = image/255 
    img_array = tf.expand_dims(img, 0)
    predictions = model.predict(img_array)

    print("Predictions:",predictions)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)

    return {"class": predicted_class, "confidence": confidence}