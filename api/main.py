'''from fastapi import FastAPI, File, UploadFile, requests
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
app = FastAPI()


MODEL=tf.keras.models.load_model('../saved_model/1')
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

#endpoint = "http://localhost:8501/v1/models/saved_model:predict"

@app.get("/hello")
async def hello():
    return "hello france"


def read_file_as_image(data) -> np.ndarray: #type hint
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    prediction=MODEL.predict(img_batch)
    '''
'''
    json_data={
    "instances":img_batch.tolist()

    }
    response=requests.post(endpoint,json=json_data)
    response.json()["predictions"][0]

'''
'''
    predicted_class =CLASS_NAMES[np.argmax(prediction[0])]
    confidence=float(np.max(prediction[0]))
    return {
        'class':predicted_class,
        "confidence":round(confidence*100 , 2)
    }
'''

from fastapi import FastAPI, File, UploadFile
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()


MODEL = tf.keras.models.load_model("../saved_model/1")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }