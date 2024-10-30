from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
import os


app = FastAPI()

model_dir = './saved_models/1.01'
model_path = os.path.join(model_dir, 'model.keras') # model directory


origins = [
    "http://localhost",
    "http://localhost:3000", # handling requests from different hosts/ CORS
]

# needed to enable support for backend and frontend with different hosts/ports 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model(model_path) # load model

CLASS_NAMES = ["Bacterial spot", "Target spot", "Mosaic virus", "YellowLeaf curl virus", "Healthy"] # labels for predictions


def convert_image_to_numpy_array(image) -> np.array: 
    image = Image.open(BytesIO(image)) # Open the image from bytes
    rgb_image = image.convert('RGB') # Convert to RGB to comply with the format (x,x,3-RGB Channels)
    resized_image = rgb_image.resize((256, 256)) # resize the image
    return np.array(resized_image)
 
@app.post("/predict") # get predictions endpoint
async def predict(file: UploadFile = File(...)): # requiring files only as a parameter
    image = convert_image_to_numpy_array(await file.read()) # reads the uploaded file and converts it into a numpy array

    img_batch= np.expand_dims(image, axis=0) # images were read in batches in the model, this makes a single image into 2dim - batch
    
    print('shape', img_batch.shape) # 
    
    prediction = model.predict(img_batch)
   
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])] # images were fed as a batch so we take the first batch, [0]
     
    confidence = np.max(prediction[0]) # returns the highest value in the array 
    
    return {
        'class': predicted_class, # class
        'confidence': float(confidence) # probability score 
    }
    
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000, log_level="info") 
