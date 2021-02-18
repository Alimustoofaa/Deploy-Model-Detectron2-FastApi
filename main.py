'''
API prediction container number v1.0 with detectron2

@Author     : Ali Mustofa HALOTEC
@Created on : 16 Feb 2021
'''

import io
import cv2
import base64
import numpy as np
from PIL import Image
from app.crop import crop_img
from fastapi import FastAPI, File
from app.prediction import Segmentor
from fastapi.responses import JSONResponse

model = Segmentor()

app = FastAPI(title='Prediction Container Number',
              description='''Get the image decomposed into instances using the neural
              network model Faster RCNN implemented in detectron2 library.''',
              version='1.0')

def encode_image(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer)

# Router prediction image container number
@app.post('/prediction')
def prediction_container_number(file: bytes = File(...)):

    image = Image.open(io.BytesIO(file)).convert("RGB")
    image = np.array(image)
    image = image[:,:,::-1].copy()
    output = model.predict(image)

    if len(output["instances"].to("cpu").pred_boxes) != 0:
        image =  crop_img(outputs=output, im=image)
    
    encoded_image = encode_image(image)
    code = 200 if len(output["instances"].to("cpu").pred_boxes) != 0 else 404
    message = 'Got prediction container number' if code == 200 else 'No predictiron container number'
    
    return {
        'status': 'success',
        'code': code,
        'message': message,
        'results': encoded_image 
    }