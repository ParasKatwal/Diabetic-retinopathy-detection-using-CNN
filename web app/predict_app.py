import base64
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential 
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask
import json
import h5py
from waitress import serve
import tensorflow as tf 

app = Flask(__name__)

def fix_layer0(filename, batch_input_shape, dtype):
    with h5py.File(filename, 'r+') as f:
        model_config = json.loads(f.attrs['model_config'].decode('utf-8'))
        layer0 = model_config['config']['layers'][0]['config']
        layer0['batch_input_shape'] = batch_input_shape
        layer0['dtype'] = dtype
        f.attrs['model_config'] = json.dumps(model_config).encode('utf-8')

def get_model():
    global model,graph
    model = load_model('diabetic_retinopathy.h5')
    print(" * Model Loaded!!")
    graph = tf.get_default_graph()
    
def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    
    return image

print(" * Loading keras model.....")
fix_layer0('diabetic_retinopathy.h5', [None, 224, 224,3], 'float32')
get_model()

@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    with graph.as_default(): 
        image = Image.open(io.BytesIO(decoded))
        processed_image = preprocess_image(image, target_size=(224, 224))
        prediction = model.predict(processed_image).tolist()
        response = {
            'prediction': {
                'dr': prediction[0][0],
                'nodr': prediction[0][1]
            }
        }
        return jsonify(response)

# serve(app, host='0.0.0.0', port=8080)