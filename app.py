from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from flask_cors import CORS
import base64

app = Flask(__name__)
APP_URL = 'localhost'
CORS(app, origins=APP_URL)

IMAGE_SHAPE = (224, 224)
model_v1 = tf.keras.models.load_model("./models/mobilenet_model.h5")
# model_v2 = tf.keras.models.load_model("./models/NIH_Seresnet152_model.h5")

model_v1_classes = {
    0: 'Atelectasis',
    1: 'Cardiomegaly',
    2: 'Effusion',
    3: 'Infiltration',
    4: 'Mass',
    5: 'Nodule',
    6: 'Pneumonia',
    7: 'Pneumothorax',
    8: 'Consolidation',
    9: 'Edema',
    10: 'Emphysema',
    11: 'Fibrosis',
    12: 'Pleural_Thickening',
    13: 'Hernia'
}
model_v2_classes = [
    'Cardiomegaly', 'Hernia', 'Infiltration', 'Nodule', 'Emphysema', 'Effusion',
    'Atelectasis', 'Pleural_Thickening', 'Pneumothorax', 'Mass', 'Fibrosis',
    'Consolidation', 'Edema', 'Pneumonia'
]


def decode_image64(image64):
    try:
        if image64 is None:
            raise (Exception('Image64 is Null'))
        prefix, image64 = image64.split(',', 1)
        image_type = prefix.split(';')[0].split(':')[1]
        image64 += '=' * ((4 - len(image64) % 4) % 4)
        image_data = base64.b64decode(image64)
        return image_data
    except Exception as e:
        return jsonify({'error': 'decoding base64 image failed'}), 400


@app.route('/', methods=['GET'])
def welcome():
    response = "<h1 style='color:#04aa6d'>welcome to doctor ai collab web api</h1>"
    return response


@app.route('/info', methods=['GET'])
def info():
    return {
        "model_v1": {
            "classes": "http://localhost:5000/model_v1",
            "predict": "http://localhost:5000/model_v1/predict",
        },
        # "model_v2": {
        #     "classes": "http://localhost:5000/model_v2",
        #     "predict": "http://localhost:5000/model_v2/predict",
        # },
    }


@app.route('/model_v1', methods=['GET'])
def model_v1_info():
    response = "<h1 style='color:#04aa6d'>Supported classes by model-v1:</h1>"
    response += "<ul>"
    for val in model_v1_classes.values():
        response += f"<li>{val}</li>"
    response += "</ul>"
    return response


@app.route('/model_v1/predict', methods=['POST'])
def model_v1_predict():
    try:
        data = request.get_json()
        image64 = data.get('image64')
        if image64 is None:
            return jsonify({'error': 'Base64 image not provided'}), 400
    except Exception as e:
        return jsonify({'error': 'Bad request format'}), 400

    try:
        image_data = decode_image64(image64)
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({'error': 'Image Processing Failed!'}), 400

    resized_image = cv2.resize(image, IMAGE_SHAPE)
    processed_image = np.expand_dims(resized_image, axis=0)
    predictions = model_v1.predict(processed_image)
    results = {
        model_v1_classes[class_index]: "{:.2f}".format(float(predictions[0][class_index]) * 100)
        for class_index in range(len(model_v1_classes))
    }
    return jsonify(results)


@app.route('/model_v2', methods=['GET'])
def model_v2_info():
    response = "<h1 style='color:#04aa6d'>Supported classes by model-v1:</h1>"
    response += "<ul>"
    for val in model_v2_classes:
        response += f"<li>{val}</li>"
    response += "</ul>"
    return response


@app.route('/model_v2/predict', methods=['POST'])
def model_v2_predict():
    img_size = 600
    try:
        data = request.get_json()
        image64 = data.get('image64')
        if image64 is None:
            return jsonify({'error': 'Base64 image not provided'}), 400
    except Exception as e:
        return jsonify({'error': 'Bad request format'}), 400

    try:
        image_data = decode_image64(image64)
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({'error': 'Image Processing Failed!'}), 400

    resized_image = cv2.resize(image, (img_size, img_size))
    processed_image = np.expand_dims(resized_image, axis=0)

    # Use the loaded model for prediction
    predictions = model_v2.predict(processed_image)

    results = {
        model_v2_classes[class_index]: "{:.2f}".format(float(predictions[0][class_index]) * 100)
        for class_index in range(len(model_v2_classes))
    }

    return jsonify(results)


if __name__ == '__main__':
    app.run(host=APP_URL, port=5000)
