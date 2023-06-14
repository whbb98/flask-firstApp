from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import mysql.connector
from flask_cors import CORS


app = Flask(__name__)
CORS(app, origins='http://doctoraicollab.test')

# Define the necessary constants
IMAGE_SHAPE = (224, 224)

xray_class_names = {
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


model = tf.keras.models.load_model("mobilenet_model.h5")

@app.route('/', methods=['GET'])
def welcome():
    response = "<h1 style='color:#04aa6d'>welcome to doctor ai collab web api</h1>"
    return response

@app.route('/info', methods=['GET'])
def modelinfo():
    response = "<h1 style='color:#04aa6d'>Supported Classes by Our API:</h1>"
    response += "<pre>"
    for val in xray_class_names.values():
        response += val + "\n"
    return response

@app.route('/predict', methods=['GET'])
def predict():
    image_id = request.args.get('id')
    if image_id is None or not image_id.isdigit() or int(image_id) <= 0:
        return "Bad Request!"

    # Connect to the MySQL database
    try:
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="doctoraicollab"
        )
    except:
        return "Couldn't Connect to Database!"

    cursor = db.cursor()

    # Fetching the image from the database
    query = "SELECT image_binary FROM blog_images WHERE id = %s"
    cursor.execute(query, (image_id,))
    result = cursor.fetchone()

    if result is None:
        return jsonify({'error': 'Image not found'}), 404

    # binary ---> numpy array
    image_data = np.frombuffer(result[0], np.uint8)

    # Decode the image to cv2 obj
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    # Resizing the image
    resized_image = cv2.resize(image, IMAGE_SHAPE)
    processed_image = np.expand_dims(resized_image, axis=0)

    # generate predictions
    predictions = model.predict(processed_image)

    # results = {xray_class_names[class_index]: float(predictions[0][class_index]) for class_index in range(len(xray_class_names))}
    results = {
        xray_class_names[class_index]: "{:.2f}".format(float(predictions[0][class_index]) * 100)
        for class_index in range(len(xray_class_names))
    }

    return jsonify(results)

if __name__ == '__main__':
    app.run(host='api.doctoraicollab.test', port=5000)
