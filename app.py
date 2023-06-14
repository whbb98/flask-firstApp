from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import mysql.connector

app = Flask(__name__)

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


@app.route('/predict', methods=['GET'])
def predict():
    image_id = request.args.get('id')

    if image_id is None or not image_id.isdigit() or int(image_id) <= 0:
        return "Bad Request!"

    # return image_id

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

    # Fetch the image from the database based on the ID
    query = "SELECT image_binary FROM blog_images WHERE id = %s"
    cursor.execute(query, (image_id,))
    result = cursor.fetchone()

    # Check if the image exists
    if result is None:
        return jsonify({'error': 'Image not found'}), 404

    # Convert the binary data to numpy array
    image_data = np.frombuffer(result[0], np.uint8)

    # Decode the image
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    # Preprocess the image
    resized_image = cv2.resize(image, IMAGE_SHAPE)
    processed_image = np.expand_dims(resized_image, axis=0)

    # Make predictions using the loaded model
    predictions = model.predict(processed_image)

    # Convert the predictions to a dictionary with class names and their ratios
    results = {xray_class_names[class_index]: float(predictions[0][class_index]) for class_index in range(len(xray_class_names))}

    # Return the results as a JSON response
    return jsonify(results)

if __name__ == '__main__':
    app.run()
