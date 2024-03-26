from flask import Flask, render_template, jsonify, request
from tensorflow.keras.models import load_model
import cv2
import numpy as np


app = Flask(__name__)


model = load_model('models/model_inceptionV3')
print("Model is loaded")


@app.route('/')
def index():
    return render_template("VegVision.html") 

@app.route("/prediction", methods=["POST"])
def prediction():
    img = request.files['img']
    img.save("img.jpg")

    image = cv2.imread("img.jpg")
    resize = cv2.resize(image, (224,224))
    predicted_probs = model.predict(np.expand_dims(resize/255, 0), verbose=False)
    predicted_class = np.argmax(predicted_probs)
    
    categories = ["Bell Pepper","Broccoli","Carrot","Lettuce","Mushroom","Onion","Peas","Tomato"]
    
    return render_template("prediction.html",data=categories[predicted_class])


if __name__ == '__main__':
    app.run(debug=True)
