from flask import Flask, request, render_template
import numpy as np
import pickle
import cv2

app = Flask(__name__)

# Load the model from the pickle file
model_path = 'rice_leaf_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

class_names = ['bacterial leaf blight', 'brown spot', 'leaf smut']

def preprocess_image(image):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', error='No file has been uploaded')
    
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return render_template('index.html', error='Invalid file type. Please upload an image file.')

    # Process the file
    image = preprocess_image(file)
    prediction = model.predict(image)
    predicted_class = class_names[np.argmax(prediction)]
    return render_template('index.html', prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
