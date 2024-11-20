from flask import Flask, request, render_template, send_file
import os
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)

# Load YOLOv8 model
model = YOLO('model/best.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Run inference
        results = model(file_path)
        results_img = results[0].plot()  # Render results on the image
        result_img_path = os.path.join(RESULT_FOLDER, f'result_{file.filename}')
        Image.fromarray(results_img).save(result_img_path)

        return send_file(result_img_path, mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True)
