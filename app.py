from flask import Flask, render_template, redirect, request, jsonify, Response
from io import BytesIO
import cv2
from PIL import Image

import os
from ultralytics import YOLO

app = Flask(__name__)
#DAWNedVision
MODEL_PATH = 'yolov8m.pt'
model = YOLO(MODEL_PATH)


UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = "static/results"
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['RESULTS_FOLDER'] = RESULTS_FOLDER


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

@app.route('/')
def index():
    examples = []

    for filename in sorted(os.listdir(UPLOAD_FOLDER)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue

        result_filename = filename + "_result.jpg"
        result_path = os.path.join(RESULTS_FOLDER, result_filename)

        if os.path.isfile(result_path):
            examples.append({
                "original": filename,
                "result": result_filename
            })

    return render_template("index.html", examples=examples)

def find_result(filepath, filename):
    result = model.predict(filepath)
    result_name = RESULTS_FOLDER +'/'+ filename + '_result.jpg'
    result[0].save(filename=result_name)
    return result_name

@app.route('/upload', methods=['POST'])
def upload():
    if 'Image' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400

    file = request.files['Image']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading.'}), 400
    
    file.save(f'{UPLOAD_FOLDER}/{file.filename}')
    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    result_img = find_result(save_path,file.filename)
    #return jsonify({'message': 'Model works successfully.'})
    return render_template('results.html', file = save_path, result = result_img )
@app.route('/detect_frame', methods=['POST'])
def detect_frame():
    if 'frame' not in request.files:
        return jsonify({'error': 'No webcam frame received.'}), 400

    try:
        image = Image.open(BytesIO(request.files['frame'].read())).convert('RGB')
        result = model.predict(source=image, verbose=False)[0]
        success, encoded = cv2.imencode('.jpg', result.plot())

        if not success:
            return jsonify({'error': 'Could not encode the detection result.'}), 500

        return Response(encoded.tobytes(), mimetype='image/jpeg')

    except Exception:
        app.logger.exception('Webcam frame detection failed')
        return jsonify({'error': 'Frame detection failed. Check the VS Code terminal.'}), 500


@app.route('/metrics')
def metrics():
    return render_template('metrics.html')


if __name__ == "__main__":
    app.run(debug=False)
