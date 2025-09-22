from flask import Blueprint, render_template, request, jsonify
from PIL import Image
from app.model_ai.model_service import run_inference
import base64
import numpy as np
import io

def home():
    return render_template("index.html")

def howitworks():
    return render_template("howitworks.html")

def upload():
    return render_template("upload.html")

def result():
    return render_template("result.html")

def condition():
    return render_template("condition.html")

# Prediction part
def predict():
    file = request.files["image"]
    image = Image.open(file).convert("RGB")
    detections, base64_image, detection_data = run_inference(image)

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    original_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({
        "detections": detections,
        "annotated_image": base64_image,
        "original_image": original_base64,
        "detection_data": detection_data
    })