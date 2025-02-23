import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenet_v2
from PIL import Image
import matplotlib.pyplot as plt
from flask import Flask, request, render_template
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Create the 'uploads' folder if it doesn't exist
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Define the same model architecture as in training
def load_model(model_path):
    device = torch.device("cpu")  # Force CPU usage
    model = mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(1280, 2)  # Adjust final layer for binary classification
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load trained weights
    model.eval()  # Set model to evaluation mode
    return model

# ✅ Load the trained model
model_path = "model/deepfake_detector.pth"
model = load_model(model_path)

# ✅ Define image preprocessing steps
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ✅ Function to predict if an image is real or fake
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Load image
    image = transform(image).unsqueeze(0)  # Apply transformations and add batch dimension

    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)  # Convert to probabilities
        confidence_scores = probabilities.squeeze().tolist()  # Get scores

    predicted_label = torch.argmax(output, dim=1).item()
    class_names = ["Real", "Fake"]
    
    return class_names[predicted_label], confidence_scores

# ✅ Flask Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", message="No file uploaded!")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", message="No file selected!")

        # Save uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Run prediction
        label, confidence_scores = predict_image(file_path)

        # Generate confidence bar chart
        fig, ax = plt.subplots()
        ax.bar(["Real", "Fake"], confidence_scores, color=["green", "red"])
        ax.set_xlabel("Class")
        ax.set_ylabel("Confidence Score")
        ax.set_title("Deepfake Detection Confidence")
        ax.set_ylim(0, 1)
        
        # Save chart
        chart_path = os.path.join("static", "confidence_chart.png")
        plt.savefig(chart_path)
        plt.close()

        return render_template("index.html", label=label, confidence_scores=confidence_scores, chart_path=chart_path, file_path=file_path)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
