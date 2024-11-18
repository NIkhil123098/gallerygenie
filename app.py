from flask import Flask, request, jsonify
import os
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from werkzeug.utils import secure_filename

app = Flask(__name__)

"""
# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Path to save uploaded images
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to encode image
def encode_image(image):
    inputs = processor(images=image, return_tensors="pt", padding=True)
    return model.get_image_features(**inputs)

# Function to encode text
def encode_text(text):
    inputs = processor(text=text, return_tensors="pt", padding=True)
    return model.get_text_features(**inputs)
"""


@app.route('/testing', methods=['POST'])
def testing():
    return "Test succeed"

"""
@app.route('/find-images', methods=['POST'])
def find_images():
    # Get the text prompt from the request
    text = request.form['text']
    
    # List to hold the image paths
    best_image = None
    best_similarity = -1

    # Process each image in the batch
    for image_file in request.files.getlist('images[]'):
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_file.filename))
        image_file.save(image_path)

        image = Image.open(image_path)

        # Encode the image and the text
        image_features = encode_image(image)
        text_features = encode_text(text)

        # Calculate similarity between image and text
        similarity = torch.cosine_similarity(text_features, image_features)

        if similarity.item() > best_similarity:
            best_similarity = similarity.item()
            best_image = image_path

    if best_image:
        return jsonify({'image_url': best_image})
    else:
        return jsonify({'error': 'No matching image found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
"""
