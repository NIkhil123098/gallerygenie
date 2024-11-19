from flask import Flask, request, jsonify
import os
import torch
from PIL import Image
from werkzeug.utils import secure_filename
from transformers import CLIPProcessor, CLIPModel

app = Flask(__name__)

# Load the CLIP model and processor
try:
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    print("Model and processor loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Folder to store uploaded images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed file extensions (images)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to compute cosine similarity
def calculate_similarity(image, text):
    try:
        # Use the CLIP processor to encode both image and text into tensors
        inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
        
        # Get the embeddings for image and text from the model
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract image and text features
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds

        # Calculate cosine similarity between image and text
        similarity = torch.cosine_similarity(image_features, text_features)

        return similarity.item()
    except Exception as e:
        print(f"Error during similarity calculation: {e}")
        return None

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the 'file' part is present in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    # Get the file from the request
    file = request.files['file']

    # If no file is selected, the filename will be empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Check if the file is allowed and save it
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Open the image using PIL
        image = Image.open(filepath).convert("RGB")

        # Get the text prompt from the request
        text = request.form['text']
        
        # Calculate the similarity between text and image
        similarity = calculate_similarity(image, text)

        if similarity is not None:
            # Return the filename and the similarity score
            return jsonify({
                'filename': filename,
                'similarity': similarity
            }), 200
        else:
            return jsonify({'error': 'Error calculating similarity'}), 500
    else:
        return jsonify({'error': 'Invalid file type. Only image files are allowed.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
