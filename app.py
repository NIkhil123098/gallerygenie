from flask import Flask, request, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image, UnidentifiedImageError
import io
import torch

app = Flask(__name__)

# Load the pre-trained BLIP model
try:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

@app.route('/caption', methods=['POST'])
def generate_caption():
    try:
        # Check for image in the request
        if 'image' not in request.files:
            return jsonify({"error": "No image file uploaded"}), 400

        # Read and validate the image
        image_file = request.files['image']
        try:
            image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
        except UnidentifiedImageError:
            return jsonify({"error": "Invalid image format"}), 400

        # Preprocess and generate caption
        inputs = processor(images=image, return_tensors="pt")

        # Check device compatibility
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Generate output
        output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)

        return jsonify({"caption": caption}), 200

    except RuntimeError as e:
        return jsonify({"error": f"Runtime error: {str(e)}"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Image-to-Text Captioning API!"

if __name__ == "__main__":
    app.run(debug=True)
