from flask import Flask, request, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io

app = Flask(__name__)

# Load the pre-trained BLIP model for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

@app.route('/caption', methods=['POST'])
def generate_caption():
    try:
        # Check if an image is uploaded
        if 'image' not in request.files:
            return jsonify({"error": "No image file uploaded"}), 400

        # Load image from the request
        image_file = request.files['image']
        image = Image.open(io.BytesIO(image_file.read()))

        # Preprocess the image and generate captions
        inputs = processor(images=image, return_tensors="pt")
        output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)

        return jsonify({"caption": caption}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Image-to-Text Captioning API!"

if __name__ == "__main__":
    app.run(debug=True)
