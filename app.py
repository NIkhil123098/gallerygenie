from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Folder to store uploaded images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed file extensions (images)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

        # Return the filename as the response
        return jsonify({'filename': filename}), 200
    else:
        return jsonify({'error': 'Invalid file type. Only image files are allowed.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
