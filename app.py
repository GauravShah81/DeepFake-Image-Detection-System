from flask import Flask, request, jsonify,render_template
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

# initializing the flask app
app = Flask(__name__)

# define the model and processor directory
model_dir = "DeepFake_Image_detection/deepfake_vs_real_image_detection/checkpoint-7141"

# Load the model using safetensor
model = ViTForImageClassification.from_pretrained(
    model_dir,
    local_files_only = True,
    trust_remote_code = True
)

# Loading the processor
processor = ViTImageProcessor.from_pretrained(model_dir)

print("Model and processor loaded successfullly")

@app.route('/')
def home():
    return render_template('index.html')

# defining the prediction endpoint
@app.route('/predict', methods = ['POST'])
def predict():
    try:
        # check if an image file is included in the request
        if 'image' not in request.files:
            return jsonify({"error" : "No Image file found in the request"}) , 400
        
        # Read the image file from the request
        file = request.files['image']
        image = Image.open(file.stream).convert('RGB')

        # Preprocess the image
        inputs = processor(images=image, return_tensors = 'pt')

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = logits.argmax(-1).item()

        # Map the predict class id to the coressponding label
        predicted_label = model.config.id2label[predicted_class_id]

        # Returning the prediction as JSON
        return jsonify({
            "prediction" : predicted_label,
            "confidence" : torch.softmax(logits, dim=-1)[0][predicted_class_id].item()
        })
    except Exception as e:
        return jsonify({"error" : str(e)}) , 500

# Running the flask app
if __name__ == "__main__":
    app.run(host='0.0.0.0' , port=5000 , debug=True)