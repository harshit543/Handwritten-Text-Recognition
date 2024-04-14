from flask import Flask, render_template, request, send_file
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from line_segmentation import segment_lines
import cv2
import numpy as np

app = Flask(__name__)

# Initialize TrOCR processor and model
output_dir_handwritten = "./saved_model_handwritten"
processor_handwritten = TrOCRProcessor.from_pretrained(output_dir_handwritten)
model_handwritten = VisionEncoderDecoderModel.from_pretrained(output_dir_handwritten)


def ocr_handwritten(image, processor, model):

    # Perform OCR
    pixel_values = processor(image, return_tensors='pt').pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html')
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html')
    if file:
        nparr = np.frombuffer(file.read(), np.uint8)
        image_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        segmented_lines = segment_lines(image_cv2)

        recognized_texts = []
        for i, line in enumerate(segmented_lines):
    # Convert segmented line to PIL image
            line_pil = Image.fromarray(line)
            recognized_text = ocr_handwritten(line_pil, processor_handwritten, model_handwritten)
            recognized_texts.append(recognized_text)
            print(f"Line {i+1} - Recognized Text:", recognized_text)
    
        return render_template('index.html', extracted_text=recognized_texts)
    
@app.route('/download', methods=['GET', 'POST'])
def download():
    if 'extracted_text' not in request.args:
        return "No extracted text to download"

    extracted_text = request.args.get('extracted_text')  # Get extracted text
    if not extracted_text:
        return "No extracted text to download"

    # Preprocess the string to remove square brackets and single quotes
    extracted_text = extracted_text.strip("[]").replace("'", "")

    # Split the string into a list of sentences
    extracted_text_list = extracted_text.split(", ")

    # Convert list of sentences to a single string
    extracted_text_str = '\n'.join(extracted_text_list)

    # Create a text file with extracted text
    with open("extracted_text.txt", "w") as file:
        file.write(extracted_text_str)

    # Send the text file to the user for download
    return send_file("extracted_text.txt", as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
