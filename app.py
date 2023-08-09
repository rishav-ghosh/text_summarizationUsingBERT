from flask import Flask, render_template, request, jsonify
import pdfplumber
from transformers import BartTokenizer, BartForConditionalGeneration
import re

app = Flask(__name__)

# Load the BART model and tokenizer
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

@app.route('/')
def index():
    return render_template('index.html')

def preprocess_text(text):
    # Remove special characters, digits, and convert to lowercase
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    return text

def generate_summary(text, num_sentences=3):
    # Preprocess the text
    preprocessed_text = preprocess_text(text)

    # Tokenize the input text
    inputs = tokenizer(preprocessed_text, max_length=1024, return_tensors="pt", truncation=True)

    # Generate summary using the BART model
    summary_ids = model.generate(inputs.input_ids, num_beams=4, max_length=500, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

@app.route('/summarize', methods=['POST'])
def summarize():
    file = request.files['file']
    if file and file.filename.endswith('.pdf'):
        # Open the PDF file using pdfplumber
        pdf = pdfplumber.open(file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        pdf.close()

        # Generate summary using BART-based summarization
        summary = generate_summary(text)

        return jsonify({'summary': summary})
    else:
        return jsonify({'error': 'Invalid file format. Only .pdf files are accepted.'})

if __name__ == '__main__':
    app.run(debug=True)
