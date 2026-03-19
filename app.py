from flask import Flask, render_template, request, jsonify
import pdfplumber
from transformers import BartTokenizer, BartForConditionalGeneration
import re

app = Flask(__name__)

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

@app.route('/')
def index():
    return render_template('index.html')

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    return text

def generate_summary(text, num_sentences=3):

    preprocessed_text = preprocess_text(text)


    inputs = tokenizer(preprocessed_text, max_length=1024, return_tensors="pt", truncation=True)


    summary_ids = model.generate(inputs.input_ids, num_beams=4, max_length=500, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

@app.route('/summarize', methods=['POST'])
def summarize():
    file = request.files['file']
    if file and file.filename.endswith('.pdf'):

        pdf = pdfplumber.open(file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        pdf.close()


        summary = generate_summary(text)

        return jsonify({'summary': summary})
    else:
        return jsonify({'error': 'Invalid file format. Only .pdf files are accepted.'})

if __name__ == '__main__':
    app.run(debug=True)
