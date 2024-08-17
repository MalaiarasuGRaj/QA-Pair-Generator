import warnings

# Suppress specific warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message="Special tokens have been added in the vocabulary")

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import re
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from huggingface_hub import login

# Hugging Face authentication
api_key = "your_huggingface_token"
login(token=api_key)

# Load the pre-trained Valhalla model and tokenizer for question generation
model_name = "valhalla/t5-small-qg-hl"  # Valhalla model from Hugging Face
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t5_model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
t5_tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)

pdf_path = "pdf_path"

# Function to extract text and images from PDF
def extract_text_and_images(pdf_path):
    doc = fitz.open(pdf_path)
    text_content = ""

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text_content += page.get_text()

        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))

            # Extract text from image using Tesseract
            text_content += pytesseract.image_to_string(image)

    return text_content

# Function to clean extracted text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    return text.strip()

# Function to segment cleaned text into smaller chunks
def segment_text(text, max_words=100):
    words = text.split()
    segments = []
    current_segment = []
    current_length = 0

    for word in words:
        current_segment.append(word)
        current_length += 1
        if current_length >= max_words:
            segments.append(' '.join(current_segment))
            current_segment = []
            current_length = 0

    if current_segment:
        segments.append(' '.join(current_segment))

    return segments

# Function to generate questions from text segments
def generate_question(segment):
    input_text = f"generate questions from the following text: {segment.replace('</s>', '')}"
    input_ids = t5_tokenizer.encode(input_text, return_tensors='pt').to(device)
    outputs = t5_model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True, do_sample=True)
    question = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return question

# Extract, clean, and segment the text from the PDF
combined_text = extract_text_and_images(pdf_path)
cleaned_text = clean_text(combined_text)
segments = segment_text(cleaned_text, max_words=100)

print("Combined Text:")
print(combined_text)

print("Cleaned Text:")
print(cleaned_text)

# Print all segments
print("Text Segments:")
for idx, segment in enumerate(segments, 1):
    print(f"Segment {idx}:\n{segment}\n")

# Generate questions from segments
questions = []
for segment in segments:
    question = generate_question(segment)
    questions.append(question)

# Print questions with numbering
for idx, question in enumerate(questions, 1):
    print(f"Question {idx}: {question}")
    print()
