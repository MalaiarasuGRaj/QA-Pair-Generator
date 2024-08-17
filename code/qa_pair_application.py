import streamlit as st
import tempfile
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import re
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import pipeline
from huggingface_hub import login
from pyngrok import ngrok
import subprocess

# Initialize Streamlit app
st.title('QA Pair Generator 🧾')
st.markdown('<style>h1{color: orange; text-align: center;}</style>', unsafe_allow_html=True)
st.subheader('Upload a PDF to generate QA pairs')

# File upload widget
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Authenticate with Hugging Face
hugging_face_token = 'your_huggingface_token'
login(token=hugging_face_token)

# Load the pre-trained Valhalla model and tokenizer for question generation
model_name = "valhalla/t5-small-qg-hl"  # Valhalla model from Hugging Face
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t5_model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
t5_tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)

# Load the RoBERTa model and tokenizer for difficulty classification
classification_model_name = "roberta-base"
classification_tokenizer = RobertaTokenizer.from_pretrained(classification_model_name)
classification_model = RobertaForSequenceClassification.from_pretrained(classification_model_name).to(device)

# Function to extract text and images from PDF
def extract_text_and_images(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    image_text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
        images = page.get_images(full=True)
        for img in images:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            try:
                image_text += pytesseract.image_to_string(image)
            except pytesseract.TesseractNotFoundError as e:
                st.error(f"Tesseract not found: {e}")
            except Exception as e:
                st.error(f"Error processing image: {e}")
    return text + " " + image_text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    return text.strip()

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

def generate_question(segment):
    input_text = f"generate questions from the following text: {segment.replace('</s>', '')}"
    input_ids = t5_tokenizer.encode(input_text, return_tensors='pt').to(device)
    outputs = t5_model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True, do_sample=True)
    question = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return question

def classify_difficulty(question):
    inputs = classification_tokenizer(question, return_tensors='pt').to(device)
    outputs = classification_model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    difficulty = ["Easy", "Medium", "Hard"][predicted_class]
    return difficulty

def extract_answer(question, context):
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2",
                           tokenizer="deepset/roberta-base-squad2")
    result = qa_pipeline(question=question, context=context)
    return result['answer']

# Main app logic
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        file_path = temp_file.name

        combined_text = extract_text_and_images(file_path)
        cleaned_text = clean_text(combined_text)
        segments = segment_text(cleaned_text, max_words=100)

        qa_pairs = []
        for segment in segments:
            question = generate_question(segment)
            difficulty = classify_difficulty(question)
            answer = extract_answer(question, segment)
            qa_pairs.append({'question': question, 'answer': answer, 'difficulty': difficulty})

        # Display the QA pairs
        for idx, qa_pair in enumerate(qa_pairs, 1):
            st.write(f"**Difficulty:** {qa_pair['difficulty']}")
            st.write(f"**Question {idx}:** {qa_pair['question']}")
            st.write(f"**Answer {idx}:** {qa_pair['answer']}")
            st.write("---")

# Set up ngrok and run the Streamlit app
if __name__ == '__main__':
    # Set up ngrok tunnel
    ngrok.set_auth_token("your_ngrok_auth_token")
    public_url = ngrok.connect(8501)
    print(f"ngrok tunnel created: {public_url}")

    # Run the Streamlit app
    subprocess.run(["streamlit", "run", __file__, "--server.port", "8501"])
