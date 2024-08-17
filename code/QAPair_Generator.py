import warnings

# Suppress specific warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message="Special tokens have been added in the vocabulary")

# Import libraries
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import fitz
import pytesseract
from PIL import Image
import io
import re
from huggingface_hub import login
import torch
from transformers import pipeline

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


# Define function to extract text and images from PDF
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
                print(f"Tesseract not found: {e}")
            except Exception as e:
                print(f"Error processing image: {e}")
    return text + " " + image_text


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    return text.strip()


def segment_text(text, max_words=100):  # Reduce max_words to increase granularity
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
    # Assuming you have 3 classes: 0 = Easy, 1 = Medium, 2 = Hard
    difficulty = ["Easy", "Medium", "Hard"][predicted_class]
    return difficulty


# Define function to extract answers from text segments
def extract_answer(question, context):
    # Specify the model name for the QA pipeline
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2",
                           tokenizer="deepset/roberta-base-squad2")
    result = qa_pipeline(question=question, context=context)
    return result['answer']


# Provide the path to your PDF file (update this path as needed)
pdf_path = "pdf_path"

# Extract, clean, and segment the text from the PDF
combined_text = extract_text_and_images(pdf_path)
cleaned_text = clean_text(combined_text)
segments = segment_text(cleaned_text, max_words=100)

# Generate, classify, and answer questions
qa_pairs = []
for segment in segments:
    question = generate_question(segment)
    difficulty = classify_difficulty(question)
    answer = extract_answer(question, segment)  # Extract answer based on the segment
    qa_pairs.append({'question': question, 'answer': answer, 'difficulty': difficulty})

# Print QA pairs with numbering and difficulty classification
for idx, qa_pair in enumerate(qa_pairs, 1):
    print(f"Difficulty: {qa_pair['difficulty']}")
    print(f"Question {idx}: {qa_pair['question']}")
    print(f"Answer {idx}: {qa_pair['answer']}")
    print()
