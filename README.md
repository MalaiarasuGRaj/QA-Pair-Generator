# QA Pair Generator 🧠🔍

## Overview 📚

Welcome to the **QA Pair Generator** project! This powerful tool uses cutting-edge natural language processing (NLP) models to generate insightful question-answer pairs from PDF documents. Whether you're working with academic papers, business reports, or any other document type, our application makes it easy to extract and understand key information.

## Features ✨

- **PDF Upload**: Effortlessly upload your PDF files through an intuitive interface.
- **Text Extraction**: Extract text and images from PDFs with advanced OCR technology.
- **Question Generation**: Create relevant questions from extracted text using the Valhalla model.
- **Difficulty Classification**: Categorize questions into difficulty levels (Easy, Medium, Hard) with the RoBERTa model.
- **Answer Extraction**: Get precise answers to your questions based on the document context.

## Getting Started 🚀

Follow these steps to set up and run the QA Pair Generator locally:

### Prerequisites 📋

- **Python 3.7+** 🐍
- **Pip** (Python package installer) 📦
- **Unix-based system** (for `install_dependencies.sh`)

## Installation 🔧

To set up the QA Pair Generator on your local machine, follow these steps:

### 1. Clone the Repository

Start by cloning the repository to your local machine:

```bash
git clone https://github.com/MalaiarasuGRaj/QA-Pair-Generator/tree/main
cd qa_generator
``` 

### 2. Install System Dependencies
Execute the '**install_dependencies.sh**' script to install the necessary system packages:
```bash
./install_dependencies.sh
``` 

### 3. Install Python Dependencies
Install the required Python packages using '**requirements.txt**':
```bash
pip install -r requirements.txt
``` 

### 4. Run the Application
Launch the Streamlit app with the following command:
```bash
streamlit run QAPair_Generator.py
``` 

## Usage 📈

### Upload PDFs:

Use the sidebar to upload your PDF files. Multiple files can be uploaded at once.

### Generate QA Pairs:

After uploading, the app will process the documents to extract text, generate questions, classify their difficulty, and find answers.

### View Results:

The generated QA pairs, along with their difficulty levels, will be displayed in the main content area.

## Example 📝

Here are some examples of how the QA Pair Generator works:

**Sample 1:**

- **Input Text:** "The capital of France is Paris. It is one of the most populous cities in Europe and is known for its art, fashion, and culture."
- **Generated Question:** "What is the capital of France?"
- **Generated Answer:** "Paris"

**Sample 2:**

- **Input Text:** "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll."
- **Generated Question:** "What process do green plants use to synthesize foods?"
- **Generated Answer:** "Photosynthesis"

**Sample 3:**

- **Input Text:** "Albert Einstein was a theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics."
- **Generated Question:** "What was Albert Einstein's theory of relativity?"
- **Generated Answer:** "one of the two pillars of modern physics"


## Contributing 🤝
We welcome contributions to enhance the QA Pair Generator! To contribute:

1. **Fork the Repository** and create a new branch.
2. **Make Changes** and test thoroughly.
3. **Submit a Pull Request** with a clear description of your changes.


## 📫 Contact Me
- **Email**: [govindarajmalaiarasu@gmail.com](mailto:govindarajmalaiarasu@gmail.com)
- **LinkedIn**: [Malaiarasu G Raj](https://www.linkedin.com/in/malaiarasu-g-raj/)
- **Portfolio**: [My Blog](https://malaiarasu07.wordpress.com/)
