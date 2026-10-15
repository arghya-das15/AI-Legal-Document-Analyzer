# AI Legal Document Analyzer

An AI-powered application designed to simplify the understanding of Indian legal documents using Natural Language Processing and Machine Learning.

## About the Project

AI Legal Document Analyzer is an academic project developed to help users understand complex legal documents through simplified summaries, clause identification, and potential risk analysis.

The application uses document-processing techniques and AI-based analysis to make legal information easier to understand.

## Features

* **Document Processing:** Support for PDF, DOCX, and image-based documents, depending on the configured parsers.
* **Legal Document Summarization:** Generate simplified summaries of legal content.
* **Clause Identification:** Identify important clauses in documents.
* **Risk Analysis:** Highlight potential risks and concerning terms.
* **OCR Support:** Extract text from scanned documents when OCR is configured.
* **Legal Reference Search:** Retrieve relevant legal information when the reference search system is configured.
* **Interactive Interface:** Built with Streamlit.

## Technology Stack

* Python
* Streamlit
* Natural Language Processing (NLP)
* Machine Learning
* OCR and document processing
* Semantic search and vector retrieval, where configured

## Project Structure

```text
AI-Legal-Document-Analyzer/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── modules/
├── models/
├── assets/
├── data/
├── training_data/
└── tests/
```

This is an illustrative structure. Update it to match the files actually included in the repository.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/arghya-das15/AI-Legal-Document-Analyzer.git
```

### 2. Navigate to the project directory

```bash
cd AI-Legal-Document-Analyzer
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

**Windows:**

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
streamlit run app.py
```

Ensure that the required model files, configuration, and dependencies are available before running the application.

## Future Improvements

* Improve legal clause extraction and summarization.
* Expand the legal reference collection.
* Enhance risk detection and evaluation.
* Improve the user interface and accessibility.
* Add more document formats and testing.

These are potential improvements and do not necessarily represent completed features.

## Author

**Arghya Das**

B.Tech — Artificial Intelligence and Machine Learning
Adamas University

## Links

* GitHub: https://github.com/arghya-das15
* LinkedIn: https://www.linkedin.com/in/arghyadas015

## Disclaimer

This project is intended for educational and informational purposes only. AI-generated summaries and risk assessments may be inaccurate or incomplete. The application does not replace advice from a qualified legal professional.
