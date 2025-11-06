# Resume Evaluator

A lightweight, production-ready web application that evaluates resumes using deterministic heuristics. Users can upload a PDF or DOCX file or paste raw text, optionally add a job description, and receive a structured analysis with actionable feedback.

## Features
- In-browser drag-and-drop or text input for resumes (≤ 5 MB).
- Optional job description field to tailor keyword coverage.
- Rule-based scoring with category breakdowns and high-impact fixes.
- Bullet rewriting suggestions with action verbs and metric placeholders.
- Keyword coverage highlighting matched and missing skills.
- Spell-checking, readability metrics, and formatting heuristics.
- Downloadable JSON report and PDF (or HTML) summary generated on demand.

## Tech Stack
- **Backend:** Python 3.11, Flask
- **Analysis:** pdfminer.six, python-docx, nltk, pyspellchecker, textstat
- **Frontend:** HTML + Tailwind CSS (CDN) + vanilla JavaScript

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
pip install -r requirements.txt
```

## Running the App
```bash
export FLASK_APP=app.py  # On Windows use `set FLASK_APP=app.py`
flask run
```
Open http://127.0.0.1:5000/ to use the interface.

## Reports & PDF Generation
Download buttons trigger an API call that writes temporary files to `reports/`. If `wkhtmltopdf` (and `pdfkit`) are installed, the summary is exported as a PDF. Otherwise, the app gracefully falls back to serving an HTML file with the same content. Temporary report files older than 12 hours are cleaned up automatically on startup.

## Testing
```bash
pytest
```
