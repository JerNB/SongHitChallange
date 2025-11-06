import os
from datetime import datetime, timedelta
from pathlib import Path

from flask import (Flask, jsonify, render_template, request, send_file,
                   send_from_directory)

from analyze import (AnalysisError, analyze_text, extract_text,
                     generate_report_files)

BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "reports"
ALLOWED_EXTENSIONS = {"pdf", "docx"}


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024 + 512

    _cleanup_reports()

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.post("/api/evaluate")
    def evaluate():
        try:
            payload = _parse_request_payload(request)
            analysis = analyze_text(payload["text"], payload.get("job_description"))
            return jsonify(analysis)
        except AnalysisError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:  # pragma: no cover - defensive
            return jsonify({"error": "Unexpected server error."}), 500

    @app.post("/api/report/json")
    def report_json():
        data = request.get_json()
        if not data or "analysis" not in data:
            return jsonify({"error": "Analysis payload required."}), 400
        analysis = data["analysis"]
        json_path, pdf_path = generate_report_files(analysis, str(REPORT_DIR))

        def remove_file(path):
            try:
                os.remove(path)
            except OSError:
                pass

        @request.after_this_request
        def cleanup(response):  # pragma: no cover - flask lifecycle
            remove_file(json_path)
            if pdf_path != json_path:
                remove_file(pdf_path)
            return response

        return send_file(json_path, as_attachment=True, download_name="resume_report.json")

    @app.post("/api/report/pdf")
    def report_pdf():
        data = request.get_json()
        if not data or "analysis" not in data:
            return jsonify({"error": "Analysis payload required."}), 400
        analysis = data["analysis"]
        json_path, pdf_path = generate_report_files(analysis, str(REPORT_DIR))

        def remove_file(path):
            try:
                os.remove(path)
            except OSError:
                pass

        @request.after_this_request
        def cleanup(response):  # pragma: no cover - flask lifecycle
            remove_file(json_path)
            if pdf_path != json_path:
                remove_file(pdf_path)
            return response

        filename = "resume_report.pdf" if pdf_path.endswith(".pdf") else "resume_report.html"
        return send_file(pdf_path, as_attachment=True, download_name=filename)

    @app.route("/static/<path:filename>")
    def static_files(filename: str):
        return send_from_directory(BASE_DIR / "static", filename)

    return app


def _parse_request_payload(req) -> dict:
    text_content = ""
    job_description = None
    if req.content_type and "application/json" in req.content_type:
        data = req.get_json() or {}
        text_content = data.get("resume_text", "")
        job_description = data.get("job_description")
    else:
        if "file" in req.files and req.files["file"]:
            file_storage = req.files["file"]
            filename = file_storage.filename or "uploaded"
            if not _allowed_file(filename):
                raise AnalysisError("Unsupported file type. Upload PDF or DOCX.")
            text_content = extract_text(file_storage, filename)
        if not text_content:
            text_content = req.form.get("resume_text", "")
        job_description = req.form.get("job_description")
    if not text_content:
        raise AnalysisError("Provide a resume file or paste text.")
    return {"text": text_content, "job_description": job_description}


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _cleanup_reports(max_age_hours: int = 12) -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
    for path in REPORT_DIR.glob("*"):
        try:
            modified = datetime.utcfromtimestamp(path.stat().st_mtime)
            if modified < cutoff:
                path.unlink()
        except OSError:
            continue


app = create_app()


if __name__ == "__main__":
    app.run(debug=False)
