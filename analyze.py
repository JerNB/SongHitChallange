import json
import os
import re
from collections import Counter
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional, Tuple

from nltk import word_tokenize
from pdfminer.high_level import extract_text as pdf_extract_text
from pdfminer.pdfparser import PDFSyntaxError
from spellchecker import SpellChecker
from textstat import textstat

try:
    word_tokenize("test")
except LookupError:  # pragma: no cover - runtime data fetch
    import nltk

    nltk.download("punkt", quiet=True)

try:
    from docx import Document
except ImportError:  # pragma: no cover - dependency issue
    Document = None  # type: ignore


MAX_FILE_SIZE = 5 * 1024 * 1024

SECTION_KEYWORDS = {
    "summary": ["summary", "objective", "profile"],
    "experience": ["experience", "employment", "work history", "professional experience"],
    "education": ["education", "academic"],
    "projects": ["projects", "project experience", "selected projects"],
    "skills": ["skills", "technical skills", "core competencies", "technologies"],
    "certifications": ["certifications", "licenses"],
}

ACTION_VERBS = {
    "achieved", "acquired", "adapted", "administered", "advised", "analyzed", "applied",
    "architected", "arranged", "assembled", "assessed", "automated", "built", "calculated",
    "captured", "chaired", "clarified", "coached", "collaborated", "compiled", "completed",
    "computed", "conceived", "conducted", "configured", "conserved", "consolidated",
    "constructed", "consulted", "controlled", "coordinated", "created", "cultivated",
    "customized", "debugged", "decreased", "defined", "delivered", "designed", "developed",
    "devised", "directed", "discovered", "doubled", "drafted", "earned", "edited", "educated",
    "elevated", "eliminated", "enabled", "engineered", "enhanced", "ensured", "established",
    "evaluated", "executed", "expanded", "expedited", "facilitated", "forecasted", "forged",
    "formulated", "founded", "generated", "guided", "implemented", "improved", "increased",
    "influenced", "initiated", "innovated", "inspected", "installed", "integrated",
    "introduced", "invented", "launched", "led", "leveraged", "maintained", "managed",
    "maximized", "measured", "mentored", "merged", "modernized", "monitored", "negotiated",
    "optimized", "orchestrated", "organized", "overhauled", "oversaw", "partnered", "pioneered",
    "planned", "prepared", "presented", "prioritized", "produced", "programmed", "projected",
    "promoted", "proposed", "reduced", "refined", "rehabilitated", "remodeled", "reorganized",
    "replaced", "reported", "researched", "resolved", "restored", "restructured", "revamped",
    "reviewed", "revitalized", "saved", "scheduled", "simplified", "spearheaded", "standardized",
    "streamlined", "strengthened", "supervised", "supported", "surpassed", "sustained",
    "tested", "trained", "transformed", "translated", "upgraded", "validated", "won",
    "accelerated", "aligned", "boosted", "brokered", "budgeted", "calibrated", "charted",
    "coded", "communicated", "conceived", "constructed", "delivered", "diagnosed", "directed",
    "educated", "enabled", "enriched", "envisioned", "estimated", "exceeded", "exercised",
    "expanded", "explored", "formalized", "fortified", "fulfilled", "harnessed", "identified",
    "illustrated", "immersed", "imparted", "implemented", "improvised", "incorporated",
    "increased", "inspired", "instilled", "instituted", "invented", "investigated", "mapped",
    "mobilized", "modeled", "motivated", "navigated", "outpaced", "outsold", "overcame",
    "performed", "piloted", "pinpointed", "reinforced", "rejuvenated", "remediated",
    "repositioned", "researched", "revived", "secured", "solicited", "strategized", "synthesized",
    "targeted", "tracked", "transformed", "uncovered", "utilized", "visualized", "won", "wrote"
}

GENERIC_KEYWORDS = [
    "python", "sql", "excel", "tableau", "power bi", "java", "c++", "html", "css",
    "javascript", "pandas", "numpy", "sklearn", "aws", "git", "finance", "dcf", "ddm",
    "accounting", "statistics"
]

FLUFFY_PHRASES = {
    "hardworking", "passionate", "responsible for", "team player", "dynamic", "motivated",
    "fast learner", "self-starter", "detail-oriented", "results-driven"
}

DATE_PATTERNS = [
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{4}\b",
    r"\b\d{1,2}/\d{4}\b",
    r"\b\d{4}\b",
    r"\b\d{4}\s*(?:-|to|–|—)\s*(?:\d{4}|present)\b",
]

QUANT_TOKENS = {"%", "percent", "percentage", "#", "$", "usd", "million", "k", "increase",
                 "decrease", "reduced", "improved", "growth", "roi", "time", "hours", "days"}


class AnalysisError(Exception):
    """Raised for analysis failures."""


def extract_text_from_pdf(file_bytes: bytes) -> str:
    buffer = BytesIO(file_bytes)
    try:
        text = pdf_extract_text(buffer)
    except PDFSyntaxError as exc:  # pragma: no cover - dependent on pdfminer internals
        raise AnalysisError("Could not parse PDF file") from exc
    if not text.strip():
        raise AnalysisError("Could not extract text from PDF. Ensure the PDF is text-based.")
    return text


def extract_text_from_docx(file_bytes: bytes) -> str:
    if Document is None:
        raise AnalysisError("python-docx is not available")
    buffer = BytesIO(file_bytes)
    try:
        doc = Document(buffer)
    except Exception as exc:  # pragma: no cover - docx parsing errors
        raise AnalysisError("Could not parse DOCX file") from exc
    text = "\n".join(p.text for p in doc.paragraphs)
    if not text.strip():
        raise AnalysisError("The DOCX file did not contain extractable text")
    return text


def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def detect_sections(text: str) -> List[str]:
    sections_found = []
    for section, keywords in SECTION_KEYWORDS.items():
        pattern = re.compile(r"^\s*(?:" + "|".join(re.escape(k) for k in keywords) + r")\b",
                             re.IGNORECASE | re.MULTILINE)
        if pattern.search(text):
            sections_found.append(section.title())
    return sections_found


def detect_dates(text: str) -> List[str]:
    matches: List[str] = []
    for pattern in DATE_PATTERNS:
        matches.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    # normalize
    normalized = []
    for match in matches:
        normalized.append(match.strip())
    return normalized


def split_bullets(text: str) -> List[str]:
    bullet_pattern = re.compile(r"^(?:[-•\u2022\*]+\s+)(.+)$", re.MULTILINE)
    bullets = bullet_pattern.findall(text)
    if not bullets:
        lines = [line.strip() for line in text.split("\n") if len(line.strip().split()) > 6]
        bullets = lines
    bullets = [re.sub(r"\s+", " ", b).strip() for b in bullets if b.strip()]
    return bullets


def is_action_verb(word: str) -> bool:
    return word.lower() in ACTION_VERBS


def analyze_bullets(bullets: List[str]) -> Dict[str, float]:
    if not bullets:
        return {
            "count": 0,
            "action_ratio": 0.0,
            "quant_ratio": 0.0,
            "long_bullets": [],
        }
    action_count = 0
    quant_count = 0
    long_bullets = []
    for bullet in bullets:
        words = bullet.split()
        if words and is_action_verb(words[0].strip("-•")):
            action_count += 1
        if any(token in QUANT_TOKENS or re.search(r"\d", token) for token in words):
            quant_count += 1
        if len(words) > 35:
            long_bullets.append(bullet)
    return {
        "count": len(bullets),
        "action_ratio": action_count / len(bullets),
        "quant_ratio": quant_count / len(bullets),
        "long_bullets": long_bullets,
    }


def compute_metadata(text: str) -> Dict[str, object]:
    words = word_tokenize(text)
    word_count = len([w for w in words if re.search(r"\w", w)])
    estimated_pages = max(1.0, round(word_count / 450.0, 1))
    bullets = split_bullets(text)
    bullet_info = analyze_bullets(bullets)
    sections = detect_sections(text)
    dates = detect_dates(text)
    avg_bullet_length = 0.0
    if bullet_info["count"]:
        avg_bullet_length = round(sum(len(b.split()) for b in bullets) / bullet_info["count"], 1)
    return {
        "words": word_count,
        "estimated_pages": estimated_pages,
        "detected_sections": sections,
        "bullet_count": bullet_info["count"],
        "average_bullet_length": avg_bullet_length,
        "dates": dates,
        "long_bullets": bullet_info["long_bullets"],
        "action_ratio": bullet_info["action_ratio"],
        "quant_ratio": bullet_info["quant_ratio"],
    }


def evaluate_structure(metadata: Dict[str, object], text: str) -> Tuple[int, List[Dict[str, str]]]:
    score = 0
    findings: List[Dict[str, str]] = []
    # Contact info check
    top_lines = "\n".join(text.splitlines()[:4])
    contact_ok = bool(re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", top_lines)) and (
        re.search(r"@[\w.-]+", top_lines) or
        re.search(r"\b\d{3}[\)\s.-]*\d{3}[\s.-]*\d{4}\b", top_lines) or
        re.search(r"linkedin.com|github.com|portfolio|http", top_lines, re.IGNORECASE)
    )
    if contact_ok:
        score += 6
    else:
        findings.append({
            "section": "Structure",
            "severity": "high",
            "message": "Contact information is incomplete or missing in the header.",
            "evidence": top_lines.strip(),
            "fix": "Include your full name and at least one contact method (email, phone, LinkedIn).",
        })
    # Length check
    pages = metadata.get("estimated_pages", 1)
    if pages <= 2:
        score += 6
    else:
        findings.append({
            "section": "Structure",
            "severity": "medium",
            "message": "Resume appears longer than 2 pages.",
            "evidence": f"Estimated pages: {pages}",
            "fix": "Condense content to the most relevant experience, targeting 1-2 pages.",
        })
    required_sections = {"Skills", "Education", "Experience"}
    present_sections = set(metadata.get("detected_sections", []))
    missing = required_sections - present_sections
    if not missing:
        score += 8
    else:
        findings.append({
            "section": "Structure",
            "severity": "high",
            "message": f"Missing expected sections: {', '.join(sorted(missing))}.",
            "evidence": f"Detected sections: {', '.join(sorted(present_sections))}",
            "fix": "Add clearly labeled sections for the missing areas to help recruiters skim.",
        })
    return score, findings


def evaluate_experience(metadata: Dict[str, object], bullets: List[str]) -> Tuple[int, List[Dict[str, str]]]:
    findings: List[Dict[str, str]] = []
    score = 0
    action_ratio = metadata.get("action_ratio", 0)
    quant_ratio = metadata.get("quant_ratio", 0)
    if action_ratio >= 0.7:
        score += 10
    else:
        findings.append({
            "section": "Experience",
            "severity": "high",
            "message": "Many bullets do not start with strong action verbs.",
            "evidence": "Sample bullet: " + (bullets[0] if bullets else "N/A"),
            "fix": "Start each bullet with a powerful verb (e.g., Led, Built, Improved).",
        })
    if quant_ratio >= 0.6:
        score += 10
    else:
        findings.append({
            "section": "Experience",
            "severity": "medium",
            "message": "Few bullets include measurable outcomes (numbers or percentages).",
            "evidence": "Consider adding metrics to demonstrate impact.",
            "fix": "Add specific metrics (%, $, #, time saved) to quantify achievements.",
        })
    dates = metadata.get("dates", [])
    if dates:
        score += 5
    else:
        findings.append({
            "section": "Experience",
            "severity": "low",
            "message": "No date patterns detected; ensure experience shows timelines.",
            "evidence": "Could not detect months/years in the resume.",
            "fix": "Add clear date ranges in a consistent format for each role.",
        })
    return score, findings


def keyword_analysis(skills_tokens: List[str], job_description: Optional[str]) -> Dict[str, object]:
    if job_description:
        jd_tokens = [t.lower() for t in word_tokenize(job_description) if re.search(r"[a-zA-Z]", t)]
        counter = Counter(jd_tokens)
        common = [word for word, _ in counter.most_common() if len(word) > 2]
        keywords = []
        for word in common:
            if word not in keywords and word not in {"and", "the", "for", "with", "you", "your", "will"}:
                keywords.append(word)
            if len(keywords) >= 15:
                break
        matched = sorted(set(skills_tokens) & set(keywords))
        missing = [k for k in keywords if k not in matched]
        mode = "jd"
        notes = "Focus on incorporating the missing keywords if they reflect your actual experience."
    else:
        keywords = GENERIC_KEYWORDS
        matched = sorted(set(skills_tokens) & set(keywords))
        missing = [k for k in keywords if k not in matched][:10]
        mode = "generic"
        notes = "Add relevant skills that align with your background and target roles."
    return {
        "mode": mode,
        "matched": matched,
        "missing": missing,
        "notes": notes,
    }


def evaluate_skills(metadata: Dict[str, object], text: str, job_description: Optional[str]) -> Tuple[int, List[Dict[str, str]], Dict[str, object]]:
    findings: List[Dict[str, str]] = []
    skills_section = extract_section_text(text, "skills")
    if skills_section:
        score = 12
        skills_tokens = sorted({token.lower() for token in re.split(r"[,\n;]+", skills_section) if token.strip()})
    else:
        skills_tokens = []
        score = 4
        findings.append({
            "section": "Skills",
            "severity": "high",
            "message": "No Skills section detected.",
            "evidence": "Resume lacks a clearly labeled Skills section.",
            "fix": "Add a Skills section highlighting tools, technologies, and methods you use.",
        })
    keyword_info = keyword_analysis(skills_tokens, job_description)
    if keyword_info["missing"]:
        findings.append({
            "section": "Skills",
            "severity": "medium",
            "message": "Missing important keywords.",
            "evidence": f"Missing: {', '.join(keyword_info['missing'][:10])}",
            "fix": "Include relevant keywords that reflect your experience and match the role.",
        })
        score += 8 if len(keyword_info["missing"]) < 5 else 4
    else:
        score += 8
    return min(score, 20), findings, keyword_info


def evaluate_writing(text: str, bullets: List[str], metadata: Dict[str, object]) -> Tuple[int, List[Dict[str, str]], Dict[str, object]]:
    findings: List[Dict[str, str]] = []
    spell = SpellChecker()
    tokens = [t.lower() for t in word_tokenize(text) if re.search(r"[a-zA-Z]", t)]
    misspelled = spell.unknown(tokens)
    misspellings = []
    for word in sorted(misspelled)[:20]:
        misspellings.append({"word": word, "suggestions": spell.candidates(word)})
    flesch = round(textstat.flesch_reading_ease(text), 2) if len(text.split()) > 50 else 0
    smog = round(textstat.smog_index(text), 2) if len(text.split()) > 50 else 0
    score = 15
    if misspellings:
        findings.append({
            "section": "Writing",
            "severity": "high",
            "message": "Spelling issues detected.",
            "evidence": ", ".join(f["word"] for f in misspellings[:5]),
            "fix": "Review spelling and run spellcheck to correct these terms.",
        })
        score -= 5
    long_bullets = metadata.get("long_bullets", [])
    if long_bullets:
        findings.append({
            "section": "Writing",
            "severity": "medium",
            "message": "Some bullets exceed 35 words.",
            "evidence": long_bullets[0],
            "fix": "Break long bullets into concise statements or split into two bullets.",
        })
        score -= 2
    readability = {"flesch": flesch, "smog": smog}
    return max(score, 0), findings, {"misspellings": misspellings}, readability


def evaluate_formatting(text: str, bullets: List[str]) -> Tuple[int, List[Dict[str, str]]]:
    findings: List[Dict[str, str]] = []
    score = 10
    bullet_symbols = {re.match(r"^([-•\u2022\*])", line.strip()).group(1)
                      for line in text.splitlines()
                      if re.match(r"^([-•\u2022\*])", line.strip())}
    if len(bullet_symbols) > 1:
        findings.append({
            "section": "Formatting",
            "severity": "low",
            "message": "Multiple bullet styles detected.",
            "evidence": f"Symbols: {', '.join(bullet_symbols)}",
            "fix": "Use a consistent bullet style across sections.",
        })
        score -= 2
    date_formats = set()
    for match in re.finditer(r"(\b\w+\s+\d{4}\b|\b\d{1,2}/\d{4}\b)", text, flags=re.IGNORECASE):
        date_formats.add(match.group(0))
    if len(date_formats) and len({len(d) for d in date_formats}) > 1:
        findings.append({
            "section": "Formatting",
            "severity": "medium",
            "message": "Inconsistent date formats detected.",
            "evidence": ", ".join(list(date_formats)[:4]),
            "fix": "Choose a single date format (e.g., Apr 2022) and apply consistently.",
        })
        score -= 2
    text_lower = text.lower()
    fluffy_instances = [phrase for phrase in FLUFFY_PHRASES if phrase in text_lower]
    if fluffy_instances:
        findings.append({
            "section": "Formatting",
            "severity": "low",
            "message": "Fluffy adjectives detected.",
            "evidence": ", ".join(fluffy_instances[:5]),
            "fix": "Remove fluff and replace with concrete achievements.",
        })
        score -= 2
    pronoun_count = len(re.findall(r"\bI\b", text))
    if pronoun_count > 0:
        findings.append({
            "section": "Formatting",
            "severity": "low",
            "message": "First-person pronouns detected.",
            "evidence": f"Found {pronoun_count} instances of 'I'.",
            "fix": "Resume bullets should be written in implied first person without 'I'.",
        })
        score -= 2
    return max(score, 0), findings


def evaluate_ats(text: str) -> Tuple[int, List[Dict[str, str]]]:
    findings: List[Dict[str, str]] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    short_line_clusters = sum(1 for line in lines if len(line) < 20)
    unique_symbols = {ch for ch in text if not ch.isalnum() and ch not in {" ", "\n", "-", "•", "&", "/", ".", ",", "(" , ")"}}
    score = 10
    if short_line_clusters > len(lines) * 0.6:
        findings.append({
            "section": "ATS",
            "severity": "medium",
            "message": "Layout may use columns or tables which ATS can misread.",
            "evidence": "Many very short lines detected.",
            "fix": "Use a single-column layout for ATS compatibility.",
        })
        score -= 3
    if len(unique_symbols) > 5:
        findings.append({
            "section": "ATS",
            "severity": "low",
            "message": "Uncommon symbols detected that may confuse ATS.",
            "evidence": ", ".join(list(unique_symbols)[:5]),
            "fix": "Limit decorative symbols and use standard characters.",
        })
        score -= 2
    return max(score, 0), findings


def extract_section_text(text: str, section_name: str) -> str:
    pattern = re.compile(rf"(^{section_name}\b.*?)(?=\n\s*[A-Z][\w &/-]{2,}\n|\Z)",
                         re.IGNORECASE | re.MULTILINE | re.DOTALL)
    match = pattern.search(text)
    if match:
        section = match.group(1)
        lines = section.splitlines()[1:]
        return "\n".join(lines).strip()
    return ""


def generate_bullet_improvement(bullet: str) -> Dict[str, str]:
    words = bullet.split()
    if not words:
        return {"original": bullet, "improved": bullet, "rationale": ""}
    core = re.sub(r"^[^a-zA-Z0-9]+", "", bullet)
    tokens = word_tokenize(core)
    action = tokens[0] if tokens else "Improved"
    if not is_action_verb(action):
        action = "Improved"
    remainder = " ".join(tokens[1:])
    if not re.search(r"\d", remainder):
        remainder += " [X%]"
    improved = f"{action.title()} {remainder.strip()} to drive measurable results."
    rationale = "Adds an action verb, introduces a metric placeholder, and clarifies impact."
    return {"original": bullet, "improved": improved, "rationale": rationale}


def build_high_impact_fixes(findings: List[Dict[str, str]]) -> List[str]:
    high_priority = [f for f in findings if f.get("severity") == "high"]
    medium_priority = [f for f in findings if f.get("severity") == "medium"]
    selected = high_priority[:5]
    if len(selected) < 5:
        selected.extend(medium_priority[:5 - len(selected)])
    fixes = []
    for finding in selected:
        fixes.append(f"{finding['message']} — {finding['fix']}")
    return fixes


def quantify_score(structure: int, experience: int, skills: int, writing: int, formatting: int, ats: int) -> Tuple[int, Dict[str, int]]:
    categories = {
        "Structure": structure,
        "Experience": experience,
        "Skills": skills,
        "Writing": writing,
        "Formatting": formatting,
        "ATS": ats,
    }
    overall = sum(categories.values())
    return overall, categories


def analyze_text(text: str, job_description: Optional[str] = None) -> Dict[str, object]:
    if not text or not text.strip():
        raise AnalysisError("No text content found in resume.")
    text = normalize_text(text)
    metadata = compute_metadata(text)
    bullets = split_bullets(text)
    structure_score, structure_findings = evaluate_structure(metadata, text)
    experience_score, experience_findings = evaluate_experience(metadata, bullets)
    skills_score, skills_findings, keyword_info = evaluate_skills(metadata, text, job_description)
    writing_score, writing_findings, spelling_info, readability = evaluate_writing(text, bullets, metadata)
    formatting_score, formatting_findings = evaluate_formatting(text, bullets)
    ats_score, ats_findings = evaluate_ats(text)

    overall, category_scores = quantify_score(
        structure_score, experience_score, skills_score, writing_score, formatting_score, ats_score
    )

    findings = structure_findings + experience_findings + skills_findings + writing_findings + formatting_findings + ats_findings

    bullet_suggestions = []
    weak_bullets = [b for b in bullets if len(b.split()) < 12 or not re.search(r"\d", b) or not is_action_verb(b.split()[0].lower())]
    for bullet in weak_bullets[:5]:
        bullet_suggestions.append(generate_bullet_improvement(bullet))

    high_impact_fixes = build_high_impact_fixes(findings)

    return {
        "overall_score": overall,
        "category_scores": category_scores,
        "metadata": {
            "words": metadata.get("words"),
            "estimated_pages": metadata.get("estimated_pages"),
            "detected_sections": metadata.get("detected_sections"),
            "bullet_count": metadata.get("bullet_count"),
            "average_bullet_length": metadata.get("average_bullet_length"),
        },
        "findings": findings,
        "bullet_suggestions": bullet_suggestions,
        "keyword_coverage": keyword_info,
        "spelling": spelling_info,
        "readability": readability,
        "high_impact_fixes": high_impact_fixes,
        "metadata_details": metadata,
    }


def extract_text(file_storage, filename: str) -> str:
    # Reset file pointer to beginning if supported
    if hasattr(file_storage, 'seek'):
        file_storage.seek(0)
    file_bytes = file_storage.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise AnalysisError("File exceeds 5 MB limit.")
    if filename.lower().endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    if filename.lower().endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    raise AnalysisError("Unsupported file format. Please upload a PDF or DOCX.")


def generate_report_files(payload: Dict[str, object], report_dir: str) -> Tuple[str, str]:
    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    json_path = os.path.join(report_dir, f"resume_report_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    html_content = _render_html_report(payload)
    html_path = os.path.join(report_dir, f"resume_report_{timestamp}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    pdf_path = html_path
    try:
        import pdfkit  # type: ignore

        pdf_path = os.path.join(report_dir, f"resume_report_{timestamp}.pdf")
        pdfkit.from_string(html_content, pdf_path)
    except Exception:
        pdf_path = html_path
    return json_path, pdf_path


def _render_html_report(payload: Dict[str, object]) -> str:
    findings_html = "".join(
        f"<li><strong>{f['section']}</strong>: {f['message']}<br><em>Fix:</em> {f['fix']}</li>"
        for f in payload.get("findings", [])
    )
    bullets_html = "".join(
        f"<li><strong>Original:</strong> {b['original']}<br><strong>Improved:</strong> {b['improved']}</li>"
        for b in payload.get("bullet_suggestions", [])
    )
    keywords = payload.get("keyword_coverage", {})
    matched_html = "".join(f"<span class='badge'>{k}</span> " for k in keywords.get("matched", []))
    missing_html = "".join(f"<span class='badge missing'>{k}</span> " for k in keywords.get("missing", []))
    return f"""
    <html>
    <head>
        <meta charset='utf-8'>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 24px; }}
            h1 {{ color: #1f2937; }}
            ul {{ line-height: 1.6; }}
            .badge {{ display: inline-block; padding: 4px 8px; margin: 2px; background: #d1fae5; }}
            .badge.missing {{ background: #fee2e2; }}
        </style>
    </head>
    <body>
        <h1>Resume Evaluation Summary</h1>
        <p><strong>Overall Score:</strong> {payload['overall_score']}</p>
        <h2>Category Scores</h2>
        <ul>
            {''.join(f'<li>{k}: {v}</li>' for k, v in payload['category_scores'].items())}
        </ul>
        <h2>High-Impact Fixes</h2>
        <ul>
            {''.join(f'<li>{fix}</li>' for fix in payload.get('high_impact_fixes', []))}
        </ul>
        <h2>Findings</h2>
        <ul>{findings_html}</ul>
        <h2>Bullet Suggestions</h2>
        <ul>{bullets_html}</ul>
        <h2>Keyword Coverage</h2>
        <p><strong>Matched:</strong> {matched_html}</p>
        <p><strong>Missing:</strong> {missing_html}</p>
    </body>
    </html>
    """


__all__ = [
    "AnalysisError",
    "analyze_text",
    "extract_text",
    "generate_report_files",
    "extract_text_from_pdf",
    "extract_text_from_docx",
    "normalize_text",
    "detect_sections",
    "detect_dates",
    "is_action_verb",
    "keyword_analysis",
]
