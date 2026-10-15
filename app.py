import streamlit as st
import re
import hashlib
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted

from modules.pdf_parser import extract_text_from_pdf
from modules.docx_parser import extract_text_from_docx


# =========================================================
# LAZY / FAST MODEL HELPERS
# =========================================================

def extract_image_text_fast(uploaded_file):
    # EasyOCR is loaded only when an image is actually uploaded.
    from modules.ocr_parser import extract_text_from_image
    return extract_text_from_image(uploaded_file)


def run_ai_analysis(text):
    # Keep generation intentionally short for local CPU execution.
    try:
        from models.legal_analyzer import run_own_legal_ai
        return run_own_legal_ai(text, max_new_tokens=24)
    except (ImportError, ModuleNotFoundError) as e:
        # Never crash the demo if the local ML package is temporarily broken.
        # The deterministic legal analysis remains available immediately.
        score, level, risks = calculate_risk(text)
        return (
            "INSTANT DOCUMENT-GROUNDED ANALYSIS:\n"
            f"Summary: {compact_summary(create_summary(text), 500)}\n"
            f"Clauses: {', '.join(pretty_items(find_clauses(text))) or 'Not stated in the document.'}\n"
            f"Risks: {', '.join(pretty_items(risks)) or 'No configured risk indicators detected.'}\n"
            f"Risk Score: {score}/100 ({level})\n\n"
            "Note: The local Legal AI runtime needs its Transformers package repaired. "
            "The document analysis above is available without waiting for the model."
        )


def pretty_items(items, limit=8):
    cleaned = []
    for item in items or []:
        value = str(item).strip()
        if value and value not in cleaned:
            cleaned.append(value)
    return cleaned[:limit]


def compact_summary(value, limit=700):
    value = str(value or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit].rsplit(" ", 1)[0] + "…"


def run_legal_question(text, question):
    # Use a fast document-grounded shortcut for common questions;
    # only fall back to the language model when real generation is needed.
    answer = fast_question_answer(text, question)
    if answer:
        return answer
    try:
        from models.legal_analyzer import ask_legal_question
        return ask_legal_question(text, question, max_new_tokens=28)
    except (ImportError, ModuleNotFoundError):
        return (
            "The requested detail is not available in the instant document index. "
            "Please try a question about the case name, parties, dates, payment, "
            "clauses, obligations, risks or summary."
        )


def fast_question_answer(text, question):
    q = question.lower().strip()
    low = text.lower()
    # Common case-title / matter-name questions.
    if any(k in q for k in ["case name", "case title", "matter name", "what is the case"]):
        m = re.search(r"(?:in the matter of|case title|case name)[:\s]+(.{10,180}?)(?:\n|\.|$)", text, re.I)
        if m:
            return "Case Name: " + m.group(1).strip(" .")
        # High Court orders often contain a title after the first page marker.
        lines = [x.strip() for x in text.splitlines() if x.strip()]
        for line in lines[:25]:
            if " v. " in line or " vs. " in line or " vs " in line:
                return "Case Name: " + line[:220]
    if any(k in q for k in ["who are the parties", "parties", "party name", "parties involved"]):
        # Reuse the already-fast party detector.
        ps = find_parties(text)
        return "Parties: " + (", ".join(ps) if ps else "Not stated in the document.")
    if any(k in q for k in ["date", "dates", "when", "effective date", "hearing date"]):
        ds = find_dates(text)
        return "Important Dates: " + (", ".join(ds[:10]) if ds else "Not stated in the document.")
    if any(k in q for k in ["amount", "money", "payment", "price", "rent", "fee", "financial"]):
        ms = find_money(text)
        return "Financial Information: " + (", ".join(ms[:10]) if ms else "Not stated in the document.")
    if any(k in q for k in ["clause", "clauses", "termination", "notice", "liability", "arbitration", "penalty", "confidentiality"]):
        cs = find_clauses(text)
        return "Important Clauses: " + (", ".join(cs) if cs else "No supported clause detected.")
    if any(k in q for k in ["risk", "risks", "red flag", "danger"]):
        score, level, risks = calculate_risk(text)
        return f"Risk Score: {score}/100 ({level}). Risk Indicators: " + (", ".join(risks) if risks else "No configured risk indicators detected.")
    if any(k in q for k in ["summary", "summarize", "what is this document", "what does this document"]):
        return create_summary(text) or "Not stated in the document."
    return None


# =========================================================
# 1. FIND PARTIES
# =========================================================

def find_parties(text):
    """Detect likely party names while avoiding surrounding sentence text."""
    parties = []

    # Pattern 1: "between X and Y"
    between_pattern = re.compile(
        r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:[.,;:]|\s+(?:hereinafter|effective|dated|the agreement)\b|$)",
        re.IGNORECASE
    )

    for match in between_pattern.findall(text):
        first = match[0].strip(" .,:;")
        second = match[1].strip(" .,:;")

        if first and second:
            parties.extend([first, second])

    # Pattern 2: explicit party labels such as Client: ABC Ltd.
    label_pattern = re.compile(
        r"\b(?:client|customer|employee|employer|buyer|seller|tenant|landlord|party)\s*[:\-]\s*([^\n,;.]+)",
        re.IGNORECASE
    )

    for match in label_pattern.findall(text):
        name = match.strip(" .,:;")
        if name:
            parties.append(name)

    # Clean common trailing words that are not part of a name.
    cleaned = []

    for name in parties:
        name = re.sub(
            r"\s+(?:shall|will|may|must|is|are|has|have)\b.*$",
            "",
            name,
            flags=re.IGNORECASE
        ).strip(" .,:;")

        if name and len(name) >= 2 and name not in cleaned:
            cleaned.append(name)

    return cleaned[:10]


# =========================================================
# 2. FIND DATES
# =========================================================

def find_dates(text):
    dates = []

    patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|"
        r"August|September|October|November|December)\s+\d{4}\b",
        r"\b(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{1,2},\s+\d{4}\b"
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)

        for date in matches:
            if date not in dates:
                dates.append(date)

    return dates[:20]


# =========================================================
# 3. FIND MONEY
# =========================================================

def find_money(text):
    amounts = []

    patterns = [
        r"(?:₹|Rs\.?|INR)\s?[\d,]+(?:\.\d+)?",
        r"\$\s?[\d,]+(?:\.\d+)?",
        r"\b\d[\d,]*(?:\.\d+)?\s?(?:rupees|INR|USD|dollars)\b"
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)

        for amount in matches:
            amount = amount.strip()

            if amount not in amounts:
                amounts.append(amount)

    return amounts[:20]


# =========================================================
# 4. FIND LEGAL TERMS
# =========================================================

def find_legal_terms(text):
    legal_terms = [
        "agreement",
        "contract",
        "party",
        "parties",
        "liability",
        "indemnity",
        "termination",
        "breach",
        "penalty",
        "damages",
        "dispute",
        "arbitration",
        "jurisdiction",
        "confidentiality",
        "intellectual property",
        "force majeure",
        "governing law",
        "notice",
        "obligation",
        "warranty",
        "representation",
        "consideration",
        "default",
        "compliance",
        "privacy",
        "data protection"
    ]

    found = []

    text_lower = text.lower()

    for term in legal_terms:
        if term in text_lower:
            found.append(term.title())

    return found


# =========================================================
# 5. FIND IMPORTANT CLAUSES
# =========================================================

def find_clauses(text):
    clauses = []

    clause_keywords = {
        "Termination Clause": [
            "termination",
            "terminate",
            "terminated"
        ],
        "Penalty Clause": [
            "penalty",
            "penalties"
        ],
        "Liability Clause": [
            "liability",
            "liable"
        ],
        "Confidentiality Clause": [
            "confidential",
            "confidentiality"
        ],
        "Dispute Resolution Clause": [
            "dispute",
            "arbitration"
        ],
        "Indemnity Clause": [
            "indemnity",
            "indemnify"
        ],
        "Notice Clause": [
            "notice",
            "written notice"
        ],
        "Force Majeure Clause": [
            "force majeure"
        ],
        "Intellectual Property Clause": [
            "intellectual property",
            "copyright",
            "trademark"
        ],
        "Governing Law Clause": [
            "governing law",
            "jurisdiction"
        ]
    }

    text_lower = text.lower()

    for clause, keywords in clause_keywords.items():

        for keyword in keywords:

            if keyword in text_lower:
                clauses.append(clause)
                break

    return clauses


# =========================================================
# 6. FIND OBLIGATIONS
# =========================================================

def find_obligations(text):
    obligations = []

    obligation_keywords = [
        "shall",
        "must",
        "required to",
        "responsible for",
        "agrees to",
        "undertakes to",
        "will provide",
        "will pay",
        "will deliver",
        "is obligated to"
    ]

    sentences = re.split(r"(?<=[.!?])\s+", text)

    for sentence in sentences:

        sentence_clean = sentence.strip()

        if not sentence_clean:
            continue

        sentence_lower = sentence_clean.lower()

        for keyword in obligation_keywords:

            if keyword in sentence_lower:

                if sentence_clean not in obligations:
                    obligations.append(sentence_clean)

                break

    return obligations[:20]


# =========================================================
# 7. CALCULATE RISK
# =========================================================

def calculate_risk(text):

    text_lower = text.lower()

    score = 0
    indicators = []

    risk_rules = {
        "Penalty": (20, ["penalty", "penalties"]),
        "Termination": (15, ["termination", "terminate"]),
        "Liability": (15, ["liability", "liable"]),
        "Dispute": (15, ["dispute", "arbitration", "litigation"]),
        "Breach": (15, ["breach", "default"]),
        "Damage": (10, ["damages", "damage"]),
        "Notice": (5, ["notice"]),
        "Indemnity": (5, ["indemnity", "indemnify"])
    }

    for risk_name, (points, keywords) in risk_rules.items():

        found = False

        for keyword in keywords:

            if keyword in text_lower:
                found = True
                break

        if found:
            score += points
            indicators.append(risk_name)

    score = min(score, 100)

    if score >= 60:
        level = "HIGH"
    elif score >= 30:
        level = "MEDIUM"
    else:
        level = "LOW"

    return score, level, indicators


# =========================================================
# 8. CREATE SIMPLE SUMMARY
# =========================================================

def create_summary(text):

    sentences = re.split(r"(?<=[.!?])\s+", text)

    clean_sentences = []

    for sentence in sentences:

        sentence = sentence.strip()

        if sentence:
            clean_sentences.append(sentence)

    if not clean_sentences:
        return "No summary could be generated."

    summary = " ".join(clean_sentences[:5])

    if len(summary) > 1200:
        summary = summary[:1200] + "..."

    return summary


# =========================================================
# 9. CREATE RECOMMENDATIONS
# =========================================================

def create_recommendations(text):

    text_lower = text.lower()

    recommendations = []

    if "termination" in text_lower:
        recommendations.append(
            "Review the termination conditions and notice period carefully."
        )

    if "penalty" in text_lower:
        recommendations.append(
            "Check whether the penalty amount is reasonable and clearly defined."
        )

    if "liability" in text_lower:
        recommendations.append(
            "Review the liability limits and responsibilities of each party."
        )

    if "indemnity" in text_lower:
        recommendations.append(
            "Carefully review the indemnity obligations before accepting the agreement."
        )

    if "dispute" in text_lower or "arbitration" in text_lower:
        recommendations.append(
            "Review the dispute resolution and arbitration procedure."
        )

    if "confidential" in text_lower:
        recommendations.append(
            "Check the confidentiality obligations and their duration."
        )

    if "jurisdiction" in text_lower:
        recommendations.append(
            "Verify that the selected jurisdiction is appropriate."
        )

    if not recommendations:
        recommendations.append(
            "Review the agreement carefully before signing."
        )

    recommendations.append(
        "For important legal decisions, consult a qualified legal professional."
    )

    return recommendations


# =========================================================
# 10. ENHANCED LEGAL ANALYSIS HELPERS
# =========================================================

def split_sentences(text):
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text)
        if s.strip()
    ]


def find_clause_evidence(text):
    """Return document sentences that support detected clause categories."""
    rules = {
        "Termination Clause": ["termination", "terminate", "terminated"],
        "Payment / Fee Clause": ["payment", "pay", "fee", "amount", "price"],
        "Penalty Clause": ["penalty", "penalties"],
        "Liability Clause": ["liability", "liable"],
        "Confidentiality Clause": ["confidential", "confidentiality"],
        "Dispute Resolution Clause": ["dispute", "arbitration", "litigation"],
        "Indemnity Clause": ["indemnity", "indemnify"],
        "Notice Clause": ["notice", "written notice"],
        "Force Majeure Clause": ["force majeure"],
        "Intellectual Property Clause": ["intellectual property", "copyright", "trademark"],
        "Governing Law Clause": ["governing law", "jurisdiction", "indian law"],
        "Delivery / Performance Clause": ["deliver", "delivery", "perform", "performance"],
    }

    evidence = {}
    sentences = split_sentences(text)
    for clause, keywords in rules.items():
        matches = []
        for sentence in sentences:
            low = sentence.lower()
            if any(k in low for k in keywords):
                if sentence not in matches:
                    matches.append(sentence)
        if matches:
            evidence[clause] = matches[:3]
    return evidence


def build_risk_details(text, risk_indicators):
    """Create transparent, document-grounded explanations for heuristic risk flags."""
    rules = {
        "Penalty": (20, ["penalty", "penalties"], "A penalty-related term or condition appears in the document."),
        "Termination": (15, ["termination", "terminate"], "The document contains a termination condition or right."),
        "Liability": (15, ["liability", "liable"], "The document contains liability-related language."),
        "Dispute": (15, ["dispute", "arbitration", "litigation"], "A dispute-resolution mechanism is mentioned."),
        "Breach": (15, ["breach", "default"], "The document refers to breach or default."),
        "Damage": (10, ["damages", "damage"], "The document contains damages-related language."),
        "Notice": (5, ["notice"], "A notice requirement is mentioned."),
        "Indemnity": (5, ["indemnity", "indemnify"], "An indemnity-related obligation is mentioned."),
    }

    sentences = split_sentences(text)
    details = []
    for name in risk_indicators:
        if name not in rules:
            continue
        points, keywords, explanation = rules[name]
        evidence = []
        for sentence in sentences:
            if any(k in sentence.lower() for k in keywords):
                evidence.append(sentence)
        if points >= 20:
            severity = "High indicator"
        elif points >= 15:
            severity = "Medium indicator"
        else:
            severity = "Low indicator"
        details.append({
            "name": name,
            "points": points,
            "severity": severity,
            "explanation": explanation,
            "evidence": evidence[:2],
        })
    return details


def find_missing_sections(text):
    """Check whether common agreement sections are present; this is a completeness check, not legal advice."""
    checks = {
        "Parties": ["between", "party", "client", "customer"],
        "Payment / Consideration": ["payment", "pay", "fee", "price", "consideration"],
        "Term / Duration": ["term", "duration", "effective", "valid until"],
        "Termination": ["termination", "terminate"],
        "Confidentiality": ["confidential", "confidentiality"],
        "Dispute Resolution": ["dispute", "arbitration", "litigation"],
        "Governing Law": ["governing law", "indian law", "jurisdiction"],
        "Liability": ["liability", "liable"],
        "Force Majeure": ["force majeure"],
        "Intellectual Property": ["intellectual property", "copyright", "trademark"],
    }
    low = text.lower()
    present, missing = [], []
    for label, keywords in checks.items():
        if any(k in low for k in keywords):
            present.append(label)
        else:
            missing.append(label)
    return present, missing


def get_law_references(text, top_k=3):
    """Retrieve local Indian-law references and expose similarity for transparency."""
    try:
        from models.rag_search import search_law
        results = search_law(text[:3000], top_k=top_k)
    except Exception as e:
        return [], f"RAG unavailable: {e}"

    formatted = []
    for item in results:
        section = str(item.get("section", "")).strip()
        if not section:
            continue
        lines = [x.strip() for x in section.splitlines() if x.strip()]
        title = lines[0] if lines else "Indian-law reference"
        explanation = " ".join(lines[1:])[:500] if len(lines) > 1 else "Local knowledge-base reference."
        formatted.append({
            "title": title,
            "explanation": explanation,
            "score": float(item.get("score", 0.0)),
        })
    return formatted, ""


# =========================================================

# =========================================================
# STREAMLIT PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="LegalDoc AI — AI Legal Document Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# SESSION STATE
# =========================================================

st.session_state.setdefault("page", "Home")
st.session_state.setdefault("dark_mode", True)
st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("ai_result", None)
st.session_state.setdefault("qa_answer", None)
st.session_state.setdefault("legal_question", "")
st.session_state.setdefault("law_references", [])
st.session_state.setdefault("rag_error", "")
st.session_state.setdefault("rag_loaded_for", None)
st.session_state.setdefault("full_analysis_done", False)


def go_to(page):
    st.session_state.page = page
    st.rerun()


def clear_session():
    st.session_state.uploader_key += 1
    st.session_state["ai_result"] = None
    st.session_state["qa_answer"] = None
    st.session_state.pop("document_id", None)
    st.session_state["document_text"] = ""
    st.session_state["document_name"] = ""
    st.session_state["document_extension"] = ""
    st.session_state["document_type"] = "General Legal Agreement"
    st.session_state["analysis_cache"] = None
    st.session_state["law_references"] = []
    st.session_state["rag_error"] = ""
    st.session_state["rag_loaded_for"] = None
    st.session_state["full_analysis_done"] = False
    st.session_state.legal_question = ""
    st.session_state.page = "Home"
    st.rerun()


def run_all_document_analysis(document_text):
    """Run the complete deterministic document analysis once and cache everything."""
    low = document_text.lower()
    if "employment" in low:
        doc_type = "Employment Agreement"
    elif "rent" in low or "lease" in low:
        doc_type = "Rental / Lease Agreement"
    elif "sale" in low and "purchase" in low:
        doc_type = "Sale / Purchase Agreement"
    elif "service" in low:
        doc_type = "Service Agreement"
    elif "non-disclosure" in low or "nda" in low:
        doc_type = "Non-Disclosure Agreement"
    else:
        doc_type = "General Legal Agreement"

    summary = create_summary(document_text)
    parties = find_parties(document_text)
    dates = find_dates(document_text)
    money = find_money(document_text)
    legal_terms = find_legal_terms(document_text)
    clauses = find_clauses(document_text)
    obligations = find_obligations(document_text)
    risk = calculate_risk(document_text)
    recommendations = create_recommendations(document_text)
    clause_evidence = find_clause_evidence(document_text)
    risk_details = build_risk_details(document_text, risk[2])
    sections = find_missing_sections(document_text)
    return {
        "summary": summary, "parties": parties, "dates": dates, "money": money,
        "legal_terms": legal_terms, "clauses": clauses, "obligations": obligations,
        "risk": risk, "recommendations": recommendations,
        "clause_evidence": clause_evidence, "risk_details": risk_details,
        "sections": sections, "document_type": doc_type,
    }


# =========================================================
# PREMIUM LEGAL-TECH UI
# =========================================================

mode = "dark" if st.session_state.dark_mode else "light"

st.markdown(f"""
<style>
:root {{
    --gold: #f7c96b;
    --gold2: #ffdc8d;
    --navy: #071426;
    --navy2: #0c1d35;
    --panel: rgba(13, 30, 55, .88);
    --line: rgba(160, 190, 230, .18);
    --muted: #aebbd0;
    --text: #f4f7fb;
    --green: #45e0a1;
    --blue: #58a8ff;
    --red: #ff5964;
}}

.stApp {{
    background:
        radial-gradient(circle at 12% 10%, rgba(66,105,170,.24), transparent 28%),
        radial-gradient(circle at 88% 18%, rgba(247,201,107,.10), transparent 22%),
        linear-gradient(135deg, #061225 0%, #0a1a31 48%, #071426 100%);
    color: var(--text);
}}

[data-testid="stHeader"] {{
    background: rgba(4, 13, 27, .72);
}}

[data-testid="stToolbar"] {{ visibility: hidden; }}

.block-container {{
    max-width: 1500px;
    padding-top: 1.0rem;
    padding-bottom: 2rem;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #071426 0%, #091b33 100%);
    border-right: 1px solid rgba(160,190,230,.14);
}}
section[data-testid="stSidebar"] > div {{ padding-top: 1.0rem; }}
section[data-testid="stSidebar"] .stRadio > label {{
    color: #aebbd0 !important;
    font-weight: 600;
}}
section[data-testid="stSidebar"] [role="radiogroup"] {{ gap: 6px; }}
section[data-testid="stSidebar"] [role="radio"] {{
    background: rgba(255,255,255,.025);
    border: 1px solid rgba(160,190,230,.08);
    border-radius: 10px;
    padding: 10px 12px;
    margin: 2px 0;
}}
section[data-testid="stSidebar"] [role="radio"]:hover {{
    background: rgba(247,201,107,.08);
    border-color: rgba(247,201,107,.28);
}}
section[data-testid="stSidebar"] [role="radio"] div:first-child {{ display:none; }}

/* Top navbar */
.topbar {{
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding: 8px 4px 13px 4px;
    border-bottom:1px solid rgba(160,190,230,.15);
    margin-bottom: 10px;
}}
.brand {{ display:flex; align-items:center; gap:12px; }}
.brand-mark {{ font-size:2.25rem; line-height:1; }}
.brand-name {{ font-size:1.85rem; font-weight:800; letter-spacing:-.6px; }}
.brand-name span {{ color:var(--gold); }}
.brand-tag {{ color:#9eacc1; font-size:.92rem; border-left:1px solid #506078; padding-left:16px; }}

/* Hero */
.hero {{
    min-height: 225px;
    border:1px solid rgba(160,190,230,.12);
    border-radius: 0 0 18px 18px;
    background:
      linear-gradient(90deg, rgba(6,18,37,.98) 0%, rgba(8,24,45,.84) 44%, rgba(7,20,38,.92) 100%),
      radial-gradient(ellipse at 14% 50%, rgba(247,201,107,.14), transparent 28%),
      radial-gradient(ellipse at 88% 50%, rgba(89,150,255,.14), transparent 28%);
    display:flex;
    align-items:center;
    justify-content:center;
    text-align:center;
    padding: 28px 24px;
    position:relative;
    overflow:hidden;
}}
.hero:before {{
    content:"⚖";
    position:absolute;
    left:7%;
    top:10px;
    font-size:12rem;
    color:rgba(247,201,107,.035);
    transform:rotate(-6deg);
}}
.hero:after {{
    content:"🏛";
    position:absolute;
    right:7%;
    bottom:-40px;
    font-size:12rem;
    color:rgba(130,165,210,.035);
}}
.hero-inner {{ position:relative; z-index:2; }}
.hero h1 {{ font-size:3.05rem; margin:0; font-weight:850; letter-spacing:-1.4px; }}
.hero h1 span {{ color:var(--gold); }}
.hero-sub {{ color:#b7c2d4; font-size:1.22rem; margin-top:6px; }}
.hero-points {{ color:var(--gold2); font-size:.85rem; font-weight:700; letter-spacing:.08em; margin-top:20px; }}

/* Feature row */
.feature-grid {{
    display:grid;
    grid-template-columns:repeat(6, 1fr);
    gap:12px;
    margin: 0 0 20px 0;
}}
.feature {{
    background:linear-gradient(145deg, rgba(18,39,70,.92), rgba(9,24,45,.92));
    border:1px solid rgba(160,190,230,.15);
    border-radius:14px;
    padding:14px 13px;
    min-height:72px;
    box-shadow:0 12px 28px rgba(0,0,0,.16);
}}
.feature-icon {{ font-size:1.65rem; float:left; margin-right:10px; }}
.feature-title {{ font-weight:750; font-size:.92rem; }}
.feature-sub {{ color:#a7b5c9; font-size:.75rem; margin-top:3px; }}

/* Main cards */
.panel {{
    background:linear-gradient(145deg, rgba(17,37,67,.96), rgba(8,24,45,.94));
    border:1px solid rgba(160,190,230,.17);
    border-radius:16px;
    padding:20px;
    box-shadow:0 16px 40px rgba(0,0,0,.18);
}}
.panel-title {{ font-size:1.5rem; font-weight:800; margin-bottom:3px; }}
.panel-sub {{ color:#aebbd0; margin-bottom:15px; }}
.privacy {{
    display:inline-block;
    padding:9px 14px;
    border-radius:10px;
    background:linear-gradient(135deg, rgba(247,201,107,.22), rgba(247,201,107,.08));
    border:1px solid rgba(247,201,107,.22);
    color:#ffe1a0;
    font-weight:700;
}}
.upload-shell {{
    border:1.5px dashed rgba(194,211,235,.65);
    border-radius:14px;
    padding:34px 22px 22px;
    text-align:center;
    background:rgba(255,255,255,.018);
}}
.upload-icon {{ font-size:3.8rem; color:var(--gold); }}
.upload-title {{ font-size:1.38rem; font-weight:800; margin-top:6px; }}
.upload-sub {{ color:#b5c0d0; margin-top:5px; }}
.upload-meta {{ color:#9aaac0; margin-top:18px; font-size:.85rem; }}
.status-row {{
    display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-top:18px;
    border-top:1px solid rgba(160,190,230,.12); padding-top:16px;
}}
.status-item {{ color:#dbe5f4; font-weight:650; font-size:.82rem; }}
.status-item span {{ display:block; color:#9caac0; font-weight:400; margin-top:3px; }}

/* Right-side How it works */
.how-card {{
    background:linear-gradient(145deg, rgba(16,36,65,.96), rgba(7,23,43,.95));
    border:1px solid rgba(160,190,230,.17);
    border-radius:16px;
    padding:18px;
}}
.step {{
    display:flex; align-items:center; gap:14px; padding:12px 10px; margin:8px 0;
    border:1px solid rgba(160,190,230,.13); border-radius:13px; background:rgba(255,255,255,.025);
}}
.step-no {{ font-size:1.2rem; font-weight:850; color:#b8c6d9; width:38px; }}
.step-icon {{ font-size:1.65rem; }}
.step-title {{ font-weight:750; }}
.step-sub {{ color:#aab7ca; font-size:.78rem; margin-top:2px; }}
.quote-card {{ margin-top:14px; padding:18px; border-radius:15px; background:linear-gradient(135deg, rgba(247,201,107,.16), rgba(13,30,55,.88)); border:1px solid rgba(247,201,107,.22); }}
.quote-card .quote {{ color:#ffe1a0; font-style:italic; font-size:1rem; }}

/* Buttons */
.stButton > button, .stDownloadButton > button {{
    border-radius:10px !important;
    border:1px solid rgba(160,190,230,.18) !important;
    font-weight:750 !important;
}}
.stButton > button[kind="primary"] {{
    background:linear-gradient(135deg, #f7c96b, #ffd98d) !important;
    color:#071426 !important;
    border:0 !important;
}}

/* Inputs */
.stTextInput input, .stTextArea textarea {{
    background:#0a1930 !important;
    color:#f5f7fb !important;
    border:1px solid rgba(160,190,230,.18) !important;
    border-radius:10px !important;
}}

/* Analysis cards */
.analysis-card {{
    background:rgba(12,29,52,.92); border:1px solid rgba(160,190,230,.16);
    border-radius:14px; padding:18px; margin:10px 0;
}}
.metric-card {{
    background:linear-gradient(145deg, rgba(19,43,76,.95), rgba(9,25,46,.95));
    border:1px solid rgba(160,190,230,.15); border-radius:13px; padding:15px;
}}
.footer {{ border-top:1px solid rgba(160,190,230,.14); margin-top:28px; padding:15px 5px; color:#aab7ca; font-size:.82rem; display:flex; justify-content:space-between; }}

@media (max-width: 1000px) {{
  .feature-grid {{ grid-template-columns:repeat(2,1fr); }}
  .hero h1 {{ font-size:2.2rem; }}
}}
</style>
""", unsafe_allow_html=True)

if not st.session_state.dark_mode:
    st.markdown("""
    <style>
      .stApp { background: linear-gradient(135deg,#eef3f9 0%,#f7f9fc 55%,#eef2f7 100%) !important; color:#17243a !important; }
      [data-testid="stHeader"] { background:rgba(255,255,255,.88) !important; }
      section[data-testid="stSidebar"] { background:#f4f7fb !important; }
      .topbar,.panel,.how-card,.analysis-card,.metric-card,.feature,.upload-shell,.quote-card,.hero {
        background:rgba(255,255,255,.94) !important; color:#17243a !important;
        border-color:rgba(40,70,110,.18) !important;
      }
      .brand-name,.hero h1,.panel-title,.step-title,.feature-title { color:#17243a !important; }
      .brand-tag,.hero-sub,.panel-sub,.feature-sub,.step-sub,.status-item span,.upload-sub,.upload-meta,.footer { color:#5b6b80 !important; }
      .stTextInput input,.stTextArea textarea { background:#fff !important; color:#17243a !important; }
    </style>
    """, unsafe_allow_html=True)


# =========================================================
# SIDEBAR — STABLE NAVIGATION
# Use buttons instead of a persistent radio widget. A persistent radio can
# restore its previous value after a rerun and immediately force the page back
# to the old tab. Buttons update the page state directly.

with st.sidebar:
    st.markdown("<div style='text-align:center;font-size:2.2rem;'>⚖️</div>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;margin:0;color:#fff;'>LegalDoc <span style='color:#f7c96b;'>AI</span></h2>", unsafe_allow_html=True)
    st.caption("Domain-specific Indian legal workspace")
    st.divider()
    st.markdown("**NAVIGATION**")

    sidebar_pages = [
        ("🏠 Home", "Home"),
        ("📄 Document Analysis", "Document Analysis"),
        ("💬 Ask Legal AI", "Ask Legal AI"),
        ("⚖️ Indian Law Search", "Indian Law Search"),
        ("📑 Reports", "Reports"),
        ("ℹ️ About the Project", "About the Project"),
    ]

    for label, target in sidebar_pages:
        is_current = st.session_state.page == target
        if st.button(
            ("✓ " if is_current else "") + label,
            key=f"side_{target}",
            use_container_width=True,
            type="primary" if is_current else "secondary",
        ):
            if st.session_state.page != target:
                st.session_state.page = target
                st.rerun()

    st.divider()
    if st.button("🗑️ Clear Document & Session", use_container_width=True, key="side_clear"):
        clear_session()
    if st.button("🌙 Toggle Dark / Light", use_container_width=True, key="side_theme"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
    st.markdown("<div style='margin-top:25px;padding:14px;border:1px solid rgba(247,201,107,.18);border-radius:12px;background:rgba(247,201,107,.05);color:#f4d58e;font-style:italic;'>“Legal knowledge should empower everyone.”</div>", unsafe_allow_html=True)


# TOP BAR + WORKING NAV BUTTONS
# =========================================================

st.markdown("""
<div class="topbar">
  <div class="brand">
    <div class="brand-mark">⚖️</div>
    <div class="brand-name">LegalDoc <span>AI</span></div>
    <div class="brand-tag">Complex Legal Documents. Clearer Possibilities.</div>
  </div>
  <div style="color:#aebbd0;font-size:.85rem;">Privacy-first • Indian context • Document-grounded</div>
</div>
""", unsafe_allow_html=True)

nav_cols = st.columns([1,1,1,1,1,1,1])
nav_items = [
    ("Home", "Home"),
    ("About", "About the Project"),
    ("How It Works", "Home"),
    ("Indian Laws", "Indian Law Search"),
    ("Help", "Ask Legal AI"),
    ("Document Analysis", "Document Analysis"),
    ("Reports", "Reports"),
]
for nav_index, (col, (label, target)) in enumerate(zip(nav_cols, nav_items), start=1):
    with col:
        # Use a unique key for every top navigation button.
        # "Home" is intentionally used by both Home and How It Works,
        # so the target name alone cannot be the widget key.
        if st.button(label, key=f"top_nav_{nav_index}", use_container_width=True):
            go_to(target)


# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">
  <div class="hero-inner">
    <h1>AI Legal Document <span>Analyzer</span></h1>
    <div class="hero-sub">From Legal Complexity to Human Clarity</div>
    <div class="hero-points">READ &nbsp; • &nbsp; UNDERSTAND &nbsp; • &nbsp; ASSESS RISK &nbsp; • &nbsp; GET GUIDANCE &nbsp; • &nbsp; EMPOWER YOUR RIGHTS</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="feature-grid">
  <div class="feature"><div class="feature-icon">📄</div><div class="feature-title">PDF & DOCX</div><div class="feature-sub">Multi-format Support</div></div>
  <div class="feature"><div class="feature-icon">🔍</div><div class="feature-title">OCR TECHNOLOGY</div><div class="feature-sub">Scanned Documents</div></div>
  <div class="feature"><div class="feature-icon">🧠</div><div class="feature-title">YOUR LEGAL AI</div><div class="feature-sub">Built for Indian Context</div></div>
  <div class="feature"><div class="feature-icon">🏛️</div><div class="feature-title">INDIAN LAW RAG</div><div class="feature-sub">Trusted Legal Sources</div></div>
  <div class="feature"><div class="feature-icon">🛡️</div><div class="feature-title">RISK ANALYSIS</div><div class="feature-sub">Identify Red Flags</div></div>
  <div class="feature"><div class="feature-icon">📑</div><div class="feature-title">PROFESSIONAL REPORTS</div><div class="feature-sub">PDF & Text Export</div></div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# UPLOAD / PROCESS ONCE
# =========================================================

uploaded_file = st.file_uploader(
    "Choose a legal document",
    type=["pdf", "docx", "jpg", "jpeg", "png"],
    label_visibility="collapsed",
    key=f"uploader_{st.session_state.uploader_key}",
)

# Persistent document workspace: navigation/reruns no longer lose the uploaded text.
for _k, _v in {
    "document_text": "",
    "document_name": "",
    "document_extension": "",
    "document_type": "General Legal Agreement",
    "analysis_cache": None,
}.items():
    st.session_state.setdefault(_k, _v)

text = st.session_state.get("document_text", "")
file_name = st.session_state.get("document_name", "")
file_extension = st.session_state.get("document_extension", "")
document_type = st.session_state.get("document_type", "General Legal Agreement")

if uploaded_file is not None:
    incoming_name = uploaded_file.name
    incoming_ext = incoming_name.lower().split(".")[-1]
    # Process only when a genuinely new file arrives.
    if incoming_name != st.session_state.get("document_name") or not st.session_state.get("document_text"):
        try:
            if incoming_ext == "pdf":
                new_text = extract_text_from_pdf(uploaded_file)
            elif incoming_ext == "docx":
                new_text = extract_text_from_docx(uploaded_file)
            elif incoming_ext in ["jpg", "jpeg", "png"]:
                with st.spinner("🔍 OCR is reading the image... Please wait."):
                    new_text = extract_image_text_fast(uploaded_file)
            else:
                new_text = ""
        except Exception as e:
            st.error(f"Could not process the document: {e}")
            new_text = ""

        if new_text and new_text.strip():
            text = new_text.strip()
            current_doc_id = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
            st.session_state["document_id"] = current_doc_id
            st.session_state["document_text"] = text
            st.session_state["document_name"] = incoming_name
            st.session_state["document_extension"] = incoming_ext
            st.session_state["ai_result"] = None
            st.session_state["qa_answer"] = None
            st.session_state["legal_question"] = ""
            st.session_state["law_references"] = []
            st.session_state["rag_error"] = ""
            st.session_state["rag_loaded_for"] = None
            st.session_state["analysis_cache"] = None
            file_name = incoming_name
            file_extension = incoming_ext

# Restore the persistent document after navigation reruns.
text = st.session_state.get("document_text", "")
file_name = st.session_state.get("document_name", "")
file_extension = st.session_state.get("document_extension", "")
document_type = st.session_state.get("document_type", "General Legal Agreement")

# Compute deterministic analysis once per document, then reuse it on every tab.
if text:
    if st.session_state.get("analysis_cache") is None:
        st.session_state["analysis_cache"] = run_all_document_analysis(text)
        st.session_state["document_type"] = st.session_state["analysis_cache"]["document_type"]
    cache = st.session_state["analysis_cache"]
    summary = cache["summary"]
    parties = cache["parties"]
    dates = cache["dates"]
    money = cache["money"]
    legal_terms = cache["legal_terms"]
    clauses = cache["clauses"]
    obligations = cache["obligations"]
    risk_score, risk_level, risk_indicators = cache["risk"]
    recommendations = cache["recommendations"]
    clause_evidence = cache["clause_evidence"]
    risk_details = cache["risk_details"]
    present_sections, missing_sections = cache["sections"]
    law_references = st.session_state.get("law_references", [])
    rag_error = st.session_state.get("rag_error", "")
else:
    summary = ""
    parties, dates, money, legal_terms, clauses, obligations = [], [], [], [], [], []
    risk_score, risk_level, risk_indicators = 0, "LOW", []
    recommendations, clause_evidence, risk_details = [], {}, []
    present_sections, missing_sections = [], []
    law_references, rag_error = [], ""


def show_upload_panel():
    st.markdown("""
    <div class="panel">
      <div style="display:flex;justify-content:space-between;gap:20px;align-items:flex-start;">
        <div>
          <div class="panel-title">📄 Upload Your Legal Document</div>
          <div class="panel-sub">Secure • Temporary Processing • AI-Powered Analysis</div>
        </div>
        <div class="privacy">🔒 Privacy-First Processing<br><small style="font-weight:500;">Your document stays in the current session</small></div>
      </div>
      <div class="upload-shell">
        <div class="upload-icon">⇧</div>
        <div class="upload-title">Drag & drop your legal document here</div>
        <div class="upload-sub">or click the Browse files button below</div>
        <div class="upload-meta">Supports PDF, DOCX, PNG, JPG &nbsp; | &nbsp; Max size depends on Streamlit configuration</div>
        <div class="status-row">
          <div class="status-item">🛡️ Your Privacy Matters<span>Documents are not intentionally stored permanently.</span></div>
          <div class="status-item">⚡ Fast Analysis<span>Rule-based + RAG + your Legal AI.</span></div>
          <div class="status-item">🔒 Secure & Temporary<span>Use Clear Document & Session when finished.</span></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if uploaded_file is not None and text:
        st.success(f"✅ {file_name} processed successfully.")
        a, b = st.columns([1,1])
        with a:
            if st.button("📄 Open Document Analysis", type="primary", use_container_width=True):
                go_to("Document Analysis")
        with b:
            if st.button("🗑️ Clear Document & Session", use_container_width=True):
                clear_session()


def show_how_it_works():
    st.markdown("""
    <div class="how-card">
      <div style="display:flex;justify-content:space-between;align-items:center;"><h2 style="margin:0;">How It Works</h2><span style="background:rgba(88,168,255,.12);padding:6px 10px;border-radius:20px;color:#b8d6ff;font-size:.75rem;">A SIMPLE 4-STEP PROCESS</span></div>
      <div class="step"><div class="step-no">01</div><div class="step-icon">📄</div><div><div class="step-title">Upload Document</div><div class="step-sub">PDF, DOCX or image document</div></div></div>
      <div class="step"><div class="step-no">02</div><div class="step-icon">🧠</div><div><div class="step-title">AI Analysis</div><div class="step-sub">Your Legal AI reads the document</div></div></div>
      <div class="step"><div class="step-no">03</div><div class="step-icon">🛡️</div><div><div class="step-title">Legal Insights</div><div class="step-sub">Summary, clauses, risk indicators and references</div></div></div>
      <div class="step"><div class="step-no">04</div><div class="step-icon">💬</div><div><div class="step-title">Ask & Explore</div><div class="step-sub">Ask questions and search Indian-law references</div></div></div>
    </div>
    <div class="quote-card"><div class="quote">“Legal knowledge should empower everyone.”</div><div style="color:#aebbd0;font-size:.8rem;margin-top:8px;">A small step towards a fairer society.</div></div>
    """, unsafe_allow_html=True)


# =========================================================
# PAGE: HOME
# =========================================================

if st.session_state.page == "Home":
    left, right = st.columns([2.05, 1])
    with left:
        show_upload_panel()
    with right:
        show_how_it_works()

    if uploaded_file is not None and text:
        st.markdown("<div class='analysis-card'><b>📊 Quick Document Status</b><br>Document is ready. All analysis sections below are enabled and connected to the current document.</div>", unsafe_allow_html=True)
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Words", len(text.split()))
        q2.metric("Clauses", len(clauses))
        q3.metric("Risk", f"{risk_score}/100")
        q4.metric("Law refs", len(law_references))

        # ========================================================
        # FULL POINT-BY-POINT VIEW ON HOME
        # ========================================================
        st.markdown("<div class='analysis-card'><h2 style='margin:0;'>📋 Complete Point-by-Point Analysis</h2><div style='color:#aebbd0;margin-top:5px;'>This is the same detailed section used in the Document Analysis page.</div></div>", unsafe_allow_html=True)
        home_points = [
            ("01", "📝 Simple Summary", compact_summary(summary) or "Not stated in the document."),
            ("02", "👥 Parties", "\n".join(["• "+x for x in pretty_items(parties)]) if parties else "Not stated in the document."),
            ("03", "📅 Important Dates", "\n".join(["• "+x for x in pretty_items(dates)]) if dates else "Not stated in the document."),
            ("04", "💰 Financial Information", "\n".join(["• "+x for x in pretty_items(money)]) if money else "Not stated in the document."),
            ("05", "⚖️ Legal Terms", ", ".join(pretty_items(legal_terms)) if legal_terms else "Not stated in the document."),
            ("06", "📌 Important Clauses", "\n".join(["• "+x for x in pretty_items(clauses)]) if clauses else "No supported clause detected."),
            ("07", "📋 Obligations", "\n".join(["• "+x for x in pretty_items(obligations)]) if obligations else "Not stated in the document."),
            ("08", "⚠️ Risk Indicators", "\n".join(["• "+x for x in pretty_items(risk_indicators)]) if risk_indicators else "No configured risk indicators detected."),
            ("09", "📊 Risk Score", f"{risk_score}/100 — {risk_level}"),
            ("10", "💡 Recommendations", "\n".join(["• "+x for x in pretty_items(recommendations)]) if recommendations else "No recommendations generated."),
        ]
        for num, title, body in home_points:
            st.markdown(
                f"<div class='analysis-card' style='margin:8px 0;padding:14px 16px;'>"
                f"<div style='font-weight:800;font-size:1.02rem;margin-bottom:7px;'>{num} &nbsp; {title}</div>"
                f"<div style='color:#e8eef7;line-height:1.6;white-space:pre-wrap;'>{escape(str(body))}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

        # ========================================================
        # QUICK ACTIONS — ALL MAIN FUNCTIONS ENABLED
        # ========================================================
        st.markdown("<div class='analysis-card'><h3 style='margin:0;'>🚀 Quick Actions</h3><div style='color:#aebbd0;margin-top:5px;'>Run the complete analysis once, then open any section instantly.</div></div>", unsafe_allow_html=True)

        # One-click full run: deterministic analysis + Legal AI + Indian-law RAG.
        if st.button("⚡ RUN COMPLETE ANALYSIS — ALL SECTIONS", type="primary", use_container_width=True, key="home_run_all"):
            try:
                with st.spinner("⚡ Running all document analysis together..."):
                    st.session_state["analysis_cache"] = run_all_document_analysis(text)
                    st.session_state["document_type"] = st.session_state["analysis_cache"]["document_type"]
                    # Run AI and law retrieval concurrently so CPU waits are not stacked.
                    from concurrent.futures import ThreadPoolExecutor
                    def _ai():
                        return run_ai_analysis(text)
                    def _law():
                        return retrieve_law_refrences(text, top_k=3)
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        f_ai = pool.submit(_ai)
                        f_law = pool.submit(_law)
                        st.session_state["ai_result"] = f_ai.result()
                        refs, err = f_law.result()
                        st.session_state["law_references"] = refs
                        st.session_state["rag_error"] = err
                    st.session_state["full_analysis_done"] = True
                st.success("✅ Complete analysis finished. All sections are ready.")
            except Exception as e:
                st.error(f"Complete analysis failed: {e}")

        a1,a2,a3,a4 = st.columns(4)
        with a1:
            if st.button("🤖 Run Your Legal AI", type="primary", use_container_width=True, key="home_run_ai"):
                try:
                    with st.spinner("🤖 Your Legal AI is analyzing... "):
                        st.session_state.ai_result = run_ai_analysis(text)
                    st.success("AI analysis completed.")
                except Exception as e:
                    st.error(f"AI analysis failed: {e}")
        with a2:
            if st.button("💬 Ask Legal AI", use_container_width=True, key="home_ask_ai"):
                go_to("Ask Legal AI")
        with a3:
            if st.button("⚖️ Indian Law Search", use_container_width=True, key="home_law"):
                go_to("Indian Law Search")
        with a4:
            if st.button("📑 Reports", use_container_width=True, key="home_reports"):
                go_to("Reports")

        if st.session_state.get("ai_result"):
            st.markdown("<div class='analysis-card'><h3>🧠 Your Legal AI V3 Result</h3></div>", unsafe_allow_html=True)
            st.write(st.session_state.get("ai_result"))

        # ========================================================
        # DOCUMENT Q&A DIRECTLY ON HOME
        # ========================================================
        st.markdown("<div class='analysis-card'><h3 style='margin:0;'>💬 Ask a Question About This Document</h3><div style='color:#aebbd0;margin-top:5px;'>Ask questions exactly like the previous version.</div></div>", unsafe_allow_html=True)
        hq1, hq2 = st.columns([4,1])
        with hq1:
            home_question = st.text_input("Question", placeholder="Example: What is the termination notice period?", key="home_question")
        with hq2:
            st.write("")
            st.write("")
            if st.button("Ask", type="primary", use_container_width=True, key="home_question_btn"):
                if not home_question.strip():
                    st.warning("Enter a question first.")
                else:
                    try:
                        with st.spinner("💬 Finding the answer..."):
                            st.session_state.qa_answer = run_legal_question(text, home_question.strip())
                        st.success("Answer ready.")
                    except Exception as e:
                        st.error(f"Question failed: {e}")
        if st.session_state.get("qa_answer"):
            st.info(st.session_state.get("qa_answer"))

        with st.expander("🔎 Evidence & Document Completeness", expanded=False):
            st.markdown("**Detected Sections**")
            st.write(", ".join(present_sections) if present_sections else "None")
            st.markdown("**Not Detected**")
            st.write(", ".join(missing_sections) if missing_sections else "None")

        with st.expander("📄 Extracted Document Text", expanded=False):
            st.text_area("Extracted text", text, height=360, key="home_extracted_text")


# =========================================================
# PAGE: DOCUMENT ANALYSIS
# =========================================================

elif st.session_state.page == "Document Analysis":
    st.markdown("<div class='panel'><div class='panel-title'>📄 Document Analysis</div><div class='panel-sub'>Document-grounded analysis using extraction, rules, your Legal AI and Indian-law RAG.</div></div>", unsafe_allow_html=True)
    if not text:
        st.info("Upload a PDF, DOCX or image from Home first.")
        if st.button("🏠 Go to Home", type="primary"):
            go_to("Home")
    else:
        st.success(f"Processed: {file_name} • {document_type}")
        if st.button("🤖 Run Your Legal AI Analysis", type="primary", use_container_width=True):
            try:
                with st.spinner("🤖 Your Legal AI is analyzing the document... Please wait."):
                    st.session_state.ai_result = run_ai_analysis(text)
                st.success("✅ Your Legal AI analysis completed.")
            except Exception as e:
                st.error("❌ Your Legal AI could not analyze the document.")
                st.code(str(e))

        if st.session_state.get("ai_result"):
            st.markdown("<div class='analysis-card'><h3>🧠 Your Legal AI — Structured Analysis</h3></div>", unsafe_allow_html=True)
            st.markdown("**📝 AI Summary**")
            st.write(st.session_state.get("ai_result"))

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Detected Clauses", len(clauses))
        m2.metric("Risk Indicators", len(risk_indicators))
        m3.metric("Law References", len(law_references))
        m4.metric("Risk Score", f"{risk_score}/100")

        # ========================================================
        # POINT-BY-POINT ANALYSIS (simple, visible, viva-friendly)
        # ========================================================
        st.markdown("<div class='analysis-card'><h3>📌 Point-by-Point Document Analysis</h3><div style='color:#aebbd0;'>All detected information is shown step-by-step below.</div></div>", unsafe_allow_html=True)

        point_items = [
            ("01", "📝 Simple Summary", compact_summary(summary) or "Not stated in the document."),
            ("02", "👥 Parties", "\n".join(["• " + x for x in pretty_items(parties)]) if parties else "Not stated in the document."),
            ("03", "📅 Important Dates", "\n".join(["• " + x for x in pretty_items(dates)]) if dates else "Not stated in the document."),
            ("04", "💰 Financial Information", "\n".join(["• " + x for x in pretty_items(money)]) if money else "Not stated in the document."),
            ("05", "⚖️ Legal Terms", ", ".join(pretty_items(legal_terms)) if legal_terms else "Not stated in the document."),
            ("06", "📌 Important Clauses", "\n".join(["• " + x for x in pretty_items(clauses)]) if clauses else "No supported clause detected."),
            ("07", "📋 Obligations", "\n".join(["• " + x for x in pretty_items(obligations)]) if obligations else "Not stated in the document."),
            ("08", "⚠️ Risk Indicators", "\n".join(["• " + x for x in pretty_items(risk_indicators)]) if risk_indicators else "No configured risk indicators detected."),
            ("09", "📊 Risk Score", f"{risk_score}/100 — {risk_level}"),
            ("10", "💡 Recommendations", "\n".join(["• " + x for x in pretty_items(recommendations)]) if recommendations else "No recommendations generated."),
        ]
        for num, title, body in point_items:
            with st.expander(f"{num}  {title}", expanded=(num == "01")):
                st.write(body)

        st.markdown("<div class='analysis-card'><b>🔎 Evidence & Completeness</b></div>", unsafe_allow_html=True)
        e1, e2 = st.columns(2)
        with e1:
            st.markdown("**Detected Sections**")
            st.write(", ".join(present_sections) if present_sections else "None")
        with e2:
            st.markdown("**Not Detected**")
            st.write(", ".join(missing_sections) if missing_sections else "None")

        tabs = st.tabs(["📝 Summary", "📌 Clauses", "⚠️ Risks", "📋 Obligations", "⚖️ Indian Law", "📄 Extracted Text"])
        with tabs[0]:
            st.write(summary)
            st.markdown("**Parties**")
            st.write("\n".join(["• "+x for x in parties]) if parties else "Not stated in the document.")
            st.markdown("**Important Dates**")
            st.write("\n".join(["• "+x for x in dates]) if dates else "Not stated in the document.")
            st.markdown("**Financial Information**")
            st.write("\n".join(["• "+x for x in money]) if money else "Not stated in the document.")
        with tabs[1]:
            if clause_evidence:
                for name, evidence in clause_evidence.items():
                    st.markdown(f"**{name}**")
                    for e in evidence: st.info(f"Evidence: {e}")
            else: st.info("No supported clause evidence detected.")
        with tabs[2]:
            if risk_details:
                for item in risk_details:
                    st.markdown(f"**{item['name']} — {item['severity']} ({item['points']} points)**")
                    st.write(item['explanation'])
                    for e in item['evidence']: st.caption(f"Evidence: {e}")
            else: st.success("No configured risk indicators detected.")
            st.progress(risk_score / 100)
            st.caption("Risk levels are heuristic document indicators, not legal conclusions.")
        with tabs[3]:
            st.write("\n".join(["• "+x for x in obligations]) if obligations else "Not stated in the document.")
            st.markdown("**Recommendations**")
            for r in recommendations: st.write(f"• {r}")
        with tabs[4]:
            if not st.session_state.get("rag_loaded_for") == st.session_state.get("document_id"):
                try:
                    with st.spinner("⚖️ Loading relevant Indian-law references..."):
                        refs, err = get_law_references(text, top_k=3)
                    st.session_state["law_references"] = refs
                    st.session_state["rag_error"] = err
                    st.session_state["rag_loaded_for"] = st.session_state.get("document_id")
                    law_references, rag_error = refs, err
                except Exception as e:
                    rag_error = f"RAG unavailable: {e}"
            else:
                law_references = st.session_state.get("law_references", [])
                rag_error = st.session_state.get("rag_error", "")
            if rag_error: st.warning(rag_error)
            for ref in law_references:
                st.markdown(f"**{ref['title']}**")
                st.write(ref['explanation'])
                st.caption(f"Semantic similarity: {ref['score']:.3f}")
            st.caption("References come from the project's local legal knowledge base and should be verified against authoritative sources.")
        with tabs[5]:
            st.text_area("Extracted document text", text, height=420, key="analysis_extracted_text")

        with st.expander("📋 Document Completeness Check"):
            st.write("**Detected sections:**")
            st.write(", ".join(present_sections) if present_sections else "None")
            st.write("**Not detected:**")
            st.write(", ".join(missing_sections) if missing_sections else "None")


# =========================================================
# PAGE: ASK LEGAL AI
# =========================================================

elif st.session_state.page == "Ask Legal AI":
    top_home, top_info = st.columns([1,5])
    with top_home:
        if st.button("🏠 Home", type="primary", use_container_width=True, key="ask_home_top"):
            go_to("Home")
    with top_info:
        st.markdown("<div style='padding-top:8px;color:#aebbd0;'>Current document stays available while you move between sections.</div>", unsafe_allow_html=True)
    st.markdown("<div class='panel'><div class='panel-title'>💬 Ask Your Legal AI</div><div class='panel-sub'>Ask questions grounded in the uploaded document and local Indian-law retrieval context.</div></div>", unsafe_allow_html=True)
    if not text:
        st.info("Upload a document from Home first.")
        if st.button("🏠 Go to Home", type="primary"): go_to("Home")
    else:
        st.markdown(f"**Current document:** {file_name}")
        question = st.text_input("Your question", value=st.session_state.get("legal_question", ""), placeholder="Example: What is the termination notice period?", key="qa_input")
        if st.button("💬 Ask Legal AI", type="primary", use_container_width=True):
            if not question.strip():
                st.warning("Please enter a question first.")
            else:
                st.session_state.legal_question = question.strip()
                try:
                    with st.spinner("🤖 Finding the answer..."):
                        st.session_state.qa_answer = run_legal_question(text, question.strip())
                except Exception as e:
                    st.error("❌ Could not answer the question.")
                    st.code(str(e))
        if st.session_state.get("qa_answer"):
            st.markdown("<div class='analysis-card'><h3>💡 AI Answer</h3></div>", unsafe_allow_html=True)
            st.write(st.session_state.get("qa_answer"))
        with st.expander("🧩 How Your Legal AI Works"):
            st.code("Document → Text Extraction → Your Fine-Tuned Legal AI\n                         ↓\n              Summary / Explanation\n                         ↓\nClause Detection + Risk Indicators + Evidence\n                         ↓\n               Sentence Embeddings\n                         ↓\n              FAISS Indian-Law RAG\n                         ↓\n          Grounded Legal Information")


# =========================================================
# PAGE: INDIAN LAW SEARCH
# =========================================================

elif st.session_state.page == "Indian Law Search":
    st.markdown("<div class='panel'><div class='panel-title'>⚖️ Indian Law Search</div><div class='panel-sub'>Semantic search across the project's local Indian legal knowledge base.</div></div>", unsafe_allow_html=True)
    query = st.text_input("Search legal topic", placeholder="Example: penalty for breach of contract")
    if st.button("🔎 Search Indian Law", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Enter a legal topic first.")
        else:
            try:
                from models.rag_search import search_law
                results = search_law(query.strip(), top_k=5)
                if results:
                    for result in results:
                        section = str(result.get("section", "")).strip()
                        lines = [x.strip() for x in section.splitlines() if x.strip()]
                        title = lines[0] if lines else "Indian-law reference"
                        body = " ".join(lines[1:]) if len(lines) > 1 else section
                        st.markdown(f"<div class='analysis-card'><h4>⚖️ {title}</h4><p>{body}</p><small>Semantic similarity: {float(result.get('score',0)):.3f}</small></div>", unsafe_allow_html=True)
                else:
                    st.info("No local law reference found.")
            except Exception as e:
                st.error(f"Indian-law search unavailable: {e}")
    if text:
        st.divider()
        st.markdown("**Suggested from current document:**")
        st.write(", ".join(legal_terms) if legal_terms else "No legal terms detected.")


# =========================================================
# PAGE: REPORTS
# =========================================================

elif st.session_state.page == "Reports":
    st.markdown("<div class='panel'><div class='panel-title'>📑 Professional Reports</div><div class='panel-sub'>Export your current analysis as PDF or TXT.</div></div>", unsafe_allow_html=True)
    if not text:
        st.info("Upload a document from Home first.")
        if st.button("🏠 Go to Home", type="primary"): go_to("Home")
    else:
        ai_report = st.session_state.get("ai_result") or "AI analysis was not run."
        report = f"""AI LEGAL DOCUMENT SIMPLIFIER\n=============================\n\nDocument Name: {file_name}\nGenerated: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\nDocument Type: {document_type}\n\nYOUR LEGAL AI ANALYSIS\n----------------------\n{ai_report}\n\nCLAUSE EVIDENCE\n---------------\n{chr(10).join([clause + ': ' + ' | '.join(items) for clause, items in clause_evidence.items()]) if clause_evidence else 'No clause evidence detected.'}\n\nRISK ANALYSIS\n-------------\n{chr(10).join([item['name'] + ' - ' + item['severity'] + ': ' + item['explanation'] for item in risk_details]) if risk_details else 'No risk indicators detected.'}\n\nINDIAN LAW RAG REFERENCES\n--------------------------\n{chr(10).join([ref['title'] + ' (similarity ' + f"{ref['score']:.3f}" + '): ' + ref['explanation'] for ref in law_references]) if law_references else 'No local Indian-law reference retrieved.'}\n\nSUMMARY\n-------\n{summary}\n\nPARTIES\n-------\n{chr(10).join(parties) if parties else 'No parties detected.'}\n\nDATES\n-----\n{chr(10).join(dates) if dates else 'No dates detected.'}\n\nFINANCIAL INFORMATION\n----------------------\n{chr(10).join(money) if money else 'No financial amounts detected.'}\n\nLEGAL TERMS\n-----------\n{chr(10).join(legal_terms) if legal_terms else 'No legal terms detected.'}\n\nCLAUSES\n-------\n{chr(10).join(clauses) if clauses else 'No important clauses detected.'}\n\nOBLIGATIONS\n-----------\n{chr(10).join(obligations) if obligations else 'No obligations detected.'}\n\nRISK SCORE\n----------\n{risk_score}/100 — {risk_level}\n\nRECOMMENDATIONS\n---------------\n{chr(10).join(recommendations)}\n\nDISCLAIMER\n----------\nEducational and informational use only; not a substitute for professional legal advice.\n"""

        def create_pdf_report():
            from io import BytesIO
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle("ReportTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=18, leading=22, spaceAfter=14)
            heading_style = ParagraphStyle("ReportHeading", parent=styles["Heading2"], fontSize=12, leading=15, spaceBefore=10, spaceAfter=6)
            body_style = ParagraphStyle("ReportBody", parent=styles["BodyText"], fontSize=9.5, leading=13, spaceAfter=5)
            story = [Paragraph("AI Legal Document Simplifier", title_style), Paragraph(f"<b>Document:</b> {escape(file_name)}", body_style), Paragraph(f"<b>Document Type:</b> {escape(document_type)}", body_style), Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}", body_style), Spacer(1,8)]
            sections = [("Your Legal AI Analysis", ai_report), ("Clause Detection + Evidence", "\n".join([c+": "+" | ".join(items) for c,items in clause_evidence.items()]) if clause_evidence else "No clause evidence detected."), ("Risk Analysis + Evidence", "\n".join([i['name']+" - "+i['severity']+": "+i['explanation'] for i in risk_details]) if risk_details else "No risk indicators detected."), ("Indian Law RAG References", "\n".join([r['title']+" (similarity "+f"{r['score']:.3f}"+"): "+r['explanation'] for r in law_references]) if law_references else "No local Indian-law reference retrieved."), ("Simple Summary", summary), ("Parties", "\n".join(parties) if parties else "No parties detected."), ("Important Dates", "\n".join(dates) if dates else "No dates detected."), ("Financial Information", "\n".join(money) if money else "No financial amounts detected."), ("Legal Terms", ", ".join(legal_terms) if legal_terms else "No legal terms detected."), ("Important Clauses", "\n".join(clauses) if clauses else "No important clauses detected."), ("Obligations", "\n".join(obligations) if obligations else "No obligations detected."), ("Risk Score", f"Risk Score: {risk_score}/100 | Risk Level: {risk_level}"), ("Recommendations", "\n".join(recommendations))]
            for heading, content in sections:
                story.append(Paragraph(escape(heading), heading_style))
                safe = escape(str(content).replace("₹", "Rs. ")).replace("\n", "<br/>")
                story.append(Paragraph(safe, body_style))
            story.append(PageBreak())
            story.append(Paragraph("Disclaimer", heading_style))
            story.append(Paragraph("This report provides automated analysis for educational and informational purposes only. It is not a substitute for professional legal advice.", body_style))
            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()

        c1,c2 = st.columns(2)
        with c1:
            st.download_button("📄 Download PDF Report", create_pdf_report(), "legal_analysis_report.pdf", "application/pdf", type="primary", use_container_width=True)
        with c2:
            st.download_button("📥 Download TXT Report", report, "legal_analysis_report.txt", "text/plain", use_container_width=True)
        st.markdown("<div class='analysis-card'><b>Report contents:</b> AI analysis, clause evidence, risk findings, Indian-law references, document summary, obligations and recommendations.</div>", unsafe_allow_html=True)


# =========================================================
# PAGE: ABOUT
# =========================================================

elif st.session_state.page == "About the Project":
    st.markdown("<div class='panel'><div class='panel-title'>ℹ️ About LegalDoc AI</div><div class='panel-sub'>A domain-specific Indian legal document analysis workspace.</div></div>", unsafe_allow_html=True)
    a,b = st.columns(2)
    with a:
        st.markdown("### 🎯 Project Goal")
        st.write("The system helps users understand uploaded legal documents by extracting text, identifying important clauses and obligations, detecting document-grounded risk indicators, retrieving relevant Indian-law references and providing a simple explanation.")
        st.markdown("### 🧠 AI Pipeline")
        st.code("Upload → PDF/DOCX/OCR → Text → Rule Analysis + Your Legal AI → Embeddings → FAISS → Indian Law RAG → Grounded Results → Report")
    with b:
        st.markdown("### 🧩 Main Components")
        for item in ["PDF / DOCX text extraction", "EasyOCR for scanned images", "Own domain-adapted Legal AI", "Sentence-transformer embeddings", "FAISS semantic search", "Indian legal knowledge base", "Rule-based risk and clause evidence", "Streamlit dashboard", "PDF / TXT reports"]:
            st.write(f"✅ {item}")
        st.markdown("### 🔒 Privacy")
        st.write("The application is designed for temporary, session-based document processing and does not intentionally save uploaded documents to permanent storage. Users can clear the current document and session from the sidebar.")
    st.warning("Educational and informational tool only. It does not replace advice from a qualified legal professional.")


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
  <div>🛡️ <b style="color:#45e0a1;">Privacy Protected</b> &nbsp; | &nbsp; Documents are processed temporarily and can be cleared with one click.</div>
  <div>🏛️ Built for an Indian legal context &nbsp; 🇮🇳 &nbsp; LEGAL AWARENESS • GREATER ACCESS • STRONGER SOCIETY</div>
</div>
""", unsafe_allow_html=True)
