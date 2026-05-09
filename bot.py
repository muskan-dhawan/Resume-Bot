import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

from dotenv import load_dotenv
import fitz
import google.generativeai as genai
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# Load environment variables from .env file
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

if not TELEGRAM_TOKEN:
    raise RuntimeError(
        "❌ Missing TELEGRAM_TOKEN. Please create a .env file (copy from .env.example) and add your token."
    )
if not GEMINI_API_KEY:
    raise RuntimeError(
        "❌ Missing GEMINI_API_KEY. Please create a .env file (copy from .env.example) and add your API key."
    )

genai.configure(api_key=GEMINI_API_KEY)

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

user_data_store = {}
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "").strip()


def extract_text(pdf_path: str) -> str:
    with fitz.open(pdf_path) as doc:
        return "\n".join(page.get_text("text") for page in doc).strip()


def resolve_model_name() -> str:
    if GEMINI_MODEL:
        return GEMINI_MODEL

    preferred_names = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    ]

    try:
        available = []
        for model in genai.list_models():
            methods = getattr(model, "supported_generation_methods", []) or []
            if "generateContent" in methods:
                name = getattr(model, "name", "")
                if name:
                    available.append(name.replace("models/", ""))

        for preferred in preferred_names:
            if preferred in available:
                return preferred

        flash_candidates = [name for name in available if "flash" in name.lower()]
        if flash_candidates:
            return flash_candidates[0]

        if available:
            return available[0]

    except Exception as exc:
        logger.warning("Unable to list models dynamically: %s", exc)

    return "gemini-2.0-flash"


def _strip_json_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return cleaned


def check_text_relevance(text: str) -> bool:
    prompt = f"""
Analyze the following text provided by a user.
1. Does it contain vulgar, offensive, or highly inappropriate language?
2. Is it completely random and entirely unrelated to professional employment, jobs, or resumes?

If EITHER is true, respond with exactly "INVALID".
Otherwise, if it represents acceptable professional text, respond with exactly "VALID".

Text to analyze:
{text[:2000]}
"""
    model_name = resolve_model_name()
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
    except Exception as first_exc:
        logger.warning("Relevance check primary model failed (%s): %s", model_name, first_exc)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)

    result = (response.text or "").strip().upper()
    return "INVALID" not in result


def analyze_resume(jd: str, resume: str) -> dict:
    prompt = f"""
You are an expert ATS resume reviewer.
Return ONLY valid JSON with this exact schema:
{{
  "match_score": <integer 0-100>,
    "score_breakdown": {{
        "skills_match": <integer 0-100>,
        "experience_relevance": <integer 0-100>,
        "education_fit": <integer 0-100>,
        "keyword_alignment": <integer 0-100>,
        "format_ats_readiness": <integer 0-100>
    }},
  "missing_skills": ["..."],
  "weak_areas": ["..."],
  "suggestions": ["..."],
  "improved_resume": "..."
}}

IMPORTANT: Format the improved_resume with:
- Section headers in UPPERCASE (e.g., EXPERIENCE, EDUCATION, SKILLS)
- Bullet points starting with "-"
- No extra formatting or markdown

Job Description:
{jd}

Resume:
{resume}
"""

    model_name = resolve_model_name()
    logger.info("Using Gemini model: %s", model_name)

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
    except Exception as first_exc:
        logger.warning("Primary model failed (%s): %s", model_name, first_exc)
        fallback_model = "gemini-2.0-flash"
        model = genai.GenerativeModel(fallback_model)
        response = model.generate_content(prompt)
    raw_text = (response.text or "").strip()
    payload = _strip_json_fence(raw_text)

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        logger.warning("Gemini returned non-JSON output. Falling back.")
        data = {
            "match_score": None,
            "score_breakdown": {},
            "missing_skills": [],
            "weak_areas": [],
            "suggestions": [],
            "improved_resume": "",
            "raw_output": raw_text,
        }

    data.setdefault("match_score", None)
    data.setdefault("score_breakdown", {})
    data.setdefault("missing_skills", [])
    data.setdefault("weak_areas", [])
    data.setdefault("suggestions", [])
    data.setdefault("improved_resume", "")
    return data


def answer_follow_up(jd: str, resume: str, analysis: dict, question: str) -> str:
    prompt = f"""
You are a friendly resume coach in a Telegram chat.
Answer the user's follow-up question using ONLY this context:

Job Description:
{jd}

Resume:
{resume[:12000]}

Previous Analysis JSON:
{json.dumps(analysis, ensure_ascii=False)}

User Follow-up Question:
{question}

Rules:
- Keep response concise and practical.
- If user asks for steps, provide bullet points.
- If the question is outside context, say it briefly and ask them to send /reset for new JD/resume.
"""

    model_name = resolve_model_name()
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
    except Exception as first_exc:
        logger.warning("Follow-up primary model failed (%s): %s", model_name, first_exc)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)

    return (response.text or "I couldn't generate a follow-up answer right now.").strip()


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def _extract_profile_fields(resume_text: str, improved_text: str) -> dict:
    source = f"{resume_text}\n{improved_text}"
    lines = [line.strip() for line in source.splitlines() if line.strip()]

    name = "Candidate Name"
    for line in lines[:30]:
        if re.fullmatch(r"[A-Za-z][A-Za-z .'-]{2,60}", line) and len(line.split()) <= 5:
            name = line
            break

    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", source)
    phone_match = re.search(r"(?:\+?\d[\d\-\s()]{8,}\d)", source)
    linkedin_match = re.search(r"(?:https?://)?(?:www\.)?linkedin\.com/[^\s)]+", source, flags=re.IGNORECASE)
    github_match = re.search(r"(?:https?://)?(?:www\.)?github\.com/[^\s)]+", source, flags=re.IGNORECASE)

    return {
        "name": name,
        "email": email_match.group(0) if email_match else "",
        "phone": phone_match.group(0) if phone_match else "",
        "linkedin": linkedin_match.group(0) if linkedin_match else "",
        "github": github_match.group(0) if github_match else "",
    }


def _parse_sections_from_improved_text(text: str) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    current_title = "Summary"
    current_items: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        is_header = (
            line.isupper()
            and len(line) <= 45
            and any(ch.isalpha() for ch in line)
            and not line.startswith("-")
        )

        if is_header:
            if current_items:
                sections.append((current_title.title(), current_items))
                current_items = []
            current_title = line
            continue

        if line.startswith("-"):
            current_items.append(line[1:].strip())
        else:
            current_items.append(line)

    if current_items:
        sections.append((current_title.title(), current_items))

    if not sections:
        sections = [("Summary", [text.strip() or "No improved resume generated."])]

    return sections


def _build_latex_resume(resume_text: str, improved_text: str) -> str:
    profile = _extract_profile_fields(resume_text, improved_text)
    sections = _parse_sections_from_improved_text(improved_text)

    contact_parts = []
    if profile["phone"]:
        contact_parts.append(_latex_escape(profile["phone"]))
    if profile["email"]:
        email = profile["email"]
        contact_parts.append(rf"\href{{mailto:{email}}}{{\underline{{{_latex_escape(email)}}}}}")
    if profile["linkedin"]:
        linkedin = profile["linkedin"] if profile["linkedin"].startswith("http") else f"https://{profile['linkedin']}"
        contact_parts.append(rf"\href{{{linkedin}}}{{\underline{{{_latex_escape(profile['linkedin'])}}}}}")
    if profile["github"]:
        github = profile["github"] if profile["github"].startswith("http") else f"https://{profile['github']}"
        contact_parts.append(rf"\href{{{github}}}{{\underline{{{_latex_escape(profile['github'])}}}}}")

    contact_line = " ~ ".join(contact_parts) if contact_parts else ""

    sections_latex = []
    for title, items in sections:
        safe_title = _latex_escape(title)
        bullet_lines = []
        for item in items:
            safe_item = _latex_escape(item)
            if safe_item:
                bullet_lines.append(rf"\item{{{safe_item}}}")

        if not bullet_lines:
            continue

        block = (
            rf"\section{{{safe_title}}}" + "\n"
            r"\resumeSubHeadingListStart" + "\n"
            + "\n".join(bullet_lines)
            + "\n"
            + r"\resumeSubHeadingListEnd"
            + "\n"
            + r"\vspace{-8pt}"
        )
        sections_latex.append(block)

    sections_body = "\n\n".join(sections_latex)
    template = r"""
\documentclass[letterpaper,11pt]{article}
\usepackage{latexsym}
\usepackage[empty]{fullpage}
\usepackage{titlesec}
\usepackage[usenames,dvipsnames]{color}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\usepackage{fancyhdr}
\usepackage[english]{babel}
\usepackage{tabularx}

\pagestyle{fancy}
\fancyhf{}
\fancyfoot{}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

\addtolength{\oddsidemargin}{-0.6in}
\addtolength{\evensidemargin}{-0.5in}
\addtolength{\textwidth}{1.19in}
\addtolength{\topmargin}{-.7in}
\addtolength{\textheight}{1.4in}

\urlstyle{same}
\raggedbottom
\raggedright
\setlength{\tabcolsep}{0in}

	itleformat{\section}{
  \vspace{-6pt}\scshape\raggedright\large\bfseries
}{}{0em}{}[\color{black}\titlerule \vspace{-3pt}]

\newcommand{\resumeSubHeadingListStart}{\begin{itemize}[leftmargin=*]}
\newcommand{\resumeSubHeadingListEnd}{\end{itemize}}

\begin{document}

\begin{center}
    {\Huge \scshape __NAME__} \\ \vspace{2pt}
    \small __CONTACT__
    \vspace{-6pt}
\end{center}

__SECTIONS__

\end{document}
""".strip()

    return (
        template.replace("__NAME__", _latex_escape(profile["name"]))
        .replace("__CONTACT__", contact_line)
        .replace("__SECTIONS__", sections_body)
    )


def _fallback_create_text_pdf(text: str, output_path: str) -> None:
    """Fallback non-LaTeX PDF generation when pdflatex is unavailable."""
    if not text.strip():
        text = "No improved resume generated."

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )

    styles = getSampleStyleSheet()
    story = []

    heading_style = ParagraphStyle(
        "ResumeHeading",
        parent=styles["Heading1"],
        fontSize=18,
        spaceAfter=10,
        alignment=TA_LEFT,
        fontName="Helvetica-Bold",
        textColor="#111111",
    )

    section_style = ParagraphStyle(
        "CustomSection",
        parent=styles["Heading2"],
        fontSize=11.5,
        textColor="#111111",
        spaceAfter=3,
        spaceBefore=7,
        alignment=TA_LEFT,
        fontName="Helvetica-Bold",
        borderWidth=0,
    )

    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["BodyText"],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=3,
        leading=12,
    )

    bullet_style = ParagraphStyle(
        "BulletBody",
        parent=body_style,
        leftIndent=10,
        firstLineIndent=-6,
    )

    lines = text.strip().split("\n")
    added_heading = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 0.05 * inch))
            continue

        is_section = (
            stripped.isupper()
            and len(stripped) < 40
            and any(c.isalpha() for c in stripped)
            and not stripped.startswith("-")
        )

        if not added_heading and not is_section and len(stripped.split()) <= 8:
            story.append(Paragraph(stripped, heading_style))
            added_heading = True
        elif is_section:
            story.append(Paragraph(stripped, section_style))
        elif stripped.startswith("-"):
            bullet_text = stripped[1:].strip()
            story.append(Paragraph(f"• {bullet_text}", bullet_style))
        else:
            story.append(Paragraph(stripped, body_style))

    doc.build(story)


def create_resume_pdf(resume_text: str, improved_text: str, output_path: str) -> None:
    """Generate PDF using LaTeX template, with fallback if compiler unavailable."""
    if not improved_text.strip():
        improved_text = "No improved resume generated."

    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        logger.warning("pdflatex not found. Using fallback PDF renderer.")
        _fallback_create_text_pdf(improved_text, output_path)
        return

    output_file = Path(output_path)
    work_dir = output_file.parent
    tex_file = work_dir / f"{output_file.stem}.tex"
    latex_content = _build_latex_resume(resume_text, improved_text)
    tex_file.write_text(latex_content, encoding="utf-8")

    try:
        result = subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_file.name],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not output_file.exists():
            logger.warning("LaTeX compile failed. stderr: %s", result.stderr[-1000:])
            _fallback_create_text_pdf(improved_text, output_path)
    finally:
        for ext in (".aux", ".log", ".out"):
            side_file = work_dir / f"{output_file.stem}{ext}"
            if side_file.exists():
                side_file.unlink(missing_ok=True)


def format_analysis(result: dict) -> str:
    if result.get("raw_output"):
        return (
            "⚠️ Structured parsing failed. Showing raw analysis:\n\n"
            f"{result['raw_output']}"
        )

    score = result.get("match_score")
    breakdown = result.get("score_breakdown", {}) or {}
    missing = result.get("missing_skills", [])
    weak = result.get("weak_areas", [])
    suggestions = result.get("suggestions", [])

    skill_score = breakdown.get("skills_match", "N/A")
    exp_score = breakdown.get("experience_relevance", "N/A")
    edu_score = breakdown.get("education_fit", "N/A")
    keyword_score = breakdown.get("keyword_alignment", "N/A")
    ats_score = breakdown.get("format_ats_readiness", "N/A")

    lines = [
        f"✅ Match Score: {score if score is not None else 'N/A'}/100",
        "",
        "📊 Score Breakdown:",
        f"- Skills Match: {skill_score}/100",
        f"- Experience Relevance: {exp_score}/100",
        f"- Education Fit: {edu_score}/100",
        f"- Keyword Alignment: {keyword_score}/100",
        f"- ATS Format Readiness: {ats_score}/100",
        "",
        "📌 Missing Skills:",
        *(f"- {item}" for item in (missing or ["None detected"])),
        "",
        "⚠️ Weak Areas:",
        *(f"- {item}" for item in (weak or ["None detected"])),
        "",
        "💡 Suggestions:",
        *(f"- {item}" for item in (suggestions or ["No suggestions returned"])),
    ]
    return "\n".join(lines)


def chunk_text(text: str, max_len: int = 3900) -> list[str]:
    chunks = []
    remaining = text.strip()
    while len(remaining) > max_len:
        cut = remaining.rfind("\n", 0, max_len)
        if cut <= 0:
            cut = max_len
        chunks.append(remaining[:cut])
        remaining = remaining[cut:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Send the Job Description first (text), then send your resume PDF. After analysis, you can ask follow-up questions in chat."
    )


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_data_store.pop(user_id, None)
    await update.message.reply_text("Session reset. Please send a new JD.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    user_id = update.effective_user.id

    if update.message.text:
        text = update.message.text.strip()

        user_state = user_data_store.get(user_id, {})
        has_analysis_context = all(
            key in user_state for key in ("jd", "resume_text", "analysis")
        )

        if has_analysis_context and not text.lower().startswith("jd:"):
            await update.message.reply_text("💬 Follow-up detected. Thinking...")
            try:
                reply = await asyncio.to_thread(
                    answer_follow_up,
                    user_state["jd"],
                    user_state["resume_text"],
                    user_state["analysis"],
                    text,
                )
                for part in chunk_text(reply):
                    await update.message.reply_text(part)
            except Exception as exc:
                logger.exception("Follow-up response failed")
                await update.message.reply_text(f"❌ Could not answer follow-up: {exc}")
            return

        if text.lower().startswith("jd:"):
            text = text[3:].strip()

        if len(text) < 30:
            await update.message.reply_text(
                "❌ This text is too short for a Job Description. Please send a proper JD."
            )
            return

        is_valid = await asyncio.to_thread(check_text_relevance, text)
        if not is_valid:
            await update.message.reply_text(
                "⚠️ Please keep your input professional and strictly related to Job Descriptions."
            )
            return

        user_data_store[user_id] = {"jd": text}
        await update.message.reply_text("✅ JD received. Now send your resume PDF.")
        return

    if update.message.document:
        jd = user_data_store.get(user_id, {}).get("jd", "")
        if not jd:
            await update.message.reply_text("❌ Please send JD first.")
            return

        if not (update.message.document.mime_type or "").lower().endswith("pdf"):
            await update.message.reply_text("❌ Please upload a valid PDF file.")
            return

        input_pdf = Path(f"resume_{user_id}.pdf")
        output_pdf = Path(f"improved_resume_{user_id}.pdf")

        try:
            file = await update.message.document.get_file()
            await file.download_to_drive(str(input_pdf))

            await update.message.reply_text("⏳ Analyzing resume against JD...")

            resume_text = await asyncio.to_thread(extract_text, str(input_pdf))
            result = await asyncio.to_thread(analyze_resume, jd, resume_text)

            analysis_text = format_analysis(result)
            for part in chunk_text(analysis_text):
                await update.message.reply_text(part)

            user_data_store[user_id] = {
                "jd": jd,
                "resume_text": resume_text,
                "analysis": result,
            }

            await update.message.reply_text(
                "💬 You can now ask follow-up questions like: 'Why is keyword score low?' or 'What should I change first?'"
            )

            improved_resume = result.get("improved_resume", "").strip()
            await asyncio.to_thread(create_resume_pdf, resume_text, improved_resume, str(output_pdf))

            with output_pdf.open("rb") as pdf_file:
                await update.message.reply_document(
                    document=pdf_file,
                    filename="improved_resume.pdf",
                    caption="📄 Here is your corrected resume PDF.",
                )

        except Exception as exc:
            logger.exception("Failed to process resume")
            await update.message.reply_text(f"❌ Failed to process file: {exc}")
        finally:
            if input_pdf.exists():
                input_pdf.unlink(missing_ok=True)
            if output_pdf.exists():
                output_pdf.unlink(missing_ok=True)


def main() -> None:
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.Document.PDF, handle_message))
    app.run_polling()


if __name__ == "__main__":
    main()