

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any, Iterable

import fitz
from bs4 import BeautifulSoup


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PDF_DIR = DATA_DIR / "pdfs"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_FILE = PROCESSED_DIR / "knowledge_base.json"


FACULTY_RE = re.compile(
    r"(?P<name>Dr\.|Prof\.|Mr\.|Ms\.|Mrs\.)\s+[A-Z][^\n]{2,80}?\n"
    r"(?P<role>[^\n]{3,120})\n"
    r"(?:Qualification:\n(?P<qualification>.*?))?"
    r"(?:Area of Interest:\n(?P<interest>.*?))?"
    r"(?:Email:\n(?P<email>[^\n]+))?",
    re.DOTALL,
)

BOILERPLATE_LINES = {
    "apply now",
    "notice board",
    "news",
    "events",
    "alumni",
    "qec",
    "oric",
    "cdc",
    "contacts",
    "sign up",
    "login",
    "logout",
    "main menu",
    "home",
    "about us",
    "director’s message",
    "director’s secretariat",
    "administration",
    "location",
    "governance",
    "board of governors",
    "board of trustees",
    "executive committee",
    "academic committee",
    "selection committee",
    "academics",
    "undergraduate programs",
    "graduate programs",
    "certificates and diplomas",
    "academic centers/schools",
    "academic calendar",
    "r&dd calendar",
    "faculty & research",
    "faculty",
    "journals",
    "centers",
    "r&dd",
    "books/book chapters",
    "conferences",
    "visiting faculty",
    "admissions",
    "admission open",
    "scholarships",
    "semester rules",
    "admission policy",
    "student support",
    "academic linkages (aleo)",
    "career development center",
    "societies / clubs",
    "lincoln corner peshawar",
    "facilities",
    "event gallery",
    "psychological wellness",
    "center",
    "students handbook",
}

BOILERPLATE_PHRASES = sorted(BOILERPLATE_LINES, key=len, reverse=True)
BOILERPLATE_WORDS = {
    word
    for phrase in BOILERPLATE_PHRASES
    for word in re.findall(r"[a-z]+", phrase)
}


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def normalize_whitespace(text: str) -> str:
    text = unescape(text or "")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_line(text: str) -> str:
    text = normalize_whitespace(text)
    text = text.replace("â", "’").replace("â", "–").replace("â", "—")
    return text


def is_boilerplate_line(line: str) -> bool:
    lowered = normalize_line(line).lower()
    if lowered in BOILERPLATE_LINES:
        return True
    if not lowered:
        return False

    phrase_hits = sum(1 for phrase in BOILERPLATE_PHRASES if phrase in lowered)
    if phrase_hits >= 4:
        return True

    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?|&|/", lowered)
    if len(tokens) >= 6:
        token_hits = sum(1 for token in tokens if token in {word for phrase in BOILERPLATE_PHRASES for word in phrase.split()})
        if token_hits / max(len(tokens), 1) >= 0.7:
            return True

    word_tokens = re.findall(r"[a-z]+", lowered)
    if len(word_tokens) >= 3 and all(token in BOILERPLATE_WORDS for token in word_tokens):
        return True

    return False


def clean_html(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup.find_all(["script", "style", "nav", "header", "footer", "aside", "form", "noscript", "svg", "iframe"]):
        tag.decompose()
    main = soup.find("main") or soup.find(id="content") or soup.find(id="main") or soup.find("article") or soup.body or soup

    preferred_tags = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "th", "td"]
    lines: list[str] = []

    for tag in main.find_all(preferred_tags):
        pieces = [part.strip() for part in tag.stripped_strings if part.strip()]
        if not pieces:
            continue
        line = normalize_line(" ".join(pieces))
        if not line:
            continue
        if is_boilerplate_line(line):
            continue
        if re.fullmatch(r"[\d\W_]+", line):
            continue
        lines.append(line)

    if not lines:
        text = main.get_text(separator="\n")
        lines = [normalize_line(line) for line in text.splitlines() if normalize_line(line)]

    compacted: list[str] = []
    previous = None
    for line in lines:
        if not line or line == previous:
            continue
        if is_boilerplate_line(line):
            continue
        compacted.append(line)
        previous = line

    return normalize_whitespace("\n".join(compacted))


def extract_pdf_text(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    pages: list[str] = []
    for page in doc:
        pages.append(page.get_text("text"))
    return normalize_whitespace("\n\n".join(pages))


def stable_id(*parts: str) -> str:
    joined = "||".join(parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


def detect_type(record: dict[str, Any], cleaned_text: str) -> str:
    section = (record.get("section") or "").lower()
    if section == "faculty-directory" or "faculty" in record.get("url", ""):
        return "faculty_entry"
    if record.get("url", "").lower().endswith(".pdf"):
        return "pdf"
    if "contact" in section or "contact us" in record.get("title", "").lower():
        return "page"
    if len(cleaned_text.splitlines()) <= 3 and any(token in cleaned_text.lower() for token in ["phone", "email", "ext", "position"]):
        return "contact_entry"
    return "page"


def parse_faculty_entry(text: str, source_url: str, title: str, scraped_at: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    matches = list(FACULTY_RE.finditer(text))
    if not matches:
        return entries

    for match in matches:
        name = normalize_whitespace(match.group("name"))
        role = normalize_whitespace(match.group("role"))
        qualification = normalize_whitespace(match.group("qualification") or "")
        interest = normalize_whitespace(match.group("interest") or "")
        email = normalize_whitespace(match.group("email") or "")
        parts = [f"Name: {name}", f"Role: {role}"]
        if qualification:
            parts.append(f"Qualification: {qualification}")
        if interest:
            parts.append(f"Area of Interest: {interest}")
        if email:
            parts.append(f"Email: {email}")
        entry_text = "\n".join(parts)
        entries.append({
            "id": stable_id(source_url, name, role),
            "source_url": source_url,
            "title": title,
            "type": "faculty_entry",
            "text": entry_text,
            "metadata": {
                "name": name,
                "role": role,
                "qualification": qualification,
                "area_of_interest": interest,
                "email": email,
            },
            "scraped_at": scraped_at,
        })
    return entries


def parse_contact_blocks(text: str, source_url: str, title: str, scraped_at: str) -> list[dict[str, Any]]:
    normalized = normalize_whitespace(text)
    if "director's secretariat" not in normalized.lower() and "joint director office" not in normalized.lower():
        return []

    entries: list[dict[str, Any]] = []
    lines = [line for line in normalized.splitlines() if line and line.lower() not in BOILERPLATE_LINES]
    current_block: list[str] = []

    def flush_block(block_lines: list[str]) -> None:
        if len(block_lines) < 2:
            return
        block_text = normalize_whitespace("\n".join(block_lines))
        if len(block_text) < 30:
            return
        entries.append({
            "id": stable_id(source_url, block_text),
            "source_url": source_url,
            "title": title,
            "type": "contact_entry",
            "text": block_text,
            "metadata": {},
            "scraped_at": scraped_at,
        })

    for line in lines:
        if re.match(r"^(Director's Secretariat|Joint Director Office)$", line):
            flush_block(current_block)
            current_block = [line]
            continue
        if line in {"Name", "Position", "Phone", "Ext", "Email"} and current_block:
            current_block.append(line)
            continue
        if current_block:
            current_block.append(line)
            if len(current_block) > 8 and re.match(r"^(Dr\.|Prof\.|Mr\.|Ms\.|Mrs\.)\s", line):
                flush_block(current_block)
                current_block = []
        else:
            current_block = [line]

    flush_block(current_block)
    return entries


def parse_pdf_metadata(pdf_name: str) -> dict[str, Any]:
    return {
        "filename": pdf_name,
        "type": "pdf",
    }


def build_page_entry(record: dict[str, Any], cleaned_text: str) -> dict[str, Any]:
    body_text = cleaned_text
    lines = [line for line in body_text.splitlines() if line]
    if lines:
        body_text = normalize_whitespace("\n".join(lines))
    return {
        "id": stable_id(record["url"], record.get("title", ""), record.get("section", "")),
        "source_url": record["url"],
        "title": record.get("title", ""),
        "type": detect_type(record, cleaned_text),
        "text": body_text,
        "metadata": {
            "section": record.get("section", ""),
            "depth": record.get("depth", 0),
        },
        "scraped_at": record.get("scraped_at", datetime.now(timezone.utc).isoformat()),
    }


def build_pdf_entry(pdf_path: Path, source_url: str | None, scraped_at: str | None) -> dict[str, Any] | None:
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        return None
    text = extract_pdf_text(pdf_path)
    if not text:
        return None
    return {
        "id": stable_id(str(pdf_path), text[:200]),
        "source_url": source_url or pdf_path.as_posix(),
        "title": pdf_path.stem.replace("_", " "),
        "type": "pdf",
        "text": text,
        "metadata": parse_pdf_metadata(pdf_path.name),
        "scraped_at": scraped_at or datetime.now(timezone.utc).isoformat(),
    }


def dedupe_entries(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for entry in entries:
        fingerprint = re.sub(r"\s+", " ", entry["text"].lower()).strip()
        fingerprint = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped.append(entry)
    return deduped


def load_raw_pages() -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    for path in sorted(RAW_DIR.glob("*.json")):
        try:
            pages.append(read_json(path))
        except json.JSONDecodeError:
            continue
    return pages


def load_pdf_index() -> list[dict[str, Any]]:
    index_path = DATA_DIR / "pdf_index.json"
    if index_path.exists():
        return read_json(index_path)
    return []


def main() -> None:
    ensure_dirs()

    entries: list[dict[str, Any]] = []
    for record in load_raw_pages():
        cleaned_text = clean_html(record.get("raw_html", ""))
        if not cleaned_text:
            cleaned_text = normalize_whitespace(record.get("text", ""))
        if not cleaned_text:
            continue

        entries.append(build_page_entry(record, cleaned_text))

        if record.get("section") == "faculty-directory" or "faculty" in record.get("url", ""):
            entries.extend(parse_faculty_entry(cleaned_text, record["url"], record.get("title", ""), record.get("scraped_at", "")))

        entries.extend(parse_contact_blocks(cleaned_text, record["url"], record.get("title", ""), record.get("scraped_at", "")))

    for item in load_pdf_index():
        local_path = item.get("local_path", "")
        candidate = BASE_DIR / local_path if not Path(local_path).is_absolute() else Path(local_path)
        pdf_entry = build_pdf_entry(candidate, item.get("source_page"), item.get("discovered_at"))
        if pdf_entry:
            entries.append(pdf_entry)

    entries = dedupe_entries(entries)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entry_count": len(entries),
        "entries": entries,
    }
    write_json(OUTPUT_FILE, payload)
    print(f"Wrote {len(entries)} entries to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
