import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import yaml

from . import config

FRONTMATTER_PATTERN = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)
HEADING_PATTERN = re.compile(r"^##\s+(.*)$", re.MULTILINE)
TITLE_PATTERN = re.compile(r"^#\s+.*$", re.MULTILINE)


@dataclass
class Chunk:
    chunk_id: str
    source_id: str
    source_type: str
    title: str
    section: str
    status: str
    text: str


def _split_sections(body: str) -> List[Tuple[str, str]]:
    body = TITLE_PATTERN.sub("", body, count=1)
    positions = [(match.start(), match.group(1).strip()) for match in HEADING_PATTERN.finditer(body)]

    if not positions:
        return [("Overview", body.strip())]

    sections: List[Tuple[str, str]] = []
    if positions[0][0] > 0:
        intro = body[: positions[0][0]].strip()
        if intro:
            sections.append(("Overview", intro))

    for index, (start, heading) in enumerate(positions):
        end = positions[index + 1][0] if index + 1 < len(positions) else len(body)
        section_text = body[start:end]
        section_text = HEADING_PATTERN.sub("", section_text, count=1).strip()
        sections.append((heading, section_text))

    return sections


def _parse_markdown(path: Path) -> List[Chunk]:
    raw = path.read_text(encoding="utf-8")
    match = FRONTMATTER_PATTERN.match(raw)
    if not match:
        raise ValueError(f"Missing frontmatter block in {path}")

    front_raw, body = match.groups()
    meta = yaml.safe_load(front_raw)
    document_id = meta["document_id"]
    title = meta["title"]
    status = meta.get("status", "current")

    chunks: List[Chunk] = []
    for index, (heading, text) in enumerate(_split_sections(body)):
        cleaned = text.strip()
        if not cleaned:
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{document_id}#{index}",
                source_id=document_id,
                source_type="knowledge_base",
                title=title,
                section=heading,
                status=status,
                text=cleaned,
            )
        )
    return chunks


def load_knowledge_base_chunks() -> List[Chunk]:
    chunks: List[Chunk] = []
    for path in sorted(config.KB_DIR.glob("*.md")):
        chunks.extend(_parse_markdown(path))
    return chunks


def load_resolved_case_chunks() -> List[Chunk]:
    payload = json.loads(config.RESOLVED_CASES_PATH.read_text(encoding="utf-8"))
    chunks: List[Chunk] = []
    for case in payload["cases"]:
        lines = [f"Title: {case['title']}", f"Status: {case['status']}"]
        if case.get("symptoms"):
            lines.append("Symptoms: " + "; ".join(case["symptoms"]))
        if case.get("resolution"):
            lines.append("Resolution steps: " + "; ".join(case["resolution"]))
        if case.get("important_limit"):
            lines.append("Important limit: " + case["important_limit"])
        if case.get("superseded_reason"):
            lines.append("Superseded reason: " + case["superseded_reason"])

        chunks.append(
            Chunk(
                chunk_id=case["case_id"],
                source_id=case["case_id"],
                source_type="resolved_case",
                title=case["title"],
                section=case["status"],
                status=case["status"],
                text="\n".join(lines),
            )
        )
    return chunks


def load_all_chunks() -> List[Chunk]:
    return load_knowledge_base_chunks() + load_resolved_case_chunks()
