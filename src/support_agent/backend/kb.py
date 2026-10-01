"""Tiny help-center search (keyword overlap over markdown sections). Good enough for a policy FAQ;
swap for embeddings/vector search without touching the agent - it only sees the tool contract."""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources

_STOP = set(
    "a an the and or of to in on for is are was were be been it its this that my i me we you your our can do does "
    "how what when where why with from at by as if not no yes will would should could have has had get got".split()
)


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP and len(t) > 1}


@dataclass(frozen=True)
class Section:
    article: str
    heading: str
    text: str


class KnowledgeBase:
    def __init__(self, sections: list[Section]):
        self.sections = sections

    @classmethod
    def load_default(cls) -> KnowledgeBase:
        sections: list[Section] = []
        for entry in sorted(resources.files("support_agent.kb").iterdir(), key=lambda p: p.name):
            if not entry.name.endswith(".md"):
                continue
            article, heading, lines = "", "", []
            for line in entry.read_text(encoding="utf-8").splitlines():
                if line.startswith("# "):
                    article = line[2:].strip()
                elif line.startswith("## "):
                    if heading:
                        sections.append(Section(article, heading, " ".join(lines).strip()))
                    heading, lines = line[3:].strip(), []
                elif line.strip():
                    lines.append(line.strip())
            if heading:
                sections.append(Section(article, heading, " ".join(lines).strip()))
        return cls(sections)

    def search(self, query: str, k: int = 3) -> list[Section]:
        q = _tokens(query)
        scored = []
        for s in self.sections:
            heading_hits = len(q & _tokens(s.heading))
            body_hits = len(q & _tokens(s.text))
            score = 2 * heading_hits + body_hits
            if score:
                scored.append((score, s))
        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored[:k]]
