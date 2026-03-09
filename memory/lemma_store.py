
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from proof.lemmas import Lemma


class LemmaStore:
    def __init__(self) -> None:
        self.lemmas: Dict[str, Lemma] = {}

    def add(self, lemma: Lemma) -> None:
        if lemma.name in self.lemmas:
            self.lemmas[lemma.name].success_count += 1
        else:
            self.lemmas[lemma.name] = lemma

    def retrieve(self, domain: str, text: str, limit: int = 5) -> List[Lemma]:
        scored = []
        text_low = text.lower()
        for lemma in self.lemmas.values():
            score = lemma.success_count
            if domain in lemma.domains:
                score += 2
            if lemma.pattern.lower() in text_low:
                score += 3
            scored.append((score, lemma))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [lemma for _, lemma in scored[:limit]]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {name: lemma.to_dict() for name, lemma in self.lemmas.items()}
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def load(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            return
        data = json.loads(p.read_text(encoding="utf-8"))
        self.lemmas = {name: Lemma(**payload) for name, payload in data.items()}
