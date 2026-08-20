from __future__ import annotations

import re
from typing import Dict, Iterable, List

COLOR_TERMS={
    "red":"red","đỏ":"red","blue":"blue","xanh":"blue","green":"green","xanh lá":"green",
    "yellow":"yellow","vàng":"yellow","black":"black","đen":"black","white":"white","trắng":"white",
    "brown":"brown","nâu":"brown","gray":"gray","grey":"gray","xám":"gray","orange":"orange","cam":"orange",
}

class AttributeExtractor:
    def extract_from_text(self, text: str) -> Dict[str, List[str]]:
        low=(text or '').lower()
        colors=[]
        for term,canon in COLOR_TERMS.items():
            if re.search(r"(?<!\w)"+re.escape(term)+r"(?!\w)", low):
                colors.append(canon)
        return {"colors": list(dict.fromkeys(colors))}

    def enrich_scene(self, scene: dict, caption: str = '') -> dict:
        attrs=self.extract_from_text(caption)
        out=dict(scene or {})
        out["attributes"]={**out.get("attributes",{}), **attrs}
        return out
