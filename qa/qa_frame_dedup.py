from __future__ import annotations

from typing import List, Sequence


def _fingerprint(image, size=16):
    small=image.convert("L").resize((size,size))
    data = small.get_flattened_data() if hasattr(small, "get_flattened_data") else small.getdata()
    vals=list(data)
    mean=sum(vals)/len(vals)
    return tuple(1 if v>=mean else 0 for v in vals)

def hamming(a,b):
    return sum(x!=y for x,y in zip(a,b)) / max(1,len(a))

class FrameDeduplicator:
    def __init__(self, threshold: float = 0.08):
        self.threshold=threshold
    def select(self, records: Sequence[dict]) -> List[dict]:
        kept=[]; fps=[]
        for rec in records:
            img=rec.get("image")
            if img is None:
                kept.append(rec); continue
            fp=_fingerprint(img)
            if any(hamming(fp, old) <= self.threshold for old in fps):
                continue
            kept.append(rec); fps.append(fp)
        return kept
