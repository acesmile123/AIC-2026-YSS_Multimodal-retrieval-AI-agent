from pathlib import Path
import inspect
import importlib


ROOT = Path(__file__).resolve().parents[2]


def test_kis_modules_import_from_repo_root():
    solve_kis = importlib.import_module("KISBranche.solve_kis")
    fusion_clip = importlib.import_module("KISBranche.fusion_clip")
    assert callable(solve_kis.solve_kis)
    assert callable(fusion_clip.fusion_clip_caption)


def test_kis_fusion_uses_supported_caption_weight_keyword():
    solve_source = (ROOT / "KISBranche" / "solve_kis.py").read_text(encoding="utf-8")
    assert "caption_weight=0.7" in solve_source
    assert "alpha=0.7" not in solve_source


def test_kis_adapter_points_to_packaged_implementation():
    source = (ROOT / "qa" / "kis_adapter.py").read_text(encoding="utf-8")
    assert "from KISBranche.clip_encoder import ClipEncoder" in source
    assert "from KISBranche.caption_retriever import CaptionRetriever" in source
    assert "from KISBranche.solve_kis import solve_kis" in source
    assert "from KISBranche.object_filter import ObjectMetadataLookup" in source


def test_solve_kis_executes_with_injected_retrieval_stubs():
    from KISBranche.solve_kis import solve_kis

    class Encoder:
        def encode_text(self, query):
            return [1.0, 0.0]

    class Retriever:
        def search(self, vector, top_k=10):
            return [
                {"video_id": "v1", "frame_id": 1, "score": 0.9},
                {"video_id": "v1", "frame_id": 2, "score": 0.8},
            ][:top_k]

    class CaptionRetriever:
        def search(self, query, top_k=10):
            return [
                {"video_id": "v1", "frame_id": 1, "score": 0.7},
            ][:top_k]

    class ObjectLookup:
        def get(self, key):
            return None

    out = solve_kis(
        {"query_variants": ["people on a stage"], "entities": []},
        Retriever(),
        Encoder(),
        CaptionRetriever(),
        ObjectLookup(),
    )
    assert out
    assert out[0][0] == "v1"


def test_kis_notebook_imports_are_package_safe():
    import json
    nb = json.loads((ROOT / "KISBranche" / "runbranch.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in nb.get("cells", []))
    assert "from solve_kis import" not in source
    assert "from clip_encoder import" not in source
    assert "from caption_retriever import" not in source
    assert "from object_filter import" not in source
    assert "from apply_object_filter import" not in source
    assert "from KISBranche.solve_kis import" in source
