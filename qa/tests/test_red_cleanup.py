from pathlib import Path
import json

from qa.qa_ocr import TesseractOCR
from qa.evaluate_qa import evaluate


def test_ocr_provider_is_graceful_when_backend_missing_or_present():
    ocr = TesseractOCR()
    assert isinstance(ocr.available(), bool)
    assert ocr.extract(None) == ""


def test_evaluator_reports_tolerant_frame_and_type_metrics():
    gt = [
        {"question": "q1", "video_id": "v1", "frame_id": 100, "answer": "3", "answer_type": "COUNT"},
        {"question": "q2", "video_id": "v2", "frame_id": 200, "answer": "red", "answer_type": "COLOR"},
    ]
    pred = [
        {"question": "q1", "video_id": "v1", "frame_id": 102, "answer": "3"},
        {"question": "q2", "video_id": "v2", "frame_id": 250, "answer": "blue"},
    ]
    result = evaluate(gt, pred, frame_tolerance=3)
    assert result["answer_accuracy"] == 0.5
    assert result["frame_accuracy_exact"] == 0.0
    assert result["frame_accuracy_tolerant"] == 0.5
    assert result["video_recall"] == 1.0
    assert result["by_answer_type"]["COUNT"]["answer_accuracy"] == 1.0
