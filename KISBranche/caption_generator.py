import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv
import json
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering, BlipForConditionalGeneration

# 1. LOAD CSV MAPPING
def load_frame_mapping(csv_path):
    mapping = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                n = int(row.get("n", row.get("keyframe_n", i)))
                frame_id = int(row.get("frame_idx", row.get("frame_id", 0)))
                mapping[n] = frame_id
            except (ValueError, TypeError):
                continue
    return mapping


# 2. GENERAL CAPTION (giữ nguyên model captioning cũ, cho caption tổng quát)
def generate_caption(image, processor, model, device, max_new_tokens=50):
    inputs = processor(image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=5,
            do_sample=False,
            repetition_penalty=1.1,
        )
    return processor.decode(output[0], skip_special_tokens=True).strip()


# 3. VQA — hỏi từng câu ngắn, đúng sở trường của BLIP
VQA_QUESTIONS = {
    "people": "Are there any people in the image?",
    "clothing_color": "What color clothes are the people wearing?",
    "main_object": "What is the main object in the image?",
    "object_color": "What color is the main object?",
    "action": "What action is happening in the image?",
    "setting": "Is this indoor or outdoor?",
    "location": "Where does this scene take place?",
}


def ask_blip_vqa(image, question, processor, model, device, max_new_tokens=15):
    inputs = processor(image, question, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)

    return processor.decode(output[0], skip_special_tokens=True).strip()


def generate_detailed_caption_vqa(
    image, cap_processor, cap_model, vqa_processor, vqa_model, device
):
    general_caption = generate_caption(image, cap_processor, cap_model, device)

    answers = {}
    for key, question in VQA_QUESTIONS.items():
        answers[key] = ask_blip_vqa(image, question, vqa_processor, vqa_model, device)

    # Ghép câu trả lời thành đoạn text để phục vụ retrieval / filter
    detail_parts = []
    if answers.get("people") and answers["people"].lower() not in ("no", "none"):
        detail_parts.append(f"people present, wearing {answers.get('clothing_color', 'unknown')} clothes")
    if answers.get("main_object"):
        detail_parts.append(f"main object: {answers['main_object']} ({answers.get('object_color', 'unknown')} color)")
    if answers.get("action"):
        detail_parts.append(f"action: {answers['action']}")
    if answers.get("setting"):
        detail_parts.append(f"setting: {answers['setting']}")
    if answers.get("location"):
        detail_parts.append(f"location: {answers['location']}")

    details_text = "; ".join(detail_parts)
    retrieval_text = f"{general_caption}. {details_text}"

    return {
        "caption": general_caption,
        "vqa_answers": answers,
        "details": details_text,
        "retrieval_text": retrieval_text,
    }


# 4. MAIN GENERATION
def generate_captions_with_mapping(image_root, csv_root, output_json):
    cap_model_id = "Salesforce/blip-image-captioning-large"
    vqa_model_id = "Salesforce/blip-vqa-base"

    print(f"[*] Loading captioning model: {cap_model_id}...")
    cap_processor = BlipProcessor.from_pretrained(cap_model_id)
    cap_model = BlipForConditionalGeneration.from_pretrained(cap_model_id)

    print(f"[*] Loading VQA model: {vqa_model_id}...")
    vqa_processor = BlipProcessor.from_pretrained(vqa_model_id)
    vqa_model = BlipForQuestionAnswering.from_pretrained(vqa_model_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cap_model.to(device)
    vqa_model.to(device)

    if device == "cuda":
        cap_model = cap_model.half()
        vqa_model = vqa_model.half()

    cap_model.eval()
    vqa_model.eval()

    print(f"[*] Device: {device}")
    if device == "cuda":
        print(f"[*] GPU: {torch.cuda.get_device_name(0)}")

    # Load existing checkpoint
    if os.path.exists(output_json):
        print(f"[*] Loading existing captions: {output_json}")
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                caption_data = json.load(f)
        except Exception:
            caption_data = []
    else:
        caption_data = []

    processed = {(x["video_id"], x["frame_id"]) for x in caption_data}
    print(f"[*] Already processed: {len(processed)} frames")

    valid_extensions = (".jpg", ".jpeg", ".png")
    video_folders = sorted(
        d for d in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, d))
    )
    print(f"[*] Found {len(video_folders)} video folders")

    for video_id in video_folders:
        image_dir = os.path.join(image_root, video_id)
        csv_path = os.path.join(csv_root, video_id + ".csv")

        if not os.path.exists(csv_path):
            print(f"[Skip] Missing CSV: {video_id}")
            continue

        frame_mapping = load_frame_mapping(csv_path)
        image_files = sorted(
            f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)
        )
        print(f"\n[*] {video_id}: {len(image_files)} images")

        for img_name in image_files:
            try:
                keyframe_n = int(os.path.splitext(img_name)[0])
            except ValueError:
                print(f"[Skip] Invalid filename: {img_name}")
                continue

            frame_id = frame_mapping.get(keyframe_n)
            if frame_id is None:
                print(f"[Skip] No mapping: {video_id}/{keyframe_n}")
                continue

            if (video_id, frame_id) in processed:
                continue

            img_path = os.path.join(image_dir, img_name)
            try:
                image = Image.open(img_path).convert("RGB")
                result = generate_detailed_caption_vqa(
                    image, cap_processor, cap_model, vqa_processor, vqa_model, device
                )

                item = {
                    "video_id": video_id,
                    "frame_id": frame_id,
                    "keyframe_n": keyframe_n,
                    "image_path": img_path,
                    "caption": result["caption"],
                    "vqa_answers": result["vqa_answers"],
                    "details": result["details"],
                    "retrieval_text": result["retrieval_text"],
                }

                caption_data.append(item)
                processed.add((video_id, frame_id))

                print(f"[{video_id}] keyframe={keyframe_n} frame={frame_id}")
                print(f"  Caption: {result['caption']}")
                print(f"  Details: {result['details']}")

                # Checkpoint sau mỗi ảnh
                with open(output_json, "w", encoding="utf-8") as f:
                    json.dump(caption_data, f, ensure_ascii=False, indent=2)

                if device == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"[Error] {video_id}/{img_name}: {e}")

    print(f"\n[*] Saved {len(caption_data)} captions")
    print(f"[*] Output: {output_json}")


if __name__ == "__main__":
    generate_captions_with_mapping(image_root="data", csv_root="data", output_json="data/captions.json")