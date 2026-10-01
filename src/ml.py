import os
import io
import cv2
import torch
import numpy as np
from typing import List, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageOps
from transformers import CLIPProcessor, CLIPModel
from pillow_heif import register_heif_opener

register_heif_opener()

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_ID = "openai/clip-vit-base-patch32"
_model = None
_processor = None


def get_model_and_processor():
    global _model, _processor
    if _model is None:
        print(f"Loading CLIP model onto {device.upper()}...")
        try:
            _model = CLIPModel.from_pretrained(MODEL_ID, use_safetensors=True, local_files_only=True).to(device)
            _processor = CLIPProcessor.from_pretrained(MODEL_ID, local_files_only=True)
            print("Successfully loaded CLIP model from local cache.")
        except Exception as e:
            print(f"Cache miss or load error ({e}). Downloading from Hugging Face...")
            _model = CLIPModel.from_pretrained(MODEL_ID, use_safetensors=True).to(device)
            _processor = CLIPProcessor.from_pretrained(MODEL_ID)
            print("Download complete. Model saved to cache.")
            
        _model.eval()
    return _model, _processor


def is_model_loaded() -> bool:
    return _model is not None


def _extract_tensor(features):
    if hasattr(features, "pooler_output") and features.pooler_output is not None:
        return features.pooler_output
    if hasattr(features, "image_embeds") and features.image_embeds is not None:
        return features.image_embeds
    if hasattr(features, "text_embeds") and features.text_embeds is not None:
        return features.text_embeds
    if not isinstance(features, torch.Tensor):
        return features[0]
    return features


def extract_video_frame(video_path: str) -> Optional[Image.Image]:
    """Extracts a video frame at ~1 second for dynamic thumbnail generation and embedding."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        target_frame = int(fps * 1.0)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        success, frame = cap.read()

        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = cap.read()

        cap.release()

        if success and frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)
    except Exception as e:
        print(f"Error extracting frame from video {video_path}: {e}")
    return None


def generate_dynamic_thumbnail_bytes(media_path: str, media_type: str, max_size: int = 360) -> Optional[bytes]:
    """
    Generates a JPEG thumbnail dynamically in-memory.
    Saves zero files to disk.
    """
    try:
        if media_type == "video":
            frame = extract_video_frame(media_path)
            if frame:
                frame.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                frame.save(buf, format="JPEG", quality=75)
                return buf.getvalue()
        else:
            with Image.open(media_path) as img:
                img = ImageOps.exif_transpose(img)
                try:
                    img.draft("RGB", (max_size, max_size))
                except Exception:
                    pass
                rgb = img.convert("RGB")
                rgb.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                rgb.save(buf, format="JPEG", quality=75)
                return buf.getvalue()
    except Exception as e:
        print(f"Dynamic thumbnail generation error for {media_path}: {e}")
    return None


def generate_dynamic_full_image_bytes(media_path: str) -> Optional[bytes]:
    """
    Generates a full-resolution JPEG dynamically in-memory for formats 
    browsers cannot natively render (like HEIC/HEIF/TIFF/BMP).
    """
    try:
        with Image.open(media_path) as img:
            img = ImageOps.exif_transpose(img)
            rgb = img.convert("RGB")
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=85)
            return buf.getvalue()
    except Exception as e:
        print(f"Dynamic full image generation error for {media_path}: {e}")
    return None


VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.m4v', '.3gp', '.mpeg', '.mpg', '.ts', '.webm', '.mkv', '.insv'}


def _load_image_for_clip(path_or_img: Union[str, Image.Image]) -> Optional[Image.Image]:
    if isinstance(path_or_img, Image.Image):
        return path_or_img.convert("RGB")

    ext = os.path.splitext(str(path_or_img))[1].lower()
    if ext in VIDEO_EXTS:
        return extract_video_frame(path_or_img)

    try:
        with Image.open(path_or_img) as img:
            img = ImageOps.exif_transpose(img)
            try:
                img.draft("RGB", (224, 224))
            except Exception:
                pass
            return img.convert("RGB")
    except Exception:
        try:
            return extract_video_frame(path_or_img)
        except Exception:
            return None


def get_image_embeddings_batch(
    items: List[Union[str, Image.Image]], 
    batch_size: int = 32
) -> List[Optional[List[float]]]:
    """Computes CLIP embeddings using streaming chunk processing and GPU acceleration."""
    if not items:
        return []

    model, processor = get_model_and_processor()
    results: List[Optional[List[float]]] = [None] * len(items)
    num_workers = min(8, os.cpu_count() or 4)

    for i in range(0, len(items), batch_size):
        chunk_items = items[i:i + batch_size]

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            chunk_imgs = list(executor.map(_load_image_for_clip, chunk_items))

        valid_pairs = [(idx, img) for idx, img in enumerate(chunk_imgs) if img is not None]
        if not valid_pairs:
            continue

        valid_indices = [p[0] for p in valid_pairs]
        valid_images = [p[1] for p in valid_pairs]

        try:
            inputs = processor(images=valid_images, return_tensors="pt", padding=True).to(device)
            
            with torch.inference_mode():
                if device == "cuda":
                    with torch.autocast("cuda"):
                        features = model.get_image_features(**inputs)
                else:
                    features = model.get_image_features(**inputs)

                features = _extract_tensor(features)
                features = features / features.norm(p=2, dim=-1, keepdim=True)
                emb_matrix = features.cpu().float().numpy()

            for sub_idx, orig_sub_idx in enumerate(valid_indices):
                global_idx = i + orig_sub_idx
                results[global_idx] = emb_matrix[sub_idx].astype(float).tolist()

        except Exception as e:
            print(f"Batch embedding error for chunk {i}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


def get_image_embedding(image_path: str) -> Optional[List[float]]:
    batch = get_image_embeddings_batch([image_path], batch_size=1)
    return batch[0] if batch else None


def get_text_embedding(text: str) -> Optional[List[float]]:
    try:
        model, processor = get_model_and_processor()
        inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            if device == "cuda":
                with torch.autocast("cuda"):
                    features = model.get_text_features(**inputs)
            else:
                features = model.get_text_features(**inputs)

            features = _extract_tensor(features)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().float().numpy().flatten().astype(float).tolist()
    except Exception as e:
        print(f"Error generating text embedding for '{text}': {e}")
        return None
