"""
Advanced Fundus Enhancement Pipeline — with per-stage image saving for visualization.
Each call to enhance_fundus() returns both the final enhanced image path
AND a list of {stage_name, image_path, description} dicts for UI display.
"""

import os
import uuid
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np

try:
    from skimage.filters import frangi
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

UPLOAD_DIR = Path("static/uploads")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def _save_stage(img: np.ndarray, run_id: str, stage_num: int, name: str) -> str:
    """Save a pipeline stage image and return its web-accessible path."""
    stage_dir = UPLOAD_DIR / "stages" / run_id
    stage_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{stage_num:02d}_{name}.png"
    path = stage_dir / fname
    cv2.imwrite(str(path), img)
    return f"uploads/stages/{run_id}/{fname}"


class FundusQualityAssessor:
    def assess(self, img: np.ndarray) -> Dict:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        mean_brightness = float(gray.mean())
        fov_mask = self._fov_mask(gray)
        roi = gray[fov_mask > 0]
        rms_contrast = float(roi.std()) if len(roi) > 0 else 0.0
        return {
            "blur_score": round(blur_score, 2),
            "mean_brightness": round(mean_brightness, 2),
            "overexposed": mean_brightness > 220,
            "underexposed": mean_brightness < 40,
            "rms_contrast": round(rms_contrast, 2),
            "quality_ok": blur_score > 50 and 40 <= mean_brightness <= 220 and rms_contrast > 15,
        }

    def _fov_mask(self, gray: np.ndarray) -> np.ndarray:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        return (mask > 0).astype(np.uint8)


def _crop_and_resize(img: np.ndarray, size: int = 1024) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return cv2.resize(img, (size, size))
    x, y, w, h = cv2.boundingRect(coords)
    pad = 10
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2 = min(img.shape[1], x + w + pad * 2)
    y2 = min(img.shape[0], y + h + pad * 2)
    crop = img[y1:y2, x1:x2]
    oh, ow = crop.shape[:2]
    scale = size / max(oh, ow)
    nh, nw = int(oh * scale), int(ow * scale)
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    dy, dx = (size - nh) // 2, (size - nw) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas


def _ben_graham(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    sigma = max(int(0.015 * min(h, w)) | 1, 1)
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    result = cv2.addWeighted(img, 4, blurred, -4, 128)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result[gray <= 10] = 128
    return result


def _clahe(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, table)


def _enhance_microaneurysms(img: np.ndarray) -> np.ndarray:
    g = img[:, :, 1]
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bth = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, se)
    # Fixed: use numpy arithmetic instead of cv2.multiply to avoid type error
    scaled = np.clip(bth.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)
    res = img.copy()
    res[:, :, 1] = cv2.add(g, scaled)
    return res


def _enhance_exudates(img: np.ndarray) -> np.ndarray:
    g = img[:, :, 1]
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    wth = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, se)
    res = img.copy()
    res[:, :, 1] = cv2.add(g, wth)
    return res


def _unsharp_mask(img: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(img, (0, 0), 0.8)
    return cv2.addWeighted(img, 1.8, blurred, -0.8, 0)


def _histogram_stretch(img: np.ndarray) -> np.ndarray:
    out = np.zeros_like(img)
    for c in range(3):
        ch = img[:, :, c].astype(np.float32)
        p_lo, p_hi = np.percentile(ch, 0.5), np.percentile(ch, 99.5)
        out[:, :, c] = np.clip((ch - p_lo) / (p_hi - p_lo + 1e-6) * 255, 0, 255).astype(np.uint8)
    return out


def _denoise(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img, None, h=6, hColor=6, templateWindowSize=7, searchWindowSize=21)


def _vessel_map(img: np.ndarray) -> Optional[np.ndarray]:
    """Returns a colour-mapped vessel visualization image."""
    if not SKIMAGE_AVAILABLE:
        return None
    green = img[:, :, 1].astype(np.float32) / 255.0
    vmap = frangi(green, sigmas=range(1, 6), alpha=0.5, beta=0.5, gamma=15, black_ridges=True)
    norm = cv2.normalize(vmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Apply a cyan colormap for visual appeal
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_OCEAN)
    return colored


# ── STAGE METADATA ────────────────────────────────────────────────────────────
STAGE_META = [
    {
        "key": "original",
        "label": "Original Image",
        "icon": "📷",
        "description": "Raw fundus photograph as uploaded. May have uneven illumination, low contrast, or border artifacts."
    },
    {
        "key": "crop_resize",
        "label": "FOV Crop & Resize",
        "icon": "✂️",
        "description": "Tight crop around the illuminated fundus disk removes black borders. Letterbox-resized to 1024×1024 using Lanczos interpolation."
    },
    {
        "key": "ben_graham",
        "label": "Ben Graham Normalization",
        "icon": "💡",
        "description": "Subtracts a large Gaussian blur to eliminate uneven illumination gradients. Makes peripheral lesions as visible as central ones. (Kaggle DR winner technique, 2015)"
    },
    {
        "key": "clahe",
        "label": "CLAHE Enhancement",
        "icon": "📊",
        "description": "Contrast Limited Adaptive Histogram Equalization on the luminance channel. Boosts local contrast without over-amplifying noise in flat regions."
    },
    {
        "key": "gamma",
        "label": "Adaptive Gamma",
        "icon": "☀️",
        "description": "Gamma correction adapted to the image's mean brightness. Dark images get γ=1.4 (brightened), bright images get γ=0.9 (toned down)."
    },
    {
        "key": "microaneurysms",
        "label": "Microaneurysm Enhancement",
        "icon": "🔴",
        "description": "Black top-hat morphological transform isolates and boosts small dark spots (microaneurysms). Structuring element sized to match 10–100µm MA diameter at 1024px."
    },
    {
        "key": "exudates",
        "label": "Exudate Enhancement",
        "icon": "🟡",
        "description": "White top-hat transform highlights bright hard exudate deposits. Critical for detecting lipid leakage from damaged retinal vessels."
    },
    {
        "key": "unsharp",
        "label": "Unsharp Masking",
        "icon": "🔬",
        "description": "Sharpens fine vascular detail and lesion edges by subtracting a blurred version. Improves vessel calibre measurement accuracy."
    },
    {
        "key": "vessels",
        "label": "Vessel Map (Frangi Filter)",
        "icon": "🩸",
        "description": "Multi-scale Frangi vesselness filter highlights the entire retinal vascular tree. Used as structural context for grading neovascularization in PDR."
    },
    {
        "key": "final",
        "label": "Final Enhanced Image",
        "icon": "✅",
        "description": "Post-processed with histogram stretching, non-local means denoising, and soft blended with the original (α=0.85) to preserve natural color while maximizing lesion visibility."
    },
]


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def enhance_fundus(input_path: str) -> Tuple[str, List[Dict]]:
    """
    Run full enhancement pipeline.

    Returns:
        (final_image_path, stages_list)
        stages_list = [{"key", "label", "icon", "description", "image_url", "metrics"}, ...]
    """
    run_id = uuid.uuid4().hex[:10]
    stages_out = []
    qa = FundusQualityAssessor()

    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    n = 0

    def save(image, key):
        nonlocal n
        url = _save_stage(image, run_id, n, key)
        n += 1
        return url

    # Stage 0: Original
    url = save(img, "original")
    pre_q = qa.assess(img)
    stages_out.append({**STAGE_META[0], "image_url": url, "metrics": pre_q})

    # Stage 1: Crop & resize
    img = _crop_and_resize(img, 1024)
    original_resized = img.copy()
    url = save(img, "crop_resize")
    stages_out.append({**STAGE_META[1], "image_url": url, "metrics": {}})

    # Stage 2: Ben Graham
    img = _ben_graham(img)
    url = save(img, "ben_graham")
    stages_out.append({**STAGE_META[2], "image_url": url, "metrics": {}})

    # Stage 3: CLAHE
    img = _clahe(img)
    url = save(img, "clahe")
    stages_out.append({**STAGE_META[3], "image_url": url, "metrics": {}})

    # Stage 4: Gamma
    brightness = pre_q["mean_brightness"]
    gamma_val = 1.4 if brightness < 80 else (0.9 if brightness > 180 else 1.1)
    img = _gamma(img, gamma_val)
    url = save(img, "gamma")
    stages_out.append({**STAGE_META[4], "image_url": url,
                       "metrics": {"gamma_applied": gamma_val, "input_brightness": brightness}})

    # Stage 5: Microaneurysm boost
    img = _enhance_microaneurysms(img)
    url = save(img, "microaneurysms")
    stages_out.append({**STAGE_META[5], "image_url": url, "metrics": {}})

    # Stage 6: Exudate boost
    img = _enhance_exudates(img)
    url = save(img, "exudates")
    stages_out.append({**STAGE_META[6], "image_url": url, "metrics": {}})

    # Stage 7: Unsharp mask
    img = _unsharp_mask(img)
    url = save(img, "unsharp")
    stages_out.append({**STAGE_META[7], "image_url": url, "metrics": {}})

    # Stage 8: Vessel map (separate visualization, doesn't modify main image)
    vmap = _vessel_map(img)
    if vmap is not None:
        url = save(vmap, "vessels")
        stages_out.append({**STAGE_META[8], "image_url": url, "metrics": {}})

    # Stage 9: Post-process and final blend
    img = _histogram_stretch(img)
    img = _denoise(img)
    img = cv2.addWeighted(img, 0.85, original_resized, 0.15, 0)

    url = save(img, "final")
    post_q = qa.assess(img)
    stages_out.append({**STAGE_META[9], "image_url": url, "metrics": post_q})

    # Save final output
    out_name = f"enhanced_{run_id}.png"
    out_path = UPLOAD_DIR / out_name
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])

    return str(out_path), stages_out


# Backwards-compatible wrapper for code that only wants the path
def enhance_fundus_simple(input_path: str) -> str:
    path, _ = enhance_fundus(input_path)
    return path