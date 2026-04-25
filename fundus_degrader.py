"""
Fundus Image Degrader
=====================
Applies realistic degradations that mirror real-world fundus image artifacts.
Each degradation type specifically tests a stage in the enhancement pipeline.

Degradations included:
  1.  Gaussian Noise          — sensor/electronic noise (tests CLAHE + denoise)
  2.  Salt & Pepper Noise     — dead pixels / bit errors (tests NLM denoise)
  3.  Poisson Noise           — photon shot noise (tests CLAHE + denoise)
  4.  Speckle Noise           — coherent illumination artifacts (tests denoise)
  5.  Uneven Illumination     — poor fundus camera alignment (tests Ben Graham)
  6.  Low Contrast            — underexposed / poor pupil dilation (tests CLAHE + gamma)
  7.  Overexposure            — flash overload / media opacity (tests gamma)
  8.  Motion Blur             — patient movement (tests unsharp mask)
  9.  Out-of-Focus Blur       — poor focusing (tests unsharp mask)
  10. JPEG Compression        — lossy storage artifacts (tests denoise)
  11. Color Cast              — camera white balance error (tests Ben Graham + CLAHE)
  12. Vignetting              — peripheral darkening (tests Ben Graham)
  13. Dust/Scratches          — lens contamination (tests morphological ops)
  14. Combined (Real-World)   — mix of multiple artifacts for stress testing
"""

import cv2
import numpy as np
import os
import uuid
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

UPLOAD_DIR = Path("static/uploads/degraded")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ── Degradation catalogue ─────────────────────────────────────────────────────

def add_gaussian_noise(img: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """
    Gaussian noise — models electronic sensor noise and readout noise.
    Common in low-light fundus imaging or high ISO settings.
    Pipeline stage tested: NLM denoising, CLAHE
    """
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = np.clip(img.astype(np.float32) + noise, 0, 255)
    return out.astype(np.uint8)


def add_salt_and_pepper(img: np.ndarray, density: float = 0.04) -> np.ndarray:
    """
    Salt & pepper noise — models dead/stuck pixels, bit-flip transmission errors.
    Pipeline stage tested: NLM denoising, median filtering within CLAHE tiles
    """
    out = img.copy()
    total = img.size // 3
    n_salt   = int(total * density / 2)
    n_pepper = int(total * density / 2)

    # Salt (white)
    ys = np.random.randint(0, img.shape[0], n_salt)
    xs = np.random.randint(0, img.shape[1], n_salt)
    out[ys, xs] = 255

    # Pepper (black)
    ys = np.random.randint(0, img.shape[0], n_pepper)
    xs = np.random.randint(0, img.shape[1], n_pepper)
    out[ys, xs] = 0

    return out


def add_poisson_noise(img: np.ndarray, scale: float = 0.6) -> np.ndarray:
    """
    Poisson (photon shot) noise — fundamental quantum noise in all optical systems.
    Worse in dim regions (periphery). Scale < 1.0 = more noise.
    Pipeline stage tested: CLAHE, NLM denoising
    """
    img_f = img.astype(np.float32) / 255.0
    noisy = np.random.poisson(img_f / scale) * scale
    return np.clip(noisy * 255, 0, 255).astype(np.uint8)


def add_speckle_noise(img: np.ndarray, variance: float = 0.05) -> np.ndarray:
    """
    Speckle (multiplicative) noise — coherent illumination artifacts,
    seen in scanning laser ophthalmoscopes and some confocal systems.
    Pipeline stage tested: NLM denoising, unsharp masking
    """
    noise = np.random.randn(*img.shape).astype(np.float32) * np.sqrt(variance)
    out = img.astype(np.float32) + img.astype(np.float32) * noise
    return np.clip(out, 0, 255).astype(np.uint8)


def add_uneven_illumination(img: np.ndarray, strength: float = 0.6) -> np.ndarray:
    """
    Uneven / gradient illumination — the single most common fundus artifact.
    Caused by misaligned light source, poor pupil dilation, media opacities.
    Pipeline stage tested: Ben Graham normalization (PRIMARY target)
    """
    h, w = img.shape[:2]
    # Radial gradient — bright center, dark periphery (or vice versa)
    cx, cy = w * np.random.uniform(0.3, 0.7), h * np.random.uniform(0.3, 0.7)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    max_dist = np.sqrt(cx**2 + cy**2)
    gradient = 1.0 - strength * (dist / max_dist)
    gradient = np.clip(gradient, 0.2, 1.2)

    # Also add a linear tilt
    tilt_x = np.linspace(1.0, 1.0 - strength * 0.4, w)
    tilt   = np.tile(tilt_x, (h, 1))
    combined = gradient * tilt
    combined = combined[:, :, np.newaxis]

    out = (img.astype(np.float32) * combined)
    return np.clip(out, 0, 255).astype(np.uint8)


def add_low_contrast(img: np.ndarray, factor: float = 0.35) -> np.ndarray:
    """
    Low contrast — underexposure, poor pupil dilation, dense cataract.
    Compresses dynamic range toward mid-grey.
    Pipeline stage tested: CLAHE, gamma correction, Ben Graham normalization
    """
    mid = 128.0
    out = mid + (img.astype(np.float32) - mid) * factor
    return np.clip(out, 0, 255).astype(np.uint8)


def add_overexposure(img: np.ndarray, gamma: float = 0.45) -> np.ndarray:
    """
    Overexposure — flash overload, anterior segment reflections.
    Burns out bright regions (optic disc, exudates).
    Pipeline stage tested: Adaptive gamma correction (gamma < 1.0 path)
    """
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, table)


def add_motion_blur(img: np.ndarray, kernel_size: int = 21, angle: float = None) -> np.ndarray:
    """
    Motion blur — patient eye movement during image capture.
    Smears fine vessels and microaneurysms.
    Pipeline stage tested: Unsharp masking, Frangi vessel filter
    """
    if angle is None:
        angle = np.random.uniform(0, 180)
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0 / kernel_size

    rot_mat = cv2.getRotationMatrix2D((kernel_size // 2, kernel_size // 2), angle, 1.0)
    kernel  = cv2.warpAffine(kernel, rot_mat, (kernel_size, kernel_size))
    kernel /= kernel.sum()

    return cv2.filter2D(img, -1, kernel)


def add_defocus_blur(img: np.ndarray, radius: int = 8) -> np.ndarray:
    """
    Out-of-focus (defocus) blur — refractive error, poor patient fixation.
    Smooths all fine structures including microaneurysms and vessel edges.
    Pipeline stage tested: Unsharp masking (PRIMARY target)
    """
    ksize = 2 * radius + 1
    return cv2.GaussianBlur(img, (ksize, ksize), radius * 0.6)


def add_jpeg_compression(img: np.ndarray, quality: int = 12) -> np.ndarray:
    """
    JPEG compression artifacts — lossy storage, telemedicine transmission.
    Creates block artifacts and ringing around high-frequency edges.
    Pipeline stage tested: NLM denoising, histogram stretch
    """
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buf = cv2.imencode('.jpg', img, encode_params)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def add_color_cast(img: np.ndarray, strength: float = 0.35) -> np.ndarray:
    """
    Color cast — white balance error, lens yellowing, camera sensor drift.
    Shifts all channels unevenly, confusing lesion colour-based detection.
    Pipeline stage tested: Ben Graham normalization, CLAHE on L channel
    """
    cast_type = np.random.choice(['warm', 'cool', 'green', 'cyan'])
    out = img.astype(np.float32).copy()

    if cast_type == 'warm':    # yellow-red cast (yellowed lens / cataract)
        out[:,:,2] = np.clip(out[:,:,2] * (1 + strength), 0, 255)  # R up
        out[:,:,0] = np.clip(out[:,:,0] * (1 - strength * 0.5), 0, 255)  # B down
    elif cast_type == 'cool':  # blue cast (cold illuminant)
        out[:,:,0] = np.clip(out[:,:,0] * (1 + strength), 0, 255)  # B up
        out[:,:,2] = np.clip(out[:,:,2] * (1 - strength * 0.5), 0, 255)  # R down
    elif cast_type == 'green': # green cast (digital sensor issue)
        out[:,:,1] = np.clip(out[:,:,1] * (1 + strength), 0, 255)
    else:                      # cyan cast
        out[:,:,0] = np.clip(out[:,:,0] * (1 + strength * 0.8), 0, 255)
        out[:,:,1] = np.clip(out[:,:,1] * (1 + strength * 0.5), 0, 255)

    return out.astype(np.uint8)


def add_vignetting(img: np.ndarray, strength: float = 0.75, radius_frac: float = 0.55) -> np.ndarray:
    """
    Vignetting — peripheral darkening from lens aperture / fundus camera alignment.
    Hides peripheral microaneurysms and haemorrhages.
    Pipeline stage tested: Ben Graham normalization, adaptive gamma
    """
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    max_r = np.sqrt(cx**2 + cy**2) * radius_frac

    mask = 1.0 - strength * np.clip(dist / max_r - 0.3, 0, 1) ** 2
    mask = np.clip(mask, 0.05, 1.0)[:, :, np.newaxis]

    return np.clip(img.astype(np.float32) * mask, 0, 255).astype(np.uint8)


def add_dust_scratches(img: np.ndarray, n_dust: int = 40, n_scratches: int = 5) -> np.ndarray:
    """
    Dust particles and scratches — lens contamination, slide-based imaging.
    Creates bright/dark circular blobs and thin lines over the image.
    Pipeline stage tested: Morphological operations (top-hat), NLM denoising
    """
    out = img.copy()
    h, w = img.shape[:2]

    # Dust blobs (dark)
    for _ in range(n_dust):
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        r  = np.random.randint(3, 18)
        alpha = np.random.uniform(0.4, 0.9)
        cv2.circle(out, (cx, cy), r, (0, 0, 0), -1)
        # Soft edge
        tmp = img.copy()
        cv2.circle(tmp, (cx, cy), r, (0, 0, 0), -1)
        out = cv2.addWeighted(out, alpha, img, 1 - alpha, 0)

    # Scratches (bright white lines)
    for _ in range(n_scratches):
        x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
        x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
        thickness = np.random.randint(1, 3)
        cv2.line(out, (x1, y1), (x2, y2), (220, 220, 220), thickness)

    return out


def add_combined_degradation(img: np.ndarray, severity: str = 'medium') -> np.ndarray:
    """
    Combined real-world degradation — applies a realistic mix of artifacts
    as seen in actual clinical fundus databases (Kaggle DR, DRIVE, STARE).
    severity: 'mild', 'medium', 'severe'
    Pipeline stage tested: ALL stages — the ultimate stress test
    """
    out = img.copy()
    rng = np.random.default_rng(42)

    if severity == 'mild':
        out = add_gaussian_noise(out, sigma=rng.uniform(8, 15))
        out = add_low_contrast(out, factor=rng.uniform(0.65, 0.85))
        out = add_vignetting(out, strength=rng.uniform(0.2, 0.4))
        out = add_jpeg_compression(out, quality=rng.integers(45, 65))

    elif severity == 'medium':
        out = add_uneven_illumination(out, strength=rng.uniform(0.35, 0.55))
        out = add_gaussian_noise(out, sigma=rng.uniform(18, 30))
        out = add_low_contrast(out, factor=rng.uniform(0.4, 0.6))
        out = add_defocus_blur(out, radius=rng.integers(3, 6))
        out = add_vignetting(out, strength=rng.uniform(0.4, 0.65))
        out = add_jpeg_compression(out, quality=rng.integers(18, 35))

    else:  # severe
        out = add_uneven_illumination(out, strength=rng.uniform(0.55, 0.8))
        out = add_motion_blur(out, kernel_size=rng.integers(15, 25))
        out = add_gaussian_noise(out, sigma=rng.uniform(30, 50))
        out = add_salt_and_pepper(out, density=rng.uniform(0.02, 0.05))
        out = add_low_contrast(out, factor=rng.uniform(0.25, 0.4))
        out = add_color_cast(out, strength=rng.uniform(0.3, 0.5))
        out = add_vignetting(out, strength=rng.uniform(0.6, 0.85))
        out = add_jpeg_compression(out, quality=rng.integers(8, 20))

    return out


# ── Degradation registry ──────────────────────────────────────────────────────

DEGRADATIONS = {
    "gaussian_noise": {
        "fn": add_gaussian_noise,
        "label": "Gaussian Noise",
        "icon": "〰️",
        "category": "Noise",
        "description": "Electronic sensor & readout noise. Common in low-light imaging or high-sensitivity settings.",
        "pipeline_target": "NLM Denoising + CLAHE",
        "params": {"sigma": (5, 60, 25, float)},
    },
    "salt_pepper": {
        "fn": add_salt_and_pepper,
        "label": "Salt & Pepper",
        "icon": "⚡",
        "category": "Noise",
        "description": "Dead/stuck pixels and bit-flip transmission errors in digital fundus systems.",
        "pipeline_target": "NLM Denoising",
        "params": {"density": (0.005, 0.12, 0.04, float)},
    },
    "poisson_noise": {
        "fn": add_poisson_noise,
        "label": "Poisson Noise",
        "icon": "✦",
        "category": "Noise",
        "description": "Photon shot noise — fundamental quantum noise, worst in peripheral/dark regions.",
        "pipeline_target": "CLAHE + NLM Denoising",
        "params": {"scale": (0.2, 1.0, 0.6, float)},
    },
    "speckle_noise": {
        "fn": add_speckle_noise,
        "label": "Speckle Noise",
        "icon": "🔆",
        "category": "Noise",
        "description": "Multiplicative coherent noise from scanning laser ophthalmoscopes.",
        "pipeline_target": "NLM Denoising + Unsharp Mask",
        "params": {"variance": (0.01, 0.15, 0.05, float)},
    },
    "uneven_illumination": {
        "fn": add_uneven_illumination,
        "label": "Uneven Illumination",
        "icon": "💡",
        "category": "Illumination",
        "description": "Off-axis light source, poor pupil dilation, or anterior segment opacity.",
        "pipeline_target": "Ben Graham Normalization",
        "params": {"strength": (0.1, 0.9, 0.6, float)},
    },
    "low_contrast": {
        "fn": add_low_contrast,
        "label": "Low Contrast",
        "icon": "🌫️",
        "category": "Illumination",
        "description": "Underexposure or dense cataract compressing the dynamic range toward mid-grey.",
        "pipeline_target": "CLAHE + Gamma Correction",
        "params": {"factor": (0.1, 0.9, 0.35, float)},
    },
    "overexposure": {
        "fn": add_overexposure,
        "label": "Overexposure",
        "icon": "☀️",
        "category": "Illumination",
        "description": "Flash overload or anterior reflections saturating bright structures.",
        "pipeline_target": "Adaptive Gamma Correction",
        "params": {"gamma": (0.2, 0.8, 0.45, float)},
    },
    "motion_blur": {
        "fn": add_motion_blur,
        "label": "Motion Blur",
        "icon": "💨",
        "category": "Blur",
        "description": "Patient saccadic eye movement during capture. Smears vessels and microaneurysms.",
        "pipeline_target": "Unsharp Masking + Frangi Filter",
        "params": {"kernel_size": (5, 35, 21, int)},
    },
    "defocus_blur": {
        "fn": add_defocus_blur,
        "label": "Defocus Blur",
        "icon": "🔵",
        "category": "Blur",
        "description": "Refractive error or poor fixation — smooths all fine structures.",
        "pipeline_target": "Unsharp Masking",
        "params": {"radius": (2, 18, 8, int)},
    },
    "jpeg_compression": {
        "fn": add_jpeg_compression,
        "label": "JPEG Artifacts",
        "icon": "📦",
        "category": "Compression",
        "description": "Block artifacts and ringing from lossy storage or telemedicine transmission.",
        "pipeline_target": "NLM Denoising + Histogram Stretch",
        "params": {"quality": (2, 40, 12, int)},
    },
    "color_cast": {
        "fn": add_color_cast,
        "label": "Color Cast",
        "icon": "🎨",
        "category": "Color",
        "description": "White balance error, yellowed lens, or camera sensor drift shifting all channels.",
        "pipeline_target": "Ben Graham + CLAHE (LAB)",
        "params": {"strength": (0.1, 0.8, 0.35, float)},
    },
    "vignetting": {
        "fn": add_vignetting,
        "label": "Vignetting",
        "icon": "🔦",
        "category": "Illumination",
        "description": "Peripheral darkening from fundus camera alignment or lens aperture.",
        "pipeline_target": "Ben Graham Normalization",
        "params": {"strength": (0.1, 1.0, 0.75, float)},
    },
    "dust_scratches": {
        "fn": add_dust_scratches,
        "label": "Dust & Scratches",
        "icon": "🧹",
        "category": "Artifacts",
        "description": "Lens contamination or slide dust — creates blobs and lines over the image.",
        "pipeline_target": "Morphological Ops + NLM Denoise",
        "params": {"n_dust": (5, 100, 40, int), "n_scratches": (0, 15, 5, int)},
    },
    "combined_mild": {
        "fn": lambda img, **kw: add_combined_degradation(img, severity='mild'),
        "label": "Combined — Mild",
        "icon": "🔀",
        "category": "Combined",
        "description": "Realistic mild multi-artifact mix: noise + mild contrast loss + vignetting + JPEG.",
        "pipeline_target": "Full Pipeline",
        "params": {},
    },
    "combined_medium": {
        "fn": lambda img, **kw: add_combined_degradation(img, severity='medium'),
        "label": "Combined — Medium",
        "icon": "⚠️",
        "category": "Combined",
        "description": "Medium stress test: uneven illumination + blur + noise + contrast + JPEG.",
        "pipeline_target": "Full Pipeline",
        "params": {},
    },
    "combined_severe": {
        "fn": lambda img, **kw: add_combined_degradation(img, severity='severe'),
        "label": "Combined — Severe",
        "icon": "🔥",
        "category": "Combined",
        "description": "Worst-case degradation matching the hardest real-world fundus images.",
        "pipeline_target": "Full Pipeline",
        "params": {},
    },
}


# ── Core API ──────────────────────────────────────────────────────────────────

def degrade_image(
    input_path: str,
    degradation_type: str,
    params: Optional[Dict] = None,
) -> Dict:
    """
    Apply a single degradation to a fundus image and save the result.

    Returns:
        {
            "success": bool,
            "output_path": str,       # path for cv2/filesystem use
            "output_url": str,        # web-accessible relative path
            "degradation": str,
            "label": str,
            "description": str,
            "pipeline_target": str,
            "params_used": dict,
        }
    """
    if degradation_type not in DEGRADATIONS:
        return {"success": False, "error": f"Unknown degradation: {degradation_type}"}

    img = cv2.imread(input_path)
    if img is None:
        return {"success": False, "error": f"Could not read image: {input_path}"}

    meta   = DEGRADATIONS[degradation_type]
    params = params or {}

    # Resolve defaults for any param not provided
    resolved = {}
    for pname, (pmin, pmax, pdefault, ptype) in meta["params"].items():
        val = params.get(pname, pdefault)
        resolved[pname] = ptype(np.clip(val, pmin, pmax))

    try:
        degraded = meta["fn"](img, **resolved)
    except Exception as e:
        log.error(f"Degradation failed: {e}")
        return {"success": False, "error": str(e)}

    run_id   = uuid.uuid4().hex[:8]
    out_name = f"deg_{degradation_type}_{run_id}.png"
    out_path = UPLOAD_DIR / out_name
    cv2.imwrite(str(out_path), degraded, [cv2.IMWRITE_PNG_COMPRESSION, 1])

    return {
        "success":        True,
        "output_path":    str(out_path),
        "output_url":     f"uploads/degraded/{out_name}",
        "degradation":    degradation_type,
        "label":          meta["label"],
        "icon":           meta["icon"],
        "description":    meta["description"],
        "pipeline_target":meta["pipeline_target"],
        "params_used":    resolved,
    }


def degrade_all(input_path: str) -> List[Dict]:
    """Apply every degradation type to an image. Returns list of result dicts."""
    results = []
    for key in DEGRADATIONS:
        result = degrade_image(input_path, key)
        results.append(result)
    return results


def get_catalogue() -> List[Dict]:
    """Return metadata for all available degradations (no image processing)."""
    return [
        {
            "key":             key,
            "label":           meta["label"],
            "icon":            meta["icon"],
            "category":        meta["category"],
            "description":     meta["description"],
            "pipeline_target": meta["pipeline_target"],
            "params":          {k: {"min":v[0],"max":v[1],"default":v[2],"type":v[3].__name__}
                                for k,v in meta["params"].items()},
        }
        for key, meta in DEGRADATIONS.items()
    ]