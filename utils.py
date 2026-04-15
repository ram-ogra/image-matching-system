"""
utils.py
========
Helper functions:
  - Image loading & resizing
  - Visualization (match lines, keypoints, side-by-side)
  - Stats formatting
"""

import cv2
import numpy as np
from PIL import Image
import io


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_image_from_upload(uploaded_file, max_size=1024):
    """
    Load image from Streamlit UploadedFile object.
    Resizes to max_size on the longest side (CPU performance constraint).

    Returns:
        img_bgr  : BGR numpy array (for OpenCV)
        img_rgb  : RGB numpy array (for display)
        img_gray : Grayscale numpy array (for feature detection)
        orig_size: (width, height) of the image AFTER resize
    """
    # Read bytes → PIL → numpy
    img_pil = Image.open(uploaded_file).convert("RGB")
    img_np  = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Resize: limit longest side to max_size
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_size:
        scale  = max_size / max(h, w)
        new_w  = int(w * scale)
        new_h  = int(h * scale)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    return img_bgr, img_rgb, img_gray, (img_bgr.shape[1], img_bgr.shape[0])


def load_image_from_path(path, max_size=1024):
    """Load image from file path."""
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise ValueError(f"Cannot load image: {path}")

    h, w = img_bgr.shape[:2]
    if max(h, w) > max_size:
        scale   = max_size / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_AREA)

    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return img_bgr, img_rgb, img_gray, (img_bgr.shape[1], img_bgr.shape[0])


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def draw_keypoints_on_image(img_rgb, kpts, color=(0, 220, 80), radius=4):
    """
    Draw keypoints as circles on image.

    Args:
        img_rgb : RGB numpy array
        kpts    : (N, 2) array of [x, y] or list of cv2.KeyPoint
        color   : BGR color tuple for circles
        radius  : circle radius in pixels

    Returns:
        img_with_kpts: RGB numpy array with keypoints drawn
    """
    img = img_rgb.copy()
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if len(kpts) == 0:
        return img

    pts = []
    if hasattr(kpts[0], 'pt'):
        pts = [(int(kp.pt[0]), int(kp.pt[1])) for kp in kpts]
    else:
        pts = [(int(p[0]), int(p[1])) for p in kpts]

    for (x, y) in pts:
        cv2.circle(img_bgr, (x, y), radius, color, -1)
        cv2.circle(img_bgr, (x, y), radius + 1, (0, 0, 0), 1)  # black border

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def draw_matches_result(img1_rgb, img2_rgb, pts1, pts2,
                         inlier_mask=None, max_lines=300, line_thickness=1):
    """
    Draw matching lines between two images side-by-side.
    Green = inlier match, Red = outlier match.

    This produces the visual similar to the Taj Mahal output in the PDF.

    Args:
        img1_rgb, img2_rgb : RGB numpy arrays
        pts1, pts2         : (N, 2) matched point arrays
        inlier_mask        : (N,) boolean — True=inlier (green), False=outlier (red)
        max_lines          : Max lines to draw (too many = unreadable)
        line_thickness     : Line width

    Returns:
        result_img : RGB numpy array of the side-by-side result
    """
    h1, w1 = img1_rgb.shape[:2]
    h2, w2 = img2_rgb.shape[:2]

    # Equalize heights
    target_h = max(h1, h2)
    if h1 != target_h:
        img1_rgb = cv2.resize(img1_rgb, (int(w1 * target_h / h1), target_h))
        w1 = img1_rgb.shape[1]
    if h2 != target_h:
        img2_rgb = cv2.resize(img2_rgb, (int(w2 * target_h / h2), target_h))
        w2 = img2_rgb.shape[1]

    # Canvas
    canvas = np.zeros((target_h, w1 + w2, 3), dtype=np.uint8)
    canvas[:, :w1]      = img1_rgb
    canvas[:, w1:w1+w2] = img2_rgb

    n = len(pts1)
    if n == 0:
        return canvas

    # Determine inlier/outlier status
    if inlier_mask is None:
        inlier_mask = np.ones(n, dtype=bool)

    # Select which matches to draw (prioritize inliers)
    inlier_indices  = [i for i in range(n) if inlier_mask[i]]
    outlier_indices = [i for i in range(n) if not inlier_mask[i]]

    # Draw outliers first (so inliers render on top)
    for i in outlier_indices[:max_lines // 4]:
        x1, y1 = int(pts1[i][0]), int(pts1[i][1])
        x2, y2 = int(pts2[i][0]) + w1, int(pts2[i][1])
        cv2.line(canvas, (x1, y1), (x2, y2), (180, 50, 50), line_thickness)

    # Draw inliers (bright green — like the Taj Mahal green lines in PDF)
    for i in inlier_indices[:max_lines]:
        x1, y1 = int(pts1[i][0]), int(pts1[i][1])
        x2, y2 = int(pts2[i][0]) + w1, int(pts2[i][1])
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 230, 80), line_thickness)
        # Draw small dots at endpoints
        cv2.circle(canvas, (x1, y1), 3, (100, 200, 255), -1)
        cv2.circle(canvas, (x2, y2), 3, (100, 200, 255), -1)

    # Draw separator line between images
    cv2.line(canvas, (w1, 0), (w1, target_h), (100, 100, 100), 2)

    return canvas


def numpy_to_pil(img_rgb):
    """Convert RGB numpy array to PIL Image."""
    return Image.fromarray(img_rgb.astype(np.uint8))


def pil_to_bytes(pil_img, fmt='PNG'):
    """Convert PIL Image to bytes for Streamlit display."""
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# STATS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def format_pipeline_stats(feature_info, match_stats, ransac_stats,
                           refinement_info, total_time_ms):
    """
    Format all pipeline stats into a clean dict for display.

    Returns:
        dict with human-readable stats
    """
    return {
        "🖼️ Image 1 Keypoints":       feature_info.get('n_kpts1', 0),
        "🖼️ Image 2 Keypoints":       feature_info.get('n_kpts2', 0),
        "🔗 Raw Matches":              match_stats.get('n_raw', 0),
        "✅ Inlier Matches (RANSAC)":  ransac_stats.get('n_inliers', 0),
        "📊 Inlier Ratio":             f"{ransac_stats.get('inlier_ratio', 0.0)}%",
        "⚡ Attention Exit Layer":     f"{match_stats.get('exit_layer', '?')} / 9",
        "🎯 Match Confidence":         f"{match_stats.get('confidence', 0.0):.3f}",
        "🔍 Refinement 0.25x":        refinement_info.get('0.25x', 0),
        "🔍 Refinement 0.5x":         refinement_info.get('0.5x', 0),
        "🔍 Refinement 1.0x":         refinement_info.get('1.0x', 0),
        "⏱️ Total Time":               f"{total_time_ms:.0f} ms",
    }
