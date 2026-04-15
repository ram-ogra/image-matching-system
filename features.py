"""
features.py
===========
Hybrid Feature Extraction Module
PDF Pipeline: Stage 1 — DISK + SuperPoint (with ORB CPU fallback)

Logic:
  - Primary: SIFT (simulates SuperPoint-quality features on CPU)
  - Enhanced: ORB for speed fallback
  - Hybrid fusion: combine both → filter best keypoints

On CPU laptop (i5): SIFT gives best quality, ORB gives speed.
Both are fused to simulate the DISK + SuperPoint hybrid from the PDF.
"""

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1A: SIFT-based extraction  (simulates SuperPoint descriptor quality)
# ─────────────────────────────────────────────────────────────────────────────

def extract_sift_features(img_gray, n_features=2000):
    """
    Extract SIFT keypoints + descriptors.
    SIFT = Scale-Invariant Feature Transform (2004, David Lowe)
    
    Why SIFT here? 
      - In the PDF, SuperPoint produces 256-D descriptors.
      - SIFT produces 128-D descriptors with similar scale/rotation invariance.
      - On CPU, SIFT is the closest classical equivalent.
    
    Returns:
        kpts_array : (N, 2) float array of [x, y] coords
        keypoints  : list of cv2.KeyPoint (for cv2 draw functions)
        descriptors: (N, 128) float array
    """
    sift = cv2.SIFT_create(
        nfeatures=n_features,
        contrastThreshold=0.03,   # Lower = more keypoints (good for textureless)
        edgeThreshold=12,          # Suppress edge responses
        sigma=1.6                  # Gaussian blur sigma
    )
    keypoints, descriptors = sift.detectAndCompute(img_gray, None)
    
    if descriptors is None or len(keypoints) == 0:
        return np.zeros((0, 2), dtype=np.float32), [], np.zeros((0, 128), dtype=np.float32)
    
    kpts_array = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)
    return kpts_array, keypoints, descriptors


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1B: ORB-based extraction  (simulates DISK dense-coverage on CPU)
# ─────────────────────────────────────────────────────────────────────────────

def extract_orb_features(img_gray, n_features=2000):
    """
    Extract ORB keypoints + descriptors.
    ORB = Oriented FAST + Rotated BRIEF
    
    Why ORB here?
      - In the PDF, DISK detects dense keypoints in textureless regions.
      - ORB's FAST detector finds corners rapidly across the full image.
      - Binary descriptors (256-bit) → ultra fast Hamming distance matching.
    
    Returns:
        kpts_array : (N, 2) float array of [x, y] coords
        keypoints  : list of cv2.KeyPoint
        descriptors: (N, 32) uint8 binary array
    """
    orb = cv2.ORB_create(
        nfeatures=n_features,
        scaleFactor=1.2,
        nlevels=8,               # Image pyramid levels (multi-scale coverage)
        edgeThreshold=15,
        fastThreshold=15         # Lower = more keypoints in flat regions
    )
    keypoints, descriptors = orb.detectAndCompute(img_gray, None)
    
    if descriptors is None or len(keypoints) == 0:
        return np.zeros((0, 2), dtype=np.float32), [], np.zeros((0, 32), dtype=np.uint8)
    
    kpts_array = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)
    return kpts_array, keypoints, descriptors


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 MAIN: HYBRID FUSION  (PDF: DISK + SuperPoint combined)
# ─────────────────────────────────────────────────────────────────────────────

def hybrid_feature_extraction(img_gray, n_features=1500, mode='hybrid'):
    """
    MAIN FUNCTION — Hybrid Feature Extraction as described in the PDF.
    
    PDF says:
      "DISK detector identifies dense keypoints → SuperPoint descriptor
       generates 256-D descriptors via bilinear interpolation."
    
    Our CPU implementation:
      - SIFT = SuperPoint equivalent (robust 128-D descriptors)
      - ORB  = DISK equivalent (dense, fast detection)
      - Fusion = combine both, deduplicate, keep best N
    
    Args:
        img_gray  : Grayscale image (H, W) numpy array
        n_features: Max keypoints per method
        mode      : 'hybrid' | 'sift_only' | 'orb_only'
    
    Returns:
        kpts  : (N, 2) keypoint coordinates
        descs : (N, 128) descriptors (normalized float)
        info  : dict with method stats
    """
    info = {'method': mode, 'n_sift': 0, 'n_orb': 0, 'n_fused': 0}
    
    if mode == 'sift_only':
        kpts, _, descs = extract_sift_features(img_gray, n_features)
        info['n_sift'] = len(kpts)
        info['n_fused'] = len(kpts)
        return kpts, descs, info
    
    if mode == 'orb_only':
        kpts, _, orb_descs = extract_orb_features(img_gray, n_features)
        # Convert binary ORB descriptors to float for uniform pipeline
        descs = orb_descs.astype(np.float32) if len(kpts) > 0 else np.zeros((0, 32), np.float32)
        info['n_orb'] = len(kpts)
        info['n_fused'] = len(kpts)
        return kpts, descs, info
    
    # ── HYBRID MODE: fuse SIFT + ORB ──────────────────────────────────────────
    sift_kpts, _, sift_descs = extract_sift_features(img_gray, n_features)
    orb_kpts, _, orb_descs   = extract_orb_features(img_gray, n_features)
    
    info['n_sift'] = len(sift_kpts)
    info['n_orb']  = len(orb_kpts)
    
    if len(sift_kpts) == 0 and len(orb_kpts) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32), info
    
    # If one method failed, use the other
    if len(sift_kpts) == 0:
        descs = orb_descs.astype(np.float32)
        kpts  = orb_kpts
        info['n_fused'] = len(kpts)
        return kpts, descs, info
    
    if len(orb_kpts) == 0:
        info['n_fused'] = len(sift_kpts)
        return sift_kpts, sift_descs, info
    
    # ── Convert ORB binary descriptors to float128 (pad with zeros to match SIFT dim) ──
    orb_descs_float = np.zeros((len(orb_kpts), 128), dtype=np.float32)
    orb_descs_float[:, :32] = orb_descs.astype(np.float32) / 255.0
    
    # ── Deduplicate: remove ORB keypoints too close to existing SIFT keypoints ──
    # This simulates the spatial proximity fusion from the PDF
    MIN_DISTANCE = 8.0  # pixels — keypoints closer than this are duplicates
    
    if len(sift_kpts) > 0 and len(orb_kpts) > 0:
        # Build spatial index using brute-force distance check (fast enough for 2000 pts)
        keep_orb_mask = np.ones(len(orb_kpts), dtype=bool)
        
        for i, op in enumerate(orb_kpts):
            dists = np.linalg.norm(sift_kpts - op, axis=1)
            if dists.min() < MIN_DISTANCE:
                keep_orb_mask[i] = False  # Too close to a SIFT point → skip
        
        orb_kpts_unique  = orb_kpts[keep_orb_mask]
        orb_descs_unique = orb_descs_float[keep_orb_mask]
    else:
        orb_kpts_unique  = orb_kpts
        orb_descs_unique = orb_descs_float
    
    # ── Concatenate SIFT + unique ORB features ────────────────────────────────
    all_kpts  = np.vstack([sift_kpts, orb_kpts_unique]) if len(orb_kpts_unique) > 0 else sift_kpts
    all_descs = np.vstack([sift_descs, orb_descs_unique]) if len(orb_kpts_unique) > 0 else sift_descs
    
    # ── Limit total keypoints ────────────────────────────────────────────────
    max_total = n_features * 2
    if len(all_kpts) > max_total:
        indices   = np.random.choice(len(all_kpts), max_total, replace=False)
        all_kpts  = all_kpts[indices]
        all_descs = all_descs[indices]
    
    # ── L2-normalize descriptors (standard for SIFT-based matching) ──────────
    norms = np.linalg.norm(all_descs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1e-8, norms)
    all_descs = all_descs / norms
    
    info['n_fused'] = len(all_kpts)
    return all_kpts, all_descs, info
