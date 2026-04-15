"""
matcher.py
==========
Pipeline Stages 2, 3, 4:
  Stage 2 — Adaptive Attention Matching  (Cross + Self attention, Early Exit)
  Stage 3 — Hierarchical Refinement      (0.25x → 0.5x → 1x coarse-to-fine)
  Stage 4 — Post-processing              (RANSAC filtering, outlier removal)

CPU-optimised: Uses numpy matrix ops to simulate transformer attention.
"""

import cv2
import numpy as np
import time


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 2: ADAPTIVE ATTENTION MATCHING
# PDF: "Transformer-based architecture with Cross-Attention, Self-Attention,
#       and Early Exit Policy governed by a lightweight MLP."
# ═════════════════════════════════════════════════════════════════════════════

class AdaptiveAttentionMatcher:
    """
    Simulates the transformer-based Adaptive Attention Matching from the PDF.

    Real LightGlue/SuperGlue requires GPU. This class replicates the LOGIC:
      - Self-Attention:  refine descriptor context within each image
      - Cross-Attention: find correspondences across image pair
      - Early Exit:      stop iterating once confidence > threshold
      - MNN Filter:      Mutual Nearest Neighbor filter for clean matches

    All operations use numpy (CPU-friendly).
    """

    def __init__(self, max_layers=9, confidence_threshold=0.85,
                 ratio_threshold=0.75):
        """
        Args:
            max_layers           : Max attention iterations (PDF uses 9)
            confidence_threshold : Early exit if avg confidence > this
            ratio_threshold      : Lowe's ratio test threshold
        """
        self.max_layers           = max_layers
        self.confidence_threshold = confidence_threshold
        self.ratio_threshold      = ratio_threshold

    # ── Self-Attention (within one image) ────────────────────────────────────
    def _self_attention(self, descriptors, temperature=0.1):
        """
        Refine descriptors using self-attention within one image.

        PDF says: "Self-Attention Layers refine contextual understanding
                   within each image individually."

        Math: Attention(Q,K,V) = softmax(Q·Kᵀ / √d) · V
        Here Q=K=V=descriptors (self-attention).

        Args:
            descriptors : (N, D) numpy array
            temperature : scaling factor (lower = sharper attention)
        Returns:
            refined_descs: (N, D) contextually refined descriptors
        """
        N, D = descriptors.shape
        if N == 0:
            return descriptors

        scale = np.sqrt(D) * temperature

        # Compute attention scores: (N, N)
        scores = descriptors @ descriptors.T / scale

        # Subtract max for numerical stability before softmax
        scores -= scores.max(axis=1, keepdims=True)
        attn_weights = np.exp(scores)
        attn_weights /= attn_weights.sum(axis=1, keepdims=True) + 1e-8

        # Weighted sum of values
        refined = attn_weights @ descriptors

        # Residual connection (like transformer): blend original + attended
        return 0.7 * descriptors + 0.3 * refined

    # ── Cross-Attention (across two images) ──────────────────────────────────
    def _cross_attention(self, desc1, desc2, temperature=0.1):
        """
        Find correspondences between two images using cross-attention.

        PDF says: "Cross-Attention Layers operate across two images,
                   allowing network to establish global correspondences."

        Math: For each descriptor in image1, attend to all descriptors in image2.

        Returns:
            desc1_enhanced: (N, D) desc1 enhanced with info from desc2
            desc2_enhanced: (M, D) desc2 enhanced with info from desc1
            cross_scores  : (N, M) raw similarity matrix
        """
        N, D = desc1.shape
        M    = desc2.shape[0]
        if N == 0 or M == 0:
            return desc1, desc2, np.zeros((N, M))

        scale = np.sqrt(D) * temperature

        # Cross similarity matrix: (N, M)
        cross_scores = desc1 @ desc2.T / scale

        # Softmax across M for each of N queries
        scores_1to2 = cross_scores - cross_scores.max(axis=1, keepdims=True)
        attn_1to2   = np.exp(scores_1to2)
        attn_1to2  /= attn_1to2.sum(axis=1, keepdims=True) + 1e-8
        desc1_enh   = 0.7 * desc1 + 0.3 * (attn_1to2 @ desc2)

        # Softmax across N for each of M queries
        scores_2to1 = cross_scores.T - cross_scores.T.max(axis=1, keepdims=True)
        attn_2to1   = np.exp(scores_2to1)
        attn_2to1  /= attn_2to1.sum(axis=1, keepdims=True) + 1e-8
        desc2_enh   = 0.7 * desc2 + 0.3 * (attn_2to1 @ desc1)

        return desc1_enh, desc2_enh, cross_scores

    # ── Confidence Estimator (MLP simulation) ────────────────────────────────
    def _estimate_confidence(self, desc1, desc2, cross_scores):
        """
        Lightweight confidence estimator — simulates the MLP in the PDF.

        PDF says: "MLP estimates confidence score after each transformer layer.
                   If confidence > 0.95, model exits early."

        We compute confidence as: mean of top-k mutual similarity scores.

        Returns:
            float: confidence in [0, 1]
        """
        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return 0.0

        # For each row, find the max score (best match score)
        max_scores = cross_scores.max(axis=1)  # (N,)

        # Apply sigmoid to map to [0, 1]
        confidence = 1.0 / (1.0 + np.exp(-max_scores.mean()))
        return float(confidence)

    # ── Mutual Nearest Neighbor Filter ───────────────────────────────────────
    def _mutual_nearest_neighbors(self, desc1, desc2):
        """
        Mutual Nearest Neighbor (MNN) matching.

        A → B match AND B → A match must agree. Removes many false positives.

        PDF says: "Initial round of descriptor matching using mutual nearest
                   neighbors provides rough alignment."

        Returns:
            matches: list of (idx1, idx2) tuples — mutual best matches
        """
        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # Cosine similarity matrix (descriptors should be L2-normalized)
        sim = desc1 @ desc2.T    # (N, M)

        # Forward: for each in desc1, find best in desc2
        nn1 = sim.argmax(axis=1)  # (N,)

        # Backward: for each in desc2, find best in desc1
        nn2 = sim.argmax(axis=0)  # (M,)

        # Mutual: keep only pairs where both agree
        matches = []
        for i, j in enumerate(nn1):
            if nn2[j] == i:
                matches.append((i, int(j)))

        return matches

    # ── Ratio Test Filter ─────────────────────────────────────────────────────
    def _ratio_test_matching(self, desc1, desc2):
        """
        Lowe's ratio test matching (used in addition to MNN for better recall).

        If best_match_dist / second_best_dist < threshold → keep (distinctive).

        Returns:
            matches: list of (idx1, idx2)
        """
        if desc1.shape[0] < 2 or desc2.shape[0] < 2:
            return []

        # Use FLANN for fast approximate nearest neighbor search
        FLANN_INDEX_KDTREE = 1
        index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        try:
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            desc1_f = desc1.astype(np.float32)
            desc2_f = desc2.astype(np.float32)
            knn = flann.knnMatch(desc1_f, desc2_f, k=2)
        except Exception:
            # Fallback: brute force
            bf  = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            knn = bf.knnMatch(desc1.astype(np.float32),
                              desc2.astype(np.float32), k=2)

        matches = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio_threshold * n.distance:
                matches.append((m.queryIdx, m.trainIdx))

        return matches

    # ── MAIN MATCH FUNCTION ───────────────────────────────────────────────────
    def match(self, kpts1, desc1, kpts2, desc2):
        """
        Full Adaptive Attention Matching pipeline.

        Steps:
          1. Initial MNN match (rough alignment)
          2. For l in range(max_layers):
               a. Self-attention on desc1 and desc2
               b. Cross-attention between desc1 and desc2
               c. Check confidence → Early Exit if confident
          3. Final ratio-test matching on refined descriptors
          4. Combine MNN + ratio-test matches

        Returns:
            matches     : list of (idx1, idx2) tuples
            exit_layer  : which layer triggered early exit
            confidence  : final confidence score
        """
        t0 = time.time()

        # Work on copies to avoid mutating original arrays
        d1 = desc1.copy().astype(np.float32)
        d2 = desc2.copy().astype(np.float32)

        # Step 1: Initial MNN matches (rough alignment — PDF's "mutual NN" step)
        initial_matches = self._mutual_nearest_neighbors(d1, d2)

        exit_layer = self.max_layers
        confidence = 0.0

        # Step 2: Iterative attention refinement
        for layer in range(1, self.max_layers + 1):

            # ── Self-attention: contextualize within each image ────────────
            d1 = self._self_attention(d1, temperature=0.15)
            d2 = self._self_attention(d2, temperature=0.15)

            # ── Cross-attention: establish global correspondences ─────────
            d1, d2, cross_scores = self._cross_attention(d1, d2, temperature=0.15)

            # ── Early Exit Check (PDF: MLP confidence > 0.95) ─────────────
            confidence = self._estimate_confidence(d1, d2, cross_scores)

            if confidence >= self.confidence_threshold:
                exit_layer = layer
                break   # 🚪 Early exit — enough confidence!

        # Step 3: Final ratio-test matching on REFINED descriptors
        ratio_matches = self._ratio_test_matching(d1, d2)

        # Step 4: Merge initial MNN + ratio-test, deduplicate
        all_match_set = set(initial_matches) | set(ratio_matches)
        matches = list(all_match_set)

        elapsed = (time.time() - t0) * 1000
        return matches, exit_layer, confidence, elapsed


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 3: HIERARCHICAL REFINEMENT
# PDF: "Multi-Scale Pyramid: 0.25x → 0.5x → 1x coarse-to-fine"
#      "Sub-Pixel MLP predicts (Δx, Δy) offsets at each level"
# ═════════════════════════════════════════════════════════════════════════════

class HierarchicalRefiner:
    """
    Coarse-to-fine refinement across three image scales.

    PDF says: "Images processed at 0.25x, 0.5x, and 1x resolution.
               Sub-Pixel MLP applied at each level to predict Δx, Δy offsets."

    Our implementation:
      - Extract features at each scale
      - Match at coarse scale for rough alignment
      - Propagate and refine at finer scales
      - Apply sub-pixel correction (quadratic interpolation approximation)
    """

    def __init__(self, scales=(0.25, 0.5, 1.0), n_features_per_scale=800):
        self.scales = scales
        self.n_features = n_features_per_scale

    def _resize_image(self, img, scale):
        """Resize image to given scale factor."""
        h, w = img.shape[:2]
        new_w = max(32, int(w * scale))
        new_h = max(32, int(h * scale))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _subpixel_correction(self, kpts, img_gray, window=3):
        """
        Sub-pixel keypoint correction using intensity gradient.

        PDF says: "Sub-Pixel MLP predicts fine-grained coordinate offsets
                   (Δx, Δy) for each keypoint."

        We simulate this with OpenCV's cornerSubPix — the standard
        gradient-based sub-pixel localization (equivalent to the MLP output
        in terms of correcting keypoint positions).

        Args:
            kpts    : (N, 2) float32 array of [x, y]
            img_gray: grayscale image at this scale
            window  : search window half-size

        Returns:
            (N, 2) refined keypoint coordinates
        """
        if len(kpts) == 0:
            return kpts

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        pts = kpts.reshape(-1, 1, 2).astype(np.float32)

        try:
            refined = cv2.cornerSubPix(
                img_gray, pts,
                winSize=(window, window),
                zeroZone=(-1, -1),
                criteria=criteria
            )
            return refined.reshape(-1, 2)
        except Exception:
            return kpts  # If fails, return original

    def _match_at_scale(self, img1_scaled, img2_scaled, scale):
        """
        Feature detection + matching at a specific scale.

        Returns:
            kpts1_scaled, kpts2_scaled : matched keypoint pairs at this scale
            kpts1_orig,  kpts2_orig    : same points scaled back to original
        """
        from features import extract_sift_features

        gray1 = img1_scaled if len(img1_scaled.shape) == 2 else \
                cv2.cvtColor(img1_scaled, cv2.COLOR_BGR2GRAY)
        gray2 = img2_scaled if len(img2_scaled.shape) == 2 else \
                cv2.cvtColor(img2_scaled, cv2.COLOR_BGR2GRAY)

        kp1, _, d1 = extract_sift_features(gray1, self.n_features)
        kp2, _, d2 = extract_sift_features(gray2, self.n_features)

        if len(kp1) < 4 or len(kp2) < 4:
            return None, None, None, None

        # Quick ratio-test matching
        bf  = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        try:
            knn = bf.knnMatch(d1.astype(np.float32), d2.astype(np.float32), k=2)
        except Exception:
            return None, None, None, None

        good = []
        for pair in knn:
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                good.append(pair[0])

        if len(good) < 4:
            return None, None, None, None

        pts1_s = np.float32([kp1[m.queryIdx] for m in good])
        pts2_s = np.float32([kp2[m.trainIdx] for m in good])

        # Sub-pixel correction at this scale (simulates PDF's MLP)
        pts1_s = self._subpixel_correction(pts1_s, gray1)
        pts2_s = self._subpixel_correction(pts2_s, gray2)

        # Scale back to original image coordinates
        pts1_orig = pts1_s / scale
        pts2_orig = pts2_s / scale

        return pts1_s, pts2_s, pts1_orig, pts2_orig

    def refine(self, img1_gray, img2_gray, initial_matches, kpts1, kpts2):
        """
        Run Hierarchical Refinement across all scales.

        PDF pipeline:
          0.25x: Coarse correspondences
          0.5x : MLP-based offset prediction
          1.0x : Final sub-pixel accuracy

        Returns:
            refined_kpts1, refined_kpts2 : (N,2) arrays at original resolution
            refinement_info              : dict with stats per scale
        """
        refinement_info = {}
        all_pts1 = []
        all_pts2 = []

        for scale in self.scales:
            # Resize to this scale
            h, w   = img1_gray.shape[:2]
            scaled1 = self._resize_image(img1_gray, scale)
            scaled2 = self._resize_image(img2_gray, scale)

            pts1_s, pts2_s, pts1_o, pts2_o = self._match_at_scale(
                scaled1, scaled2, scale)

            label = f'{scale}x'
            if pts1_o is not None:
                refinement_info[label] = len(pts1_o)
                all_pts1.append(pts1_o)
                all_pts2.append(pts2_o)
            else:
                refinement_info[label] = 0

        # Merge all scale results
        if not all_pts1:
            # Fallback: use original matches
            if initial_matches and len(initial_matches) > 0:
                valid_matches = [(i, j) for (i, j) in initial_matches
                                 if i < len(kpts1) and j < len(kpts2)]
                if valid_matches:
                    pts1 = np.array([kpts1[i] for (i, _) in valid_matches], np.float32)
                    pts2 = np.array([kpts2[j] for (_, j) in valid_matches], np.float32)
                    return pts1, pts2, refinement_info
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), refinement_info

        # Combine all scale contributions
        combined_pts1 = np.vstack(all_pts1)
        combined_pts2 = np.vstack(all_pts2)

        # Deduplicate nearby points (within 5px)
        combined_pts1, combined_pts2 = _deduplicate_matches(
            combined_pts1, combined_pts2, min_dist=5.0)

        refinement_info['total_refined'] = len(combined_pts1)
        return combined_pts1, combined_pts2, refinement_info


def _deduplicate_matches(pts1, pts2, min_dist=5.0):
    """Remove duplicate match pairs that are very close to each other."""
    if len(pts1) == 0:
        return pts1, pts2

    keep = np.ones(len(pts1), dtype=bool)
    for i in range(len(pts1)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(pts1)):
            if not keep[j]:
                continue
            d1 = np.linalg.norm(pts1[i] - pts1[j])
            d2 = np.linalg.norm(pts2[i] - pts2[j])
            if d1 < min_dist and d2 < min_dist:
                keep[j] = False   # Duplicate — remove j

    return pts1[keep], pts2[keep]


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 4: POST-PROCESSING — RANSAC FILTERING
# PDF: "RANSAC-based homography estimation to separate inliers from outliers"
# ═════════════════════════════════════════════════════════════════════════════

def ransac_filter(pts1, pts2, reproj_thresh=4.0, method='homography'):
    """
    RANSAC geometric verification.

    PDF says: "RANSAC (Random Sample Consensus) filters outliers.
               Essential Matrix computed for pose estimation."

    Two modes:
      'homography' : Find best planar transformation (for 2D scenes / same plane)
      'essential'  : Find Essential Matrix (for 3D scenes with known camera)

    Args:
        pts1, pts2     : (N, 2) matched point arrays
        reproj_thresh  : Max reprojection error for inlier (pixels)
        method         : 'homography' or 'fundamental'

    Returns:
        inlier_pts1    : (M, 2) inlier points from image 1
        inlier_pts2    : (M, 2) inlier points from image 2
        H              : 3x3 transformation matrix (or None)
        inlier_mask    : (N,) boolean array
        stats          : dict with n_total, n_inliers, inlier_ratio
    """
    stats = {'n_total': len(pts1), 'n_inliers': 0, 'inlier_ratio': 0.0}

    if len(pts1) < 4:
        mask = np.zeros(len(pts1), dtype=bool)
        return pts1, pts2, None, mask, stats

    if method == 'fundamental':
        # Fundamental Matrix (no camera intrinsics needed)
        try:
            F, mask = cv2.findFundamentalMat(
                pts1.reshape(-1, 1, 2),
                pts2.reshape(-1, 1, 2),
                cv2.FM_RANSAC,
                reproj_thresh,
                0.999
            )
            H = F
        except Exception:
            H, mask = None, None
    else:
        # Homography (best for planar scenes, monuments, buildings)
        try:
            H, mask = cv2.findHomography(
                pts1.reshape(-1, 1, 2),
                pts2.reshape(-1, 1, 2),
                cv2.RANSAC,
                reproj_thresh,
                maxIters=2000,
                confidence=0.995
            )
        except Exception:
            H, mask = None, None

    if mask is None:
        mask = np.zeros(len(pts1), dtype=bool)
    else:
        mask = mask.ravel().astype(bool)

    inlier_pts1 = pts1[mask]
    inlier_pts2 = pts2[mask]

    n_inliers = int(mask.sum())
    stats['n_inliers']    = n_inliers
    stats['inlier_ratio'] = round(n_inliers / max(len(pts1), 1) * 100, 1)

    return inlier_pts1, inlier_pts2, H, mask, stats
