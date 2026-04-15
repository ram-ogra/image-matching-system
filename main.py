"""
main.py
=======
Image Matching Pipeline Runner
Orchestrates all 4 stages from the PDF:
  Stage 1: Hybrid Feature Extraction
  Stage 2: Adaptive Attention Matching
  Stage 3: Hierarchical Refinement
  Stage 4: Post-processing (RANSAC)

Can be run standalone OR imported by app.py (Streamlit UI).
"""

import cv2
import numpy as np
import time
import os

from features import hybrid_feature_extraction
from matcher import AdaptiveAttentionMatcher, HierarchicalRefiner, ransac_filter
from utils import (load_image_from_path, draw_keypoints_on_image,
                   draw_matches_result, numpy_to_pil, format_pipeline_stats)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_matching_pipeline(
    img1_rgb, img1_gray,
    img2_rgb, img2_gray,
    n_features=1500,
    feature_mode='hybrid',
    max_attention_layers=9,
    confidence_threshold=0.85,
    ratio_threshold=0.75,
    ransac_thresh=4.0,
    use_hierarchical=True,
    progress_callback=None
):
    """
    Run the complete 4-stage image matching pipeline from the PDF.

    Args:
        img1_rgb, img2_rgb   : RGB images (H,W,3) numpy arrays
        img1_gray, img2_gray : Grayscale images (H,W) numpy arrays
        n_features           : Max keypoints per image per method
        feature_mode         : 'hybrid' | 'sift_only' | 'orb_only'
        max_attention_layers : Max transformer layers (PDF uses 9)
        confidence_threshold : Early exit confidence threshold
        ratio_threshold      : Lowe's ratio test threshold
        ransac_thresh        : RANSAC reprojection error threshold
        use_hierarchical     : Whether to run Stage 3 (slower but more accurate)
        progress_callback    : Optional function(stage: str, pct: float) for UI

    Returns:
        result dict with all outputs + stats
    """
    t_total_start = time.time()

    def progress(stage, pct):
        if progress_callback:
            progress_callback(stage, pct)

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 1: HYBRID FEATURE EXTRACTION
    # PDF: "DISK Detector → keypoints → SuperPoint Descriptor → 256-D descriptors"
    # ══════════════════════════════════════════════════════════════════════════
    progress("🔍 Stage 1: Hybrid Feature Extraction", 10)
    t1 = time.time()

    kpts1, descs1, info1 = hybrid_feature_extraction(img1_gray, n_features, feature_mode)
    kpts2, descs2, info2 = hybrid_feature_extraction(img2_gray, n_features, feature_mode)

    t_stage1 = (time.time() - t1) * 1000

    feature_info = {
        'n_kpts1': len(kpts1),
        'n_kpts2': len(kpts2),
        'method':  feature_mode,
        'time_ms': round(t_stage1, 1),
        'info1':   info1,
        'info2':   info2,
    }

    if len(kpts1) < 4 or len(kpts2) < 4:
        return {
            'success': False,
            'error':   f"Not enough keypoints: img1={len(kpts1)}, img2={len(kpts2)}",
            'feature_info': feature_info,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2: ADAPTIVE ATTENTION MATCHING
    # PDF: "Transformer layers 1-9, Cross-Attention, Self-Attention, Early Exit"
    # ══════════════════════════════════════════════════════════════════════════
    progress("🤝 Stage 2: Adaptive Attention Matching", 35)
    t2 = time.time()

    attention_matcher = AdaptiveAttentionMatcher(
        max_layers=max_attention_layers,
        confidence_threshold=confidence_threshold,
        ratio_threshold=ratio_threshold
    )

    raw_matches, exit_layer, confidence, attn_ms = attention_matcher.match(
        kpts1, descs1, kpts2, descs2)

    t_stage2 = (time.time() - t2) * 1000

    match_stats = {
        'n_raw':       len(raw_matches),
        'exit_layer':  exit_layer,
        'confidence':  round(confidence, 4),
        'time_ms':     round(t_stage2, 1),
    }

    if len(raw_matches) < 4:
        return {
            'success': False,
            'error':   f"Too few matches after attention: {len(raw_matches)}",
            'feature_info':  feature_info,
            'match_stats':   match_stats,
        }

    # Convert match indices to point arrays
    valid_matches = [(i, j) for (i, j) in raw_matches
                     if i < len(kpts1) and j < len(kpts2)]
    raw_pts1 = np.array([kpts1[i] for (i, _) in valid_matches], dtype=np.float32)
    raw_pts2 = np.array([kpts2[j] for (_, j) in valid_matches], dtype=np.float32)

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 3: HIERARCHICAL REFINEMENT
    # PDF: "Multi-Scale Pyramid: 0.25x→coarse, 0.5x→refine, 1x→sub-pixel"
    # ══════════════════════════════════════════════════════════════════════════
    progress("🔬 Stage 3: Hierarchical Refinement", 60)
    t3 = time.time()
    refinement_info = {}

    if use_hierarchical and len(raw_matches) >= 4:
        refiner = HierarchicalRefiner(
            scales=(0.25, 0.5, 1.0),
            n_features_per_scale=min(n_features // 2, 600)
        )
        ref_pts1, ref_pts2, refinement_info = refiner.refine(
            img1_gray, img2_gray, valid_matches, kpts1, kpts2)

        # Merge refined + raw matches (refined get priority)
        if len(ref_pts1) > 0:
            # Use refined points as primary
            pts_for_ransac1 = ref_pts1
            pts_for_ransac2 = ref_pts2
        else:
            pts_for_ransac1 = raw_pts1
            pts_for_ransac2 = raw_pts2
    else:
        pts_for_ransac1 = raw_pts1
        pts_for_ransac2 = raw_pts2
        refinement_info = {'0.25x': 0, '0.5x': 0, '1.0x': 0}

    t_stage3 = (time.time() - t3) * 1000

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 4: POST-PROCESSING — RANSAC FILTERING
    # PDF: "RANSAC-based homography estimation, Essential Matrix computation"
    # ══════════════════════════════════════════════════════════════════════════
    progress("✅ Stage 4: RANSAC Post-processing", 85)
    t4 = time.time()

    inlier_pts1, inlier_pts2, H_matrix, inlier_mask, ransac_stats = ransac_filter(
        pts_for_ransac1, pts_for_ransac2,
        reproj_thresh=ransac_thresh,
        method='homography'
    )

    t_stage4 = (time.time() - t4) * 1000
    t_total   = (time.time() - t_total_start) * 1000

    progress("🎨 Generating visualizations...", 95)

    # ── Build visualizations ──────────────────────────────────────────────────

    # 1. Images with keypoints
    kpts1_img = draw_keypoints_on_image(img1_rgb, kpts1, color=(0, 220, 80), radius=4)
    kpts2_img = draw_keypoints_on_image(img2_rgb, kpts2, color=(0, 220, 80), radius=4)

    # 2. All raw matches (before RANSAC)
    raw_result_img = draw_matches_result(
        img1_rgb, img2_rgb,
        raw_pts1, raw_pts2,
        inlier_mask=None,
        max_lines=200
    )

    # 3. Final clean matches (RANSAC inliers only — PDF style green lines)
    if len(pts_for_ransac1) > 0:
        final_result_img = draw_matches_result(
            img1_rgb, img2_rgb,
            pts_for_ransac1, pts_for_ransac2,
            inlier_mask=inlier_mask,
            max_lines=300,
            line_thickness=1
        )
    else:
        final_result_img = np.hstack([img1_rgb, img2_rgb])

    progress("✨ Done!", 100)

    # ── Compile all stats ─────────────────────────────────────────────────────
    pipeline_stats = format_pipeline_stats(
        feature_info, match_stats, ransac_stats,
        refinement_info, t_total
    )

    pipeline_stats["⏱️ Stage 1 Time"] = f"{t_stage1:.0f} ms"
    pipeline_stats["⏱️ Stage 2 Time"] = f"{t_stage2:.0f} ms"
    pipeline_stats["⏱️ Stage 3 Time"] = f"{t_stage3:.0f} ms"
    pipeline_stats["⏱️ Stage 4 Time"] = f"{t_stage4:.0f} ms"

    return {
        'success':         True,
        'kpts1':           kpts1,
        'kpts2':           kpts2,
        'raw_matches':     valid_matches,
        'inlier_pts1':     inlier_pts1,
        'inlier_pts2':     inlier_pts2,
        'H_matrix':        H_matrix,
        'inlier_mask':     inlier_mask,

        # Visualizations
        'kpts1_img':       kpts1_img,
        'kpts2_img':       kpts2_img,
        'raw_result_img':  raw_result_img,
        'final_result_img':final_result_img,

        # Stats
        'feature_info':    feature_info,
        'match_stats':     match_stats,
        'ransac_stats':    ransac_stats,
        'refinement_info': refinement_info,
        'pipeline_stats':  pipeline_stats,
        'total_time_ms':   round(t_total, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse, sys

    parser = argparse.ArgumentParser(description='Image Matching Pipeline (CLI)')
    parser.add_argument('--img1',    required=True,  help='Path to image 1')
    parser.add_argument('--img2',    required=True,  help='Path to image 2')
    parser.add_argument('--output',  default='output_match.jpg', help='Output path')
    parser.add_argument('--mode',    default='hybrid',
                        choices=['hybrid', 'sift_only', 'orb_only'])
    parser.add_argument('--nfeats',  type=int, default=1500, help='Max keypoints')
    parser.add_argument('--no-hier', action='store_true',
                        help='Skip hierarchical refinement (faster)')
    args = parser.parse_args()

    print("=" * 60)
    print("  IMAGE MATCHING PIPELINE")
    print(f"  Image 1 : {args.img1}")
    print(f"  Image 2 : {args.img2}")
    print(f"  Mode    : {args.mode}")
    print("=" * 60)

    _, img1_rgb, img1_gray, _ = load_image_from_path(args.img1)
    _, img2_rgb, img2_gray, _ = load_image_from_path(args.img2)

    def cli_progress(stage, pct):
        print(f"  [{pct:3.0f}%] {stage}")

    result = run_matching_pipeline(
        img1_rgb, img1_gray,
        img2_rgb, img2_gray,
        n_features=args.nfeats,
        feature_mode=args.mode,
        use_hierarchical=not args.no_hier,
        progress_callback=cli_progress
    )

    if not result['success']:
        print(f"\n[ERROR] {result['error']}")
        sys.exit(1)

    # Save result
    out_bgr = cv2.cvtColor(result['final_result_img'], cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, out_bgr)
    print(f"\n[Saved] {args.output}")

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    for k, v in result['pipeline_stats'].items():
        print(f"  {k:<35} {v}")
    print("=" * 60)
