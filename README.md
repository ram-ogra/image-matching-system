# 🔍 Image Matching System — Hybrid Deep Learning Pipeline

**Project by:** Ramswroop Ogra (12212024), IIIT Sonepat  
**Based on:** PDF Report — Image Matching using Hybrid Deep Learning

---

## 📁 Project Structure

```
image_matching_project/
├── app.py           # Streamlit UI (run this!)
├── main.py          # Pipeline runner + CLI
├── features.py      # Stage 1: Hybrid Feature Extraction
├── matcher.py       # Stage 2+3+4: Attention + Refinement + RANSAC
├── utils.py         # Helper functions
├── requirements.txt # Dependencies
└── README.md        # This file
```

---

## ⚡ Setup & Run (VS Code + Git Bash)

### Step 1: Open VS Code and Git Bash terminal

Press `Ctrl + ~` in VS Code to open terminal, then switch to Git Bash.

---

### Step 2: Create project folder

```bash
mkdir image_matching_project
cd image_matching_project
```

---

### Step 3: Create virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate it (Git Bash)
source venv/Scripts/activate

# You should see (venv) in your terminal now ✅
```

---

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ This installs: streamlit, opencv-python, opencv-contrib-python, numpy, Pillow, matplotlib, scipy, scikit-image

---

### Step 5: Run the Streamlit App

```bash
streamlit run app.py
```

> Browser will open automatically at: `http://localhost:8501`

---

## 🖥️ How to Use the App

1. **Upload Image 1** — your reference image (e.g., Taj Mahal morning photo)
2. **Upload Image 2** — your query image (e.g., Taj Mahal evening photo)
3. Adjust settings in the **sidebar** if needed
4. Click **"🚀 Run Image Matching Pipeline"**
5. View results:
   - Detected keypoints on each image
   - Final matched lines (green = correct, red = outlier)
   - Detailed stats for all 4 pipeline stages
6. **Download** the output images

---

## 🔧 CLI Usage (without UI)

```bash
# Basic run
python main.py --img1 path/to/img1.jpg --img2 path/to/img2.jpg

# With options
python main.py \
  --img1 images/taj1.jpg \
  --img2 images/taj2.jpg \
  --mode hybrid \
  --nfeats 1500 \
  --output result.jpg

# Fast mode (skip hierarchical refinement)
python main.py --img1 img1.jpg --img2 img2.jpg --no-hier
```

---

## 🧠 Pipeline Explanation

### Stage 1: Hybrid Feature Extraction
- **SIFT** (simulates SuperPoint): Scale/rotation invariant 128-D descriptors
- **ORB** (simulates DISK): Dense binary features, ultra-fast
- **Fusion**: Combine both, deduplicate overlapping keypoints

### Stage 2: Adaptive Attention Matching
- **Self-Attention**: Refine each image's descriptors using context from neighbors
- **Cross-Attention**: Find correspondences across both images
- **Early Exit**: Stop when confidence > threshold (saves compute!)
- **MNN + Ratio Test**: Mutual nearest neighbors + Lowe's ratio test

### Stage 3: Hierarchical Refinement (0.25x → 0.5x → 1x)
- **Coarse (0.25x)**: Find approximate correspondences at low resolution
- **Medium (0.5x)**: Refine with MLP-based sub-pixel correction
- **Fine (1.0x)**: Final sub-pixel accuracy using cornerSubPix

### Stage 4: Post-processing (RANSAC)
- **RANSAC Homography**: Remove geometrically inconsistent matches
- **Inlier selection**: Keep only verified matches
- **Output**: Clean match visualization (green lines)

---

## ⚙️ Performance Tips (i5 CPU laptop)

| Setting | Fast | Balanced | Best Quality |
|---------|------|----------|--------------|
| Keypoints | 500 | 1000 | 2000 |
| Attn Layers | 3 | 6 | 9 |
| Image Size | 512 | 768 | 1024 |
| Hierarchical | OFF | ON | ON |
| Expected Time | ~2s | ~5s | ~10s |

---

## 📊 Expected Results

For two photos of the same building:
- **Keypoints detected**: 500–2000 per image
- **Raw matches**: 100–500
- **Inlier matches**: 50–300 (depends on similarity)
- **Inlier ratio**: 60–90%
- **Total time (CPU)**: 3–10 seconds

---

## 🐛 Troubleshooting

**"No matches found"**
→ Try images with more overlap (>30% shared area)

**"SIFT_create error"**
→ Run: `pip install opencv-contrib-python`

**Very slow on CPU**
→ Set Image Size to 512, Keypoints to 500, Layers to 3, disable Hierarchical

**ModuleNotFoundError**
→ Make sure venv is activated: `source venv/Scripts/activate`

---

## 📚 References

1. Sarlin et al. (2020) — SuperGlue: CVPR 2020
2. DeTone et al. (2018) — SuperPoint: CVPR 2018  
3. Lowe (2004) — SIFT: IJCV 2004
4. Rublee et al. (2011) — ORB: ICCV 2011
