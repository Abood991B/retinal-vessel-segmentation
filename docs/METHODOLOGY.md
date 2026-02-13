# Methodology — Retinal Vessel Segmentation Pipeline

## 1. Problem Statement

Retinal fundus photography captures the interior surface of the eye, revealing the network of blood vessels that supply the retina. Automated segmentation of these vessels is a foundational step in computer-aided diagnosis of:

- **Diabetic retinopathy** — Abnormal vessel growth and microaneurysms
- **Glaucoma** — Changes in vessel calibre near the optic nerve head
- **Hypertensive retinopathy** — Vessel narrowing and arteriovenous nicking

The challenge lies in accurately distinguishing thin, low-contrast vessels from the textured retinal background, under varying illumination conditions and across diverse patient populations.

## 2. Design Rationale

### Why Classical Image Processing?

While deep learning methods (e.g., U-Net, SegNet) achieve state-of-the-art vessel segmentation accuracy, they require:
- Large annotated training datasets
- GPU hardware for training and inference
- Significant compute time

This project demonstrates that a **carefully engineered classical pipeline** can achieve strong results with:
- Zero training data (unsupervised)
- CPU-only execution (<1s per image)
- Full interpretability at every stage
- Minimal dependencies (OpenCV + NumPy)

### Why the Green Channel?

In RGB fundus images, the green channel exhibits the highest contrast between vessels (dark) and retinal tissue (bright). The red channel is often saturated, and the blue channel has low signal-to-noise ratio due to absorption by the lens.

| Channel | Vessel Contrast | Noise Level | Selected? |
|---------|----------------|-------------|-----------|
| Red     | Low (saturated) | Low | ✗ |
| **Green** | **High** | **Moderate** | **✓** |
| Blue    | Moderate | High | ✗ |

## 3. Pipeline Architecture

### Stage 1: Green Channel Extraction

```
Input:  BGR image (H x W x 3), dtype uint8
Output: Single-channel image (H x W), dtype uint8
```

Simply extracts `image[:, :, 1]`. No conversion or scaling needed.

### Stage 2: Morphological Background Subtraction

**Problem solved:** Fundus images suffer from non-uniform illumination — the optic disc region is very bright while the periphery is dark. Fine vessels in both regions become invisible when the illumination gradient dominates.

**How it works:**
1. Apply morphological **closing** with a large circular structuring element (`bg_kernel_size = 21`)
2. The closing operation estimates the low-frequency illumination pattern (background envelope)
3. Subtract this estimated background from the original green channel
4. The result enhances vessels uniformly across the entire retina — the single biggest driver of improved recall

**Parameters:**
- `bg_kernel_size = 21` — Must exceed the widest vessel diameter in the fundus. Larger = smoother background estimate.

**Why this stage matters:** Without background subtraction, vessels near the bright optic disc are much harder to threshold than vessels in the dark periphery. By removing the illumination gradient, all vessels become similarly dark against a uniform background.

### Stage 3: CLAHE (Contrast-Limited Adaptive Histogram Equalization)

**Problem solved:** Even after background subtraction, local contrast can vary. CLAHE further normalizes regional differences.

**How CLAHE works:**
1. Divide the image into a grid of tiles (8x8 for fine-grained adaptation)
2. Compute the histogram for each tile
3. Clip the histogram at a threshold (`clipLimit`) to prevent over-amplification
4. Redistribute clipped pixels equally across all bins
5. Equalize each tile independently
6. Bilinearly interpolate tile boundaries for smooth transitions

**Parameters:**
- `clipLimit = 2.0` (main) / `2.5` (supplementary) — Controls contrast amplification ceiling
- `tileGridSize = (8, 8)` — Fine-grained grid reveals faint capillaries that a coarse 3x3 grid would miss

### Stage 4: Median Filtering

**Problem solved:** Impulse (salt-and-pepper) noise from camera sensors.

**Why median over Gaussian?** The median filter replaces each pixel with the median of its neighbourhood, making it optimal for removing outlier pixels while **preserving sharp vessel edges** — Gaussian blur would soften the very edges we need.

**Parameter:** `ksize = 3` — 3x3 neighbourhood

### Stage 5: Local Mean-Adaptive Thresholding

**The core segmentation step.** This is a sliding-window approach:

1. Compute the local mean in an `n x n` neighbourhood around each pixel
2. Subtract the local mean from the actual intensity
3. If the difference exceeds threshold `C`, classify as vessel

```
vessel(x,y) = 1  if  I(x,y) - mean_local(x,y) > C
              0  otherwise
```

**Why adaptive (local) thresholding?**
A global threshold fails because vessel intensity varies across the fundus. The local mean automatically adapts to regional brightness.

**Parameters:**
- `n = 18` — Neighbourhood size (approximately 2x the width of the thickest vessels)
- `C = 6` — Positive offset: after background subtraction, vessels are *brighter* than their local surroundings by at least C intensity units

**Note:** Unlike the original pipeline where C was negative (vessels darker than background), the background subtraction stage inverts the contrast relationship, making vessels bright. The offset is therefore positive.

### Stage 6: Morphological Refinement

Apply **closing followed by opening** with an elliptical structuring element:

- **Closing** (dilate -> erode): Fills narrow breaks in vessel segments, reconnecting fragmented vessels
- **Opening** (erode -> dilate): Removes small isolated noise blobs

**Parameter:** `kernel_size = 3` with elliptical shape for isotropic smoothing

### Stage 7: Connected-Component Area Filtering

**Problem solved:** After thresholding, many small isolated blobs remain that are not true vessels. These hurt precision.

**How it works:**
1. Find all connected components in the binary mask using 8-connectivity
2. Compute the area (pixel count) of each component
3. Remove any component whose area falls below `min_vessel_area`

**Parameters:**
- `min_vessel_area = 120` (main) / `50` (supplementary) — Threshold in pixels. Higher = more aggressive noise removal (better precision), but risks losing thin vessel segments.

**Impact:** This stage dramatically improves precision without significantly affecting recall, because true vessel structures are spatially connected and cover many pixels.

### Stage 8: ROI Masking

Fundus images typically show a circular retinal disc against a black background. Without masking, black background pixels may be incorrectly classified.

**Algorithm:**
1. Convert original image to grayscale
2. Apply Otsu's thresholding to separate retina from background
3. Find all external contours
4. Select the largest contour (the retinal disc)
5. Create a filled mask from this contour
6. Refine with morphological close/open

**Result:** A binary mask where 255 = inside retina, 0 = outside.

### Stage 9: Gaussian Smoothing + Final Binarisation

A light Gaussian blur (`sigma = 0.5`, kernel 3x3) softens the jagged pixel-level boundaries that result from hard thresholding. This produces more natural-looking vessel edges without significantly affecting accuracy.

The smoothed image is then thresholded at intensity 128 to produce the final binary mask: `{0, 1}`.

## 4. Evaluation Framework

### Metrics

Given binary masks **P** (prediction) and **G** (ground truth):

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Precision** | TP / (TP + FP) | % of predicted vessels that are correct |
| **Recall** | TP / (TP + FN) | % of actual vessels that are detected |
| **F1-Score** | 2·P·R / (P + R) | Harmonic mean — balances precision/recall |
| **IoU** | TP / (TP + FP + FN) | Overlap between prediction and GT |
| **Accuracy** | (TP + TN) / N | Overall pixel classification accuracy |
| **Specificity** | TN / (TN + FP) | Background classification accuracy |

Where:
- **TP** = True Positive (correctly identified vessel pixels)
- **FP** = False Positive (background pixels incorrectly marked as vessel)
- **FN** = False Negative (vessel pixels incorrectly marked as background)
- **TN** = True Negative (correctly identified background pixels)

### Why F1 and IoU Over Accuracy?

Retinal images have severe class imbalance: vessels occupy ~10–15% of pixels. A naive "predict all background" classifier would achieve ~85% accuracy. **F1 and IoU account for this imbalance** by focusing on the overlap between prediction and ground truth.

## 5. Parameter Sensitivity & Optimisation

The pipeline's performance is most sensitive to four parameters, which were optimised via automated grid search over 504 configurations per dataset:

| Parameter | Range Tested | Optimal (Main) | Optimal (Add) | Impact |
|-----------|-------------|----------------|---------------|--------|
| `bg_kernel_size` | [21, 25, 29] | 21 | 21 | Background estimation scale; smaller = more aggressive vessel enhancement |
| `clahe_clip_limit` | [1.5, 2.0, 2.5, 3.0] | 2.0 | 2.5 | Contrast amplification; too high amplifies noise |
| `threshold_offset` | [6, 7, 8, ..., 12] | 6 | 6 | Sensitivity: lower = more detections (higher recall, lower precision) |
| `min_vessel_area` | [30, 50, 80, ..., 150] | 120 | 50 | Noise removal: higher = more aggressive (higher precision, risks losing thin vessels) |

The optimiser works by evaluating each configuration on a 10-image sample subset and selecting the combination that maximises F1-score.

## 6. Limitations & Future Work

### Current Limitations
- Fixed parameters per dataset -- does not adapt to individual image quality
- No vessel-orientation-specific enhancement (e.g., Gabor/Frangi filters)
- Background subtraction kernel size must be manually set above the widest vessel

### Implemented Improvements (vs. Baseline)
1. **Morphological background subtraction** -- +6.3 pp recall on main dataset
2. **Connected-component area filtering** -- significant precision recovery
3. **Automated parameter optimisation** -- 504-configuration grid search
4. **Fine-grained CLAHE (8x8)** -- better local contrast adaptation

### Potential Future Enhancements
1. **Frangi vesselness filter** -- Hessian-based tubular structure enhancement
2. **Gabor filter bank** -- Oriented filters at multiple angles for vessel detection
3. **Multi-resolution analysis** -- Process at 2-3 scales and combine
4. **Per-image adaptive parameters** -- Use image statistics to select optimal config
5. **Deep learning fusion** -- Use classical features as input to a lightweight CNN
