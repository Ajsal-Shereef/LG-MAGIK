# VAE Quality Inspection Dataset

This directory contains visual inspection samples and diagnostic artifacts used to validate the **Text-Conditioned VAE quality analysis engine** across both valid transformations and failure modes.

---

## 1. Dataset Breakdown

| Category | Sample Count | Source | Quality Status | Primary Diagnostic Finding |
| :--- | :---: | :--- | :---: | :--- |
| **`vae_good_data`** | **64** | `MiniWorld` Target 1 (Seed 42) execution trajectory | **PASS** (100%) | Localized object swaps confirmed; dominant blob ratio $\ge 0.35$; required target color pixels verified ($\ge 80\text{ px}$). |
| **`vae_error_data`** | **5** | Relocation instructions on `000071.png` (same objects, different locations) | **FAIL** (100%) | `SEMANTIC_PRESENCE` failures: missing relocated objects, spatial sector mismatches. |

Total Manifest Entries: **69**

---

## 2. Relocated Object Error Samples (`000071.png`)

All error samples were synthesized using the real observation [`data/MiniWorld/agent/images/000071.png`](file:///weka/s222147455/LG-MAGIK/data/MiniWorld/agent/images/000071.png) without introducing any new objects. The prompt retains only the original scene objects (**`blue box`** and **`green ball`**) while instructing the VAE to synthesize them in different locations (angles, distances, and sectors).

- **Original Image `000071.png` State**: Blue box on the left ($x \approx 39$) and green ball on the far right ($x \approx 71$).
- **Original Caption**: *"The agent is in a room with grass floor and concrete walls. A blue box is found to the left at angle 0.382 at a distance of 3.8 units. A green ball is found to the right at angle 33.5 at a distance of 1.3 units."*

### Error Samples Summary

| # | Sample Name | Relocation Instruction / Caption | VAE Error Type | Failed Component | Diagnostic Reason |
| :-: | :--- | :--- | :---: | :---: | :--- |
| **1** | `MiniWorld_error_000071_swapped_sectors` | *"A green ball is found to the left at angle 0.382 at a distance of 3.8 units. A blue box is found to the right at angle 33.5 at a distance of 1.3 units."* | `VAE_SEMANTIC_MISSING_OBJECT` | `SEMANTIC_PRESENCE` | Inverted sectors: Model failed to synthesize blue box in right sector (only 2 blue px found, required $\ge 80$). |
| **2** | `MiniWorld_error_000071_inverted_dist_swap` | *"A green ball is found to the left at angle 10.5 at a distance of 1.0 units. A blue box is found to the right at angle 25.0 at a distance of 4.5 units."* | `VAE_SEMANTIC_MISSING_OBJECT` | `SEMANTIC_PRESENCE` | Distance & sector relocation: Model failed to synthesize blue box in right sector (0 blue px found). |
| **3** | `MiniWorld_error_000071_both_right_skewed` | *"A blue box is found to the right at angle 40.0 at a distance of 1.0 units. A green ball is found to the right at angle 10.0 at a distance of 3.5 units."* | `VAE_SEMANTIC_SECTOR_MISMATCH` | `SEMANTIC_PRESENCE` | Extreme angle skew: Green ball only has 47.4% of pixels in right sector (threshold $\ge 50\%$). |
| **4** | `MiniWorld_error_000071_symmetrical_swap` | *"A green ball is found to the left at angle 20.0 at a distance of 1.5 units. A blue box is found to the right at angle 20.0 at a distance of 2.0 units."* | `VAE_SEMANTIC_MISSING_OBJECT` | `SEMANTIC_PRESENCE` | Symmetrical lateral inversion: Model generated only 3 blue px in right sector. |
| **5** | `MiniWorld_error_000071_polar_opposite_swap` | *"A green ball is found to the left at angle 35.0 at a distance of 1.0 units. A blue box is found to the right at angle 15.0 at a distance of 3.5 units."* | `VAE_SEMANTIC_MISSING_OBJECT` | `SEMANTIC_PRESENCE` | Wide angle inversion: Model failed to synthesize blue box in right sector (0 blue px found). |

---

## 3. Directory Layout

```
Results/inspected_samples/
├── manifest.json                  # Complete metadata with category labels and VAE evaluations
├── README.md                      # Documentation report
├── original/                      # 200x200 upscaled original observations
├── reconstructed/                 # 200x200 upscaled imagined VAE outputs
├── combined/                      # 408x200 side-by-side composite images
├── masks/                         # 200x200 morphological binary difference masks
└── native_80x80/
    ├── original/                  # Native resolution original frames
    └── reconstructed/             # Native resolution VAE frames
```

---

## 4. Verification Check

Running `evaluate_vae_quality` across the entire manifest demonstrates:
1. **0 False Positives**: All 64 original trajectory samples pass (`verdict: PASS`).
2. **0 False Negatives**: All 5 location-relocated error samples fail (`verdict: FAIL`) with detailed diagnostics and sector-accurate reasoning.
