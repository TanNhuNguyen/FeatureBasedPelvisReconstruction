# 🦴 Feature-Based Pelvis Reconstruction Dataset

This repository contains the dataset required to run the **Feature-Based Pelvis Reconstruction** system. It includes calibration images, template models, and structured data for training, validation, and anatomical analysis.

---

## 📥 Download Instructions

You can download the dataset ZIP file from the following Google Drive link:

🔗 [Download Pelvis Reconstruction Data](https://drive.google.com/file/d/1HJXa2jRTfCUPpGTBp6S0vjX9AOLcHgO0/view?usp=sharing)

---

## 📦 Setup Instructions

After downloading the ZIP file:

1. **Extract the contents**:
   - On Windows: Right-click the ZIP file → “Extract All…”
   - On macOS: Double-click the ZIP file to unzip.
   - On Linux:
     ```bash
     unzip PelvisReconstructionData.zip
     ```

2. **Place the extracted folders** inside your project’s `./Data` directory:
   ```
   ./Data/
   ├── PelvisBoneRecon/
   └── Template/
   ```

3. Ensure your working directory includes the following subfolders (used by the code):
   ```
   ./Data/
   ├── Template/
   ├── PelvisBoneRecon/
		├──CrossValidation
			├──AffineDeformation
			├──AugmentedTestingIDs
			├──FeatureSelectionProtocol
			├──RadialBasicFunctionStrategy
			├──ShapeRelationStrategy
			└──TrainingValidTestingIDs
		├──Debugs
		└──FemalePelvisGeometries
   ```

---

## 🗂 Folder Structure and Contents

### `PelvisBoneRecon/`

Contains the core data used for model training, validation, and analysis.

#### `CrossValidation/`

Used for evaluating model performance across different strategies and data splits.

- `AffineDeformation/`: Saving the pelvic reconstruction results using the affine transform strategy.
- `AugmentedTestingIDs/`: IDs of samples used for testing with data augmentation.
- `FeatureSelectionProtocol/`: Feature selection strategies and metadata.
- `RadialBasicFunctionStrategy/`:  Saving the pelvic reconstruction results using the radial basis function strategy.
- `ShapeOptimizationStrategy/`: Saving the pelvic reconstruction results using the shape optimization strategy.
- `ShapeRelationStrategy/`: Saving the pelvic reconstruction results using the shape relation strategy.
- `TrainingValidTestingIDs/`: Lists of IDs for training, validation, and testing splits.

#### `Debugs/`

Contains debug outputs such as camera positions, intermediate visualizations, and logs captured during calibration and reconstruction.

#### `FemalePelvisGeometries/`

Stores anatomical geometry data for female pelvis models.

- `PersonalizedPelvisStructures/`: Individualized 3D pelvis reconstructions.
- `ShapeVariationAnalyses/`: Statistical shape analysis results across subjects.

---

### `Template/`

Includes template models and reference geometries used for registration and alignment during reconstruction.

---

## 🛠 Dependencies

Make sure the following Python libraries are installed before running the code:

- `numpy`, `cv2`, `SimpleITK`, `pyvista`, `trimesh`, `scikit-learn`, `matplotlib`, `pydicom`, `pymeshlab`, `tqdm`, `scipy`

You can install them using:
```bash
pip install -r requirements.txt
```

---
