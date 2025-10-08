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
     unzip Data.zip
     ```

2. **Place the extracted folders** inside your project’s `./Data` directory:
   ```
   ./Data/
   └── PelvisBoneRecon/
   ```

3. Ensure your working directory includes the following subfolders (used by the code):
   ```
   ./Data/
      ├── PelvisBoneRecon/
         ├──CrossValidation
            ├──FeatureSelectionProtocol
            └──TrainingValidTestingIDs
         └──FemalePelvisGeometries
            └──PersonalizedPelvisStructures
      └──Template
         └──PelvisBonesMuscles
            └──FeatureSelectionProtocol
   ```

---

## 🗂 Folder Structure and Contents

### `PelvisBoneRecon/`

Contains the core data used for model training, validation, and analysis.

#### `CrossValidation/`

Used for evaluating model performance across different strategies and data splits.

- `FeatureSelectionProtocol/`: Feature selection strategies, bary centric, and bary coordinates of the feature points.
- `TrainingValidTestingIDs/`: Lists of IDs for training, validation, and testing splits.

#### `FemalePelvisGeometries/`

Stores anatomical geometry data for female pelvis models.

- `PersonalizedPelvisStructures/`: Individualized 3D pelvis reconstructions.

### `Template/`

Contains the template structure of the pelvic bone and muscles, including all ROI vertex, faces, and vertex indices of each pelvic region.

#### `PelvisBonesMuscles/`

Contains the files of *.csv for saving the vertex indices for each of pelvic bone structure, *.ply file for saving the template pelvic structure mesh, and the *.pp file containing the feature points of the pelvic structure in each reigon of interest parts.

##### `FeatureSelectionProtocol/`

Contains the feature selection protocals as well as the bary centric coordinate and bary facet indices for studying the relationship between the feature points and pelvic structures.

---