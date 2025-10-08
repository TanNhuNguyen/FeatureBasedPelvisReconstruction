# Feature-Based Pelvis Reconstruction System

## Overview

This project implements a comprehensive system for reconstructing pelvis bone and muscle structures from anatomical feature points using advanced machine learning techniques. The system combines statistical shape modeling, various regression approaches, and cross-validation frameworks to achieve accurate pelvis reconstruction for medical and biomechanical applications.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Core Functions](#core-functions)
- [Usage Examples](#usage-examples)
- [Data Organization](#data-organization)
- [Reconstruction Strategies](#reconstruction-strategies)
- [Cross-Validation Framework](#cross-validation-framework)
- [Visualization Tools](#visualization-tools)
- [Clinical Applications](#clinical-applications)
- [Quality Assurance](#quality-assurance)
- [Troubleshooting](#troubleshooting)
- [Citation and References](#citation-and-references)

---

## Installation

### Required Dependencies

- [List dependencies here]

### Additional Requirements

- [List additional requirements here]

---

## Project Structure

### Main Directory Structure

### Data Folder Organization

**Template Data** (`/Data/Template/PelvisBonesMuscles/`)

- `TempPelvisBoneMuscles.ply` – Template bone-muscle mesh  
- `TempPelvisBoneMesh.ply` – Template bone-only mesh  
- `TempPelvisMuscles.ply` – Template muscle mesh  
- `TempPelvisBoneMesh_picked_points.pp` – Anatomical feature points  
- `TemplatePelvisCoarseShape.ply` – Coarse shape model  

**Cross-Validation Data** (`/Data/PelvisBoneRecon/CrossValidation/`)

- `TrainTestSplits/` – Subject ID splits for each fold  
- `ValidationErrors/` – Validation error metrics  
- `TestingErrors/` – Final testing error metrics  
- `FeatureSelectionProtocol/` – Feature selection strategies  

**Reconstruction Results**

- `PredictedTestingPelvicStructures/` – Predicted mesh outputs  
- `BestWorstPredictedCases/` – Extreme performance cases  
- `MeshFeatureMuscleErrors/` – Detailed error analysis  

---

## Core Functions

### 1. Data Processing Functions

#### `deformTemplatePelvisToTargetPelvis_allData_usingROIFeatureShapeAndMeshDeformation()`

Generates personalized pelvis structures by deforming template meshes to match target anatomical features.

**Process:**

- Loads template pelvis bone-muscle structures  
- Applies rigid SVD + affine CPD transformations  
- Uses radial basis function interpolation for non-rigid deformation  
- Processes ROI-specific deformations  
- Generates subject-specific pelvis muscle structures  

**Inputs:** Template meshes, target feature points, subject pelvis shapes  
**Outputs:** Personalized pelvis bone-muscle meshes  

#### `prepareTrainingTestingIDs()`

Creates stratified train/validation/test splits for cross-validation.

**Process:**

- Loads subject ID list  
- Generates 10-fold cross-validation splits  
- Ensures reproducible splits with seed control  
- Saves ID lists for each fold  

---

### 2. Feature Selection and Protocol Functions

#### `findROIPelvisFeatureIndices()`

Establishes mappings between anatomical features and mesh regions.

**ROI Regions:**

- Left/Right Ilium  
- Sacrum  
- Sacroiliac Joints  
- Pubic Joint  

---

### 3. Reconstruction Strategy Functions

#### `featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor_trainValidationVariousFeatures()`

Trains multi-output ridge regression models across feature selection strategies.

**Key Parameters:**

- Feature strategies  
- PCA components: 1–200  
- Cross-validation: 10-fold  

#### `featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor_testUsingOptimalNumComponents()`

Tests trained models using optimal component configurations.

---

### 4. Alternative Reconstruction Strategies

- **Affine Transform Strategy**  
- **Radial Basis Function Strategy**  
- **Shape Optimization Strategy**  
- **Regression Variants:**  
  - Ridge Linear Regression  
  - Canonical Correlation Analysis  
  - Partial Least Squares  
  - Gaussian Process Regression  

---

### 5. Analysis and Visualization Functions

#### `featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor_drawTestingErrors()`

Generates performance visualizations.

#### `featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor_estimateBestAndWorstPredictedCases()`

Identifies performance extremes for detailed analysis.

#### `featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor_visualizeBestWorstPredictedCases()`

3D visualization of extreme performance cases.

---

## Usage Examples

- Basic Training and Validation  
- Testing with Optimal Parameters  
- Generate Personalized Structures  
- Visualize Results  

---

## Reconstruction Strategies

### 1. Affine Transform-based Deformation

- **Pros:** Fast, simple  
- **Cons:** Limited non-rigid capability  

### 2. Radial Basis Function Interpolation

- **Pros:** Smooth deformations  
- **Cons:** Computationally intensive  

### 3. Statistical Shape Optimization

- **Pros:** Compact representation  
- **Cons:** Limited to training data  

### 4. Shape Relation Strategy (Recommended)

- **Pros:** Robust generalization  
- **Cons:** Requires training data  

---

## Cross-Validation Framework

### Data Splitting Strategy

- Training: 70%  
- Validation: 20%  
- Testing: 10%  

### Validation Metrics

- Point-to-Point Distance  
- Feature Reconstruction Error  
- Muscle Attachment Error  
- Mesh-to-Mesh Distance  

---

## Visualization Tools

### 3D Mesh Visualization

- Trimesh Integration  
- Color Mapping  
- Comparative Display  
- Professional Formatting  

### Statistical Charts

- Bar Charts  
- Line Plots  
- Trend Analysis  
- Performance Metrics  

---

## Clinical Applications

- Pre-surgical Planning  
- Biomechanical Analysis  
- Prosthetic Design  
- Population Studies  

---

## Quality Assurance

- Error Thresholds: <3mm  
- Validation Protocols  
- Performance Monitoring  
- Case Review  

---

## Troubleshooting

### Common Issues

- Memory Errors  
- Convergence Issues  
- Mesh Quality  
- Feature Alignment  

### Debug Functions

- [List debug utilities here]

---

## Citation and References

When using this system, please cite the relevant research publications and acknowledge the comprehensive validation framework implemented for pelvis reconstruction applications.

> Note: This system is designed for research and educational purposes. Clinical applications require additional validation and regulatory approval.
