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

```python
# Core Scientific Computing
import numpy as np
import pandas as pd
import trimesh
import matplotlib.pyplot as plt

# Machine Learning Libraries
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.cross_decomposition import CCA, PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

# Scientific Computing
from scipy.interpolate import RBFInterpolator
from scipy.optimize import least_squares
from scipy.spatial import KDTree

# System Libraries
import sys
import os
import warnings
```

### Additional Requirements

- Custom modules: SupportingTools, VisualInterface, SystemDatabaseManager
- H5PY for database operations
- Pymeshlab for mesh processing (optional)

---

## Project Structure

### Main Directory Structure

```
FeatureBasedPelvisReconstruction/
├── FeatureBasedPelvisReconstruction.py    # Main processing script
├── SupportingTools/
│   └── SupportingTools.py                 # Utility functions
├── Data/
│   ├── PelvisBoneRecon/
│   │   ├── FemalePelvisGeometries/        # Contain female pelvis geometries including muscle and bone
│   │   └── CrossValidation/               # CV data and results
│   └── Template/                          # Template meshes
└── README.md
```

### Data Folder Organization

**Cross-Validation Data** (`./Data/PelvisBoneRecon/CrossValidation/`)
- `TrainingValidTestingIDs/` – Subject ID splits for each fold
- `FeatureSelectionProtocol/` – Feature selection strategies
  - `AllFeaturePoints.pp` – Complete anatomical feature points
  - `FeatureSelectionIndexStrategies.txt` – Feature subset definitions
- Result folders: `AffineDeformation/`, `RadialBasisFunction/`, `ShapeOptimization/`, `ShapeRelation/`

**Template Data**
- `TempPelvisBoneMesh` – Template bone mesh
- `TempPelvisBoneMuscleMesh` – Template bone-muscle mesh
- Template feature points and barycentric coordinates

---

## Core Functions

### 1. Affine Transform-Based Reconstruction

#### `featureToPelvisStructureRecon_affineTransform_BoneStructure()`
#### `featureToPelvisStructureRecon_affineTransform_BoneMuscleStructure()`

Reconstructs pelvis structures using rigid SVD and affine CPD transformations.

**Command Line Usage:**
```bash
python script.py [StartFeatSelStratIndex] [EndFeatSelStratIndex] [StartValidIndex] [EndValidIndex]
```

**Process:**
1. Load template meshes and feature selection strategies
2. For each validation fold and feature strategy:
   - Extract subject pelvis vertices and feature points
   - Apply rigid SVD transformation for alignment
   - Apply affine CPD transformation for refinement
   - Compute reconstruction errors
3. Save validation results to CSV files

### 2. Radial Basis Function Interpolation

#### `featureToPelvisStructureRecon_radialBasicFunctionInterpolation_BoneStructures()`
#### `featureToPelvisStructureRecon_radialBasicFunctionInterpolation_BoneMuscleStructures()`

Non-rigid deformation using RBF interpolation for smooth anatomical reconstruction. These functions extend the affine transformation approach by adding radial basis function interpolation for final mesh deformation, providing more accurate local anatomical details.

**Process:**
1. Apply rigid SVD and affine CPD transformations (similar to affine method)
2. Use RBF interpolation to deform the template mesh to match target feature points
3. Compute feature displacement vectors for improved reconstruction accuracy

### 3. Shape Optimization Strategy

#### `featureToPelvisStructureRecon_shapeOptimizationStrategy_BoneStructures()`
#### `featureToPelvisStructureRecon_shapeOptimizationStrategy_BoneMuscleStructures()`

Optimizes shape parameters to minimize feature reconstruction error using PCA-based statistical shape models.

**Command Line Usage:**
```bash
python script.py [StartFeatSelStratIndex] [EndFeatSelStratIndex] [StartValidIndex] [EndValidIndex] [StartNumComps] [EndNumComps]
```

**Key Features:**
- Statistical shape modeling using PCA decomposition
- Optimization-based parameter estimation
- Barycentric coordinate-based feature point computation
- Multiple component analysis for optimal dimensionality

### 4. Shape Relation Strategy (Primary Method)

The shape relation strategy uses machine learning models to establish statistical relationships between anatomical feature points and pelvis structure parameters.

#### Core Regression Models:

##### `featureToPelvisStructureRecon_shapeRelationStrategy_BoneStructures()`
Basic shape relation approach for bone-only structures.

##### `featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_multivarirateLinearRegression()`
Implements multivariate linear regression for feature-to-structure mapping.

##### `featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_RidgeLinearRegression()`
Uses Ridge regression with L2 regularization to prevent overfitting and improve generalization.

##### `featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_CanonicalCorrelationAnalysis()`
Employs Canonical Correlation Analysis (CCA) to find linear relationships between feature sets and shape parameters.

##### `featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_PartialLeastSquaresRegression()`
Implements Partial Least Squares (PLS) regression for dimensionality reduction and prediction.

##### `featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_GaussianProcessRegressor()`
Advanced non-parametric regression using Gaussian Process models for uncertainty quantification.

##### `featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor()`
Multi-output regression wrapper that can be applied to various base regressors for comprehensive shape prediction.

#### Advanced Analysis and Optimization Functions:

##### `featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor_trainValidationVariousFeatures()`
Comprehensive training and validation across multiple feature selection strategies to identify optimal anatomical landmark combinations.

##### `featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor_determineOptimalNumComponents()`
Automatically determines the optimal number of PCA components for shape representation through systematic evaluation.

##### `featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor_testUsingOptimalNumComponents()`
Performs final testing using the previously determined optimal number of components for unbiased performance evaluation.

#### Comprehensive Evaluation and Visualization:

##### `featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor_drawTestingErrors()`
Generates comprehensive error visualization plots for testing results analysis.

##### `featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor_computeVariousTestingErrors()`
Computes multiple error metrics including point-to-point distances, feature alignment errors, and mesh quality measures.

##### `featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor_drawTestingErrorsInVariousMetrics()`
Creates detailed error analysis plots across different evaluation metrics for comprehensive performance assessment.

##### `featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor_estimateBestAndWorstPredictedCases()`
Identifies the best and worst performing cases from the test set for detailed analysis and quality assurance.

##### `featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor_visualizeBestWorstPredictedCases()`
Provides 3D visualization of the best and worst predicted cases for qualitative assessment.

##### `featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor_visualizeBestWorstPredictedCasesWithCTImageScans()`
Advanced visualization that overlays predicted results with original CT scan data for clinical validation.

### 5. Strategy Selection and Comparison

#### `featureToPelvisStructureRecon_selectOptimalReconstructionStrategy()`

Comprehensive comparison framework that evaluates all reconstruction approaches and selects the optimal method based on cross-validation performance. This function provides systematic comparison across:
- Affine transform-based deformation
- Radial basis function interpolation  
- Statistical shape optimization
- Various shape relation strategies
- Different regression models (Linear, Ridge, CCA, PLS, Gaussian Process)

---

## Usage Examples

### Basic Training and Validation

```bash
# Train affine transform models for feature strategies 0-5, validation folds 0-9
python FeatureBasedPelvisReconstruction.py 0 5 0 9

# Train with different reconstruction strategies
python FeatureBasedPelvisReconstruction.py 0 10 0 9
```

### Analysis and Testing

```python
# Determine optimal PCA components
featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor_determineOptimalNumComponents()

# Test with optimal parameters
featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor_testUsingOptimalNumComponents()

# Generate visualizations
featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor_drawTestingErrors()
```

---

## Data Organization

### Camera Configuration
```python
cameraIntrinsicMatrix = np.array([[1185.7,   0,   960,  0],
                                  [   0, 1185.7,  540,  0],
                                  [   0,   0,     1,    0],
                                  [   0,   0,     0,    1]])
```

### Key Data Paths
- **Main Database:** `H:/Data/PelvisBoneRecon/SystemDataBase/SystemDatabase.h5`
- **Cross-Validation:** `H:/Data/PelvisBoneRecon/CrossValidation/`
- **Debug Output:** `H:/Data/PelvisBoneRecon/Debugs/`

---

## Reconstruction Strategies

### 1. Affine Transform-based Deformation
- **Method:** Rigid SVD + Affine CPD transformations
- **Pros:** Fast, robust alignment
- **Cons:** Limited to linear deformations
- **Best for:** Initial alignment, coarse reconstruction

### 2. Radial Basis Function Interpolation
- **Method:** RBF-based mesh deformation
- **Pros:** Smooth, non-rigid deformations
- **Cons:** Computationally intensive
- **Best for:** Fine anatomical detail preservation

### 3. Shape Optimization Strategy
- **Method:** Parameter optimization with constraints
- **Pros:** Physics-based, interpretable
- **Cons:** Local minima, parameter tuning
- **Best for:** Biomechanically plausible results

### 4. Shape Relation Strategy (Recommended)
- **Method:** Statistical learning from training data
- **Available Models:** Linear, Ridge, CCA, PLS, Gaussian Process, Multi-Output
- **Pros:** Data-driven, robust generalization
- **Cons:** Requires substantial training data
- **Best for:** Production clinical applications

---

## Cross-Validation Framework

### Data Splitting
- **Training/Validation:** Multiple fold cross-validation
- **Feature Selection:** Various anatomical landmark subsets
- **Performance Metrics:** Point-to-point distance, mesh quality

### Feature Selection Strategies
Multiple feature subsets defined in protocol files:
- Complete anatomical landmarks
- Region-specific features (ilium, sacrum, joints)
- Clinically relevant subsets
- Reduced sets for computational efficiency

### Validation Process
1. Load subject IDs for each validation fold
2. Extract features and mesh vertices for training/validation subjects
3. Train models using various regression techniques
4. Evaluate reconstruction accuracy
5. Select optimal hyperparameters
6. Test on held-out subjects

---

## Visualization Tools

### 3D Mesh Visualization
- **Trimesh-based rendering** for interactive viewing
- **Color-coded error mapping** to highlight reconstruction quality
- **Side-by-side comparisons** of original vs. reconstructed
- **Best/worst case visualization** for analysis

### Statistical Analysis
- **Error distribution plots** across subjects and methods
- **Performance comparison charts** between strategies
- **Cross-validation result summaries**
- **Component analysis** for optimal dimensionality

### Clinical Integration
- **CT scan overlay visualization** for validation
- **Multi-modal comparison tools**
- **Export capabilities** for clinical software

---

## Clinical Applications

### Pre-surgical Planning
- Patient-specific anatomy reconstruction from limited imaging
- Surgical approach optimization based on individual anatomy
- Risk assessment using personalized models

### Biomechanical Analysis
- Motion simulation with subject-specific geometry
- Load distribution analysis for implant design
- Muscle attachment point estimation

### Population Studies
- Anatomical variation analysis across demographics
- Disease progression modeling
- Normal vs. pathological anatomy comparison

---

## Quality Assurance

### Performance Metrics
- **Point-to-Point Distance:** Average reconstruction error
- **Feature Alignment Error:** Landmark-specific accuracy
- **Mesh Quality:** Surface smoothness and anatomical validity
- **Clinical Relevance:** Expert evaluation of results

### Validation Standards
- **Accuracy Threshold:** <3mm average reconstruction error
- **Cross-Validation:** Statistical significance testing
- **Independent Testing:** Held-out subject evaluation
- **Expert Review:** Clinical validation of results

---

## Troubleshooting

### Common Issues

#### Memory Limitations
- **Symptom:** Out of memory during processing
- **Solution:** Process in smaller batches, reduce mesh resolution
- **Prevention:** Monitor system resources, optimize data loading

#### Convergence Problems
- **Symptom:** Optimization fails to converge
- **Solution:** Adjust tolerance parameters, validate input data
- **Prevention:** Check mesh quality, use robust initialization

#### Poor Reconstruction Quality
- **Symptom:** High reconstruction errors
- **Solution:** Review feature selection, check template alignment
- **Prevention:** Validate anatomical landmarks, expert annotation

### System Requirements
- **Memory:** 16GB+ RAM recommended
- **Storage:** 50GB+ for full datasets
- **Processing:** Multi-core CPU beneficial for cross-validation

---

## Citation and References

When using this system, please cite the relevant research publications describing the methodology and validation studies.

### Key References
- Statistical shape modeling in medical imaging
- Cross-validation methodologies for anatomical reconstruction
- Machine learning applications in medical image analysis

---

## License and Disclaimer

This software is provided for research and educational purposes. Clinical applications require additional validation and regulatory approval according to local medical device regulations.

For technical support or collaboration inquiries, please contact the development team.

> **Important:** This system is designed for research purposes only. Users are responsible for ensuring compliance with applicable regulations for clinical use.