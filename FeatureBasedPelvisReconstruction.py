#************************************************************ SUPPORTING LIBRARIES
import os;
import numpy as np;
import sys;
import matplotlib.pyplot as plt;
import trimesh;
from scipy.interpolate import RBFInterpolator;
from scipy.optimize import least_squares;
from sklearn.decomposition import PCA;
from sklearn.linear_model import LinearRegression;
from sklearn.linear_model import Ridge;
from sklearn.cross_decomposition import CCA;
from sklearn.cross_decomposition import PLSRegression;
from sklearn.gaussian_process import GaussianProcessRegressor;
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import RidgeCV
from sklearn.gaussian_process.kernels import RBF;
from sklearn.preprocessing import StandardScaler;
import matplotlib.pyplot as plt;
from matplotlib.font_manager import FontProperties;

import SupportingTools.SupportingTools as sp;
import VisualInterface.VisualInterface as vi;

import warnings
warnings.filterwarnings("ignore")

#************************************************************ SUPPORTING BUFFERS
viewer = vi.VisualInterface();
disk = "H:";


#********************************************************** PROCESSING FUNCTIONS
#************************** DATA PROCESSING FUNCTIONS
def deformTemplatePelvisToTargetPelvis_allData_usingROIFeatureShapeAndMeshDeformation():
    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 3):
        print("\t Please input the command as the following: [ProgramName] [StartIndex] [EndIndex]"); return;
    startIndex = int(sys.argv[1]); endIndex = int(sys.argv[2]);    
    processingDisk = "H:/SpinalPelvisPred";
    templateFolder = processingDisk + "/Data/Template";
    pelvisBoneMuscleTempFolder = templateFolder + "/PelvisBonesMuscles";
    pelvisReconFolder = processingDisk +  "/Data/PelvisBoneRecon";
    targetPelvisFolder = pelvisReconFolder + "/FemalePelvisGeometries/1KPelvisData";
    debugFolder = processingDisk + "/Data/PelvisBoneRecon/Debugs";

    # Reading processing IDs
    print("Reading processing IDs ...");
    processingIDs = sp.readListOfStrings(pelvisReconFolder + "/FemalePelvisGeometries/1KPelvisDataFemalePelvisIDs.txt");

    # Getting template information
    print("Getting template information ...");
    tempPelvisBoneMuscleMesh = sp.readMesh(pelvisBoneMuscleTempFolder + "/TempPelvisBoneMuscles.ply");
    tempPelvisMuscleMesh = sp.readMesh(pelvisBoneMuscleTempFolder + "/TempPelvisMuscles.ply");
    tempPelvisShape = sp.readMesh(pelvisBoneMuscleTempFolder + "/TemplatePelvisCoarseShape.ply");
    tempPelvisMesh = sp.readMesh(pelvisBoneMuscleTempFolder + "/TempPelvisBoneMesh.ply");
    tempWithoutJointPelvisMesh = sp.readMesh(pelvisBoneMuscleTempFolder + "/TempPelvisBoneMeshWithOutJoints.ply");
    tempLeftIliumMesh = sp.readMesh(pelvisBoneMuscleTempFolder + "/TempPelvisBoneMesh_LeftIlium.ply");
    tempRightIliumMesh = sp.readMesh(pelvisBoneMuscleTempFolder + "/TempPelvisBoneMesh_RightIlium.ply");
    tempSacrumMesh = sp.readMesh(pelvisBoneMuscleTempFolder + "/TempPelvisBoneMesh_Sacrum.ply");
    tempLeftSacroiliacJointMesh = sp.readMesh(pelvisBoneMuscleTempFolder + "/TempPelvisBoneMesh_LeftSacroiliacJoint.ply");
    tempRightSacroiliacJointMesh = sp.readMesh(pelvisBoneMuscleTempFolder + "/TempPelvisBoneMesh_RightSacroiliacJoint.ply");
    tempPubicJointMesh = sp.readMesh(pelvisBoneMuscleTempFolder + "/TempPelvisBoneMesh_PubicJoint.ply");
    
    tempPelvisFeatures = sp.read3DPointsFromPPFile(pelvisBoneMuscleTempFolder + "/TempPelvisBoneMesh_picked_points.pp");
    pelvisFeatureBaryIndicesOnShape, pelvisFeatureBaryCoordsOnShape = sp.computeBarycentricLandmarks(tempPelvisShape, tempPelvisFeatures);
    pelvisFeatureBaryIndicesOnMesh, pelvisFeatureBaryCoordsOnMesh = sp.computeBarycentricLandmarks(tempPelvisMesh, tempPelvisFeatures);
    
    leftIliumFeatureIndices = sp.readIndicesFromCSVFile(pelvisBoneMuscleTempFolder + "/LeftIliumFeatureIndices.csv");
    rightIliumFeatureIndices = sp.readIndicesFromCSVFile(pelvisBoneMuscleTempFolder + "/RightIliumFeatureIndices.csv");
    sacrumFeatureIndices = sp.readIndicesFromCSVFile(pelvisBoneMuscleTempFolder + "/SacrumFeatureIndices.csv");
    
    tempLeftSacroiliacJointFeatures = sp.read3DPointsFromPPFile(pelvisBoneMuscleTempFolder + "/TempPelvisBoneMesh_LeftSacroiliacJoint_picked_points.pp");
    tempRightSacroiliacJointFeatures = sp.read3DPointsFromPPFile(pelvisBoneMuscleTempFolder + "/TempPelvisBoneMesh_RightSacroiliacJoint_picked_points.pp");
    tempPubicJointFeatures = sp.read3DPointsFromPPFile(pelvisBoneMuscleTempFolder + "/TempPelvisBoneMesh_PubicJoint_picked_points.pp");
    tempPelvisMuscleFeatures = sp.read3DPointsFromPPFile(pelvisBoneMuscleTempFolder + "/TempPelvisMuscles_picked_points.pp");

    leftSacroiliacJointFeatureBaryIndices, leftSacroiliacJointFeatureBaryCoords = sp.computeBarycentricLandmarks(tempWithoutJointPelvisMesh, tempLeftSacroiliacJointFeatures);
    rightSacroiliacJointFeatureBaryIndices, rightSacroiliacJointFeatureBaryCoords = sp.computeBarycentricLandmarks(tempWithoutJointPelvisMesh, tempRightSacroiliacJointFeatures);
    pubicJointFeatureBaryIndices, pubicJointFeatureBaryCoords = sp.computeBarycentricLandmarks(tempWithoutJointPelvisMesh, tempPubicJointFeatures);
    pelvisMuscleFeatureBaryIndices, pelvisMuscleFeatureBaryCoords = sp.computeBarycentricLandmarks(tempPelvisMesh, tempPelvisMuscleFeatures);
    
    leftSacroiliacJointFeatureBaryIndices = sp.transferFaceIndicesToOtherMesh(leftSacroiliacJointFeatureBaryIndices, tempWithoutJointPelvisMesh, tempPelvisMesh);
    rightSacroiliacJointFeatureBaryIndices = sp.transferFaceIndicesToOtherMesh(rightSacroiliacJointFeatureBaryIndices, tempWithoutJointPelvisMesh, tempPelvisMesh);
    pubicJointFeatureBaryIndices = sp.transferFaceIndicesToOtherMesh(pubicJointFeatureBaryIndices, tempWithoutJointPelvisMesh, tempPelvisMesh);
        
    pelvisBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempPelvisMesh.vertices, tempPelvisBoneMuscleMesh.vertices);
    pelvisMuscleVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempPelvisMuscleMesh.vertices, tempPelvisBoneMuscleMesh.vertices);
    leftIliumVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempLeftIliumMesh.vertices, tempPelvisBoneMuscleMesh.vertices);
    rightIliumVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempRightIliumMesh.vertices, tempPelvisBoneMuscleMesh.vertices);
    sacrumVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempSacrumMesh.vertices, tempPelvisBoneMuscleMesh.vertices);
    leftSacroiliacJointVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempLeftSacroiliacJointMesh.vertices, tempPelvisBoneMuscleMesh.vertices);
    rightSacroiliacJointVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempRightSacroiliacJointMesh.vertices, tempPelvisBoneMuscleMesh.vertices);
    pubicJointVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempPubicJointMesh.vertices, tempPelvisBoneMuscleMesh.vertices);    
    
    # Checking for each subject
    print("Checking for each subject ...");
    for i in range(startIndex, endIndex + 1):
        # Debugging 
        sID = processingIDs[i];
        print("/****************************************** Processing subject: ", i, " with ID: ", sID);

        # Reading target information
        print("Reading target information ...");
        targetPelvisShape = sp.readMesh(targetPelvisFolder + f"/{sID}-PelvisBoneShape.ply");
        targetPelvisMesh = sp.readMesh(targetPelvisFolder + f"/{sID}-PelvisBoneMesh.ply");
        targetPelvisFeatures = sp.read3DPointsFromPPFile(targetPelvisFolder + f"/{sID}-PelvisBoneMesh_picked_points.pp");

        # Fixing the target information
        print("Fixing the target information ...");
        targetPelvisShape = sp.fixMesh(targetPelvisShape);
        targetPelvisMesh = sp.fixMesh(targetPelvisMesh);

        # Prepare the aligned buffers
        print("Preparing the aligned buffers ...");
        alignedPelvisShape = sp.cloneMesh(tempPelvisShape);
        alignedPelvisFeatures = tempPelvisFeatures.copy();
        
        # Deform using the rigid transform 
        print("Deform using the rigid transform ...");
        svdTransform = sp.estimateRigidSVDTransform(alignedPelvisFeatures, targetPelvisFeatures);
        alignedPelvisShape.vertices = sp.transform3DPoints(alignedPelvisShape.vertices, svdTransform);
        alignedPelvisFeatures = sp.reconstructLandmarksFromBarycentric(alignedPelvisShape, pelvisFeatureBaryIndicesOnShape, pelvisFeatureBaryCoordsOnShape);
        
        # Deform with affine transform 
        print("Deform with affine transform ...");
        affineTransform = sp.estimateAffineTransformCPD(alignedPelvisFeatures, targetPelvisFeatures);
        alignedPelvisShape = sp.transformMesh(alignedPelvisShape, affineTransform);
        del(alignedPelvisFeatures); gc.collect();
        
        # Deform using the non-rigid ICP registration
        print("Deform using the non-rigid ICP registration ...");
        alignedPelvisShapeVertices = trimesh.registration.nricp_amberg(
            source_mesh=alignedPelvisShape,
            target_geometry=targetPelvisShape,
            source_landmarks=(pelvisFeatureBaryIndicesOnShape, pelvisFeatureBaryCoordsOnShape),
            target_positions=targetPelvisFeatures
        )
        alignedPelvisShape.vertices = alignedPelvisShapeVertices.copy(); 
        del(alignedPelvisShapeVertices); gc.collect();

        # Project deformed shape to the target shape
        print("Project deformed shape to the taret shape ...");
        alignedPelvisShape = sp.projectMeshOntoMesh(alignedPelvisShape, targetPelvisShape);        

        # Deform the template pelvis mesh to def pelvis shape
        print("Deform template pelvis mesh to def pelvis shape ...");
        print("\t Using the rigid transform ...");
        deformedPelvisShape = sp.cloneMesh(tempPelvisShape);
        deformedPelvisBoneMuscleMesh = sp.cloneMesh(tempPelvisBoneMuscleMesh);
        svdTransform = sp.estimateRigidSVDTransform(deformedPelvisShape.vertices, alignedPelvisShape.vertices);
        deformedPelvisShape.vertices = sp.transform3DPoints(deformedPelvisShape.vertices, svdTransform);
        deformedPelvisBoneMuscleMesh.vertices = sp.transform3DPoints(deformedPelvisBoneMuscleMesh.vertices, svdTransform);

        print("\t Using the non-rigid transform ...");
        affineTransform = sp.estimateAffineTransformCPD(deformedPelvisShape.vertices, alignedPelvisShape.vertices);
        deformedPelvisShape.vertices = sp.transform3DPoints(deformedPelvisShape.vertices, affineTransform);
        deformedPelvisBoneMuscleMesh.vertices = sp.transform3DPoints(deformedPelvisBoneMuscleMesh.vertices, affineTransform);

        print("\t Using the radial basic function ...");
        deformedToAlignedShapeVertexDisplacements = alignedPelvisShape.vertices - deformedPelvisShape.vertices;
        rbf = RBFInterpolator(deformedPelvisShape.vertices, deformedToAlignedShapeVertexDisplacements, kernel='thin_plate_spline', smoothing=1e-3);
        deformedPelvisBoneMuscleVertexDisplacements = rbf(deformedPelvisBoneMuscleMesh.vertices);
        deformedPelvisBoneMuscleMesh.vertices = deformedPelvisBoneMuscleMesh.vertices + deformedPelvisBoneMuscleVertexDisplacements;
        del(deformedPelvisShape, deformedToAlignedShapeVertexDisplacements, svdTransform, affineTransform, rbf, deformedPelvisBoneMuscleVertexDisplacements); gc.collect();
        
        # Try to deform the pelvis mesh using the ROI shape
        print("Try to deform the pelvis mesh using the ROI shape ...");
        ## Getting the ROI meshes
        print("\t Getting the ROI meshes ...");
        deformedPelvisMesh = sp.cloneMesh(tempPelvisMesh);
        deformedPelvisMuscleMesh = sp.cloneMesh(tempPelvisMuscleMesh);
        deformedLeftIliumMesh = sp.cloneMesh(tempLeftIliumMesh);
        deformedRightIliumMesh = sp.cloneMesh(tempRightIliumMesh);
        deformedSacrumMesh = sp.cloneMesh(tempSacrumMesh);
        deformedLeftSacroiliacJointMesh = sp.cloneMesh(tempLeftSacroiliacJointMesh);
        deformedRightSacroiliacJointMesh = sp.cloneMesh(tempRightSacroiliacJointMesh);
        deformedPubicJointMesh = sp.cloneMesh(tempPubicJointMesh);        
        
        deformedPelvisMesh.vertices = deformedPelvisBoneMuscleMesh.vertices[pelvisBoneVertexIndices];
        deformedPelvisMuscleMesh.vertices = deformedPelvisBoneMuscleMesh.vertices[pelvisMuscleVertexIndices];
        deformedLeftIliumMesh.vertices = deformedPelvisBoneMuscleMesh.vertices[leftIliumVertexIndices];
        deformedRightIliumMesh.vertices = deformedPelvisBoneMuscleMesh.vertices[rightIliumVertexIndices];
        deformedSacrumMesh.vertices = deformedPelvisBoneMuscleMesh.vertices[sacrumVertexIndices];
        deformedLeftSacroiliacJointMesh.vertices = deformedPelvisBoneMuscleMesh.vertices[leftSacroiliacJointVertexIndices];
        deformedRightSacroiliacJointMesh.vertices = deformedPelvisBoneMuscleMesh.vertices[rightSacroiliacJointVertexIndices];
        deformedPubicJointMesh.vertices = deformedPelvisBoneMuscleMesh.vertices[pubicJointVertexIndices];

        ## Getting ROI features
        print("\t Getting the ROI features ...");
        deformedFullPelvisFeatures = sp.reconstructLandmarksFromBarycentric(deformedPelvisMesh, pelvisFeatureBaryIndicesOnMesh, pelvisFeatureBaryCoordsOnMesh);
        deformedLeftIliumFeatures = deformedFullPelvisFeatures[leftIliumFeatureIndices];
        deformedRightIliumFeatures = deformedFullPelvisFeatures[rightIliumFeatureIndices];
        deformedSacrumFeatures = deformedFullPelvisFeatures[sacrumFeatureIndices];        

        ## Compute bary coords for ROI features
        print("\t Compute bary coords for ROI features ...");
        leftIliumFeatureBaryIndices, leftIliumFeatureBaryCoords = sp.computeBarycentricLandmarks(deformedLeftIliumMesh, deformedLeftIliumFeatures);
        rightIliumFeatureBaryIndices, rightIliumFeatureBaryCoords = sp.computeBarycentricLandmarks(deformedRightIliumMesh, deformedRightIliumFeatures);
        sacrumFeatureBaryIndices, sacrumFeatureBaryCoords = sp.computeBarycentricLandmarks(deformedSacrumMesh, deformedSacrumFeatures);
        
        ## Compute the target ROI features
        print("\t Compute target ROI features ...");
        targetLeftIliumFeatures = targetPelvisFeatures[leftIliumFeatureIndices];
        targetRightIliumFeatures = targetPelvisFeatures[rightIliumFeatureIndices];
        targetSacrumFeatures = targetPelvisFeatures[sacrumFeatureIndices];

        ## Deform leftIliumMesh to the target mesh
        print("\t Deform leftIliumMesh to the target mesh ...");
        personalizedLeftIliumMeshVertices = trimesh.registration.nricp_amberg(
            source_mesh=deformedLeftIliumMesh,
            target_geometry=targetPelvisMesh,
            source_landmarks=(leftIliumFeatureBaryIndices, leftIliumFeatureBaryCoords),
            target_positions=targetLeftIliumFeatures
        )
        personalizedLeftIliumMesh = sp.cloneMesh(deformedLeftIliumMesh);
        personalizedLeftIliumMesh.vertices = personalizedLeftIliumMeshVertices.copy();
        del(personalizedLeftIliumMeshVertices); gc.collect();

        ## Deform rightIliumMesh to the target mesh
        print("\t Deform rightIliumMesh to the target mesh ...");
        personalizedRightIliumMeshVertices = trimesh.registration.nricp_amberg(
            source_mesh=deformedRightIliumMesh,
            target_geometry=targetPelvisMesh,
            source_landmarks=(rightIliumFeatureBaryIndices, rightIliumFeatureBaryCoords),
            target_positions=targetRightIliumFeatures
        )
        personalizedRightIliumMesh = sp.cloneMesh(deformedRightIliumMesh);
        personalizedRightIliumMesh.vertices = personalizedRightIliumMeshVertices.copy();
        del(personalizedRightIliumMeshVertices); gc.collect();

        ## Deform sacrum to the target mesh
        print("\t Deform sacrum to the target mesh ...");
        personalizedSacrumMeshVertices = trimesh.registration.nricp_amberg(
            source_mesh=deformedSacrumMesh,
            target_geometry=targetPelvisMesh,
            source_landmarks=(sacrumFeatureBaryIndices, sacrumFeatureBaryCoords),
            target_positions=targetSacrumFeatures
        )
        personalizedSacrumMesh = sp.cloneMesh(deformedSacrumMesh);
        personalizedSacrumMesh.vertices = personalizedSacrumMeshVertices.copy();
        del(personalizedSacrumMeshVertices); gc.collect();

        ## Forming personalized pelvis bone muscle mesh
        print("\t Forming personalized pelvis bone muscle mesh ...");
        personalizedPelvisBoneMuscleMesh = sp.cloneMesh(deformedPelvisBoneMuscleMesh);
        personalizedPelvisBoneMuscleMesh.vertices[leftIliumVertexIndices] = personalizedLeftIliumMesh.vertices;
        personalizedPelvisBoneMuscleMesh.vertices[rightIliumVertexIndices] = personalizedRightIliumMesh.vertices;
        personalizedPelvisBoneMuscleMesh.vertices[sacrumVertexIndices] = personalizedSacrumMesh.vertices;
        personalizedPelvisMesh = sp.cloneMesh(deformedPelvisMesh);
        personalizedPelvisMesh.vertices = personalizedPelvisBoneMuscleMesh.vertices[pelvisBoneVertexIndices];

        # Deform the pelvis muscle
        print("Deforming the pelvis muscle ...");
        deformedPelvisMuscleFeatures = sp.reconstructLandmarksFromBarycentric(deformedPelvisMesh, pelvisMuscleFeatureBaryIndices, pelvisMuscleFeatureBaryCoords);
        personalizedPelvisMusceFeatures = sp.reconstructLandmarksFromBarycentric(personalizedPelvisMesh, pelvisMuscleFeatureBaryIndices, pelvisMuscleFeatureBaryCoords);
        deformedToPersonalizedPelvisMuscleFeatureDisplacements = personalizedPelvisMusceFeatures - deformedPelvisMuscleFeatures;
        rbf = RBFInterpolator(deformedPelvisMuscleFeatures, 
                              deformedToPersonalizedPelvisMuscleFeatureDisplacements, 
                              kernel='thin_plate_spline', 
                              smoothing=1e-3);
        personalizedPelvisMuscleVertexDisplacements = rbf(deformedPelvisMuscleMesh.vertices);
        personalizedPelvisMuscleMesh = sp.cloneMesh(deformedPelvisMuscleMesh);
        personalizedPelvisMuscleMesh.vertices = deformedPelvisMuscleMesh.vertices + personalizedPelvisMuscleVertexDisplacements;
        del(deformedPelvisMuscleFeatures, personalizedPelvisMusceFeatures, deformedToPersonalizedPelvisMuscleFeatureDisplacements, 
            rbf, personalizedPelvisMuscleVertexDisplacements); gc.collect();

        # Comptue the personalized pelvis muscle meshes
        print("Computing the personalized pelvis muscle meshes ...");
        personalizedPelvisBoneMuscleMesh = sp.cloneMesh(deformedPelvisBoneMuscleMesh);
        personalizedPelvisBoneMuscleMesh.vertices[pelvisBoneVertexIndices] = personalizedPelvisMesh.vertices;
        personalizedPelvisBoneMuscleMesh.vertices[pelvisMuscleVertexIndices] = personalizedPelvisMuscleMesh.vertices;
        
        # Save the personalized results
        print("Save the personalized results ...");
        sp.saveMeshToPLY(debugFolder + f"/{sID}-PersonalizedPelvisBoneMuscleMesh.ply", personalizedPelvisBoneMuscleMesh);
        del(personalizedPelvisBoneMuscleMesh); gc.collect();

    # Finished 
    print("Finished.");
    
#************************** CROSS-VALIDATION FUNCTIONS
#************* USING AFFINE TRANSFORM FOR BONE AND BONE MUSCLE STRUCTURES
def featureToPelvisStructureRecon_affineTransform_BoneStructure():
    """
    Performs feature-to-pelvis structure reconstruction using affine transformation on bone structures.
    
    This function evaluates the accuracy of pelvis bone reconstruction by deforming a template pelvis 
    bone mesh to match target feature points using rigid (SVD) and affine (CPD) transformations.
    The evaluation is performed across multiple feature selection strategies and cross-validation folds.
    
    Key Steps:
    1. Parse command line arguments for processing range
    2. Initialize system database and load template data
    3. Load feature selection strategies 
    4. For each feature selection strategy and validation fold:
       - Load validation subject IDs
       - Extract pelvis bone vertices and feature points for each subject
       - Deform template mesh to match target features using transformations
       - Compute point-to-point distance errors
       - Save validation errors to CSV files
    
    Transformations Applied:
    - Rigid SVD transformation: Aligns template to target via rotation/translation
    - Affine CPD transformation: Further refines alignment with scaling/shearing
    
    Output:
    - CSV files containing average point-to-point distances for each configuration
    """

    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 5):
        print("Please input the command as the following: [ProgramName] [StartFeatSelStratIndex] [EndFeatSelStratIndex] [StartValidIndex] [EndValidIndex]");
        return;
    startFeatSelStratIndex = int(sys.argv[1]); endFeatSelStratIndex = int(sys.argv[2]);
    startValidIndex = int(sys.argv[3]); endValidIndex = int(sys.argv[4]);
    
    personalizedPelvicStructureFolder = disk + r"\Data\PelvisBoneRecon\FemalePelvisGeometries\PersonalizedPelvisStructures";
    templateDataFolder = disk + r"\Data\PelvisBoneRecon\Template\PelvisBonesMuscles";
    crossValidationFolder = disk + r"\Data\PelvisBoneRecon\CrossValidation";
    trainValidTestIDFolder = crossValidationFolder + r"\TrainingValidTestingIDs";
    featureSelectionProtocolFolder = crossValidationFolder + r"\FeatureSelectionProtocol";
    outFolder = crossValidationFolder + r"\AffineDeformation\BoneStructures";

    # Checking and creating output folder
    print("Checking and creating output folder ...");
    if (not os.path.exists(outFolder)): os.makedirs(outFolder);

    # Reading initial information
    print("Reading initial information ...");
    tempPelvisBoneMesh = sp.readMeshFromPLY(templateDataFolder + r"/TempPelvisBoneMesh.ply");
    tempPelvisBoneMuscleMesh = sp.readMeshFromPLY(templateDataFolder + r"/TempPelvisBoneMuscles.ply");
    pelvisBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempPelvisBoneMesh.vertices, tempPelvisBoneMuscleMesh.vertices);
    
    # Reading feature section strategies
    print("Reading feature selection strategies ...");
    allPelvicFeatures = sp.read3DPointsFromPPFile(featureSelectionProtocolFolder + r"\AllFeaturePoints.pp");
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + r"\FeatureSelectionIndexStrategies.txt");

    # Iterate for each feature selection strategy
    print("Iterate for each feature selection strategy ...");
    for featSelIndex in range(startFeatSelStratIndex, endFeatSelStratIndex + 1):
        # Debugging
        print("/********************************************** FEATURE SELECTION STRATEGY: ", featSelIndex);

        # Get the feature selection strategies
        print("\t Getting the feature selection strategy ...");
        featureSelectionIndices = list(featureSelectionStrategyDict.values())[featSelIndex];
        tempPelvicFeatures = allPelvicFeatures[featureSelectionIndices];
        pelvisFeatureBaryIndices, pelvicFeatureBaryCoords = sp.computeBarycentricLandmarks(tempPelvisBoneMesh, tempPelvicFeatures);

        # Conduct training validating throughout each time of cross-validation
        for validIndex in range(startValidIndex, endValidIndex + 1):
            # Debugging information
            print("/********************************************** VALIDATION: ", validIndex);

            # Reading training and validating subject IDs
            print("\t Reading training and validating subject IDs ...");
            validIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/ValidationIDs_{validIndex}.txt");

            # Forming the training and validating data
            print("\t Forming the training and validating data ...");
            validPelvisBoneVertexData = []; validPelvisFeatureData = [];
            for i, ID in enumerate(validIDs):
                pelvicStructure = sp.readMeshFromPLY(personalizedPelvicStructureFolder + f"/{ID}-PersonalizedPelvisBoneMuscleMesh.ply");
                pelvisBoneMesh = sp.cloneMesh(tempPelvisBoneMesh);
                pelvisBoneMesh.vertices = pelvicStructure.vertices[pelvisBoneVertexIndices];
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvisBoneMesh, pelvisFeatureBaryIndices, pelvicFeatureBaryCoords);
                validPelvisBoneVertexData.append(pelvisBoneMesh.vertices.flatten());
                validPelvisFeatureData.append(pelvicFeatures.flatten());
            validPelvisBoneVertexData = np.array(validPelvisBoneVertexData);
            validPelvisFeatureData = np.array(validPelvisFeatureData);

            # Try to deform the template pelvis bone mesh to the target pelvis bone features
            print("\t Deforming template pelvis bone mesh to target pelvis bone features ...");
            avgP2PDists = [];
            for v, validPelvisFeatures in enumerate(validPelvisFeatureData):
                # Debugging
                print("#********************************** FEATURE SELECTION INDEX: ", featSelIndex, ", VALIDATION INDEX: ", validIndex, " VALID SUBJECT INDEX: ", v);

                # Getting target information
                targetFeaturePoints = validPelvisFeatures.reshape(-1, 3);
                validPelvisBoneVertices = validPelvisBoneVertexData[v].reshape(-1, 3);
                validPelvisBoneMesh = sp.cloneMesh(tempPelvisBoneMesh);
                validPelvisBoneMesh.vertices = validPelvisBoneVertices;

                # Deform the template pelvis bone to the target feature points
                defPelvisBoneMesh = sp.cloneMesh(tempPelvisBoneMesh);
                defFeaturePoints = sp.reconstructLandmarksFromBarycentric(defPelvisBoneMesh, pelvisFeatureBaryIndices, pelvicFeatureBaryCoords);
                svdTransform = sp.estimateRigidSVDTransform(defFeaturePoints, targetFeaturePoints);
                defPelvisBoneMesh = sp.transformMesh(defPelvisBoneMesh, svdTransform);
                defFeaturePoints = sp.transform3DPoints(defFeaturePoints, svdTransform);
                affineTransform = sp.estimateAffineTransformCPD(defFeaturePoints, targetFeaturePoints);
                defPelvisBoneMesh = sp.transformMesh(defPelvisBoneMesh, affineTransform);
                defFeaturePoints = sp.transform3DPoints(defFeaturePoints, affineTransform);

                # Computing points to points distances
                avgP2PDist = sp.computeAveragePointsToPointsDistance(defPelvisBoneMesh.vertices, validPelvisBoneMesh.vertices);
                avgP2PDists.append(avgP2PDist);
            avgP2PDists = np.array(avgP2PDists);

            # Save the computed errors to file
            sp.saveMatrixToCSVFile(outFolder + f"/AveragePoint2PointDistances_{featSelIndex}_{validIndex}.csv", avgP2PDists);

    # Finished processing
    print("Finished processing.");
def featureToPelvisStructureRecon_affineTransform_BoneMuscleStructure():
    """
    Performs feature-to-pelvis structure reconstruction using affine transformation on bone structures.
    
    This function evaluates the accuracy of pelvis bone reconstruction by deforming a template pelvis 
    bone mesh to match target feature points using rigid (SVD) and affine (CPD) transformations.
    The evaluation is performed across multiple feature selection strategies and cross-validation folds.
    
    Key Steps:
    1. Parse command line arguments for processing range
    2. Initialize system database and load template data
    3. Load feature selection strategies 
    4. For each feature selection strategy and validation fold:
       - Load validation subject IDs
       - Extract pelvis bone vertices and feature points for each subject
       - Deform template mesh to match target features using transformations
       - Compute point-to-point distance errors
       - Save validation errors to CSV files
    
    Transformations Applied:
    - Rigid SVD transformation: Aligns template to target via rotation/translation
    - Affine CPD transformation: Further refines alignment with scaling/shearing
    
    Output:
    - CSV files containing average point-to-point distances for each configuration
    """

    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 5):
        print("Please input the command as the following: [ProgramName] [StartFeatSelStratIndex] [EndFeatSelStratIndex] [StartValidIndex] [EndValidIndex]");
        return;
    startFeatSelStratIndex = int(sys.argv[1]); endFeatSelStratIndex = int(sys.argv[2]);
    startValidIndex = int(sys.argv[3]); endValidIndex = int(sys.argv[4]);
    
    debugFolder = disk + r"\Data\PelvisBoneRecon\Debugs";
    personalizedPelvicStructureFolder = disk + r"\Data\PelvisBoneRecon\FemalePelvisGeometries\PersonalizedPelvisStructures";
    templateDataFolder = disk + r"\Data\PelvisBoneRecon\Template\PelvisBonesMuscles";
    crossValidationFolder = disk + r"\Data\PelvisBoneRecon\CrossValidation";
    trainValidTestIDFolder = crossValidationFolder + r"\TrainingValidTestingIDs";
    featureSelectionProtocolFolder = crossValidationFolder + r"\FeatureSelectionProtocol";
    outFolder = crossValidationFolder + r"\AffineDeformation\BoneMuscleStructures";
    
    # Checking and creating output folder
    print("Checking and creating output folder ...");
    if (not os.path.exists(outFolder)): os.makedirs(outFolder);

    # Reading initial information
    print("Reading initial information ...");
    tempPelvisBoneMesh = sp.readMeshFromPLY(templateDataFolder + r"/TempPelvisBoneMesh.ply");
    tempPelvisBoneMuscleMesh = sp.readMeshFromPLY(templateDataFolder + r"/TempPelvisBoneMuscles.ply");
    pelvisBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempPelvisBoneMesh.vertices, tempPelvisBoneMuscleMesh.vertices);
    
    # Reading feature section strategies
    print("Reading feature selection strategies ...");
    allPelvicFeatures = sp.read3DPointsFromPPFile(featureSelectionProtocolFolder + "/AllFeaturePoints.pp");
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");
    
    # Iterate for each feature selection strategy
    print("Iterate for each feature selection strategy ...");
    for featSelIndex in range(startFeatSelStratIndex, endFeatSelStratIndex + 1):
        # Debugging
        print("/********************************************** FEATURE SELECTION STRATEGY: ", featSelIndex);

        # Get the feature selection strategies
        print("\t Getting the feature selection strategy ...");
        featureSelectionIndices = list(featureSelectionStrategyDict.values())[featSelIndex];
        tempPelvicFeatures = allPelvicFeatures[featureSelectionIndices];
        pelvisFeatureBaryIndices, pelvicFeatureBaryCoords = sp.computeBarycentricLandmarks(tempPelvisBoneMuscleMesh, tempPelvicFeatures);

        # Conduct training validating throughout each time of cross-validation
        for validIndex in range(startValidIndex, endValidIndex + 1):
            # Debugging information
            print("/********************************************** VALIDATION: ", validIndex);

            # Reading training and validating subject IDs
            print("\t Reading training and validating subject IDs ...");
            validIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/ValidationIDs_{validIndex}.txt");

            # Forming the training and validating data
            print("\t Forming the training and validating data ...");
            validPelvisBoneMuscleVertexData = []; validPelvisFeatureData = [];
            for i, ID in enumerate(validIDs):
                pelvisBoneMuscleMesh = sp.readMeshFromPLY(personalizedPelvicStructureFolder + f"/{ID}-PersonalizedPelvisBoneMuscleMesh.ply");
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvisBoneMuscleMesh, pelvisFeatureBaryIndices, pelvicFeatureBaryCoords);
                validPelvisBoneMuscleVertexData.append(pelvisBoneMuscleMesh.vertices.flatten());
                validPelvisFeatureData.append(pelvicFeatures.flatten());
            validPelvisBoneMuscleVertexData = np.array(validPelvisBoneMuscleVertexData);
            validPelvisFeatureData = np.array(validPelvisFeatureData);

            # Try to deform the template pelvis bone mesh to the target pelvis bone features
            print("\t Deforming template pelvis bone mesh to target pelvis bone features ...");
            avgP2PDists = [];
            for v, validPelvisFeatures in enumerate(validPelvisFeatureData):
                # Debugging
                print("#********************************** FEATURE SELECTION INDEX: ", featSelIndex, ", VALIDATION INDEX: ", validIndex, " VALID SUBJECT INDEX: ", v);

                # Getting target information
                targetFeaturePoints = validPelvisFeatures.reshape(-1, 3);
                validPelvisBoneMuscleVertices = validPelvisBoneMuscleVertexData[v].reshape(-1, 3);
                validPelvisBoneMuscleMesh = sp.cloneMesh(tempPelvisBoneMuscleMesh);
                validPelvisBoneMuscleMesh.vertices = validPelvisBoneMuscleVertices;

                # Deform the template pelvis bone to the target feature points
                defPelvisBoneMuscleMesh = sp.cloneMesh(tempPelvisBoneMuscleMesh);
                defFeaturePoints = sp.reconstructLandmarksFromBarycentric(defPelvisBoneMuscleMesh, pelvisFeatureBaryIndices, pelvicFeatureBaryCoords);
                svdTransform = sp.estimateRigidSVDTransform(defFeaturePoints, targetFeaturePoints);
                defPelvisBoneMuscleMesh = sp.transformMesh(defPelvisBoneMuscleMesh, svdTransform);
                defFeaturePoints = sp.transform3DPoints(defFeaturePoints, svdTransform);
                affineTransform = sp.estimateAffineTransformCPD(defFeaturePoints, targetFeaturePoints);
                defPelvisBoneMuscleMesh = sp.transformMesh(defPelvisBoneMuscleMesh, affineTransform);
                defFeaturePoints = sp.transform3DPoints(defFeaturePoints, affineTransform);
                
                # Compute only the bone vertices for computing distances
                validPelvisBoneVertices = validPelvisBoneMuscleMesh.vertices[pelvisBoneVertexIndices];
                personalizedBoneVertices = defPelvisBoneMuscleMesh.vertices[pelvisBoneVertexIndices];

                # Computing points to points distances
                avgP2PDist = sp.computeAveragePointsToPointsDistance(personalizedBoneVertices, validPelvisBoneVertices);
                avgP2PDists.append(avgP2PDist);
            avgP2PDists = np.array(avgP2PDists);

            # Save the computed errors to file
            sp.saveMatrixToCSVFile(outFolder + f"/AveragePoint2PointDistances_{featSelIndex}_{validIndex}.csv", avgP2PDists);

    # Finished processing
    print("Finished processing.");

#************* USING RADIAL BASIS FUNCTION INTERPOLATION FOR BONE AND BONE MUSCLE STRUCTURES
def featureToPelvisStructureRecon_radialBasicFunctionInterpolation_BoneStructures():
    """
    Performs feature-to-pelvis structure reconstruction using radial basis function interpolation on bone structures.
    
    This function evaluates the accuracy of pelvis bone reconstruction by deforming a template pelvis 
    bone mesh to match target feature points using rigid (SVD), affine (CPD), and radial basis function 
    transformations. The evaluation is performed across multiple feature selection strategies and cross-validation folds.
    
    Key Steps:
    1. Parse command line arguments for processing range
    2. Initialize system database and load template data
    3. Load feature selection strategies 
    4. For each feature selection strategy and validation fold:
       - Load validation subject IDs
       - Extract pelvis bone vertices and feature points for each subject
       - Apply rigid SVD transformation for initial alignment
       - Apply affine CPD transformation for further refinement
       - Apply radial basis function interpolation for final deformation
       - Compute point-to-point distance errors
       - Save validation errors to CSV files
    
    Transformations Applied:
    - Rigid SVD transformation: Aligns template to target via rotation/translation
    - Affine CPD transformation: Further refines alignment with scaling/shearing
    - Radial Basis Function (RBF): Non-rigid deformation using thin plate splines
    
    Output:
    - CSV files containing average point-to-point distances for each configuration
    """

    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 5):
        print("Please input the command as the following: [ProgramName] [StartFeatSelStratIndex] [EndFeatSelStratIndex] [StartValidIndex] [EndValidIndex]");
        return;
    startFeatSelStratIndex = int(sys.argv[1]); endFeatSelStratIndex = int(sys.argv[2]);
    startValidIndex = int(sys.argv[3]); endValidIndex = int(sys.argv[4]);
    personalizedPelvicStructureFolder = disk + r"\Data\PelvisBoneRecon\FemalePelvisGeometries\PersonalizedPelvisStructures";
    templateDataFolder = disk + r"\Data\PelvisBoneRecon\Template\PelvisBonesMuscles";
    systemDatabaseFolder = disk + "/Data/PelvisBoneRecon/SystemDataBase/"
    crossValidationFolder = disk + r"/Data/PelvisBoneRecon/CrossValidation";
    trainValidTestIDFolder = crossValidationFolder + "/TrainingValidTestingIDs";
    featureSelectionProtocolFolder = crossValidationFolder + "/FeatureSelectionProtocol";    
    systemDatabaseFilePath = systemDatabaseFolder + "/SystemDatabase.h5";    
    outFolder = crossValidationFolder + "/RadialBasicFunctionStrategy/BoneStructures";
    
    # Checking and creating output folder
    print("Checking and creating output folder ...");
    if (not os.path.exists(outFolder)): os.makedirs(outFolder);

    # Reading initial information
    print("Reading initial information ...");
    tempPelvisBoneMesh = sp.readMeshFromPLY(templateDataFolder + r"/TempPelvisBoneMesh.ply");
    tempPelvisBoneMuscleMesh = sp.readMeshFromPLY(templateDataFolder + r"/TempPelvisBoneMuscles.ply");
    pelvisBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempPelvisBoneMesh.vertices, tempPelvisBoneMuscleMesh.vertices);
    
    # Reading feature section strategies
    print("Reading feature selection strategies ...");
    allPelvicFeatures = sp.read3DPointsFromPPFile(featureSelectionProtocolFolder + "/AllFeaturePoints.pp");
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");
    
    # Iterate for each feature selection strategy
    print("Iterate for each feature selection strategy ...");
    for featSelIndex in range(startFeatSelStratIndex, endFeatSelStratIndex + 1):
        # Debugging
        print("/********************************************** FEATURE SELECTION STRATEGY: ", featSelIndex);

        # Get the feature selection strategies
        print("\t Getting the feature selection strategy ...");
        featureSelectionIndices = list(featureSelectionStrategyDict.values())[featSelIndex];
        tempPelvicFeatures = allPelvicFeatures[featureSelectionIndices];
        pelvisFeatureBaryIndices, pelvicFeatureBaryCoords = sp.computeBarycentricLandmarks(tempPelvisBoneMesh, tempPelvicFeatures);

        # Conduct training validating throughout each time of cross-validation
        for validIndex in range(startValidIndex, endValidIndex + 1):
            # Debugging information
            print("/********************************************** VALIDATION: ", validIndex);

            # Reading training and validating subject IDs
            print("\t Reading training and validating subject IDs ...");
            validIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/ValidationIDs_{validIndex}.txt");

            # Forming the training and validating data
            print("\t Forming the training and validating data ...");
            validPelvisBoneVertexData = []; validPelvisFeatureData = [];
            for i, ID in enumerate(validIDs):
                pelvicStructure = sp.readMeshFromPLY(personalizedPelvicStructureFolder + f"/{ID}-PersonalizedPelvisBoneMuscleMesh.ply");
                pelvisBoneMesh = sp.cloneMesh(tempPelvisBoneMesh);
                pelvisBoneMesh.vertices = pelvicStructure.vertices[pelvisBoneVertexIndices];
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvisBoneMesh, pelvisFeatureBaryIndices, pelvicFeatureBaryCoords);
                validPelvisBoneVertexData.append(pelvisBoneMesh.vertices.flatten());
                validPelvisFeatureData.append(pelvicFeatures.flatten());
            validPelvisBoneVertexData = np.array(validPelvisBoneVertexData);
            validPelvisFeatureData = np.array(validPelvisFeatureData);

            # Try to deform the template pelvis bone mesh to the target pelvis bone features
            print("\t Deforming template pelvis bone mesh to target pelvis bone features ...");
            avgP2PDists = [];
            for v, validPelvisFeatures in enumerate(validPelvisFeatureData):
                # Debugging
                print("#********************************** FEATURE SELECTION INDEX: ", featSelIndex, ", VALIDATION INDEX: ", validIndex, " VALID SUBJECT INDEX: ", v);

                # Getting target information
                targetFeaturePoints = validPelvisFeatures.reshape(-1, 3);
                validPelvisBoneVertices = validPelvisBoneVertexData[v].reshape(-1, 3);
                validPelvisBoneMesh = sp.cloneMesh(tempPelvisBoneMesh);
                validPelvisBoneMesh.vertices = validPelvisBoneVertices;

                # Deform the template pelvis bone to the target feature points
                defPelvisBoneMesh = sp.cloneMesh(tempPelvisBoneMesh);
                defFeaturePoints = sp.reconstructLandmarksFromBarycentric(defPelvisBoneMesh, pelvisFeatureBaryIndices, pelvicFeatureBaryCoords);
                svdTransform = sp.estimateRigidSVDTransform(defFeaturePoints, targetFeaturePoints);
                defPelvisBoneMesh = sp.transformMesh(defPelvisBoneMesh, svdTransform);
                defFeaturePoints = sp.transform3DPoints(defFeaturePoints, svdTransform);
                affineTransform = sp.estimateAffineTransformCPD(defFeaturePoints, targetFeaturePoints);
                defPelvisBoneMesh = sp.transformMesh(defPelvisBoneMesh, affineTransform);
                defFeaturePoints = sp.transform3DPoints(defFeaturePoints, affineTransform);

                # Deform using the radial basic function
                deformedToPersonalizedPelvisFeatureDisplacements =  targetFeaturePoints - defFeaturePoints;
                rbf = RBFInterpolator(defFeaturePoints, deformedToPersonalizedPelvisFeatureDisplacements, kernel='thin_plate_spline');
                personalizedPelvisBoneVertexDisplacements = rbf(defPelvisBoneMesh.vertices);
                personalizedPelvisBoneMesh = sp.cloneMesh(defPelvisBoneMesh);
                personalizedPelvisBoneMesh.vertices = defPelvisBoneMesh.vertices + personalizedPelvisBoneVertexDisplacements;

                # Computing points to points distances
                avgP2PDist = sp.computeAveragePointsToPointsDistance(personalizedPelvisBoneMesh.vertices, validPelvisBoneMesh.vertices);
                avgP2PDists.append(avgP2PDist);
            avgP2PDists = np.array(avgP2PDists);

            # Save the computed errors to file
            sp.saveMatrixToCSVFile(outFolder + f"/AveragePoint2PointDistances_{featSelIndex}_{validIndex}.csv", avgP2PDists);

    # Finished processing
    print("Finished processing.");
def featureToPelvisStructureRecon_radialBasicFunctionInterpolation_BoneMuscleStructures():
    """
    Performs feature-to-pelvis structure reconstruction using radial basis function interpolation on bone-muscle structures.
    
    This function evaluates the accuracy of pelvis bone reconstruction by deforming a template pelvis 
    bone-muscle mesh to match target feature points using rigid (SVD), affine (CPD), and radial basis function 
    transformations. The evaluation is performed across multiple feature selection strategies and cross-validation folds.
    
    Key Steps:
    1. Parse command line arguments for processing range
    2. Initialize system database and load template data
    3. Load feature selection strategies 
    4. For each feature selection strategy and validation fold:
       - Load validation subject IDs
       - Extract pelvis bone-muscle vertices and feature points for each subject
       - Apply rigid SVD transformation for initial alignment
       - Apply affine CPD transformation for further refinement
       - Apply radial basis function interpolation for final deformation
       - Compute point-to-point distance errors (bone vertices only)
       - Save validation errors to CSV files
    
    Transformations Applied:
    - Rigid SVD transformation: Aligns template to target via rotation/translation
    - Affine CPD transformation: Further refines alignment with scaling/shearing
    - Radial Basis Function (RBF): Non-rigid deformation using thin plate splines
    
    Output:
    - CSV files containing average point-to-point distances for each configuration
    
    Note: Accuracy is computed only on bone vertices since muscle ground truth may not be available
    """
    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 5):
        print("Please input the command as the following: [ProgramName] [StartFeatSelStratIndex] [EndFeatSelStratIndex] [StartValidIndex] [EndValidIndex]");
        return;
    startFeatSelStratIndex = int(sys.argv[1]); endFeatSelStratIndex = int(sys.argv[2]);
    startValidIndex = int(sys.argv[3]); endValidIndex = int(sys.argv[4]);
    personalizedPelvicStructureFolder = disk + r"\Data\PelvisBoneRecon\FemalePelvisGeometries\PersonalizedPelvisStructures";
    templateDataFolder = disk + r"\Data\PelvisBoneRecon\Template\PelvisBonesMuscles";
    crossValidationFolder = disk + r"/Data/PelvisBoneRecon/CrossValidation";
    trainValidTestIDFolder = crossValidationFolder + "/TrainingValidTestingIDs";
    featureSelectionProtocolFolder = crossValidationFolder + "/FeatureSelectionProtocol";
    outFolder = crossValidationFolder + "/RadialBasicFunctionStrategy/BoneMuscleStructures";
    
    # Checking and creating output folder
    print("Checking and creating output folder ...");
    if (not os.path.exists(outFolder)): os.makedirs(outFolder);

    # Reading initial information
    print("Reading initial information ...");
    tempPelvisBoneMesh = sp.readMeshFromPLY(templateDataFolder + r"/TempPelvisBoneMesh.ply");
    tempPelvisBoneMuscleMesh = sp.readMeshFromPLY(templateDataFolder + r"/TempPelvisBoneMuscles.ply");
    pelvisBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempPelvisBoneMesh.vertices, tempPelvisBoneMuscleMesh.vertices);
    
    # Reading feature section strategies
    print("Reading feature selection strategies ...");
    allPelvicFeatures = sp.read3DPointsFromPPFile(featureSelectionProtocolFolder + "/AllFeaturePoints.pp");
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");
    
    # Iterate for each feature selection strategy
    print("Iterate for each feature selection strategy ...");
    for featSelIndex in range(startFeatSelStratIndex, endFeatSelStratIndex + 1):
        # Debugging
        print("/********************************************** FEATURE SELECTION STRATEGY: ", featSelIndex);

        # Get the feature selection strategies
        print("\t Getting the feature selection strategy ...");
        featureSelectionIndices = list(featureSelectionStrategyDict.values())[featSelIndex];
        tempPelvicFeatures = allPelvicFeatures[featureSelectionIndices];
        pelvisFeatureBaryIndices, pelvicFeatureBaryCoords = sp.computeBarycentricLandmarks(tempPelvisBoneMuscleMesh, tempPelvicFeatures);

        # Conduct training validating throughout each time of cross-validation
        for validIndex in range(startValidIndex, endValidIndex + 1):
            # Debugging information
            print("/********************************************** VALIDATION: ", validIndex);

            # Reading training and validating subject IDs
            print("\t Reading training and validating subject IDs ...");
            validIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/ValidationIDs_{validIndex}.txt");

            # Forming the training and validating data
            print("\t Forming the training and validating data ...");
            validPelvisBoneMuscleVertexData = []; validPelvisFeatureData = [];
            for i, ID in enumerate(validIDs):
                pelvisBoneMuscleMesh = sp.readMeshFromPLY(personalizedPelvicStructureFolder + f"/{ID}-PersonalizedPelvisBoneMuscleMesh.ply");
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvisBoneMuscleMesh, pelvisFeatureBaryIndices, pelvicFeatureBaryCoords);
                validPelvisBoneMuscleVertexData.append(pelvisBoneMuscleMesh.vertices.flatten());
                validPelvisFeatureData.append(pelvicFeatures.flatten());
            validPelvisBoneMuscleVertexData = np.array(validPelvisBoneMuscleVertexData);
            validPelvisFeatureData = np.array(validPelvisFeatureData);

            # Try to deform the template pelvis bone mesh to the target pelvis bone features
            print("\t Deforming template pelvis bone mesh to target pelvis bone features ...");
            avgP2PDists = [];
            for v, validPelvisFeatures in enumerate(validPelvisFeatureData):
                # Debugging
                print("#********************************** FEATURE SELECTION INDEX: ", featSelIndex, ", VALIDATION INDEX: ", validIndex, " VALID SUBJECT INDEX: ", v);

                # Getting target information
                targetFeaturePoints = validPelvisFeatures.reshape(-1, 3);
                validPelvisBoneMuscleVertices = validPelvisBoneMuscleVertexData[v].reshape(-1, 3);
                validPelvisBoneMuscleMesh = sp.cloneMesh(tempPelvisBoneMuscleMesh);
                validPelvisBoneMuscleMesh.vertices = validPelvisBoneMuscleVertices;

                # Deform the template pelvis bone to the target feature points
                defPelvisBoneMuscleMesh = sp.cloneMesh(tempPelvisBoneMuscleMesh);
                defFeaturePoints = sp.reconstructLandmarksFromBarycentric(defPelvisBoneMuscleMesh, pelvisFeatureBaryIndices, pelvicFeatureBaryCoords);
                svdTransform = sp.estimateRigidSVDTransform(defFeaturePoints, targetFeaturePoints);
                defPelvisBoneMuscleMesh = sp.transformMesh(defPelvisBoneMuscleMesh, svdTransform);
                defFeaturePoints = sp.transform3DPoints(defFeaturePoints, svdTransform);
                affineTransform = sp.estimateAffineTransformCPD(defFeaturePoints, targetFeaturePoints);
                defPelvisBoneMuscleMesh = sp.transformMesh(defPelvisBoneMuscleMesh, affineTransform);
                defFeaturePoints = sp.transform3DPoints(defFeaturePoints, affineTransform);
                
                # Deform using the radial basic function
                deformedToPersonalizedPelvisFeatureDisplacements =  targetFeaturePoints - defFeaturePoints;
                rbf = RBFInterpolator(defFeaturePoints, deformedToPersonalizedPelvisFeatureDisplacements, kernel='thin_plate_spline');
                personalizedPelvisBoneMuscleVertexDisplacements = rbf(defPelvisBoneMuscleMesh.vertices);
                personalizedPelvisBoneMuscleMesh = sp.cloneMesh(defPelvisBoneMuscleMesh);
                personalizedPelvisBoneMuscleMesh.vertices = defPelvisBoneMuscleMesh.vertices + personalizedPelvisBoneMuscleVertexDisplacements;
                
                defFeaturePoints = sp.reconstructLandmarksFromBarycentric(personalizedPelvisBoneMuscleMesh, pelvisFeatureBaryIndices, pelvicFeatureBaryCoords);
                
                # Compute only the bone vertices for computing distances
                validPelvisBoneVertices = validPelvisBoneMuscleMesh.vertices[pelvisBoneVertexIndices];
                personalizedBoneVertices = personalizedPelvisBoneMuscleMesh.vertices[pelvisBoneVertexIndices];

                # Computing points to points distances
                avgP2PDist = sp.computeAveragePointsToPointsDistance(validPelvisBoneVertices, personalizedBoneVertices);
                avgP2PDists.append(avgP2PDist);
            avgP2PDists = np.array(avgP2PDists);

            # Save the computed errors to file
            sp.saveMatrixToCSVFile(outFolder + f"/AveragePoint2PointDistances_{featSelIndex}_{validIndex}.csv", avgP2PDists);

    # Finished processing
    print("Finished processing.");

#************* USING SHAPE OPTIMIZATION STRATEGY FOR BONE AND BONE MUSCLE STRUCTURES
def featureToPelvisStructureRecon_shapeOptimizationStrategy_BoneStructures():
    """
    Performs feature-to-pelvis structure reconstruction using shape optimization strategy on bone structures.
    
    This function evaluates the accuracy of pelvis bone reconstruction by optimizing a statistical shape model 
    (SSM) of pelvis bones to match target feature points using least squares optimization.
    The evaluation is performed across multiple feature selection strategies, cross-validation folds, and 
    varying numbers of principal components.
    
    Key Steps:
    1. Parse command line arguments for processing range (feature strategies, validation folds, components)
    2. Initialize system database and load template data
    3. Load feature selection strategies 
    4. For each feature selection strategy and validation fold:
       - Load training and validation subject IDs
       - Extract pelvis bone vertices and feature points for each subject
       - Train PCA statistical shape model on training data with specified components
       - For each validation subject:
         * Initialize parameters from mean training data
         * Optimize SSM parameters to minimize feature point distances using least squares
         * Reconstruct pelvis bone mesh from optimized parameters
         * Compute point-to-point distance errors
       - Save validation errors to CSV files
    
    Optimization Strategy:
    - PCA Statistical Shape Model: Learns shape variations from training data
    - Least Squares Optimization: Minimizes distance between reconstructed and target feature points
    - Parameter Space Search: Optimizes in reduced PCA parameter space
    
    Output:
    - CSV files containing average point-to-point distances for each feature strategy, 
      validation fold, and number of components configuration
    """
    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 7):
        print("Please input the command as the following: [ProgramName] [StartFeatSelStratIndex] [EndFeatSelStratIndex] [StartValidIndex] [EndValidIndex] [StartNumComps] [EndNumComps]");
        return;
    startFeatSelStratIndex = int(sys.argv[1]); endFeatSelStratIndex = int(sys.argv[2]);
    startValidIndex = int(sys.argv[3]); endValidIndex = int(sys.argv[4]);
    startNumComps = int(sys.argv[5]); endNumComps = int(sys.argv[6]);    
    personalizedPelvicStructureFolder = disk + r"\Data\PelvisBoneRecon\FemalePelvisGeometries\PersonalizedPelvisStructures";
    templateDataFolder = disk + r"\Data\PelvisBoneRecon\Template\PelvisBonesMuscles";
    crossValidationFolder = disk + r"\Data\PelvisBoneRecon\CrossValidation";
    trainValidTestIDFolder = crossValidationFolder + r"\TrainingValidTestingIDs";
    featureSelectionProtocolFolder = crossValidationFolder + r"\FeatureSelectionProtocol";    
    outFolder = crossValidationFolder + r"\ShapeOptimizationStrategy\BoneStructures";

    # Defining helper functions
    print("Defining helper functions ...");
    def reconstructPelvisBoneMesh(pcaModel, paramData, meshFaces):
        """Reconstruct pelvis bone mesh vertices from PCA model."""
        meshVertices = pcaModel.inverse_transform(paramData);
        meshVertices = meshVertices.reshape(-1, 3);
        outMesh = sp.formMesh(meshVertices, meshFaces);
        return outMesh;
    def computeFeaturePoints(pelvisBoneMesh, baryIndices, baryCoords):
        """Compute feature points using barycentric interpolation."""
        featurePoints = sp.reconstructLandmarksFromBarycentric(pelvisBoneMesh, baryIndices, baryCoords);
        return featurePoints
    def optimizePelvisBoneSSM(initialParams, pcaModel, meshFaces, baryIndices, baryCoords, targetFeaturePoints):
        """Optimization function to align pelvis bone feature points with target points."""
        reconstructedMesh = reconstructPelvisBoneMesh(pcaModel, initialParams.reshape(1, -1), meshFaces)
        computedFeaturePoints = computeFeaturePoints(reconstructedMesh, baryIndices, baryCoords)
        return (computedFeaturePoints - targetFeaturePoints).flatten();
    
    # Checking and creating output folder
    print("Checking and creating output folder ...");
    if (not os.path.exists(outFolder)): os.makedirs(outFolder);

    # Reading initial information
    print("Reading initial information ...");
    tempPelvisBoneMesh = sp.readMeshFromPLY(templateDataFolder + r"/TempPelvisBoneMesh.ply");
    tempPelvisBoneMuscleMesh = sp.readMeshFromPLY(templateDataFolder + r"/TempPelvisBoneMuscles.ply");
    pelvisBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempPelvisBoneMesh.vertices, tempPelvisBoneMuscleMesh.vertices);
    
    # Reading feature section strategies
    print("Reading feature selection strategies ...");
    allPelvicFeatures = sp.read3DPointsFromPPFile(featureSelectionProtocolFolder + "/AllFeaturePoints.pp");
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");
    
    # Iterate for each feature selection strategy
    print("Iterate for each feature selection strategy ...");
    for featSelIndex in range(startFeatSelStratIndex, endFeatSelStratIndex + 1):
        # Debugging
        print("/********************************************** FEATURE SELECTION STRATEGY: ", featSelIndex);

        # Get the feature selection strategies
        print("\t Getting the feature selection strategy ...");
        featureSelectionIndices = list(featureSelectionStrategyDict.values())[featSelIndex];
        tempPelvicFeatures = allPelvicFeatures[featureSelectionIndices];
        pelvicFeatureBaryIndices, pelvicFeatureBaryCoords = sp.computeBarycentricLandmarks(tempPelvisBoneMesh, tempPelvicFeatures);

        # Conduct training validating throughout each time of cross-validation
        for validIndex in range(startValidIndex, endValidIndex + 1):
            # Debugging information
            print("/********************************************** VALIDATION: ", validIndex);

            # Reading training and validating subject IDs
            print("\t Reading training and validating subject IDs ...");
            trainIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/TrainingIDs_{validIndex}.txt");
            validIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/ValidationIDs_{validIndex}.txt");

            # Forming the training and validating data
            print("\t Forming the training and validating data ...");
            trainPelvisBoneVertexData = []; 
            validPelvisBoneVertexData = []; validPelvisFeatureData = [];
            for i, ID in enumerate(trainIDs):
                pelvicStructure = sp.readMeshFromPLY(personalizedPelvicStructureFolder + f"/{ID}-PersonalizedPelvisBoneMuscleMesh.ply");
                pelvisBoneMesh = sp.cloneMesh(tempPelvisBoneMesh);
                pelvisBoneMesh.vertices = pelvicStructure.vertices[pelvisBoneVertexIndices];
                trainPelvisBoneVertexData.append(pelvisBoneMesh.vertices.flatten());
            for i, ID in enumerate(validIDs):
                pelvicStructure = sp.readMeshFromPLY(personalizedPelvicStructureFolder + f"/{ID}-PersonalizedPelvisBoneMuscleMesh.ply");
                pelvisBoneMesh = sp.cloneMesh(tempPelvisBoneMesh);
                pelvisBoneMesh.vertices = pelvicStructure.vertices[pelvisBoneVertexIndices];
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvisBoneMesh, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                validPelvisBoneVertexData.append(pelvisBoneMesh.vertices.flatten());
                validPelvisFeatureData.append(pelvicFeatures.flatten());
            trainPelvisBoneVertexData = np.array(trainPelvisBoneVertexData);
            validPelvisBoneVertexData = np.array(validPelvisBoneVertexData);
            validPelvisFeatureData = np.array(validPelvisFeatureData);

            # Begin to train for each number of components
            print("\t Begin to train in each number of components ...");
            for numComps in range(startNumComps, endNumComps + 1):
                print("/***************************** FEATURE SELECTION INDEX: ", featSelIndex, ", VALIDATION INDEX: ", validIndex, ", NUM COMPS: ", numComps);

                # Train the statistial shape model of the pelvis bone structure
                pelvisBoneNumComps = min(numComps, trainPelvisBoneVertexData.shape[1]);
                pelvisBoneSSM = PCA(n_components=pelvisBoneNumComps);
                pelvisBoneSSM.fit(trainPelvisBoneVertexData);

                # Optimize the trained PCA model to the valid features
                avgP2PDists = [];
                for v, validFeatures in enumerate(validPelvisFeatureData):
                    # Getting validating feature points
                    validFeaturePoints = validFeatures.reshape(-1, 3);
                    validPelvisBoneVertices = validPelvisBoneVertexData[v];
                    validPelvisBoneVertices = validPelvisBoneVertices.reshape(-1, 3);

                    # Optimize to the target features
                    initialParams = pelvisBoneSSM.transform(trainPelvisBoneVertexData.mean(axis=0).reshape(1, -1)).flatten();
                    result = least_squares(optimizePelvisBoneSSM, initialParams, args=(pelvisBoneSSM, tempPelvisBoneMesh.faces, 
                                                                                       pelvicFeatureBaryIndices, pelvicFeatureBaryCoords, 
                                                                                       validFeaturePoints), verbose=0, ftol=1e-12);
                    optimizedParams = result.x.reshape(1, -1);
                    optimizedPelvisBoneVertices = pelvisBoneSSM.inverse_transform(optimizedParams);
                    optimizedPelvisBoneVertices = optimizedPelvisBoneVertices.reshape(-1, 3);

                    # Compute points to points distances
                    avgP2PDist = sp.computeAveragePointsToPointsDistance(optimizedPelvisBoneVertices, validPelvisBoneVertices);
                    avgP2PDists.append(avgP2PDist);
                avgP2PDist = np.array(avgP2PDist);

                # Save the comptued errors to file
                sp.saveMatrixToCSVFile(outFolder + f"/AveragePoint2PointDistances_{featSelIndex}_{validIndex}_{numComps}.csv", avgP2PDists);

    # Finished processing
    print("Finished processing.");
def featureToPelvisStructureRecon_shapeOptimizationStrategy_BoneMuscleStructures():
    """
    Performs feature-to-pelvis structure reconstruction using shape optimization strategy on bone-muscle structures.
    
    This function evaluates the accuracy of pelvis bone reconstruction by optimizing a statistical shape model 
    (SSM) of pelvis bone-muscle structures to match target feature points using least squares optimization.
    The evaluation is performed across multiple feature selection strategies, cross-validation folds, and 
    varying numbers of principal components.
    
    Key Steps:
    1. Parse command line arguments for processing range (feature strategies, validation folds, components)
    2. Initialize system database and load template data
    3. Load feature selection strategies 
    4. For each feature selection strategy and validation fold:
       - Load training and validation subject IDs
       - Extract pelvis bone-muscle vertices and feature points for each subject
       - Train PCA statistical shape model on training data with specified components
       - For each validation subject:
         * Initialize parameters from mean training data
         * Optimize SSM parameters to minimize feature point distances using least squares
         * Reconstruct pelvis bone-muscle mesh from optimized parameters
         * Compute point-to-point distance errors (bone vertices only)
       - Save validation errors to CSV files
    
    Optimization Strategy:
    - PCA Statistical Shape Model: Learns shape variations from training data
    - Least Squares Optimization: Minimizes distance between reconstructed and target feature points
    - Parameter Space Search: Optimizes in reduced PCA parameter space
    
    Output:
    - CSV files containing average point-to-point distances for each feature strategy, 
      validation fold, and number of components configuration
    
    Note: Accuracy is computed only on bone vertices since muscle ground truth may not be available
    """

    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 7):
        print("Please input the command as the following: [ProgramName] [StartFeatSelStratIndex] [EndFeatSelStratIndex] [StartValidIndex] [EndValidIndex] [StartNumComps] [EndNumComps]");
        return;
    startFeatSelStratIndex = int(sys.argv[1]); endFeatSelStratIndex = int(sys.argv[2]);
    startValidIndex = int(sys.argv[3]); endValidIndex = int(sys.argv[4]);
    startNumComps = int(sys.argv[5]); endNumComps = int(sys.argv[6]);
    personalizedPelvicStructureFolder = disk + r"\Data\PelvisBoneRecon\FemalePelvisGeometries\PersonalizedPelvisStructures";
    templateDataFolder = disk + r"\Data\PelvisBoneRecon\Template\PelvisBonesMuscles";
    crossValidationFolder = disk + r"/Data/PelvisBoneRecon/CrossValidation";
    trainValidTestIDFolder = crossValidationFolder + "/TrainingValidTestingIDs";
    featureSelectionProtocolFolder = crossValidationFolder + "/FeatureSelectionProtocol";
    outFolder = crossValidationFolder + "/ShapeOptimizationStrategy/BoneMuscleStructures";

    # Checking and creating output folder
    print("Checking and creating output folder ...");
    def reconstructPelvisBoneMesh(pcaModel, paramData, meshFaces):
        """Reconstruct pelvis bone mesh vertices from PCA model."""
        meshVertices = pcaModel.inverse_transform(paramData);
        meshVertices = meshVertices.reshape(-1, 3);
        outMesh = sp.formMesh(meshVertices, meshFaces);
        return outMesh;
    def computeFeaturePoints(pelvisBoneMesh, baryIndices, baryCoords):
        """Compute feature points using barycentric interpolation."""
        featurePoints = sp.reconstructLandmarksFromBarycentric(pelvisBoneMesh, baryIndices, baryCoords);
        return featurePoints
    def optimizePelvisBoneSSM(initialParams, pcaModel, meshFaces, baryIndices, baryCoords, targetFeaturePoints):
        """Optimization function to align pelvis bone feature points with target points."""
        reconstructedMesh = reconstructPelvisBoneMesh(pcaModel, initialParams.reshape(1, -1), meshFaces)
        computedFeaturePoints = computeFeaturePoints(reconstructedMesh, baryIndices, baryCoords)
        return (computedFeaturePoints - targetFeaturePoints).flatten();
    
    # Checking and creating output folder
    print("Checking and creating output folder ...");
    if (not os.path.exists(outFolder)): os.makedirs(outFolder);

    # Reading initial information
    print("Reading initial information ...");
    tempPelvisBoneMesh = sp.readMeshFromPLY(templateDataFolder + r"/TempPelvisBoneMesh.ply");
    tempPelvisBoneMuscleMesh = sp.readMeshFromPLY(templateDataFolder + r"/TempPelvisBoneMuscles.ply");
    pelvisBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempPelvisBoneMesh.vertices, tempPelvisBoneMuscleMesh.vertices);
    
    # Reading feature section strategies
    print("Reading feature selection strategies ...");
    allPelvicFeatures = sp.read3DPointsFromPPFile(featureSelectionProtocolFolder + "/AllFeaturePoints.pp");
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");
    
    # Iterate for each feature selection strategy
    print("Iterate for each feature selection strategy ...");
    for featSelIndex in range(startFeatSelStratIndex, endFeatSelStratIndex + 1):
        # Debugging
        print("/********************************************** FEATURE SELECTION STRATEGY: ", featSelIndex);

        # Get the feature selection strategies
        print("\t Getting the feature selection strategy ...");
        featureSelectionIndices = list(featureSelectionStrategyDict.values())[featSelIndex];
        tempPelvicFeatures = allPelvicFeatures[featureSelectionIndices];
        pelvicFeatureBaryIndices, pelvicFeatureBaryCoords = sp.computeBarycentricLandmarks(tempPelvisBoneMuscleMesh, tempPelvicFeatures);

        # Conduct training validating throughout each time of cross-validation
        for validIndex in range(startValidIndex, endValidIndex + 1):
            # Debugging information
            print("/********************************************** VALIDATION: ", validIndex);

            # Reading training and validating subject IDs
            print("\t Reading training and validating subject IDs ...");
            trainIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/TrainingIDs_{validIndex}.txt");
            validIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/ValidationIDs_{validIndex}.txt");

            # Forming the training and validating data
            print("\t Forming the training and validating data ...");
            trainPelvisBoneMuscleVertexData = []; 
            validPelvisBoneMuscleVertexData = []; validPelvisFeatureData = [];
            for i, ID in enumerate(trainIDs):
                pelvicStructure = sp.readMeshFromPLY(personalizedPelvicStructureFolder + f"/{ID}-PersonalizedPelvisBoneMuscleMesh.ply");
                trainPelvisBoneMuscleVertexData.append(pelvicStructure.vertices.flatten());
            for i, ID in enumerate(validIDs):
                pelvicStructure = sp.readMeshFromPLY(personalizedPelvicStructureFolder + f"/{ID}-PersonalizedPelvisBoneMuscleMesh.ply");
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvicStructure, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                validPelvisBoneMuscleVertexData.append(pelvicStructure.vertices.flatten());
                validPelvisFeatureData.append(pelvicFeatures.flatten());
            trainPelvisBoneMuscleVertexData = np.array(trainPelvisBoneMuscleVertexData);
            validPelvisBoneMuscleVertexData = np.array(validPelvisBoneMuscleVertexData);
            validPelvisFeatureData = np.array(validPelvisFeatureData);

            # Begin to train for each number of components
            print("\t Begin to train in each number of components ...");
            for numComps in range(startNumComps, endNumComps + 1):
                print("/***************************** FEATURE SELECTION INDEX: ", featSelIndex, ", VALIDATION INDEX: ", validIndex, ", NUM COMPS: ", numComps);

                # Train the statistial shape model of the pelvis bone structure
                pelvisBoneMuscleNumComps = min(numComps, trainPelvisBoneMuscleVertexData.shape[1]);
                pelvisBoneMuscleSSM = PCA(n_components=pelvisBoneMuscleNumComps);
                pelvisBoneMuscleSSM.fit(trainPelvisBoneMuscleVertexData);

                # Optimize the trained PCA model to the valid features
                avgP2PDists = [];
                for v, validFeatures in enumerate(validPelvisFeatureData):
                    # Getting validating feature points
                    validFeaturePoints = validFeatures.reshape(-1, 3);
                    validPelvisBoneMuscleVertices = validPelvisBoneMuscleVertexData[v];
                    validPelvisBoneMuscleVertices = validPelvisBoneMuscleVertices.reshape(-1, 3);

                    # Optimize to the target features
                    initialParams = pelvisBoneMuscleSSM.transform(trainPelvisBoneMuscleVertexData.mean(axis=0).reshape(1, -1)).flatten();
                    result = least_squares(optimizePelvisBoneSSM, initialParams, args=(pelvisBoneMuscleSSM, tempPelvisBoneMuscleMesh.faces, 
                                                                                       pelvicFeatureBaryIndices, pelvicFeatureBaryCoords, 
                                                                                       validFeaturePoints), verbose=0, ftol=1e-12);
                    optimizedParams = result.x.reshape(1, -1);
                    optimizedPelvisBoneMuscleVertices = pelvisBoneMuscleSSM.inverse_transform(optimizedParams);
                    optimizedPelvisBoneMuscleVertices = optimizedPelvisBoneMuscleVertices.reshape(-1, 3);

                    # Compute only accuracy on the pelvis bone structure
                    validPelvisBoneVertices = validPelvisBoneMuscleVertices[pelvisBoneVertexIndices];
                    optimPelvisBoneVertices = optimizedPelvisBoneMuscleVertices[pelvisBoneVertexIndices];

                    # Compute points to points distances
                    avgP2PDist = sp.computeAveragePointsToPointsDistance(optimPelvisBoneVertices, validPelvisBoneVertices);
                    avgP2PDists.append(avgP2PDist);
                avgP2PDist = np.array(avgP2PDist);

                # Save the comptued errors to file
                sp.saveMatrixToCSVFile(outFolder + f"/AveragePoint2PointDistances_{featSelIndex}_{validIndex}_{numComps}.csv", avgP2PDists);

    # Finished processing
    print("Finished processing.");

#************* USING SHAPE RELATION STRATEGY FOR BONE AND BONE MUSCLE STRUCTURES
def featureToPelvisStructureRecon_shapeRelationStrategy_BoneStructures():
    """
    Performs feature-to-pelvis structure reconstruction using shape relation strategy on bone structures.
    
    This function evaluates the accuracy of pelvis bone reconstruction by learning statistical relationships 
    between feature points and bone shape parameters using PCA dimensionality reduction and linear regression.
    The evaluation is performed across multiple feature selection strategies, cross-validation folds, and 
    varying numbers of principal components.
    
    Key Steps:
    1. Parse command line arguments for processing range (feature strategies, validation folds, components)
    2. Initialize system database and load template data
    3. Load feature selection strategies 
    4. For each feature selection strategy and validation fold:
       - Load training and validation subject IDs
       - Extract pelvis bone vertices and feature points for each subject
       - Scale and normalize both feature and vertex data using StandardScaler
       - Apply PCA dimensionality reduction to both feature and vertex spaces
       - Train linear regression model to map feature parameters to vertex parameters
       - For each validation subject:
         * Transform features to PCA parameter space
         * Predict vertex parameters using trained regression model
         * Reconstruct pelvis bone mesh from predicted parameters
         * Compute point-to-point distance errors
       - Save validation errors to CSV files
    
    Shape Relation Strategy:
    - PCA Feature Space: Reduces feature dimensionality while preserving variance
    - PCA Vertex Space: Creates compact representation of shape variations
    - Linear Regression: Learns mapping between feature and vertex parameter spaces
    - Cross-validation: Ensures robust evaluation across different data splits
    
    Output:
    - CSV files containing average point-to-point distances for each feature strategy, 
      validation fold, and number of components configuration
    """

    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 7):
        print("Please input the command as the following: [ProgramName] [StartFeatSelStratIndex] [EndFeatSelStratIndex] [StartValidIndex] [EndValidIndex] [StartNumComps] [EndNumComps]");
        return;
    startFeatSelStratIndex = int(sys.argv[1]); endFeatSelStratIndex = int(sys.argv[2]);
    startValidIndex = int(sys.argv[3]); endValidIndex = int(sys.argv[4]);
    startNumComps = int(sys.argv[5]); endNumComps = int(sys.argv[6]);
    personalizedPelvicStructureFolder = disk + r"/Data/PelvisBoneRecon/FemalePelvisGeometries/PersonalizedPelvisStructures";
    templateDataFolder = disk + r"/Data/PelvisBoneRecon/Template/PelvisBonesMuscles";
    crossValidationFolder = disk + r"/Data/PelvisBoneRecon/CrossValidation";
    trainValidTestIDFolder = crossValidationFolder + "/TrainingValidTestingIDs";
    featureSelectionProtocolFolder = crossValidationFolder + "/FeatureSelectionProtocol";
    outFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneStructures";

    # Checking and creating output folder
    print("Checking and creating output folder ...");
    if (not os.path.exists(outFolder)): os.makedirs(outFolder);

    # Reading initial information
    print("Reading initial information ...");
    tempPelvisBoneMesh = sp.readMeshFromPLY(templateDataFolder + r"/TempPelvisBoneMesh.ply");
    tempPelvisBoneMuscleMesh = sp.readMeshFromPLY(templateDataFolder + r"/TempPelvisBoneMuscles.ply");
    pelvisBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempPelvisBoneMesh.vertices, tempPelvisBoneMuscleMesh.vertices);

    # Reading feature selection strategies
    print("Reading feature selection strategies ...");
    allPelvicFeatures = sp.read3DPointsFromPPFile(featureSelectionProtocolFolder + "/AllFeaturePoints.pp");
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");
    
    # Iterate for each feature selection strategy
    print("Iterate for each feature selection strategy ...");
    for featSelIndex in range(startFeatSelStratIndex, endFeatSelStratIndex + 1):
        # Debugging
        print("/********************************************** FEATURE SELECTION STRATEGY: ", featSelIndex);

        # Get the feature selection strategies
        print("\t Getting the feature selection strategy ...");
        featureSelectionIndices = list(featureSelectionStrategyDict.values())[featSelIndex];
        tempPelvicFeatures = allPelvicFeatures[featureSelectionIndices];
        pelvicFeatureBaryIndices, pelvicFeatureBaryCoords = sp.computeBarycentricLandmarks(tempPelvisBoneMesh, tempPelvicFeatures);

        # Conduct training validating throughout each time of cross-validation
        for validIndex in range(startValidIndex, endValidIndex + 1):
            # Debugging information
            print("/********************************************** VALIDATION: ", validIndex);

            # Reading training and validating subject IDs
            print("\t Reading training and validating subject IDs ...");
            trainIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/TrainingIDs_{validIndex}.txt");
            validIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/ValidationIDs_{validIndex}.txt");

            # Forming the training and validating data
            print("\t Forming the training and validating data ...");
            trainPelvisBoneVertexData = []; validPelvicVertexData = [];
            trainPelvisFeatureData = []; validPelvicFeatureData = [];
            for i, ID in enumerate(trainIDs):
                pelvicStructure = sp.readMeshFromPLY(personalizedPelvicStructureFolder + f"/{ID}-PersonalizedPelvisBoneMuscleMesh.ply");
                pelvisBoneMesh = sp.cloneMesh(tempPelvisBoneMesh);
                pelvisBoneMesh.vertices = pelvicStructure.vertices[pelvisBoneVertexIndices];
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvisBoneMesh, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                trainPelvisBoneVertexData.append(pelvisBoneMesh.vertices.flatten());
                trainPelvisFeatureData.append(pelvicFeatures.flatten());
            for i, ID in enumerate(validIDs):
                pelvicStructure = sp.readMeshFromPLY(personalizedPelvicStructureFolder + f"/{ID}-PersonalizedPelvisBoneMuscleMesh.ply");
                pelvisBoneMesh = sp.cloneMesh(tempPelvisBoneMesh);
                pelvisBoneMesh.vertices = pelvicStructure.vertices[pelvisBoneVertexIndices];
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvisBoneMesh, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                validPelvicVertexData.append(pelvisBoneMesh.vertices.flatten());
                validPelvicFeatureData.append(pelvicFeatures.flatten());

            # Begin to train for each number of components
            print("\t Begin to train in each number of components ...");
            for numComps in range(startNumComps, endNumComps + 1):
                print("/***************************** FEATURE SELECTION INDEX: ", featSelIndex, ", VALIDATION INDEX: ", validIndex, ", NUM COMPS: ", numComps);

                # Scale the pelvic feature and vertex data
                print("\t Scaling data ...");
                yScaler = StandardScaler().fit(trainPelvisBoneVertexData);
                xScaler = StandardScaler().fit(trainPelvisFeatureData);
                scaledYData = yScaler.transform(trainPelvisBoneVertexData);
                scaledXData = xScaler.transform(trainPelvisFeatureData);

                # Control the number of components
                print("\t Controlling the number of components ...");
                targetNumComps = numComps;
                xDims = scaledXData.shape[1];
                yDims = scaledYData.shape[1];
                xNumComps = min(xDims, targetNumComps);
                yNumComps = min(yDims, targetNumComps);

                # Parameterize the scaled pelvic vertex and feature data
                print("\t Parameterization training ...");
                print("\t\t Parameterizing the scaledXData ...");
                xSSM = PCA(n_components=xNumComps);
                xSSM.fit(scaledXData);
                trainXParamData = xSSM.transform(scaledXData);
                print("\t\t Parameterizing the scaledYData ...");
                ySSM = PCA(n_components=yNumComps);
                ySSM.fit(scaledYData);
                trainYParamData = ySSM.transform(scaledYData);

                # Linear regression from feature parameters to vertex parameters
                print("\t Regression training ...");
                regressor = LinearRegression()
                regressor.fit(trainXParamData, trainYParamData);

                # Computing errors on the validating data
                print("\t Computing valiation errors ...");
                ## Scale and parameterize the validation data
                scaledValidXData = xScaler.transform(validPelvicFeatureData);
                validXParams = xSSM.transform(scaledValidXData);
                ## Try to predict the pelvic vertex params from the pelvic feature params
                predYParams = regressor.predict(validXParams);
                ## Inverse transform to the scaled Y data
                predScaledYData = ySSM.inverse_transform(predYParams);
                predYData = yScaler.inverse_transform(predScaledYData);
                ## Computing validating errors
                avgP2PDists = [];
                for v, predY in enumerate(predYData):
                    validY = validPelvicVertexData[v];
                    validPelvicStructureVertices = validY.reshape(-1, 3);
                    predPelvicStructureVertices = predY.reshape(-1, 3);
                    avgP2PDist = sp.computeAveragePointsToPointsDistance(predPelvicStructureVertices, validPelvicStructureVertices);
                    avgP2PDists.append(avgP2PDist);
                avgP2PDists = np.array(avgP2PDists);

                # Save the computed errors
                print("\t Save the computed errors ...");
                sp.saveMatrixToCSVFile(outFolder + f"/AveragePoint2PointDistances_{featSelIndex}_{validIndex}_{numComps}.csv", avgP2PDists);

    # Finished processing
    print("Finished processing.");
def featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_multivarirateLinearRegression():
    """
    Performs feature-to-pelvis structure reconstruction using multivariate linear regression on bone-muscle structures.
    
    This function evaluates the accuracy of pelvis bone reconstruction by learning statistical relationships 
    between feature points and bone-muscle shape parameters using PCA dimensionality reduction and multivariate 
    linear regression. The evaluation is performed across multiple feature selection strategies, cross-validation 
    folds, and varying numbers of principal components.
    
    Key Steps:
    1. Parse command line arguments for processing range (feature strategies, validation folds, components)
    2. Initialize system database and load template data
    3. Load feature selection strategies 
    4. For each feature selection strategy and validation fold:
       - Load training and validation subject IDs
       - Extract pelvis bone-muscle vertices and feature points for each subject
       - Scale and normalize both feature and vertex data using StandardScaler
       - Apply PCA dimensionality reduction to both feature and vertex spaces
       - Train multivariate linear regression model to map feature parameters to vertex parameters
       - For each validation subject:
         * Transform features to PCA parameter space
         * Predict vertex parameters using trained regression model
         * Reconstruct pelvis bone-muscle mesh from predicted parameters
         * Compute point-to-point distance errors (bone vertices only)
       - Save validation errors to CSV files
    
    Shape Relation Strategy:
    - PCA Feature Space: Reduces feature dimensionality while preserving variance
    - PCA Vertex Space: Creates compact representation of shape variations
    - Multivariate Linear Regression: Learns direct mapping between feature and vertex parameter spaces
    - Cross-validation: Ensures robust evaluation across different data splits
    
    Output:
    - CSV files containing average point-to-point distances for each feature strategy, 
      validation fold, and number of components configuration
    
    Note: Accuracy is computed only on bone vertices since muscle ground truth may not be available
    """
    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 7):
        print("Please input the command as the following: [ProgramName] [StartFeatSelStratIndex] [EndFeatSelStratIndex] [StartValidIndex] [EndValidIndex] [StartNumComps] [EndNumComps]");
        return;
    startFeatSelStratIndex = int(sys.argv[1]); endFeatSelStratIndex = int(sys.argv[2]);
    startValidIndex = int(sys.argv[3]); endValidIndex = int(sys.argv[4]);
    startNumComps = int(sys.argv[5]); endNumComps = int(sys.argv[6]);    
    personalizedPelvicStructureFolder = disk + r"/Data/PelvisBoneRecon/FemalePelvisGeometries/PersonalizedPelvisStructures";
    templateDataFolder = disk + r"/Data/PelvisBoneRecon/Template/PelvisBonesMuscles";
    crossValidationFolder = disk + r"/Data/PelvisBoneRecon/CrossValidation";
    trainValidTestIDFolder = crossValidationFolder + "/TrainingValidTestingIDs";
    featureSelectionProtocolFolder = crossValidationFolder + "/FeatureSelectionProtocol";
    outFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures";

    # Checking and creating output folder
    print("Checking and creating output folder ...");
    if (not os.path.exists(outFolder)): os.makedirs(outFolder);

    # Reading initial information
    print("Reading initial information ...");
    tempPelvisBoneMesh = sp.readMeshFromPLY(templateDataFolder + r"/TempPelvisBoneMesh.ply");
    tempPelvisBoneMuscleMesh = sp.readMeshFromPLY(templateDataFolder + r"/TempPelvisBoneMuscles.ply");
    pelvisBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempPelvisBoneMesh.vertices, tempPelvisBoneMuscleMesh.vertices);

    # Reading feature section strategies
    print("Reading feature selection strategies ...");
    allPelvicFeatures = sp.read3DPointsFromPPFile(featureSelectionProtocolFolder + "/AllFeaturePoints.pp");
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");    
    
    # Iterate for each feature selection strategy
    print("Iterate for each feature selection strategy ...");
    for featSelIndex in range(startFeatSelStratIndex, endFeatSelStratIndex + 1):
        # Debugging
        print("/********************************************** FEATURE SELECTION STRATEGY: ", featSelIndex);

        # Get the feature selection strategies
        print("\t Getting the feature selection strategy ...");
        featureSelectionIndices = list(featureSelectionStrategyDict.values())[featSelIndex];
        tempPelvicFeatures = allPelvicFeatures[featureSelectionIndices];
        pelvicFeatureBaryIndices, pelvicFeatureBaryCoords = sp.computeBarycentricLandmarks(tempPelvisBoneMuscleMesh, tempPelvicFeatures);

        # Conduct training validating throughout each time of cross-validation
        for validIndex in range(startValidIndex, endValidIndex + 1):
            # Debugging information
            print("/********************************************** VALIDATION: ", validIndex);

            # Reading training and validating subject IDs
            print("\t Reading training and validating subject IDs ...");
            trainIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/TrainingIDs_{validIndex}.txt");
            validIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/ValidationIDs_{validIndex}.txt");

            # Forming the training and validating data
            print("\t Forming the training and validating data ...");
            trainingPelvicVertexData = []; validPelvicVertexData = [];
            trainingPelvicFeatureData = []; validPelvicFeatureData = [];
            for i, ID in enumerate(trainIDs):
                pelvicStructure = sp.readMeshFromPLY(personalizedPelvicStructureFolder + f"/{ID}-PersonalizedPelvisBoneMuscleMesh.ply");
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvicStructure, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                trainingPelvicVertexData.append(pelvicStructure.vertices.flatten());
                trainingPelvicFeatureData.append(pelvicFeatures.flatten());
            for i, ID in enumerate(validIDs):
                pelvicStructure = sp.readMeshFromPLY(personalizedPelvicStructureFolder + f"/{ID}-PersonalizedPelvisBoneMuscleMesh.ply");
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvicStructure, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                validPelvicVertexData.append(pelvicStructure.vertices.flatten());
                validPelvicFeatureData.append(pelvicFeatures.flatten());

            # Begin to train for each number of components
            print("\t Begin to train in each number of components ...");
            for numComps in range(startNumComps, endNumComps + 1):
                print("/***************************** FEATURE SELECTION INDEX: ", featSelIndex, ", VALIDATION INDEX: ", validIndex, ", NUM COMPS: ", numComps);

                # Scale the pelvic feature and vertex data
                yScaler = StandardScaler().fit(trainingPelvicVertexData);
                xScaler = StandardScaler().fit(trainingPelvicFeatureData);
                scaledYData = yScaler.transform(trainingPelvicVertexData);
                scaledXData = xScaler.transform(trainingPelvicFeatureData);

                # Control the number of components
                targetNumComps = numComps;
                xDims = scaledXData.shape[1];
                yDims = scaledYData.shape[1];
                xNumComps = min(xDims, targetNumComps);
                yNumComps = min(yDims, targetNumComps);

                # Parameterize the scaled pelvic vertex and feature data
                xSSM = PCA(n_components=xNumComps);
                xSSM.fit(scaledXData);
                trainXParamData = xSSM.transform(scaledXData);
                ySSM = PCA(n_components=yNumComps);
                ySSM.fit(scaledYData);
                trainYParamData = ySSM.transform(scaledYData);

                # Linear regression from feature parameters to vertex parameters
                regressor = LinearRegression()
                regressor.fit(trainXParamData, trainYParamData);

                # Computing errors on the validating data
                ## Scale and parameterize the validation data
                scaledValidXData = xScaler.transform(validPelvicFeatureData);
                validXParams = xSSM.transform(scaledValidXData);
                ## Try to predict the pelvic vertex params from the pelvic feature params
                predYParams = regressor.predict(validXParams);
                ## Inverse transform to the scaled Y data
                predScaledYData = ySSM.inverse_transform(predYParams);
                predYData = yScaler.inverse_transform(predScaledYData);
                ## Computing validating errors
                avgP2PDists = [];
                for v, predY in enumerate(predYData):
                    # Getting the valiation data and predicted data
                    validY = validPelvicVertexData[v];
                    validPelvicStructureVertices = validY.reshape(-1, 3);
                    predPelvicStructureVertices = predY.reshape(-1, 3);

                    # Compute accuracy only on the bone vertices because we do not have ground truth of the muscle
                    validPelvisBoneVertices = validPelvicStructureVertices[pelvisBoneVertexIndices];
                    predPelvisBoneVertices = predPelvicStructureVertices[pelvisBoneVertexIndices];

                    # Compute points to points distances
                    avgP2PDist = sp.computeAveragePointsToPointsDistance(validPelvisBoneVertices, predPelvisBoneVertices);
                    avgP2PDists.append(avgP2PDist);
                avgP2PDists = np.array(avgP2PDists);

                # Save the computed errors
                sp.saveMatrixToCSVFile(outFolder + f"/AveragePoint2PointDistances_{featSelIndex}_{validIndex}_{numComps}.csv", avgP2PDists);

    # Finished processing
    print("Finished processing.");
def featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_RidgeLinearRegression():
    """
    Performs feature-to-pelvis structure reconstruction using ridge linear regression on bone-muscle structures.
    
    This function evaluates the accuracy of pelvis bone reconstruction by learning statistical relationships 
    between feature points and bone-muscle shape parameters using PCA dimensionality reduction and ridge 
    linear regression with L2 regularization. The evaluation is performed across multiple feature selection 
    strategies, cross-validation folds, and varying numbers of principal components.
    
    Key Steps:
    1. Parse command line arguments for processing range (feature strategies, validation folds, components)
    2. Initialize system database and load template data
    3. Load feature selection strategies 
    4. For each feature selection strategy and validation fold:
       - Load training and validation subject IDs
       - Extract pelvis bone-muscle vertices and feature points for each subject
       - Scale and normalize both feature and vertex data using StandardScaler
       - Apply PCA dimensionality reduction to both feature and vertex spaces
       - Train ridge linear regression model with L2 regularization to map feature parameters to vertex parameters
       - For each validation subject:
         * Transform features to PCA parameter space
         * Predict vertex parameters using trained ridge regression model
         * Reconstruct pelvis bone-muscle mesh from predicted parameters
         * Compute point-to-point distance errors (bone vertices only)
       - Save validation errors to CSV files
    
    Shape Relation Strategy:
    - PCA Feature Space: Reduces feature dimensionality while preserving variance
    - PCA Vertex Space: Creates compact representation of shape variations
    - Ridge Linear Regression: Learns mapping with L2 regularization to prevent overfitting
    - Cross-validation: Ensures robust evaluation across different data splits
    
    Output:
    - CSV files containing average point-to-point distances for each feature strategy, 
      validation fold, and number of components configuration
    
    Note: Ridge regression adds L2 penalty to prevent overfitting compared to standard linear regression.
    Accuracy is computed only on bone vertices since muscle ground truth may not be available.
    """
    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 7):
        print("Please input the command as the following: [ProgramName] [StartFeatSelStratIndex] [EndFeatSelStratIndex] [StartValidIndex] [EndValidIndex] [StartNumComps] [EndNumComps]");
        return;
    startFeatSelStratIndex = int(sys.argv[1]); endFeatSelStratIndex = int(sys.argv[2]);
    startValidIndex = int(sys.argv[3]); endValidIndex = int(sys.argv[4]);
    startNumComps = int(sys.argv[5]); endNumComps = int(sys.argv[6]);    
    personalizedPelvicStructureFolder = disk + r"/Data/PelvisBoneRecon/FemalePelvisGeometries/PersonalizedPelvisStructures";
    templateDataFolder = disk + r"/Data/PelvisBoneRecon/Template/PelvisBonesMuscles";
    crossValidationFolder = disk + r"/Data/PelvisBoneRecon/CrossValidation";
    trainValidTestIDFolder = crossValidationFolder + "/TrainingValidTestingIDs";
    featureSelectionProtocolFolder = crossValidationFolder + "/FeatureSelectionProtocol";
    outFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures_RidgeLinearRegression";

    # Checking and creating output folder
    print("Checking and creating output folder ...");
    if (not os.path.exists(outFolder)): os.makedirs(outFolder);

    # Reading initial information
    print("Reading initial information ...");
    tempPelvisBoneMesh = sp.readMeshFromPLY(templateDataFolder + "/TempPelvisBoneMesh.ply");
    tempPelvisBoneMuscleMesh = sp.readMeshFromPLY(templateDataFolder + "/TempPelvisBoneMuscles.ply");
    pelvisBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempPelvisBoneMesh.vertices, tempPelvisBoneMuscleMesh.vertices);

    # Reading feature section strategies
    print("Reading feature selection strategies ...");
    allPelvicFeatures = sp.read3DPointsFromPPFile(featureSelectionProtocolFolder + "/AllFeaturePoints.pp");
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");    
    
    # Iterate for each feature selection strategy
    print("Iterate for each feature selection strategy ...");
    for featSelIndex in range(startFeatSelStratIndex, endFeatSelStratIndex + 1):
        # Debugging
        print("/********************************************** FEATURE SELECTION STRATEGY: ", featSelIndex);

        # Get the feature selection strategies
        print("\t Getting the feature selection strategy ...");
        featureSelectionIndices = list(featureSelectionStrategyDict.values())[featSelIndex];
        tempPelvicFeatures = allPelvicFeatures[featureSelectionIndices];
        pelvicFeatureBaryIndices, pelvicFeatureBaryCoords = sp.computeBarycentricLandmarks(tempPelvisBoneMuscleMesh, tempPelvicFeatures);

        # Conduct training validating throughout each time of cross-validation
        for validIndex in range(startValidIndex, endValidIndex + 1):
            # Debugging information
            print("/********************************************** VALIDATION: ", validIndex);

            # Reading training and validating subject IDs
            print("\t Reading training and validating subject IDs ...");
            trainIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/TrainingIDs_{validIndex}.txt");
            validIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/ValidationIDs_{validIndex}.txt");

            # Forming the training and validating data
            print("\t Forming the training and validating data ...");
            trainingPelvicVertexData = []; validPelvicVertexData = [];
            trainingPelvicFeatureData = []; validPelvicFeatureData = [];
            for i, ID in enumerate(trainIDs):
                pelvicStructure = sp.readMeshFromPLY(personalizedPelvicStructureFolder + f"/{ID}-PersonalizedPelvisBoneMuscleMesh.ply");
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvicStructure, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                trainingPelvicVertexData.append(pelvicStructure.vertices.flatten());
                trainingPelvicFeatureData.append(pelvicFeatures.flatten());
            for i, ID in enumerate(validIDs):
                pelvicStructure = sp.readMeshFromPLY(personalizedPelvicStructureFolder + f"/{ID}-PersonalizedPelvisBoneMuscleMesh.ply");
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvicStructure, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                validPelvicVertexData.append(pelvicStructure.vertices.flatten());
                validPelvicFeatureData.append(pelvicFeatures.flatten());

            # Begin to train for each number of components
            print("\t Begin to train in each number of components ...");
            for numComps in range(startNumComps, endNumComps + 1):
                print("/***************************** FEATURE SELECTION INDEX: ", featSelIndex, ", VALIDATION INDEX: ", validIndex, ", NUM COMPS: ", numComps);

                # Scale the pelvic feature and vertex data
                yScaler = StandardScaler().fit(trainingPelvicVertexData);
                xScaler = StandardScaler().fit(trainingPelvicFeatureData);
                scaledYData = yScaler.transform(trainingPelvicVertexData);
                scaledXData = xScaler.transform(trainingPelvicFeatureData);

                # Control the number of components
                targetNumComps = numComps;
                xDims = scaledXData.shape[1];
                yDims = scaledYData.shape[1];
                xNumComps = min(xDims, targetNumComps);
                yNumComps = min(yDims, targetNumComps);

                # Parameterize the scaled pelvic vertex and feature data
                xSSM = PCA(n_components=xNumComps);
                xSSM.fit(scaledXData);
                trainXParamData = xSSM.transform(scaledXData);
                ySSM = PCA(n_components=yNumComps);
                ySSM.fit(scaledYData);
                trainYParamData = ySSM.transform(scaledYData);

                # Linear regression from feature parameters to vertex parameters
                regressor = Ridge(alpha=1.0);
                regressor.fit(trainXParamData, trainYParamData);

                # Computing errors on the validating data
                ## Scale and parameterize the validation data
                scaledValidXData = xScaler.transform(validPelvicFeatureData);
                validXParams = xSSM.transform(scaledValidXData);
                ## Try to predict the pelvic vertex params from the pelvic feature params
                predYParams = regressor.predict(validXParams);                
                ## Inverse transform to the scaled Y data
                predScaledYData = ySSM.inverse_transform(predYParams);
                predYData = yScaler.inverse_transform(predScaledYData);
                ## Computing validating errors
                avgP2PDists = [];
                for v, predY in enumerate(predYData):
                    # Getting the valiation data and predicted data
                    validY = validPelvicVertexData[v];
                    validPelvicStructureVertices = validY.reshape(-1, 3);
                    predPelvicStructureVertices = predY.reshape(-1, 3);

                    # Compute accuracy only on the bone vertices because we do not have ground truth of the muscle
                    validPelvisBoneVertices = validPelvicStructureVertices[pelvisBoneVertexIndices];
                    predPelvisBoneVertices = predPelvicStructureVertices[pelvisBoneVertexIndices];

                    # Compute points to points distances
                    avgP2PDist = sp.computeAveragePointsToPointsDistance(validPelvisBoneVertices, predPelvisBoneVertices);
                    avgP2PDists.append(avgP2PDist);
                avgP2PDists = np.array(avgP2PDists);

                # Save the computed errors
                sp.saveMatrixToCSVFile(outFolder + f"/AveragePoint2PointDistances_{featSelIndex}_{validIndex}_{numComps}.csv", avgP2PDists);

    # Finished processing
    print("Finished processing.");
def featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_CanonicalCorrelationAnalysis():
    """
    Performs feature-to-pelvis structure reconstruction using canonical correlation analysis on bone-muscle structures.
    
    This function evaluates the accuracy of pelvis bone reconstruction by learning statistical relationships 
    between feature points and bone-muscle shape parameters using PCA dimensionality reduction and canonical 
    correlation analysis (CCA). The evaluation is performed across multiple feature selection strategies, 
    cross-validation folds, and varying numbers of principal components.
    
    Key Steps:
    1. Parse command line arguments for processing range (feature strategies, validation folds, components)
    2. Initialize system database and load template data
    3. Load feature selection strategies 
    4. For each feature selection strategy and validation fold:
       - Load training and validation subject IDs
       - Extract pelvis bone-muscle vertices and feature points for each subject
       - Scale and normalize both feature and vertex data using StandardScaler
       - Apply PCA dimensionality reduction to both feature and vertex spaces
       - Train canonical correlation analysis model to find correlated components
       - Train linear regression on canonical components to map feature to vertex spaces
       - For each validation subject:
         * Transform features to CCA canonical space
         * Predict vertex canonical components using trained regression model
         * Reconstruct pelvis bone-muscle mesh from predicted parameters
         * Compute point-to-point distance errors (bone vertices only)
       - Save validation errors to CSV files
    
    Shape Relation Strategy:
    - PCA Feature Space: Reduces feature dimensionality while preserving variance
    - PCA Vertex Space: Creates compact representation of shape variations
    - Canonical Correlation Analysis: Finds maximally correlated linear combinations between spaces
    - Linear Regression: Maps between canonical components of feature and vertex spaces
    - Cross-validation: Ensures robust evaluation across different data splits
    
    Output:
    - CSV files containing average point-to-point distances for each feature strategy, 
      validation fold, and number of components configuration
    
    Note: CCA finds linear combinations that maximize correlation between feature and vertex spaces.
    Accuracy is computed only on bone vertices since muscle ground truth may not be available.
    """
    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 7):
        print("Please input the command as the following: [ProgramName] [StartFeatSelStratIndex] [EndFeatSelStratIndex] [StartValidIndex] [EndValidIndex] [StartNumComps] [EndNumComps]");
        return;
    startFeatSelStratIndex = int(sys.argv[1]); endFeatSelStratIndex = int(sys.argv[2]);
    startValidIndex = int(sys.argv[3]); endValidIndex = int(sys.argv[4]);
    startNumComps = int(sys.argv[5]); endNumComps = int(sys.argv[6]);    
    personalizedPelvicStructureFolder = disk + r"/Data/PelvisBoneRecon/FemalePelvisGeometries/PersonalizedPelvisStructures";
    templateDataFolder = disk + r"/Data/PelvisBoneRecon/Template/PelvisBonesMuscles";
    crossValidationFolder = disk + r"/Data/PelvisBoneRecon/CrossValidation";
    trainValidTestIDFolder = crossValidationFolder + "/TrainingValidTestingIDs";
    featureSelectionProtocolFolder = crossValidationFolder + "/FeatureSelectionProtocol";
    outFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures_CanonicalCorrelationAnalysis";

    # Checking and creating output folder
    print("Checking and creating output folder ...");
    if (not os.path.exists(outFolder)): os.makedirs(outFolder);

    # Reading initial information
    print("Reading initial information ...");
    tempPelvisBoneMesh = sp.readMeshFromPLY(templateDataFolder + "/TempPelvisBoneMesh.ply");
    tempPelvisBoneMuscleMesh = sp.readMeshFromPLY(templateDataFolder + "/TempPelvisBoneMuscleMesh.ply");
    pelvisBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempPelvisBoneMesh.vertices, tempPelvisBoneMuscleMesh.vertices);

    # Reading feature section strategies
    print("Reading feature selection strategies ...");
    allPelvicFeatures = sp.read3DPointsFromPPFile(featureSelectionProtocolFolder + "/AllFeaturePoints.pp");
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");    
    
    # Iterate for each feature selection strategy
    print("Iterate for each feature selection strategy ...");
    for featSelIndex in range(startFeatSelStratIndex, endFeatSelStratIndex + 1):
        # Debugging
        print("/********************************************** FEATURE SELECTION STRATEGY: ", featSelIndex);

        # Get the feature selection strategies
        print("\t Getting the feature selection strategy ...");
        featureSelectionIndices = list(featureSelectionStrategyDict.values())[featSelIndex];
        tempPelvicFeatures = allPelvicFeatures[featureSelectionIndices];
        pelvicFeatureBaryIndices, pelvicFeatureBaryCoords = sp.computeBarycentricLandmarks(tempPelvisBoneMuscleMesh, tempPelvicFeatures);

        # Conduct training validating throughout each time of cross-validation
        for validIndex in range(startValidIndex, endValidIndex + 1):
            # Debugging information
            print("/********************************************** VALIDATION: ", validIndex);

            # Reading training and validating subject IDs
            print("\t Reading training and validating subject IDs ...");
            trainIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/TrainingIDs_{validIndex}.txt");
            validIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/ValidationIDs_{validIndex}.txt");

            # Forming the training and validating data
            print("\t Forming the training and validating data ...");
            trainingPelvicVertexData = []; validPelvicVertexData = [];
            trainingPelvicFeatureData = []; validPelvicFeatureData = [];
            for i, ID in enumerate(trainIDs):
                pelvicStructure = sp.readMeshFromPLY(personalizedPelvicStructureFolder + f"/{ID}-PersonalizedPelvisBoneMuscleMesh.ply");
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvicStructure, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                trainingPelvicVertexData.append(pelvicStructure.vertices.flatten());
                trainingPelvicFeatureData.append(pelvicFeatures.flatten());
            for i, ID in enumerate(validIDs):
                pelvicStructure = sp.readMeshFromPLY(personalizedPelvicStructureFolder + f"/{ID}-PersonalizedPelvisBoneMuscleMesh.ply");
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvicStructure, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                validPelvicVertexData.append(pelvicStructure.vertices.flatten());
                validPelvicFeatureData.append(pelvicFeatures.flatten());

            # Begin to train for each number of components
            print("\t Begin to train in each number of components ...");
            for numComps in range(startNumComps, endNumComps + 1):
                print("/***************************** FEATURE SELECTION INDEX: ", featSelIndex, ", VALIDATION INDEX: ", validIndex, ", NUM COMPS: ", numComps);

                # Scale the pelvic feature and vertex data
                yScaler = StandardScaler().fit(trainingPelvicVertexData);
                xScaler = StandardScaler().fit(trainingPelvicFeatureData);
                scaledYData = yScaler.transform(trainingPelvicVertexData);
                scaledXData = xScaler.transform(trainingPelvicFeatureData);

                # Control the number of components
                targetNumComps = numComps;
                xDims = scaledXData.shape[1];
                yDims = scaledYData.shape[1];
                xNumComps = min(xDims, targetNumComps);
                yNumComps = min(yDims, targetNumComps);
                targetNumComps = min(xNumComps, yNumComps);

                # Parameterize the scaled pelvic vertex and feature data
                xSSM = PCA(n_components=targetNumComps);
                xSSM.fit(scaledXData);
                trainXParamData = xSSM.transform(scaledXData);
                ySSM = PCA(n_components=targetNumComps);
                ySSM.fit(scaledYData);
                trainYParamData = ySSM.transform(scaledYData);

                # Linear regression from feature parameters to vertex parameters
                cca = CCA(n_components=targetNumComps);
                cca.fit(trainXParamData, trainYParamData);
                X_c, Y_c = cca.transform(trainXParamData, trainYParamData);
                reg = LinearRegression();
                reg.fit(X_c, Y_c);

                # Computing errors on the validating data
                ## Scale and parameterize the validation data
                scaledValidXData = xScaler.transform(validPelvicFeatureData);
                validXParams = xSSM.transform(scaledValidXData);
                ## Try to predict the pelvic vertex params from the pelvic feature params
                X_valid_c = cca.transform(validXParams);
                Y_pred_c = reg.predict(X_valid_c);
                predYParams = cca.inverse_transform(Y_pred_c);
                ## Inverse transform to the scaled Y data
                predScaledYData = ySSM.inverse_transform(predYParams);                
                predYData = yScaler.inverse_transform(predScaledYData);
                ## Computing validating errors
                avgP2PDists = [];
                for v, predY in enumerate(predYData):
                    # Getting the valiation data and predicted data
                    validY = validPelvicVertexData[v];
                    validPelvicStructureVertices = validY.reshape(-1, 3);
                    predPelvicStructureVertices = predY.reshape(-1, 3);

                    # Compute accuracy only on the bone vertices because we do not have ground truth of the muscle
                    validPelvisBoneVertices = validPelvicStructureVertices[pelvisBoneVertexIndices];
                    predPelvisBoneVertices = predPelvicStructureVertices[pelvisBoneVertexIndices];

                    # Compute points to points distances
                    avgP2PDist = sp.computeAveragePointsToPointsDistance(validPelvisBoneVertices, predPelvisBoneVertices);
                    avgP2PDists.append(avgP2PDist);
                avgP2PDists = np.array(avgP2PDists);

                # Save the computed errors
                sp.saveMatrixToCSVFile(outFolder + f"/AveragePoint2PointDistances_{featSelIndex}_{validIndex}_{numComps}.csv", avgP2PDists);

    # Finished processing
    print("Finished processing.");
def featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_PartialLeastSquaresRegression():
    """
    Performs feature-to-pelvis structure reconstruction using partial least squares regression on bone-muscle structures.
    
    This function evaluates the accuracy of pelvis bone reconstruction by learning statistical relationships 
    between feature points and bone-muscle shape parameters using PCA dimensionality reduction and partial 
    least squares (PLS) regression. The evaluation is performed across multiple feature selection strategies, 
    cross-validation folds, and varying numbers of principal components.
    
    Key Steps:
    1. Parse command line arguments for processing range (feature strategies, validation folds, components)
    2. Initialize system database and load template data
    3. Load feature selection strategies 
    4. For each feature selection strategy and validation fold:
       - Load training and validation subject IDs
       - Extract pelvis bone-muscle vertices and feature points for each subject
       - Scale and normalize both feature and vertex data using StandardScaler
       - Apply PCA dimensionality reduction to both feature and vertex spaces
       - Train partial least squares regression model to map feature parameters to vertex parameters
       - For each validation subject:
         * Transform features to PCA parameter space
         * Predict vertex parameters using trained PLS regression model
         * Reconstruct pelvis bone-muscle mesh from predicted parameters
         * Compute point-to-point distance errors (bone vertices only)
       - Save validation errors to CSV files
    
    Shape Relation Strategy:
    - PCA Feature Space: Reduces feature dimensionality while preserving variance
    - PCA Vertex Space: Creates compact representation of shape variations
    - Partial Least Squares Regression: Finds linear combinations that maximize covariance between spaces
    - Cross-validation: Ensures robust evaluation across different data splits
    
    Output:
    - CSV files containing average point-to-point distances for each feature strategy, 
      validation fold, and number of components configuration
    
    Note: PLS regression maximizes covariance between feature and vertex parameter spaces.
    Accuracy is computed only on bone vertices since muscle ground truth may not be available.
    """
    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 7):
        print("Please input the command as the following: [ProgramName] [StartFeatSelStratIndex] [EndFeatSelStratIndex] [StartValidIndex] [EndValidIndex] [StartNumComps] [EndNumComps]");
        return;
    startFeatSelStratIndex = int(sys.argv[1]); endFeatSelStratIndex = int(sys.argv[2]);
    startValidIndex = int(sys.argv[3]); endValidIndex = int(sys.argv[4]);
    startNumComps = int(sys.argv[5]); endNumComps = int(sys.argv[6]);    
    crossValidationFolder = disk + r"/Data/PelvisBoneRecon/CrossValidation";
    trainValidTestIDFolder = crossValidationFolder + "/TrainingValidTestingIDs";
    featureSelectionProtocolFolder = crossValidationFolder + "/FeatureSelectionProtocol";
    personalizedPelvicStructureFolder = disk + r"/Data/PelvisBoneRecon/FemalePelvisGeometries/PersonalizedPelvisStructures";
    templateDataFolder = disk + r"/Data/PelvisBoneRecon/Template/PelvisBonesMuscles";
    outFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures_PartialLeastSquaresRegression";

    # Checking and creating output folder
    print("Checking and creating output folder ...");
    if (not os.path.exists(outFolder)): os.makedirs(outFolder);

    # Reading initial information
    print("Reading initial information ...");
    tempPelvisBoneMesh = sp.readMeshFromPLY(templateDataFolder + "/TempPelvisBoneMesh.ply");
    tempPelvisBoneMuscleMesh = sp.readMeshFromPLY(templateDataFolder + "/TempPelvisBoneMuscles.ply");
    pelvisBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempPelvisBoneMesh.vertices, tempPelvisBoneMuscleMesh.vertices);

    # Reading feature section strategies
    print("Reading feature selection strategies ...");
    allPelvicFeatures = sp.read3DPointsFromPPFile(featureSelectionProtocolFolder + "/AllFeaturePoints.pp");
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");    
    
    # Iterate for each feature selection strategy
    print("Iterate for each feature selection strategy ...");
    for featSelIndex in range(startFeatSelStratIndex, endFeatSelStratIndex + 1):
        # Debugging
        print("/********************************************** FEATURE SELECTION STRATEGY: ", featSelIndex);

        # Get the feature selection strategies
        print("\t Getting the feature selection strategy ...");
        featureSelectionIndices = list(featureSelectionStrategyDict.values())[featSelIndex];
        tempPelvicFeatures = allPelvicFeatures[featureSelectionIndices];
        pelvicFeatureBaryIndices, pelvicFeatureBaryCoords = sp.computeBarycentricLandmarks(tempPelvisBoneMuscleMesh, tempPelvicFeatures);

        # Conduct training validating throughout each time of cross-validation
        for validIndex in range(startValidIndex, endValidIndex + 1):
            # Debugging information
            print("/********************************************** VALIDATION: ", validIndex);

            # Reading training and validating subject IDs
            print("\t Reading training and validating subject IDs ...");
            trainIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/TrainingIDs_{validIndex}.txt");
            validIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/ValidationIDs_{validIndex}.txt");

            # Forming the training and validating data
            print("\t Forming the training and validating data ...");
            trainingPelvicVertexData = []; validPelvicVertexData = [];
            trainingPelvicFeatureData = []; validPelvicFeatureData = [];
            for i, ID in enumerate(trainIDs):
                pelvicStructure = sp.readMeshFromPLY(personalizedPelvicStructureFolder + f"/{ID}-PersonalizedPelvisBoneMuscleMesh.ply");
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvicStructure, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                trainingPelvicVertexData.append(pelvicStructure.vertices.flatten());
                trainingPelvicFeatureData.append(pelvicFeatures.flatten());
            for i, ID in enumerate(validIDs):
                pelvicStructure = sp.readMeshFromPLY(personalizedPelvicStructureFolder + f"/{ID}-PersonalizedPelvisBoneMuscleMesh.ply");
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvicStructure, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                validPelvicVertexData.append(pelvicStructure.vertices.flatten());
                validPelvicFeatureData.append(pelvicFeatures.flatten());

            # Begin to train for each number of components
            print("\t Begin to train in each number of components ...");
            for numComps in range(startNumComps, endNumComps + 1):
                print("/***************************** FEATURE SELECTION INDEX: ", featSelIndex, ", VALIDATION INDEX: ", validIndex, ", NUM COMPS: ", numComps);

                # Scale the pelvic feature and vertex data
                yScaler = StandardScaler().fit(trainingPelvicVertexData);
                xScaler = StandardScaler().fit(trainingPelvicFeatureData);
                scaledYData = yScaler.transform(trainingPelvicVertexData);
                scaledXData = xScaler.transform(trainingPelvicFeatureData);

                # Control the number of components
                targetNumComps = numComps;
                xDims = scaledXData.shape[1];
                yDims = scaledYData.shape[1];
                xNumComps = min(xDims, targetNumComps);
                yNumComps = min(yDims, targetNumComps);
                targetNumComps = min(xNumComps, yNumComps);

                # Parameterize the scaled pelvic vertex and feature data
                xSSM = PCA(n_components=targetNumComps);
                xSSM.fit(scaledXData);
                trainXParamData = xSSM.transform(scaledXData);
                ySSM = PCA(n_components=targetNumComps);
                ySSM.fit(scaledYData);
                trainYParamData = ySSM.transform(scaledYData);

                # Linear regression from feature parameters to vertex parameters
                pls = PLSRegression(n_components=targetNumComps);
                pls.fit(trainXParamData, trainYParamData);

                # Computing errors on the validating data
                ## Scale and parameterize the validation data
                scaledValidXData = xScaler.transform(validPelvicFeatureData);
                validXParams = xSSM.transform(scaledValidXData);
                ## Try to predict the pelvic vertex params from the pelvic feature params
                predYParams = pls.predict(validXParams);
                ## Inverse transform to the scaled Y data
                predScaledYData = ySSM.inverse_transform(predYParams);                
                predYData = yScaler.inverse_transform(predScaledYData);
                ## Computing validating errors
                avgP2PDists = [];
                for v, predY in enumerate(predYData):
                    # Getting the valiation data and predicted data
                    validY = validPelvicVertexData[v];
                    validPelvicStructureVertices = validY.reshape(-1, 3);
                    predPelvicStructureVertices = predY.reshape(-1, 3);

                    # Compute accuracy only on the bone vertices because we do not have ground truth of the muscle
                    validPelvisBoneVertices = validPelvicStructureVertices[pelvisBoneVertexIndices];
                    predPelvisBoneVertices = predPelvicStructureVertices[pelvisBoneVertexIndices];

                    # Compute points to points distances
                    avgP2PDist = sp.computeAveragePointsToPointsDistance(validPelvisBoneVertices, predPelvisBoneVertices);
                    avgP2PDists.append(avgP2PDist);
                avgP2PDists = np.array(avgP2PDists);

                # Save the computed errors
                sp.saveMatrixToCSVFile(outFolder + f"/AveragePoint2PointDistances_{featSelIndex}_{validIndex}_{numComps}.csv", avgP2PDists);

    # Finished processing
    print("Finished processing.");
def featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_GaussianProcessRegressor():
    """
    Performs feature-to-pelvis structure reconstruction using Gaussian process regression on bone-muscle structures.
    
    This function evaluates the accuracy of pelvis bone reconstruction by learning statistical relationships 
    between feature points and bone-muscle shape parameters using PCA dimensionality reduction and Gaussian 
    process regression with RBF kernel. The evaluation is performed across multiple feature selection strategies, 
    cross-validation folds, and varying numbers of principal components.
    
    Key Steps:
    1. Parse command line arguments for processing range (feature strategies, validation folds, components)
    2. Initialize system database and load template data
    3. Load feature selection strategies 
    4. For each feature selection strategy and validation fold:
       - Load training and validation subject IDs
       - Extract pelvis bone-muscle vertices and feature points for each subject
       - Scale and normalize both feature and vertex data using StandardScaler
       - Apply PCA dimensionality reduction to both feature and vertex spaces
       - Train Gaussian process regressor with RBF kernel to map feature parameters to vertex parameters
       - For each validation subject:
         * Transform features to PCA parameter space
         * Predict vertex parameters using trained Gaussian process model
         * Reconstruct pelvis bone-muscle mesh from predicted parameters
         * Compute point-to-point distance errors (bone vertices only)
       - Save validation errors to CSV files
    
    Shape Relation Strategy:
    - PCA Feature Space: Reduces feature dimensionality while preserving variance
    - PCA Vertex Space: Creates compact representation of shape variations
    - Gaussian Process Regression: Non-parametric Bayesian approach with RBF kernel
    - Cross-validation: Ensures robust evaluation across different data splits
    
    Output:
    - CSV files containing average point-to-point distances for each feature strategy, 
      validation fold, and number of components configuration
    
    Note: Gaussian processes provide uncertainty quantification and handle non-linear relationships.
    Accuracy is computed only on bone vertices since muscle ground truth may not be available.
    """
    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 7):
        print("Please input the command as the following: [ProgramName] [StartFeatSelStratIndex] [EndFeatSelStratIndex] [StartValidIndex] [EndValidIndex] [StartNumComps] [EndNumComps]");
        return;
    startFeatSelStratIndex = int(sys.argv[1]); endFeatSelStratIndex = int(sys.argv[2]);
    startValidIndex = int(sys.argv[3]); endValidIndex = int(sys.argv[4]);
    startNumComps = int(sys.argv[5]); endNumComps = int(sys.argv[6]);    
    personalizedPelvicStructureFolder = disk + r"/Data/PelvisBoneRecon/FemalePelvisGeometries/PersonalizedPelvisStructures";
    templateDataFolder = disk + r"/Data/PelvisBoneRecon/Template/PelvisBonesMuscles";
    crossValidationFolder = disk + r"/Data/PelvisBoneRecon/CrossValidation";
    trainValidTestIDFolder = crossValidationFolder + "/TrainingValidTestingIDs";
    featureSelectionProtocolFolder = crossValidationFolder + "/FeatureSelectionProtocol";
    outFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures_GaussianProcessRegressor";

    # Checking and creating output folder
    print("Checking and creating output folder ...");
    if (not os.path.exists(outFolder)): os.makedirs(outFolder);

    # Reading initial information
    print("Reading initial information ...");
    tempPelvisBoneMesh = sp.readMeshFromPLY(templateDataFolder + "/TempPelvisBoneMesh.ply");
    tempPelvisBoneMuscleMesh = sp.readMeshFromPLY(templateDataFolder + "/TempPelvisBoneMuscleMesh.ply");
    pelvisBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempPelvisBoneMesh.vertices, tempPelvisBoneMuscleMesh.vertices);

    # Reading feature section strategies
    print("Reading feature selection strategies ...");
    allPelvicFeatures = sp.read3DPointsFromPPFile(featureSelectionProtocolFolder + "/AllFeaturePoints.pp");
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");    
    
    # Iterate for each feature selection strategy
    print("Iterate for each feature selection strategy ...");
    for featSelIndex in range(startFeatSelStratIndex, endFeatSelStratIndex + 1):
        # Debugging
        print("/********************************************** FEATURE SELECTION STRATEGY: ", featSelIndex);

        # Get the feature selection strategies
        print("\t Getting the feature selection strategy ...");
        featureSelectionIndices = list(featureSelectionStrategyDict.values())[featSelIndex];
        tempPelvicFeatures = allPelvicFeatures[featureSelectionIndices];
        pelvicFeatureBaryIndices, pelvicFeatureBaryCoords = sp.computeBarycentricLandmarks(tempPelvisBoneMuscleMesh, tempPelvicFeatures);

        # Conduct training validating throughout each time of cross-validation
        for validIndex in range(startValidIndex, endValidIndex + 1):
            # Debugging information
            print("/********************************************** VALIDATION: ", validIndex);

            # Reading training and validating subject IDs
            print("\t Reading training and validating subject IDs ...");
            trainIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/TrainingIDs_{validIndex}.txt");
            validIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/ValidationIDs_{validIndex}.txt");

            # Forming the training and validating data
            print("\t Forming the training and validating data ...");
            trainingPelvicVertexData = []; validPelvicVertexData = [];
            trainingPelvicFeatureData = []; validPelvicFeatureData = [];
            for i, ID in enumerate(trainIDs):
                pelvicStructure = sp.readMeshFromPLY(personalizedPelvicStructureFolder + f"/{ID}-PersonalizedPelvisBoneMuscleMesh.ply");
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvicStructure, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                trainingPelvicVertexData.append(pelvicStructure.vertices.flatten());
                trainingPelvicFeatureData.append(pelvicFeatures.flatten());
            for i, ID in enumerate(validIDs):
                pelvicStructure = sp.readMeshFromPLY(personalizedPelvicStructureFolder + f"/{ID}-PersonalizedPelvisBoneMuscleMesh.ply");
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvicStructure, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                validPelvicVertexData.append(pelvicStructure.vertices.flatten());
                validPelvicFeatureData.append(pelvicFeatures.flatten());

            # Begin to train for each number of components
            print("\t Begin to train in each number of components ...");
            for numComps in range(startNumComps, endNumComps + 1):
                print("/***************************** FEATURE SELECTION INDEX: ", featSelIndex, ", VALIDATION INDEX: ", validIndex, ", NUM COMPS: ", numComps);

                # Scale the pelvic feature and vertex data
                yScaler = StandardScaler().fit(trainingPelvicVertexData);
                xScaler = StandardScaler().fit(trainingPelvicFeatureData);
                scaledYData = yScaler.transform(trainingPelvicVertexData);
                scaledXData = xScaler.transform(trainingPelvicFeatureData);

                # Control the number of components
                targetNumComps = numComps;
                xDims = scaledXData.shape[1];
                yDims = scaledYData.shape[1];
                xNumComps = min(xDims, targetNumComps);
                yNumComps = min(yDims, targetNumComps);

                # Parameterize the scaled pelvic vertex and feature data
                xSSM = PCA(n_components=xNumComps);
                xSSM.fit(scaledXData);
                trainXParamData = xSSM.transform(scaledXData);
                ySSM = PCA(n_components=yNumComps);
                ySSM.fit(scaledYData);
                trainYParamData = ySSM.transform(scaledYData);

                # Linear regression from feature parameters to vertex parameters
                gpr = GaussianProcessRegressor(kernel=RBF())
                gpr.fit(trainXParamData, trainYParamData);

                # Computing errors on the validating data
                ## Scale and parameterize the validation data
                scaledValidXData = xScaler.transform(validPelvicFeatureData);
                validXParams = xSSM.transform(scaledValidXData);
                ## Try to predict the pelvic vertex params from the pelvic feature params
                predYParams = gpr.predict(validXParams);
                ## Inverse transform to the scaled Y data
                predYParams = predYParams.reshape(-1, yNumComps);
                predScaledYData = ySSM.inverse_transform(predYParams);
                predYData = yScaler.inverse_transform(predScaledYData);
                ## Computing validating errors
                avgP2PDists = [];
                for v, predY in enumerate(predYData):
                    # Getting the valiation data and predicted data
                    validY = validPelvicVertexData[v];
                    validPelvicStructureVertices = validY.reshape(-1, 3);
                    predPelvicStructureVertices = predY.reshape(-1, 3);

                    # Compute accuracy only on the bone vertices because we do not have ground truth of the muscle
                    validPelvisBoneVertices = validPelvicStructureVertices[pelvisBoneVertexIndices];
                    predPelvisBoneVertices = predPelvicStructureVertices[pelvisBoneVertexIndices];

                    # Compute points to points distances
                    avgP2PDist = sp.computeAveragePointsToPointsDistance(validPelvisBoneVertices, predPelvisBoneVertices);
                    avgP2PDists.append(avgP2PDist);
                avgP2PDists = np.array(avgP2PDists);

                # Save the computed errors
                sp.saveMatrixToCSVFile(outFolder + f"/AveragePoint2PointDistances_{featSelIndex}_{validIndex}_{numComps}.csv", avgP2PDists);

    # Finished processing
    print("Finished processing.");
def featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor():
    """
    Performs feature-to-pelvis structure reconstruction using multi-output ridge regression on bone-muscle structures.
    
    This function evaluates the accuracy of pelvis bone reconstruction by learning statistical relationships 
    between feature points and bone-muscle shape parameters using PCA dimensionality reduction and multi-output 
    ridge regression with cross-validation. The evaluation is performed across multiple feature selection 
    strategies, cross-validation folds, and varying numbers of principal components.
    
    Key Steps:
    1. Parse command line arguments for processing range (feature strategies, validation folds, components)
    2. Initialize system database and load template data
    3. Load feature selection strategies 
    4. For each feature selection strategy and validation fold:
       - Load training and validation subject IDs
       - Extract pelvis bone-muscle vertices and feature points for each subject
       - Scale and normalize both feature and vertex data using StandardScaler
       - Apply PCA dimensionality reduction to both feature and vertex spaces
       - Train multi-output ridge regression model with cross-validation to map feature parameters to vertex parameters
       - For each validation subject:
         * Transform features to PCA parameter space
         * Predict vertex parameters using trained regression model
         * Reconstruct pelvis bone-muscle mesh from predicted parameters
         * Compute point-to-point distance errors (bone vertices only)
       - Save validation errors to CSV files
    
    Shape Relation Strategy:
    - PCA Feature Space: Reduces feature dimensionality while preserving variance
    - PCA Vertex Space: Creates compact representation of shape variations
    - Multi-Output Ridge Regression: Uses RidgeCV for automatic hyperparameter tuning across multiple outputs
    - Cross-validation: Ensures robust evaluation across different data splits
    
    Output:
    - CSV files containing average point-to-point distances for each feature strategy, 
      validation fold, and number of components configuration
    
    Note: Multi-output regression handles multiple target variables simultaneously with built-in cross-validation.
    Accuracy is computed only on bone vertices since muscle ground truth may not be available.
    """
    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 7):
        print("Please input the command as the following: [ProgramName] [StartFeatSelStratIndex] [EndFeatSelStratIndex] [StartValidIndex] [EndValidIndex] [StartNumComps] [EndNumComps]");
        return;
    startFeatSelStratIndex = int(sys.argv[1]); endFeatSelStratIndex = int(sys.argv[2]);
    startValidIndex = int(sys.argv[3]); endValidIndex = int(sys.argv[4]);
    startNumComps = int(sys.argv[5]); endNumComps = int(sys.argv[6]);    
    templateDataFolder = disk + r"/Data/PelvisBoneRecon/Template/PelvisBonesMuscles";
    personalizedPelvicStructureFolder = disk + r"/Data/PelvisBoneRecon/FemalePelvisGeometries/PersonalizedPelvisStructures";
    crossValidationFolder = disk + r"/Data/PelvisBoneRecon/CrossValidation";
    trainValidTestIDFolder = crossValidationFolder + "/TrainingValidTestingIDs";
    featureSelectionProtocolFolder = crossValidationFolder + "/FeatureSelectionProtocol";
    outFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures_MultiOutputRegressor";

    # Checking and creating output folder
    print("Checking and creating output folder ...");
    if (not os.path.exists(outFolder)): os.makedirs(outFolder);

    # Reading initial information
    print("Reading initial information ...");
    tempPelvisBoneMesh = sp.readMeshFromPLY(templateDataFolder + "/TempPelvisBoneMesh.ply");
    tempPelvisBoneMuscleMesh = sp.readMeshFromPLY(templateDataFolder + "/TempPelvisBoneMuscles.ply");
    pelvisBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempPelvisBoneMesh.vertices, tempPelvisBoneMuscleMesh.vertices);

    # Reading feature section strategies
    print("Reading feature selection strategies ...");
    allPelvicFeatures = sp.read3DPointsFromPPFile(featureSelectionProtocolFolder + "/AllFeaturePoints.pp");
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");    
    
    # Iterate for each feature selection strategy
    print("Iterate for each feature selection strategy ...");
    for featSelIndex in range(startFeatSelStratIndex, endFeatSelStratIndex + 1):
        # Debugging
        print("/********************************************** FEATURE SELECTION STRATEGY: ", featSelIndex);

        # Get the feature selection strategies
        print("\t Getting the feature selection strategy ...");
        featureSelectionIndices = list(featureSelectionStrategyDict.values())[featSelIndex];
        tempPelvicFeatures = allPelvicFeatures[featureSelectionIndices];
        pelvicFeatureBaryIndices, pelvicFeatureBaryCoords = sp.computeBarycentricLandmarks(tempPelvisBoneMuscleMesh, tempPelvicFeatures);

        # Conduct training validating throughout each time of cross-validation
        for validIndex in range(startValidIndex, endValidIndex + 1):
            # Debugging information
            print("/********************************************** VALIDATION: ", validIndex);

            # Reading training and validating subject IDs
            print("\t Reading training and validating subject IDs ...");
            trainIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/TrainingIDs_{validIndex}.txt");
            validIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/ValidationIDs_{validIndex}.txt");

            # Forming the training and validating data
            print("\t Forming the training and validating data ...");
            trainingPelvicVertexData = []; validPelvicVertexData = [];
            trainingPelvicFeatureData = []; validPelvicFeatureData = [];
            for i, ID in enumerate(trainIDs):
                pelvicStructure = sp.readMeshFromPLY(personalizedPelvicStructureFolder + f"/{ID}-PersonalizedPelvisBoneMuscleMesh.ply");
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvicStructure, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                trainingPelvicVertexData.append(pelvicStructure.vertices.flatten());
                trainingPelvicFeatureData.append(pelvicFeatures.flatten());
            for i, ID in enumerate(validIDs):
                pelvicStructure = sp.readMeshFromPLY(personalizedPelvicStructureFolder + f"/{ID}-PersonalizedPelvisBoneMuscleMesh.ply");
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvicStructure, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                validPelvicVertexData.append(pelvicStructure.vertices.flatten());
                validPelvicFeatureData.append(pelvicFeatures.flatten());

            # Begin to train for each number of components
            print("\t Begin to train in each number of components ...");
            for numComps in range(startNumComps, endNumComps + 1):
                print("/***************************** FEATURE SELECTION INDEX: ", featSelIndex, ", VALIDATION INDEX: ", validIndex, ", NUM COMPS: ", numComps);

                # Scale the pelvic feature and vertex data
                yScaler = StandardScaler().fit(trainingPelvicVertexData);
                xScaler = StandardScaler().fit(trainingPelvicFeatureData);
                scaledYData = yScaler.transform(trainingPelvicVertexData);
                scaledXData = xScaler.transform(trainingPelvicFeatureData);

                # Control the number of components
                targetNumComps = numComps;
                xDims = scaledXData.shape[1];
                yDims = scaledYData.shape[1];
                xNumComps = min(xDims, targetNumComps);
                yNumComps = min(yDims, targetNumComps);

                # Parameterize the scaled pelvic vertex and feature data
                xSSM = PCA(n_components=xNumComps);
                xSSM.fit(scaledXData);
                trainXParamData = xSSM.transform(scaledXData);
                ySSM = PCA(n_components=yNumComps);
                ySSM.fit(scaledYData);
                trainYParamData = ySSM.transform(scaledYData);

                # Linear regression from feature parameters to vertex parameters
                model = MultiOutputRegressor(RidgeCV())
                model.fit(trainXParamData, trainYParamData)

                # Computing errors on the validating data
                ## Scale and parameterize the validation data
                scaledValidXData = xScaler.transform(validPelvicFeatureData);
                validXParams = xSSM.transform(scaledValidXData);
                ## Try to predict the pelvic vertex params from the pelvic feature params
                predYParams = model.predict(validXParams);
                ## Inverse transform to the scaled Y data
                predYParams = predYParams.reshape(-1, yNumComps);
                predScaledYData = ySSM.inverse_transform(predYParams);
                predYData = yScaler.inverse_transform(predScaledYData);
                ## Computing validating errors
                avgP2PDists = [];
                for v, predY in enumerate(predYData):
                    # Getting the valiation data and predicted data
                    validY = validPelvicVertexData[v];
                    validPelvicStructureVertices = validY.reshape(-1, 3);
                    predPelvicStructureVertices = predY.reshape(-1, 3);

                    # Compute accuracy only on the bone vertices because we do not have ground truth of the muscle
                    validPelvisBoneVertices = validPelvicStructureVertices[pelvisBoneVertexIndices];
                    predPelvisBoneVertices = predPelvicStructureVertices[pelvisBoneVertexIndices];

                    # Compute points to points distances
                    avgP2PDist = sp.computeAveragePointsToPointsDistance(validPelvisBoneVertices, predPelvisBoneVertices);
                    avgP2PDists.append(avgP2PDist);
                avgP2PDists = np.array(avgP2PDists);

                # Save the computed errors
                sp.saveMatrixToCSVFile(outFolder + f"/AveragePoint2PointDistances_{featSelIndex}_{validIndex}_{numComps}.csv", avgP2PDists);

    # Finished processing
    print("Finished processing.");
def featureToPelvisStructureRecon_selectOptimalReconstructionStrategy():
    """
    Analyzes and compares the performance of different feature-to-pelvis reconstruction strategies to identify the optimal approach.
    
    This function evaluates multiple reconstruction methodologies by loading validation/testing error data 
    and generating comparative visualizations to determine the best performing strategy for pelvis structure 
    reconstruction from anatomical feature points.
    
    Key Analysis Components:
    1. Affine Transform-based Non-rigid Deformation
    2. Radial Basis Function Interpolation-based Non-rigid Deformation  
    3. Statistical Shape Optimization
    4. Statistical Shape Relation with various regression models
    5. Performance comparison across different feature selection strategies
    
    Reconstruction Strategies Evaluated:
    - Affine Transform: Uses rigid SVD + affine CPD transformations
    - RBF Interpolation: Adds radial basis function deformation for non-rigid alignment
    - Shape Optimization: PCA-based statistical shape model with least squares optimization
    - Shape Relation: PCA + regression mapping between feature and shape parameter spaces
    
    Regression Models Analyzed:
    - Multivariate Linear Regression
    - Ridge Linear Regression
    - Canonical Correlation Analysis
    - Partial Least Squares Regression
    - Gaussian Process Regression
    - Multi-output Ridge Regression
    
    Visualization Outputs:
    - Box plots showing error distributions for each method
    - Line plots showing optimal number of components vs validation errors
    - Bar charts comparing mean errors across different strategies
    - Statistical trend analysis with polynomial fitting
    
    Output:
    - Comparative performance charts identifying the optimal reconstruction strategy
    - Error trend analysis across different numbers of feature points
    - Statistical summaries for method selection guidance
    
    Note: This function serves as a comprehensive evaluation tool to guide the selection
    of the most effective feature-to-pelvis reconstruction methodology based on empirical
    validation results across multiple cross-validation folds.
    """

    # Initializing
    print("Initializing ...");    
    crossValidationFolder = disk + r"/Data/PelvisBoneRecon/CrossValidation";
    featureSelIndex = 10;
    optimNumCompBoneMuscleOptim = 18;

    # Define error drawing function
    def affineTransformNonRigidDeformation():
        """
        Analyzes and visualizes the performance of affine transform-based non-rigid deformation strategy.
        
        This function evaluates the reconstruction accuracy of pelvis bone-muscle structures using 
        affine transformation methods by loading validation error data and generating statistical 
        visualizations to assess the effectiveness of this approach.
        
        Key Steps:
        1. Load validation error data from affine transform bone-muscle structure results
        2. Aggregate mean validation errors across all cross-validation folds
        3. Convert error measurements to millimeters for presentation
        4. Generate box plot visualization showing error distribution
        5. Annotate plot with mean error values and statistical information
        
        Methodology Analysis:
        - Affine Transform Strategy: Uses rigid SVD transformation followed by affine CPD transformation
        - Validation Framework: Processes errors across multiple cross-validation folds (0-9)
        - Error Metrics: Point-to-point distance measurements between predicted and ground truth
        - Feature Selection: Uses feature selection strategy index 10 for consistent evaluation
        
        Visualization Output:
        - Box plot showing error distribution for pelvic structure reconstruction
        - Mean error annotation with precise measurements
        - Professional formatting with bold fonts and clear labeling
        - Title indicating the specific deformation strategy being analyzed
        
        Statistical Summary:
        - Computes mean validation errors across all folds
        - Converts measurements from meters to millimeters for clinical relevance
        - Provides visual assessment of method performance and variability
        
        Note: This analysis focuses specifically on bone-muscle structure reconstruction
        using affine transformation as the primary deformation strategy, providing
        baseline performance metrics for comparison with other reconstruction approaches.
        """
        # Initializing
        print("\t Initializing ...");
        validErrorFolder = crossValidationFolder + "/AffineDeformation/BoneMuscleStructures";

        # Reading validation errors for bone and muscle structure
        print("\t Reading validating errors for bone and muscle structures ...");
        boneMuscleStructureValidErrorData = [];    
        minValidIndex = 0; maxValidIndex = 9;
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            validErrors = sp.readMatrixFromCSVFile(validErrorFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}.csv");
            meanValidError = np.mean(validErrors);
            boneMuscleStructureValidErrorData.append(meanValidError);
        boneMuscleStructureValidErrorData = np.array(boneMuscleStructureValidErrorData);
        
        print("\t Drawing box plots for errors distribution ...")
        # Convert errors to millimeters
        drawingData = boneMuscleStructureValidErrorData * 1000
        # Create box plot
        plt.boxplot([drawingData],
                    labels=["Pelvic Structure"],
                    medianprops=dict(color="red"))
        # Set title with bold and larger font
        plt.title("Affine Transform-based\nNon-rigid Deformation",
                fontsize=16, fontweight='bold')
        # Set y-axis label
        plt.ylabel("Validation Errors (mm)", fontsize=14, fontweight='bold')
        # Set x-axis label
        plt.xticks(fontsize=12, fontweight='bold')
        # Set y-ticks (index numbers) font size and weight
        plt.yticks(fontsize=12, fontweight='bold')
        # No grid
        plt.grid(False)
        # Add mean values
        mean2 = np.mean(drawingData)
        # Annotate means (adjust position as needed)
        plt.text(1, mean2, f"{mean2:.3f}", ha='center', va='bottom', fontsize=12, fontweight='bold', color='blue')
        plt.show()
    def radialBasisFunctionInterpolation():
        """
        Analyzes and visualizes the performance of radial basis function interpolation-based non-rigid deformation strategy.
        
        This function evaluates the reconstruction accuracy of pelvis bone-muscle structures using 
        radial basis function interpolation methods by loading validation error data and generating statistical 
        visualizations to assess the effectiveness of this approach.
        
        Key Steps:
        1. Load validation error data from radial basis function bone-muscle structure results
        2. Aggregate mean validation errors across all cross-validation folds
        3. Convert error measurements to millimeters for presentation
        4. Generate box plot visualization showing error distribution
        5. Annotate plot with mean error values and statistical information
        
        Methodology Analysis:
        - RBF Interpolation Strategy: Uses rigid SVD + affine CPD + radial basis function transformations
        - Validation Framework: Processes errors across multiple cross-validation folds (0-9)
        - Error Metrics: Point-to-point distance measurements between predicted and ground truth
        - Feature Selection: Uses feature selection strategy index 10 for consistent evaluation
        
        Visualization Output:
        - Box plot showing error distribution for pelvic structure reconstruction
        - Mean error annotation with precise measurements
        - Professional formatting with bold fonts and clear labeling
        - Title indicating the specific deformation strategy being analyzed
        
        Statistical Summary:
        - Computes mean validation errors across all folds
        - Converts measurements from meters to millimeters for clinical relevance
        - Provides visual assessment of method performance and variability
        
        Note: This analysis focuses specifically on bone-muscle structure reconstruction
        using radial basis function interpolation as the primary deformation strategy, providing
        enhanced non-rigid deformation capabilities compared to affine transformation alone.
        """
            
        # Initializing
        print("\t Initializing ...");
        boneStructureValidErrorFolder = crossValidationFolder + "/RadialBasicFunctionStrategy/BoneStructures";
        boneMuscleStructureValidErrorFolder = crossValidationFolder + "/RadialBasicFunctionStrategy/BoneMuscleStructures";

        # Reading validation errors for bone and muscle structure
        print("\t Reading validating errors for bone and muscle structures ...");
        validErrorData = [];
        minValidIndex = 0; maxValidIndex = 9;
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            validErrors = sp.readMatrixFromCSVFile(boneMuscleStructureValidErrorFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}.csv");
            meanValidError = np.mean(validErrors);
            validErrorData.append(meanValidError);
        validErrorData = np.array(validErrorData);
        
        print("\t Drawing box plots for errors distribution ...")
        # Convert errors to millimeters
        drawingData = validErrorData * 1000
        # Create box plot
        plt.boxplot([drawingData],
                    labels=["Pelvic Structure"],
                    medianprops=dict(color="red"))
        # Set title with bold and larger font
        plt.title("Radial Basis Function Interpolation-based\nNon-rigid Deformation",
                fontsize=16, fontweight='bold')
        # Set y-axis label
        plt.ylabel("Validation Errors (mm)", fontsize=14, fontweight='bold')
        # Set x-axis label
        plt.xticks(fontsize=12, fontweight='bold')
        # Set y-ticks (index numbers) font size and weight
        plt.yticks(fontsize=12, fontweight='bold')
        # No grid
        plt.grid(False)
        # Add mean values
        mean2 = np.mean(drawingData)
        # Annotate means (adjust position as needed)
        plt.text(1, mean2, f"{mean2:.3f}", ha='center', va='bottom', fontsize=12, fontweight='bold', color='blue')
        plt.show()
    def statisticalShapeOptimization():
        """
        Analyzes and visualizes the performance of statistical shape optimization strategy for pelvis reconstruction.
        
        This function evaluates the reconstruction accuracy of pelvis bone-muscle structures using 
        statistical shape optimization methods by loading validation error data across varying numbers 
        of principal components and generating performance visualizations.
        
        Key Steps:
        1. Load validation error data from shape optimization bone-muscle structure results
        2. Aggregate validation errors across all cross-validation folds (0-9)
        3. Process errors for varying numbers of components (1-80)
        4. Convert error measurements to millimeters for clinical presentation
        5. Generate line plot visualization showing component count vs validation errors
        6. Identify and annotate optimal number of components with minimum error
        
        Methodology Analysis:
        - Shape Optimization Strategy: Uses PCA-based statistical shape models with least squares optimization
        - Validation Framework: Processes errors across multiple cross-validation folds
        - Component Analysis: Evaluates performance across 1-80 principal components
        - Error Metrics: Point-to-point distance measurements between predicted and ground truth
        - Feature Selection: Uses feature selection strategy index 10 for consistent evaluation
        
        Visualization Output:
        - Line plot showing validation errors vs number of components
        - Minimum error point identification and annotation
        - Professional formatting with bold fonts and clear axis labeling
        - Title indicating the specific optimization strategy being analyzed
        
        Statistical Summary:
        - Computes mean validation errors across all folds for each component count
        - Converts measurements from meters to millimeters for clinical relevance
        - Identifies optimal number of components for minimum reconstruction error
        - Provides visual assessment of method performance and parameter sensitivity
        
        Note: This analysis focuses on statistical shape optimization using PCA-based models
        to determine the optimal balance between model complexity and reconstruction accuracy.
        """
        # Initializing
        print("\t Initializing ...");
        validErrorFolder = crossValidationFolder + "/ShapeOptimizationStrategy/BoneMuscleStructures";
    
        print("\t Reading valiating errors for bone and muscle structures ...");
        maxNumComps = 80; minNumComps = 1;
        minValidIndex = 0; maxValidIndex = 9;
        validErrorData = np.zeros((10, maxNumComps));         
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            for numComps in range(minNumComps, maxNumComps + 1):
                validErrors = sp.readMatrixFromCSVFile(validErrorFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}_{numComps}.csv");
                validError = validErrors.mean();
                validErrorData[validIndex,  numComps - 1] = validError;
        validErrorData = np.array(validErrorData);
        
        # Drawing the graphs
        print("\t Drawing the graph ...");
        drawingData = np.mean(validErrorData, axis=0) * 1000
        N = validErrorData.shape[1]
        x = np.arange(1, N + 1)
        y = drawingData

        # Find minimum points
        minIndex = np.argmin(y)
        minX = x[minIndex]
        minY = y[minIndex]

        # Plotting the curves
        line2, = plt.plot(x, y, label=f'Pelvic Structure (min: {minX}, {minY:.2f} mm)')

        # Axis labels and title
        plt.xlabel('Numbers of Components', fontdict={'fontsize': 14, 'fontweight': 'bold'})
        plt.ylabel('Validation Errors (mm)', fontdict={'fontsize': 14, 'fontweight': 'bold'})
        plt.title('Statistical Shape Optimization', fontdict={'fontsize': 16, 'fontweight': 'bold'})
        plt.xticks(fontsize=12, fontweight='bold', fontname='Arial')
        plt.yticks(fontsize=12, fontweight='bold', fontname='Arial')

        # Mark and annotate minimum points
        plt.scatter(minX, minY, color=line2.get_color(), zorder=3)

        # Add legend in top-left
        plt.legend(loc='upper left', fontsize=12, frameon=False)

        plt.show()
    def statisticalShapeRelation():
        """
        Analyzes and visualizes the performance of statistical shape relation strategy for pelvis reconstruction.
        
        This function evaluates the reconstruction accuracy of pelvis bone-muscle structures using 
        statistical shape relation methods by loading validation error data across varying numbers 
        of principal components and generating performance visualizations with trend analysis.
        
        Key Steps:
        1. Load validation error data from shape relation bone-muscle structure results
        2. Aggregate validation errors across all cross-validation folds (0-9)
        3. Process errors for varying numbers of components (1-80)
        4. Convert error measurements to millimeters for clinical presentation
        5. Generate line plot visualization showing component count vs validation errors
        6. Identify and annotate optimal number of components with minimum error
        
        Methodology Analysis:
        - Shape Relation Strategy: Uses PCA + regression mapping between feature and shape parameter spaces
        - Validation Framework: Processes errors across multiple cross-validation folds
        - Component Analysis: Evaluates performance across 1-80 principal components
        - Error Metrics: Point-to-point distance measurements between predicted and ground truth
        - Feature Selection: Uses feature selection strategy index 10 for consistent evaluation
        
        Visualization Output:
        - Line plot showing validation errors vs number of components
        - Minimum error point identification and annotation
        - Professional formatting with bold fonts and clear axis labeling
        - Title indicating the specific relation strategy being analyzed
        
        Statistical Summary:
        - Computes mean validation errors across all folds for each component count
        - Converts measurements from meters to millimeters for clinical relevance
        - Identifies optimal number of components for minimum reconstruction error
        - Provides visual assessment of method performance and parameter sensitivity
        
        Note: This analysis focuses on statistical shape relation using regression-based models
        to determine the optimal balance between feature dimensionality and reconstruction accuracy.
        """
        # Initializing
        print("\t Initializing ...");
        validErrorFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures";
    
        print("\t Reading valiating errors for bone and muscle structures ...");
        maxNumComps = 80; minNumComps = 1;
        minValidIndex = 0; maxValidIndex = 9;
        validErrorData = np.zeros((maxValidIndex + 1, maxNumComps));
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            for numComps in range(minNumComps, maxNumComps + 1):
                validErrors = sp.readMatrixFromCSVFile(validErrorFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}_{numComps}.csv");
                validError = validErrors.mean();
                validErrorData[validIndex,  numComps - 1] = validError;
        validErrorData = np.array(validErrorData);

        # Drawing the graphs
        print("\t Drawing the graph ...");
        drawingData2 = np.mean(validErrorData, axis=0) * 1000
        N = validErrorData.shape[1]
        x = np.arange(1, N + 1)
        y = drawingData2

        # Find minimum points
        minIndex = np.argmin(y)
        minX = x[minIndex]
        minY = y[minIndex]

        # Plotting the curves
        line2, = plt.plot(x, y, label=f'Pelvic Structure (min: {minX}, {minY:.2f} mm)')

        # Axis labels and title
        plt.xlabel('Numbers of Components', fontdict={'fontsize': 14, 'fontweight': 'bold'})
        plt.ylabel('Validation Errors (mm)', fontdict={'fontsize': 14, 'fontweight': 'bold'})
        plt.title('Statistical Shape Relation', fontdict={'fontsize': 16, 'fontweight': 'bold'})
        plt.xticks(fontsize=12, fontweight='bold', fontname='Arial')
        plt.yticks(fontsize=12, fontweight='bold', fontname='Arial')

        # Mark and annotate minimum points
        plt.scatter(minX, minY, color=line2.get_color(), zorder=3)

        # Add legend in top-left
        plt.legend(loc='upper left', fontsize=12, frameon=False)

        plt.show()
    def variousReconstructionStrategies():
        """
        Analyzes and compares performance across multiple pelvis reconstruction strategies.
        
        This function evaluates the reconstruction accuracy of pelvis bone-muscle structures by 
        loading validation error data from different reconstruction methodologies and generating 
        comparative bar chart visualizations to assess the effectiveness of each approach.
        
        Key Steps:
        1. Load validation error data from four different reconstruction strategies
        2. Aggregate mean validation errors across all cross-validation folds (0-9)
        3. Process errors for optimal component configurations for each strategy
        4. Convert error measurements to millimeters for clinical presentation
        5. Generate comparative bar chart showing mean errors with standard deviations
        6. Annotate bars with precise error values for quantitative comparison
        
        Reconstruction Strategies Compared:
        - Affine Transform: Uses rigid SVD + affine CPD transformations
        - Radial Basis Function: Adds RBF interpolation for enhanced non-rigid deformation
        - Statistical Shape Optimization: PCA-based shape models with least squares optimization
        - Statistical Shape Relation: PCA + regression mapping between feature and shape spaces
        
        Methodology Analysis:
        - Validation Framework: Processes errors across multiple cross-validation folds
        - Optimal Components: Uses predetermined optimal number of components for fair comparison
        - Error Metrics: Point-to-point distance measurements between predicted and ground truth
        - Feature Selection: Uses feature selection strategy index 10 for consistent evaluation
        
        Visualization Output:
        - Bar chart showing mean validation errors with error bars (standard deviations)
        - Precise error value annotations on each bar for quantitative assessment
        - Professional formatting with bold fonts and clear axis labeling
        - Title indicating comprehensive strategy comparison
        
        Statistical Summary:
        - Computes mean and standard deviation validation errors across all folds
        - Converts measurements from meters to millimeters for clinical relevance
        - Provides visual ranking of reconstruction strategies by performance
        - Enables identification of the most effective reconstruction approach
        
        Note: This comprehensive comparison helps determine the optimal reconstruction strategy
        by evaluating performance across multiple validation folds using consistent evaluation metrics.
        """
        # Initializing
        print("\t Initializing ...");
        affineTransformValidFolder = crossValidationFolder + "/AffineDeformation/BoneMuscleStructures";
        rbfBasedValidFolder = crossValidationFolder + "/RadialBasicFunctionStrategy/BoneMuscleStructures";
        shapeOptimBasedValidFolder = crossValidationFolder + "/ShapeOptimizationStrategy/BoneMuscleStructures";
        shapeRelationBasedValidFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures";
        maxNumComps = 80; minNumComps = 1; minValidIndex = 0; maxValidIndex = 9;
        shapeOptimBasedOptimNumComps = 21; shapeRelationBasedOptimComps = 38;
    
        # Reading errors
        print("\t Reading errors ...");
        print("\t\t For affine base ...");
        affineTransformBasedValidErrors = [];
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            validErrors = sp.readMatrixFromCSVFile(affineTransformValidFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}.csv");
            meanValidError = np.mean(validErrors);
            affineTransformBasedValidErrors.append(meanValidError);
        affineTransformBasedValidErrors = np.array(affineTransformBasedValidErrors);

        print("\t\t For radial basis function based ...");
        radialBasisFunctionBasedValidErrors = [];
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            validErrors = sp.readMatrixFromCSVFile(rbfBasedValidFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}.csv");
            meanValidError = np.mean(validErrors);
            radialBasisFunctionBasedValidErrors.append(meanValidError);
        radialBasisFunctionBasedValidErrors = np.array(radialBasisFunctionBasedValidErrors);

        print("\t\t For shape optimization based ...");
        shapeOptimizationBasedValidErrors = np.zeros((maxValidIndex + 1, maxNumComps));
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            for numComps in range(minNumComps, maxNumComps + 1):
                validErrors = sp.readMatrixFromCSVFile(shapeOptimBasedValidFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}_{numComps}.csv");
                validError = validErrors.mean();
                shapeOptimizationBasedValidErrors[validIndex,  numComps - 1] = validError;
        shapeOptimizationBasedValidErrors = np.array(shapeOptimizationBasedValidErrors);
        optimalValidErrors_shapeOptim = shapeOptimizationBasedValidErrors[:, shapeOptimBasedOptimNumComps - 1];
    
        print("\t\t For shape relation based ...");
        shapeRelationBasedValidErrors = np.zeros((maxValidIndex + 1, maxNumComps));
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            for numComps in range(minNumComps, maxNumComps + 1):
                validErrors = sp.readMatrixFromCSVFile(shapeRelationBasedValidFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}_{numComps}.csv");
                validError = validErrors.mean();
                shapeRelationBasedValidErrors[validIndex,  numComps - 1] = validError;
        shapeRelationBasedValidErrors = np.array(shapeRelationBasedValidErrors);
        optimalValidErrors_shapeRelation = shapeRelationBasedValidErrors[:, shapeRelationBasedOptimComps - 1];
    
        print("\t Drawing charts ...");
        ## Organize data
        groups = [
            ("Affine Transform", affineTransformBasedValidErrors*1000),
            ("Radial Basic \nFunction", radialBasisFunctionBasedValidErrors*1000),
            ("Statistical Shape \nOptimization", optimalValidErrors_shapeOptim*1000),
            ("Statistical Shape \nRelation", optimalValidErrors_shapeRelation*1000)
        ]
        ## Computing means and stds
        means = [[np.mean(data)] for _, data in groups]
        stds = [[np.std(data)] for _, data in groups]
        ## Plotting
        labels = [g[0] for g in groups]
        x = np.arange(len(labels))  # label locations
        width = 0.35  # width of the bars
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(x, [m[0] for m in means], width, yerr=[s[0] for s in stds], capsize=5)
        ## Add text labels for the means and decoration
        def add_labels(bars, values):
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points", ha='center', va='bottom', fontsize=14, fontweight='bold')
        add_labels(bars, [m[0] for m in means])
        ax.set_ylabel('Validation Errors (mm)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Feature-to-Structure Reconstruction Strategies', fontsize=14, fontweight='bold')
        ax.set_title('Comparison of Error Metrics Across Methods', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, ha='center', fontsize=12, fontweight='bold')
        for label in ax.get_yticklabels():
            label.set_fontsize(14)
            label.set_fontweight('bold')
        boldFont = FontProperties(weight='bold', size=14)
        ax.legend(prop=boldFont)
        plt.tight_layout()
        plt.show()
    def boneMuscleStructureWithMultiLinearRegression():
        """
        Analyzes and visualizes the performance of multivariate linear regression on bone-muscle structures.
        
        This function evaluates the reconstruction accuracy of pelvis bone-muscle structures using 
        multivariate linear regression by loading validation error data across varying numbers 
        of principal components and generating performance visualizations with trend analysis.
        
        Key Steps:
        1. Load validation error data from multivariate linear regression bone-muscle structure results
        2. Aggregate validation errors across all cross-validation folds (0-9)
        3. Process errors for varying numbers of components (1-80)
        4. Convert error measurements to millimeters for clinical presentation
        5. Generate line plot visualization showing component count vs validation errors
        6. Identify and annotate optimal number of components with minimum error
        
        Methodology Analysis:
        - Shape Relation Strategy: Uses PCA + multivariate linear regression mapping between feature and shape parameter spaces
        - Validation Framework: Processes errors across multiple cross-validation folds
        - Component Analysis: Evaluates performance across 1-80 principal components
        - Error Metrics: Point-to-point distance measurements between predicted and ground truth
        - Feature Selection: Uses feature selection strategy index 10 for consistent evaluation
        
        Visualization Output:
        - Line plot showing validation errors vs number of components
        - Minimum error point identification and annotation with coordinates
        - Professional formatting with bold fonts and clear axis labeling
        - Title indicating the specific regression method being analyzed
        
        Statistical Summary:
        - Computes mean validation errors across all folds for each component count
        - Converts measurements from meters to millimeters for clinical relevance
        - Identifies optimal number of components for minimum reconstruction error
        - Provides visual assessment of method performance and parameter sensitivity
        
        Note: This analysis focuses on multivariate linear regression as the mapping function
        between PCA-reduced feature and vertex parameter spaces for pelvis reconstruction.
        """
        # Initializing
        print("Initializing ...");
        multiLinearRegresionFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures";
        maxNumComps = 80; minNumComps = 1; minValidIndex = 0; maxValidIndex = 9;

        # Reading validating errors 
        print("Reading valiating errors ...");
        validationErrors_multiLinear = np.zeros((maxValidIndex + 1, maxNumComps));
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            for numComps in range(minNumComps, maxNumComps + 1):
                validErrors = sp.readMatrixFromCSVFile(multiLinearRegresionFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}_{numComps}.csv");
                validError = validErrors.mean();
                validationErrors_multiLinear[validIndex,  numComps - 1] = validError;
        validationErrors_multiLinear = np.array(validationErrors_multiLinear);

        # Draw graphs
        print("Drawing the graphs ...");
        ## Forming drawing data
        drawingData = np.mean(validationErrors_multiLinear, axis=0) * 1000;
        N = validationErrors_multiLinear.shape[1];
        x = np.arange(1, N + 1);
        y1 = drawingData;
        ## Find minimum points
        min_index = np.argmin(y1);
        min_x = x[min_index];
        min_y = y1[min_index];
        ## Plotting the curves
        line1, = plt.plot(x, y1, label=f'Bone Structure (min: {min_x}, {min_y:.2f} mm)');
        ## Axis labels and title
        plt.xlabel('Numbers of Components', fontdict={'fontsize': 14, 'fontweight': 'bold'});
        plt.ylabel('Validation Errors (mm)', fontdict={'fontsize': 14, 'fontweight': 'bold'});
        plt.title('Multivariate Linear Regression', fontdict={'fontsize': 16, 'fontweight': 'bold'});
        plt.xticks(fontsize=12, fontweight='bold', fontname='Arial');
        plt.yticks(fontsize=12, fontweight='bold', fontname='Arial');
        ## Mark and annotate minimum points
        plt.scatter(min_x, min_y, color=line1.get_color(), zorder=3);
        plt.annotate(f'({min_x}, {min_y:.2f})', xy=(min_x, min_y), xytext=(min_x + 0.05, min_y + 0.05),  fontsize=14, color='red', ha='center');
        plt.show();

        # Finished
        print("\t Finished.");
    def boneMuscleStructureWithRidgeLinearRegression():
        """
        Analyzes and visualizes the performance of ridge linear regression on bone-muscle structures.
        
        This function evaluates the reconstruction accuracy of pelvis bone-muscle structures using 
        ridge linear regression by loading validation error data across varying numbers 
        of principal components and generating performance visualizations with trend analysis.
        
        Key Steps:
        1. Load validation error data from ridge linear regression bone-muscle structure results
        2. Aggregate validation errors across all cross-validation folds (0-9)
        3. Process errors for varying numbers of components (1-80)
        4. Convert error measurements to millimeters for clinical presentation
        5. Generate line plot visualization showing component count vs validation errors
        6. Identify and annotate optimal number of components with minimum error
        
        Methodology Analysis:
        - Shape Relation Strategy: Uses PCA + ridge linear regression mapping between feature and shape parameter spaces
        - Validation Framework: Processes errors across multiple cross-validation folds
        - Component Analysis: Evaluates performance across 1-80 principal components
        - Error Metrics: Point-to-point distance measurements between predicted and ground truth
        - Feature Selection: Uses feature selection strategy index 10 for consistent evaluation
        
        Visualization Output:
        - Line plot showing validation errors vs number of components
        - Minimum error point identification and annotation with coordinates
        - Professional formatting with bold fonts and clear axis labeling
        - Title indicating the specific regression method being analyzed
        
        Statistical Summary:
        - Computes mean validation errors across all folds for each component count
        - Converts measurements from meters to millimeters for clinical relevance
        - Identifies optimal number of components for minimum reconstruction error
        - Provides visual assessment of method performance and parameter sensitivity
        
        Note: Ridge regression adds L2 regularization to prevent overfitting compared to standard linear regression.
        Accuracy is computed only on bone vertices since muscle ground truth may not be available.
        """
        # Initializing
        print("Initializing ...");
        ridgeLinearRegresionFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures_RidgeLinearRegression";
        maxNumComps = 80; minNumComps = 1; minValidIndex = 0; maxValidIndex = 9;

        # Reading validating errors 
        print("Reading valiating errors ...");
        validationErrors_ridgeLinearRegression = np.zeros((maxValidIndex + 1, maxNumComps));
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            for numComps in range(minNumComps, maxNumComps + 1):
                validErrors = sp.readMatrixFromCSVFile(ridgeLinearRegresionFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}_{numComps}.csv");
                validError = validErrors.mean();
                validationErrors_ridgeLinearRegression[validIndex,  numComps - 1] = validError;
        validationErrors_ridgeLinearRegression = np.array(validationErrors_ridgeLinearRegression);

        # Draw graphs
        print("Drawing the graphs ...");
        ## Forming drawing data
        drawingData = np.mean(validationErrors_ridgeLinearRegression, axis=0) * 1000;
        N = validationErrors_ridgeLinearRegression.shape[1];
        x = np.arange(1, N + 1);
        y1 = drawingData;
        ## Find minimum points
        min_index = np.argmin(y1);
        min_x = x[min_index];
        min_y = y1[min_index];
        ## Plotting the curves
        line1, = plt.plot(x, y1, label=f'Bone Structure (min: {min_x}, {min_y:.2f} mm)');
        ## Axis labels and title
        plt.xlabel('Numbers of Components', fontdict={'fontsize': 14, 'fontweight': 'bold'});
        plt.ylabel('Validation Errors (mm)', fontdict={'fontsize': 14, 'fontweight': 'bold'});
        plt.title('Ridge Linear Regression', fontdict={'fontsize': 16, 'fontweight': 'bold'});
        plt.xticks(fontsize=12, fontweight='bold', fontname='Arial');
        plt.yticks(fontsize=12, fontweight='bold', fontname='Arial');
        ## Mark and annotate minimum points
        plt.scatter(min_x, min_y, color=line1.get_color(), zorder=3);
        plt.annotate(f'({min_x}, {min_y:.2f})', xy=(min_x, min_y), xytext=(min_x + 0.05, min_y + 0.05),  fontsize=14, color='red', ha='center');
        plt.show();
    def boneMuscleStructureWithCanonicalCorrelationAnalysisRegression():
        """
        Analyzes and visualizes the performance of canonical correlation analysis regression on bone-muscle structures.
        
        This function evaluates the reconstruction accuracy of pelvis bone-muscle structures using 
        canonical correlation analysis (CCA) regression by loading validation error data across varying numbers 
        of principal components and generating performance visualizations with trend analysis.
        
        Key Steps:
        1. Load validation error data from canonical correlation analysis bone-muscle structure results
        2. Aggregate validation errors across all cross-validation folds (0-9)
        3. Process errors for varying numbers of components (1-80)
        4. Convert error measurements to millimeters for clinical presentation
        5. Generate line plot visualization showing component count vs validation errors
        6. Identify and annotate optimal number of components with minimum error
        
        Methodology Analysis:
        - Shape Relation Strategy: Uses PCA + canonical correlation analysis mapping between feature and shape parameter spaces
        - Validation Framework: Processes errors across multiple cross-validation folds
        - Component Analysis: Evaluates performance across 1-80 principal components
        - Error Metrics: Point-to-point distance measurements between predicted and ground truth
        - Feature Selection: Uses feature selection strategy index 10 for consistent evaluation
        
        Visualization Output:
        - Line plot showing validation errors vs number of components
        - Minimum error point identification and annotation with coordinates
        - Professional formatting with bold fonts and clear axis labeling
        - Title indicating the specific regression method being analyzed
        
        Statistical Summary:
        - Computes mean validation errors across all folds for each component count
        - Converts measurements from meters to millimeters for clinical relevance
        - Identifies optimal number of components for minimum reconstruction error
        - Provides visual assessment of method performance and parameter sensitivity
        
        Note: Canonical correlation analysis finds linear combinations that maximize correlation between 
        feature and vertex parameter spaces, providing enhanced modeling of inter-space relationships.
        Accuracy is computed only on bone vertices since muscle ground truth may not be available.
        """
        # Initializing
        print("Initializing ...");
        canonicalCorrelationRegresionFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures_CanonicalCorrelationAnalysis";
        maxNumComps = 80; minNumComps = 1; minValidIndex = 0; maxValidIndex = 9;
        
        # Reading validating errors 
        print("Reading valiating errors ...");
        validationErrors_canonicalCorrelationRegression = np.zeros((maxValidIndex + 1, maxNumComps));
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            for numComps in range(minNumComps, maxNumComps + 1):
                validErrors = sp.readMatrixFromCSVFile(canonicalCorrelationRegresionFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}_{numComps}.csv");
                validError = validErrors.mean();
                validationErrors_canonicalCorrelationRegression[validIndex,  numComps - 1] = validError;
        validationErrors_canonicalCorrelationRegression = np.array(validationErrors_canonicalCorrelationRegression);
        
        # Draw graphs
        print("Drawing the graphs ...");
        ## Forming drawing data
        drawingData = np.mean(validationErrors_canonicalCorrelationRegression, axis=0) * 1000;
        N = validationErrors_canonicalCorrelationRegression.shape[1];
        x = np.arange(1, N + 1); y = drawingData;
        ## Find minimum points
        min_index = np.argmin(y);
        min_x = x[min_index]; min_y = y[min_index];
        ## Plotting the curves
        line1, = plt.plot(x, y, label=f'Bone Structure (min: {min_x}, {min_y:.2f} mm)');
        ## Axis labels and title
        plt.xlabel('Numbers of Components', fontdict={'fontsize': 14, 'fontweight': 'bold'});
        plt.ylabel('Validation Errors (mm)', fontdict={'fontsize': 14, 'fontweight': 'bold'});
        plt.title('Canonical Correlation Analysis', fontdict={'fontsize': 16, 'fontweight': 'bold'});
        plt.xticks(fontsize=12, fontweight='bold', fontname='Arial');
        plt.yticks(fontsize=12, fontweight='bold', fontname='Arial');
        ## Mark and annotate minimum points
        plt.scatter(min_x, min_y, color=line1.get_color(), zorder=3);
        plt.annotate(f'({min_x}, {min_y:.2f})', xy=(min_x, min_y), xytext=(min_x + 0.05, min_y + 0.05),  fontsize=14, color='red', ha='right');
        print("minx = ", min_x, "miny = ", min_y);
        plt.show();

        # Finished
        print("\t Finished.");
    def boneMuscleStructureWithPartialLeastSquaresRegression():
        """
        Analyzes and visualizes the performance of partial least squares regression on bone-muscle structures.
        
        This function evaluates the reconstruction accuracy of pelvis bone-muscle structures using 
        partial least squares (PLS) regression by loading validation error data across varying numbers 
        of principal components and generating performance visualizations with trend analysis.
        
        Key Steps:
        1. Load validation error data from partial least squares regression bone-muscle structure results
        2. Aggregate validation errors across all cross-validation folds (0-9)
        3. Process errors for varying numbers of components (1-80)
        4. Convert error measurements to millimeters for clinical presentation
        5. Generate line plot visualization showing component count vs validation errors
        6. Identify and annotate optimal number of components with minimum error
        
        Methodology Analysis:
        - Shape Relation Strategy: Uses PCA + partial least squares regression mapping between feature and shape parameter spaces
        - Validation Framework: Processes errors across multiple cross-validation folds
        - Component Analysis: Evaluates performance across 1-80 principal components
        - Error Metrics: Point-to-point distance measurements between predicted and ground truth
        - Feature Selection: Uses feature selection strategy index 10 for consistent evaluation
        
        Visualization Output:
        - Line plot showing validation errors vs number of components
        - Minimum error point identification and annotation with coordinates
        - Professional formatting with bold fonts and clear axis labeling
        - Title indicating the specific regression method being analyzed
        
        Statistical Summary:
        - Computes mean validation errors across all folds for each component count
        - Converts measurements from meters to millimeters for clinical relevance
        - Identifies optimal number of components for minimum reconstruction error
        - Provides visual assessment of method performance and parameter sensitivity
        
        Note: Partial least squares regression maximizes covariance between feature and vertex parameter spaces.
        Accuracy is computed only on bone vertices since muscle ground truth may not be available.
        """

        # Initializing
        print("Initializing ...");
        partialLeastSquaresRegresionFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures_PartialLeastSquaresRegression";
        maxNumComps = 80; minNumComps = 1; minValidIndex = 0; maxValidIndex = 9;

        # Reading validating errors 
        print("Reading valiating errors ...");
        validationErrors_partialLeastSquaresRegression = np.zeros((maxValidIndex + 1, maxNumComps));
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            for numComps in range(minNumComps, maxNumComps + 1):
                validErrors = sp.readMatrixFromCSVFile(partialLeastSquaresRegresionFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}_{numComps}.csv");
                validError = validErrors.mean();
                validationErrors_partialLeastSquaresRegression[validIndex,  numComps - 1] = validError;
        validationErrors_partialLeastSquaresRegression = np.array(validationErrors_partialLeastSquaresRegression);
        optimalValidErrors_partialLeastSquaresRegression = validationErrors_partialLeastSquaresRegression[:, optimNumCompBoneMuscleOptim - 1];

        # Draw graphs
        print("Drawing the graphs ...");
        ## Forming drawing data
        drawingData = np.mean(validationErrors_partialLeastSquaresRegression, axis=0) * 1000;
        N = validationErrors_partialLeastSquaresRegression.shape[1];
        x = np.arange(1, N + 1);
        y1 = drawingData;
        ## Find minimum points
        min_index = np.argmin(y1);
        min_x = x[min_index];
        min_y = y1[min_index];
        ## Plotting the curves
        line1, = plt.plot(x, y1, label=f'Bone Structure (min: {min_x}, {min_y:.2f} mm)');
        ## Axis labels and title
        plt.xlabel('Numbers of Components', fontdict={'fontsize': 14, 'fontweight': 'bold'});
        plt.ylabel('Validation Errors (mm)', fontdict={'fontsize': 14, 'fontweight': 'bold'});
        plt.title('Partial Least Squares Regression', fontdict={'fontsize': 16, 'fontweight': 'bold'});
        plt.xticks(fontsize=12, fontweight='bold', fontname='Arial');
        plt.yticks(fontsize=12, fontweight='bold', fontname='Arial');
        ## Mark and annotate minimum points
        plt.scatter(min_x, min_y, color=line1.get_color(), zorder=3);
        plt.annotate(f'({min_x}, {min_y:.2f})', xy=(min_x, min_y), xytext=(min_x + 0.05, min_y + 0.05),  fontsize=14, color='red', ha='right');
        print("minx = ", min_x, "miny = ", min_y);
        plt.show();

        # Finished
        print("\t Finished.");
    def boneMuscleStructureWithGaussianProcessRegression():
        """
        Analyzes and visualizes the performance of Gaussian process regression on bone-muscle structures.
        
        This function evaluates the reconstruction accuracy of pelvis bone-muscle structures using 
        Gaussian process regression by loading validation error data across varying numbers 
        of principal components and generating performance visualizations with trend analysis.
        
        Key Steps:
        1. Load validation error data from Gaussian process regression bone-muscle structure results
        2. Aggregate validation errors across all cross-validation folds (0-9)
        3. Process errors for varying numbers of components (1-80)
        4. Convert error measurements to millimeters for clinical presentation
        5. Generate line plot visualization showing component count vs validation errors
        6. Identify and annotate optimal number of components with minimum error
        
        Methodology Analysis:
        - Shape Relation Strategy: Uses PCA + Gaussian process regression mapping between feature and shape parameter spaces
        - Validation Framework: Processes errors across multiple cross-validation folds
        - Component Analysis: Evaluates performance across 1-80 principal components
        - Error Metrics: Point-to-point distance measurements between predicted and ground truth
        - Feature Selection: Uses feature selection strategy index 10 for consistent evaluation
        
        Visualization Output:
        - Line plot showing validation errors vs number of components
        - Minimum error point identification and annotation with coordinates
        - Professional formatting with bold fonts and clear axis labeling
        - Title indicating the specific regression method being analyzed
        
        Statistical Summary:
        - Computes mean validation errors across all folds for each component count
        - Converts measurements from meters to millimeters for clinical relevance
        - Identifies optimal number of components for minimum reconstruction error
        - Provides visual assessment of method performance and parameter sensitivity
        
        Note: Gaussian process regression provides probabilistic predictions and can capture 
        non-linear relationships between feature and vertex parameter spaces with uncertainty quantification.
        Accuracy is computed only on bone vertices since muscle ground truth may not be available.
        """

        # Initializing
        print("Initializing ...");
        gaussianProcessRegresionFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures_GaussianProcessRegressor";
        maxNumComps = 80; minNumComps = 1; minValidIndex = 0; maxValidIndex = 9;

        # Reading validating errors 
        print("Reading valiating errors ...");
        validationErrors_gaussianProcessRegression = np.zeros((maxValidIndex + 1, maxNumComps));
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            for numComps in range(minNumComps, maxNumComps + 1):
                validErrors = sp.readMatrixFromCSVFile(gaussianProcessRegresionFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}_{numComps}.csv");
                validError = validErrors.mean();
                validationErrors_gaussianProcessRegression[validIndex,  numComps - 1] = validError;
        validationErrors_gaussianProcessRegression = np.array(validationErrors_gaussianProcessRegression);
        optimalValidErrors_gaussianProcessRegression = validationErrors_gaussianProcessRegression[:, optimNumCompBoneMuscleOptim - 1];

        # Draw graphs
        print("Drawing the graphs ...");
        ## Forming drawing data
        drawingData = np.mean(validationErrors_gaussianProcessRegression, axis=0) * 1000;
        N = validationErrors_gaussianProcessRegression.shape[1];
        x = np.arange(1, N + 1);
        y1 = drawingData;
        ## Find minimum points
        min_index = np.argmin(y1);
        min_x = x[min_index];
        min_y = y1[min_index];
        ## Plotting the curves
        line1, = plt.plot(x, y1, label=f'Bone Structure (min: {min_x}, {min_y:.2f} mm)');
        ## Axis labels and title
        plt.xlabel('Numbers of Components', fontdict={'fontsize': 14, 'fontweight': 'bold'});
        plt.ylabel('Validation Errors (mm)', fontdict={'fontsize': 14, 'fontweight': 'bold'});
        plt.title('Gaussian Process Regression', fontdict={'fontsize': 16, 'fontweight': 'bold'});
        plt.xticks(fontsize=12, fontweight='bold', fontname='Arial');
        plt.yticks(fontsize=12, fontweight='bold', fontname='Arial');
        ## Mark and annotate minimum points
        plt.scatter(min_x, min_y, color=line1.get_color(), zorder=3);
        plt.annotate(f'({min_x}, {min_y:.2f})', xy=(min_x, min_y), xytext=(min_x + 0.05, min_y + 0.05),  fontsize=14, color='red', ha='right');
        print("minx = ", min_x, "miny = ", min_y);
        plt.show();

        # Finished
        print("\t Finished.");
    def boneMuscleStructureWithRidgeMultiOutputRegression():
        """
        Analyzes and visualizes the performance of ridge multi-output regression on bone-muscle structures.
        
        This function evaluates the reconstruction accuracy of pelvis bone-muscle structures using 
        ridge multi-output regression by loading validation error data across varying numbers 
        of principal components and generating performance visualizations with trend analysis.
        
        Key Steps:
        1. Load validation error data from ridge multi-output regression bone-muscle structure results
        2. Aggregate validation errors across all cross-validation folds (0-9)
        3. Process errors for varying numbers of components (1-80)
        4. Convert error measurements to millimeters for clinical presentation
        5. Generate line plot visualization showing component count vs validation errors
        6. Identify and annotate optimal number of components with minimum error
        
        Methodology Analysis:
        - Shape Relation Strategy: Uses PCA + ridge multi-output regression mapping between feature and shape parameter spaces
        - Validation Framework: Processes errors across multiple cross-validation folds
        - Component Analysis: Evaluates performance across 1-80 principal components
        - Error Metrics: Point-to-point distance measurements between predicted and ground truth
        - Feature Selection: Uses feature selection strategy index 10 for consistent evaluation
        
        Visualization Output:
        - Line plot showing validation errors vs number of components
        - Minimum error point identification and annotation with coordinates
        - Professional formatting with bold fonts and clear axis labeling
        - Title indicating the specific regression method being analyzed
        
        Statistical Summary:
        - Computes mean validation errors across all folds for each component count
        - Converts measurements from meters to millimeters for clinical relevance
        - Identifies optimal number of components for minimum reconstruction error
        - Provides visual assessment of method performance and parameter sensitivity
        
        Note: Ridge multi-output regression combines ridge regularization with multi-output capability
        to handle multiple target variables simultaneously while preventing overfitting.
        Accuracy is computed only on bone vertices since muscle ground truth may not be available.
        """
        # Initializing
        print("Initializing ...");
        ridgeMultiOutputRegresionFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures_MultiOutputRegressor";
        maxNumComps = 80; minNumComps = 1; minValidIndex = 0; maxValidIndex = 9;
        
        # Reading validating errors 
        print("Reading valiating errors ...");
        validationErrors_ridgeMultiOutputRegression = np.zeros((maxValidIndex + 1, maxNumComps));
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            for numComps in range(minNumComps, maxNumComps + 1):
                validErrors = sp.readMatrixFromCSVFile(ridgeMultiOutputRegresionFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}_{numComps}.csv");
                validError = validErrors.mean();
                validationErrors_ridgeMultiOutputRegression[validIndex,  numComps - 1] = validError;
        validationErrors_ridgeMultiOutputRegression = np.array(validationErrors_ridgeMultiOutputRegression);
        optimalValidErrors_ridgeMultiOutputRegression = validationErrors_ridgeMultiOutputRegression[:, optimNumCompBoneMuscleOptim - 1];

        # Draw graphs
        print("Drawing the graphs ...");
        ## Forming drawing data
        drawingData = np.mean(validationErrors_ridgeMultiOutputRegression, axis=0) * 1000;
        N = validationErrors_ridgeMultiOutputRegression.shape[1];
        x = np.arange(1, N + 1);
        y1 = drawingData;
        ## Find minimum points
        min_index = np.argmin(y1);
        min_x = x[min_index];
        min_y = y1[min_index];
        ## Plotting the curves
        line1, = plt.plot(x, y1, label=f'Bone Structure (min: {min_x}, {min_y:.2f} mm)');
        ## Axis labels and title
        plt.xlabel('Numbers of Components', fontdict={'fontsize': 14, 'fontweight': 'bold'});
        plt.ylabel('Validation Errors (mm)', fontdict={'fontsize': 14, 'fontweight': 'bold'});
        plt.title('Ridge Multiple Ouput Regression', fontdict={'fontsize': 16, 'fontweight': 'bold'});
        plt.xticks(fontsize=12, fontweight='bold', fontname='Arial');
        plt.yticks(fontsize=12, fontweight='bold', fontname='Arial');
        ## Mark and annotate minimum points
        plt.scatter(min_x, min_y, color=line1.get_color(), zorder=3);
        plt.annotate(f'({min_x}, {min_y:.2f})', xy=(min_x, min_y), xytext=(min_x + 0.05, min_y + 0.05),  fontsize=14, color='red', ha='right');
        print("minx = ", min_x, "miny = ", min_y);
        plt.show();

        # Finished
        print("\t Finished.");
    def boneMuscleStructureWithWithVariousRegressions():
        """
        Analyzes and compares the performance of various regression models on bone-muscle structures.
        
        This function evaluates the reconstruction accuracy of pelvis bone-muscle structures by 
        loading validation error data from multiple regression approaches and generating comparative 
        bar chart visualizations to assess the effectiveness of each regression method.
        
        Key Steps:
        1. Load validation error data from six different regression model results
        2. Aggregate validation errors across all cross-validation folds (0-9)
        3. Process errors for optimal component configurations for each regression method
        4. Convert error measurements to millimeters for clinical presentation
        5. Generate comparative bar chart showing mean errors with standard deviations
        6. Annotate bars with precise error values for quantitative comparison
        
        Regression Models Compared:
        - Canonical Correlation Analysis: Finds maximally correlated linear combinations between spaces
        - Gaussian Process Regression: Non-parametric Bayesian approach with uncertainty quantification
        - Multivariate Linear Regression: Direct linear mapping between feature and vertex parameters
        - Partial Least Squares Regression: Maximizes covariance between feature and vertex spaces
        - Ridge Linear Regression: Linear regression with L2 regularization to prevent overfitting
        - Multi-output Ridge Regression: Ridge regression with multi-output capability
        
        Methodology Analysis:
        - Validation Framework: Processes errors across multiple cross-validation folds (0-9)
        - Optimal Components: Uses predetermined optimal number of components for each method
        - Error Metrics: Point-to-point distance measurements between predicted and ground truth
        - Feature Selection: Uses feature selection strategy index 10 for consistent evaluation
        
        Visualization Output:
        - Bar chart showing mean validation errors with error bars (standard deviations)
        - Precise error value annotations on each bar for quantitative assessment
        - Professional formatting with bold fonts and clear axis labeling
        - Title indicating comprehensive regression method comparison
        
        Statistical Summary:
        - Computes mean and standard deviation validation errors across all folds
        - Converts measurements from meters to millimeters for clinical relevance
        - Provides visual ranking of regression methods by performance
        - Enables identification of the most effective regression approach
        
        Note: This comprehensive comparison helps determine the optimal regression methodology
        for feature-to-pelvis reconstruction by evaluating performance across multiple validation 
        folds using consistent evaluation metrics. Accuracy is computed only on bone vertices 
        since muscle ground truth may not be available.
        """
        # Initializing
        print("Initializing ...");
        multiLinearRegresionFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures";
        ridgeLinearRegresionFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures_RidgeLinearRegression";
        canonicalCorrelationRegresionFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures_CanonicalCorrelationAnalysis";
        partialLeastSquaresRegresionFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures_PartialLeastSquaresRegression";
        gaussianProcessRegresionFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures_GaussianProcessRegressor";
        ridgeMultiOutputRegresionFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures_MultiOutputRegressor";
        optimNumComps_multiLinerRegression = 38;
        optimNumComps_ridgeLinearRegression = 79;
        optimNumComps_canonicalRegression = 12;
        optimNumComps_partialLeastSquaresRegression = 38;
        optimNumComps_gaussianProcessRegression = 38;
        optimNumComps_ridgeMultiOutputRegression = 79;
        maxNumComps = 80; minNumComps = 1; minValidIndex = 0; maxValidIndex = 9;

        # Reading validating errors 
        print("Reading valiating errors ...");
        validationErrors_multiLinear = np.zeros((maxValidIndex + 1, maxNumComps));
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            for numComps in range(minNumComps, maxNumComps + 1):
                validErrors = sp.readMatrixFromCSVFile(multiLinearRegresionFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}_{numComps}.csv");
                validError = validErrors.mean();
                validationErrors_multiLinear[validIndex,  numComps - 1] = validError;
        validationErrors_multiLinear = np.array(validationErrors_multiLinear);
        optimalValidErrors_multiLinear = validationErrors_multiLinear[:, optimNumComps_multiLinerRegression - 1];

        validationErrors_ridgeLinearRegression = np.zeros((maxValidIndex + 1, maxNumComps));
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            for numComps in range(minNumComps, maxNumComps + 1):
                validErrors = sp.readMatrixFromCSVFile(ridgeLinearRegresionFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}_{numComps}.csv");
                validError = validErrors.mean();
                validationErrors_ridgeLinearRegression[validIndex,  numComps - 1] = validError;
        validationErrors_ridgeLinearRegression = np.array(validationErrors_ridgeLinearRegression);
        optimalValidErrors_ridgeLinearRegression = validationErrors_ridgeLinearRegression[:, optimNumComps_ridgeLinearRegression - 1];
    
        validationErrors_canonicalCorrelationRegression = np.zeros((maxValidIndex + 1, maxNumComps));
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            for numComps in range(minNumComps, maxNumComps + 1):
                validErrors = sp.readMatrixFromCSVFile(canonicalCorrelationRegresionFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}_{numComps}.csv");
                validError = validErrors.mean();
                validationErrors_canonicalCorrelationRegression[validIndex,  numComps - 1] = validError;
        validationErrors_canonicalCorrelationRegression = np.array(validationErrors_canonicalCorrelationRegression);
        optimalValidErrors_canonicalCorrelationRegression = validationErrors_canonicalCorrelationRegression[:, optimNumComps_canonicalRegression - 1];
    
        validationErrors_partialLeastSquaresRegression = np.zeros((maxValidIndex + 1, maxNumComps));
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            for numComps in range(minNumComps, maxNumComps + 1):
                validErrors = sp.readMatrixFromCSVFile(partialLeastSquaresRegresionFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}_{numComps}.csv");
                validError = validErrors.mean();
                validationErrors_partialLeastSquaresRegression[validIndex,  numComps - 1] = validError;
        validationErrors_partialLeastSquaresRegression = np.array(validationErrors_partialLeastSquaresRegression);
        optimalValidErrors_partialLeastSquaresRegression = validationErrors_partialLeastSquaresRegression[:, optimNumComps_partialLeastSquaresRegression - 1];
    
        validationErrors_gaussianProcessRegression = np.zeros((maxValidIndex + 1, maxNumComps));
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            for numComps in range(minNumComps, maxNumComps + 1):
                validErrors = sp.readMatrixFromCSVFile(gaussianProcessRegresionFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}_{numComps}.csv");
                validError = validErrors.mean();
                validationErrors_gaussianProcessRegression[validIndex,  numComps - 1] = validError;
        validationErrors_gaussianProcessRegression = np.array(validationErrors_gaussianProcessRegression);
        optimalValidErrors_gaussianProcessRegression = validationErrors_gaussianProcessRegression[:, optimNumComps_gaussianProcessRegression - 1];

        validationErrors_ridgeMultiOutputRegression = np.zeros((maxValidIndex + 1, maxNumComps));
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            for numComps in range(minNumComps, maxNumComps + 1):
                validErrors = sp.readMatrixFromCSVFile(ridgeMultiOutputRegresionFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}_{numComps}.csv");
                validError = validErrors.mean();
                validationErrors_ridgeMultiOutputRegression[validIndex,  numComps - 1] = validError;
        validationErrors_ridgeMultiOutputRegression = np.array(validationErrors_ridgeMultiOutputRegression);
        optimalValidErrors_ridgeMultiOutputRegression = validationErrors_ridgeMultiOutputRegression[:, optimNumComps_ridgeMultiOutputRegression - 1];
    
        print("\t Drawing charts ...");

        ## Organize data
        groups = [
            ("Canonical\nCorrelation", optimalValidErrors_canonicalCorrelationRegression * 1000),
            ("Gaussian\nProcess", optimalValidErrors_gaussianProcessRegression * 1000),
            ("Multivariate\nLinear", optimalValidErrors_multiLinear * 1000),
            ("Partial\nLeastSquare", optimalValidErrors_partialLeastSquaresRegression * 1000),
            ("Ridge\nLinear", optimalValidErrors_ridgeLinearRegression * 1000),
            ("Multioutput\nRidge", optimalValidErrors_ridgeMultiOutputRegression * 1000)
        ]

        ## Compute means and standard deviations
        means = [np.mean(data) for _, data in groups]
        stds = [np.std(data) for _, data in groups]
        labels = [name for name, _ in groups]

        ## Set x locations and bar width
        x = np.arange(len(labels))  # Centered positions
        width = 0.6

        ## Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(x, means, width=width, yerr=stds, capsize=5)

        ## Add value labels on top of each bar
        for bar, val in zip(bars, means):
            height = bar.get_height()
            ax.annotate(f'{val:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=13, fontweight='bold')

        ## Axis labels and ticks
        ax.set_ylabel('Validation Errors (mm)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Multiple Regression Models', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12, fontweight='bold', ha='center')
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels([f'{tick:.1f}' for tick in ax.get_yticks()], fontsize=13, fontweight='bold')

        ## No legend needed (you have no groups/colors)
        ax.grid(False)

        ## Layout
        plt.tight_layout()
        plt.style.use('default');
        plt.show()
    
    # Running
    print("Running ...");
    boneMuscleStructureWithWithVariousRegressions();

    # Finished processing
    print("Finished processing.");
def featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor_trainValidationVariousFeatures():
    """
    Performs comprehensive training and validation of multi-output ridge regression for pelvis bone-muscle reconstruction across various feature selection strategies.
    
    This function evaluates the accuracy of pelvis bone reconstruction by training multi-output ridge regression models 
    on different feature selection strategies and varying numbers of principal components. The evaluation is performed 
    across multiple cross-validation folds to ensure robust performance assessment.
    
    Key Steps:
    1. Parse command line arguments for processing range (feature strategies, validation folds, components)
    2. Initialize file paths and load template data from personalized pelvis structures
    3. Load feature selection strategies and template mesh information
    4. For each feature selection strategy and validation fold:
       - Load training and validation subject IDs from cross-validation splits
       - Extract pelvis bone-muscle vertices and feature points for each subject
       - Scale and normalize both feature and vertex data using StandardScaler
       - Apply PCA dimensionality reduction to both feature and vertex spaces
       - Train multi-output ridge regression model with cross-validation
       - For each number of components:
         * Transform features to PCA parameter space
         * Predict vertex parameters using trained regression model
         * Reconstruct pelvis bone-muscle mesh from predicted parameters
         * Compute point-to-point distance errors (bone vertices only)
       - Save validation errors to CSV files
    
    Command Line Arguments:
    - StartFeatureIndex: Starting index for feature selection strategies
    - EndFeatureIndex: Ending index for feature selection strategies  
    - StartValidationIndex: Starting validation fold index
    - EndValidationIndex: Ending validation fold index
    - StartNumComponents: Starting number of PCA components
    - EndNumComponents: Ending number of PCA components
    
    Data Processing:
    - Uses personalized pelvis bone-muscle meshes as ground truth
    - Applies barycentric landmark reconstruction for feature point extraction
    - Implements standardized scaling for numerical stability
    - Uses PCA for dimensionality reduction in both input and output spaces
    
    Model Training:
    - Multi-output ridge regression with cross-validation (RidgeCV)
    - Automatic hyperparameter tuning through cross-validation
    - Handles multiple target variables simultaneously
    - Prevents overfitting through L2 regularization
    
    Output:
    - CSV files containing average point-to-point distances for each configuration
    - Validation errors organized by feature strategy, validation fold, and component count
    - Performance metrics for systematic evaluation of reconstruction accuracy
    
    Note: This function focuses on bone vertex accuracy since muscle ground truth may not be available.
    The comprehensive evaluation across multiple parameters enables optimal model selection and performance assessment.
    """

    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 7):
        print("Please input the command as the following: [ProgramName] [StartFeatureIndex] [EndFeatureIndex] [StartValidationIndex] [EndValidationIndex] [StartNumComponents] [EndNumComponents]");
        return;
    startFeatureIndex = int(sys.argv[1]);
    endFeatureIndex = int(sys.argv[2]);
    startValidationIndex = int(sys.argv[3]);
    endValidationIndex = int(sys.argv[4]);
    startNumComponents = int(sys.argv[5]);
    endNumComponents = int(sys.argv[6]);    
    mainFolder = disk + r"/SpinalPelvisPred";
    pelvisReconFolder = mainFolder + r"/Data/PelvisBoneRecon";
    femalePelvisFolder = pelvisReconFolder + r"/FemalePelvisGeometries";
    personalizedFemalePelvisFolder = femalePelvisFolder + r"/PersonalizedPelvisStructures";
    crossValidationFolder = mainFolder + r"/Data/PelvisBoneRecon/CrossValidation/ShapeRelationStrategy/OptimalTrainValidTest";
    trainTestSplitFolder = crossValidationFolder + "/TrainTestSplits";
    validationErrorFolder = crossValidationFolder + "/ValidationErrors";
    featureSelectionProtocolFolder = mainFolder + r"/Data/PelvisBoneRecon/CrossValidation/FeatureSelectionProtocol";
    templateFolder = mainFolder + r"/Data/Template";

    # Reading structural feature strategies
    print("Reading structural feature strategies ...");
    allPelvicFeatures = sp.read3DPointsFromPPFile(featureSelectionProtocolFolder + "/AllFeaturePoints.pp");
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");

    # Reading template information
    print("Reading template information ...");
    templatePelvisBoneMuscleMesh = sp.readMesh(templateFolder + "/PelvisBonesMuscles/TempPelvisBoneMuscles.ply");
    templatePelvisBoneMesh = sp.readMesh(templateFolder + "/PelvisBonesMuscles/TempPelvisBoneMesh.ply");
    pelvisBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(templatePelvisBoneMesh.vertices, templatePelvisBoneMuscleMesh.vertices);

    # Iterate for each feature selection strategy
    print("Iterating for each feature selection strategy ...");
    for featureSelIndex in range(startFeatureIndex, endFeatureIndex + 1):
        # Get the feature selection strategies
        print("\t Getting the feature selection strategy ...");
        featureSelectionIndices = list(featureSelectionStrategyDict.values())[featureSelIndex];
        tempPelvicFeatures = allPelvicFeatures[featureSelectionIndices];
        pelvicFeatureBaryIndices, pelvicFeatureBaryCoords = sp.computeBarycentricLandmarks(templatePelvisBoneMuscleMesh, tempPelvicFeatures);

        # Iterate for each validation index
        for validIndex in range(startValidationIndex, endValidationIndex + 1):
            # Debugging information
            print("/********************************************** VALIDATION: ", validIndex);

            # Reading training and validating subject IDs
            print("\t Reading training and validating subject IDs ...");
            trainIDs = sp.readListOfStrings(trainTestSplitFolder + f"/TrainingIDs_{validIndex}.txt");
            validIDs = sp.readListOfStrings(trainTestSplitFolder + f"/ValidationIDs_{validIndex}.txt");

            # Forming the training and validating data
            print("\t Forming the training and validating data ...");
            trainingPelvicVertexData = []; validPelvicVertexData = [];
            trainingPelvicFeatureData = []; validPelvicFeatureData = [];
            for i, ID in enumerate(trainIDs):
                pelvicStructure = sp.readMesh(personalizedFemalePelvisFolder + f"/{ID}-PersonalizedPelvisBoneMuscleMesh.ply");
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvicStructure, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                trainingPelvicVertexData.append(pelvicStructure.vertices.flatten());
                trainingPelvicFeatureData.append(pelvicFeatures.flatten());
            for i, ID in enumerate(validIDs):
                pelvicStructure = sp.readMesh(personalizedFemalePelvisFolder + f"/{ID}-PersonalizedPelvisBoneMuscleMesh.ply");
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvicStructure, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                validPelvicVertexData.append(pelvicStructure.vertices.flatten());
                validPelvicFeatureData.append(pelvicFeatures.flatten());
            
            # Iterate for each number of components
            for numComps in range(startNumComponents, endNumComponents + 1):
                print("/***************************** FEATURE SELECTION INDEX: ", featureSelIndex, ", VALIDATION INDEX: ", validIndex, ", NUM COMPS: ", numComps);

                # Scale the pelvic feature and vertex data
                print("\t\t Scaling the pelvic feature and vertex data ...");
                yScaler = StandardScaler().fit(trainingPelvicVertexData);
                xScaler = StandardScaler().fit(trainingPelvicFeatureData);
                scaledYData = yScaler.transform(trainingPelvicVertexData);
                scaledXData = xScaler.transform(trainingPelvicFeatureData);

                # Control the number of components
                print("\t\t Controlling the number of components ...");
                targetNumComps = numComps;
                xDims = scaledXData.shape[1];
                yDims = scaledYData.shape[1];
                xNumComps = min(xDims, targetNumComps);
                yNumComps = min(yDims, targetNumComps);

                # Parameterize the scaled pelvic vertex and feature data
                print("\t\t Parameterizing the scaled pelvic vertex and feature data ...");
                xSSM = PCA(n_components=xNumComps);
                xSSM.fit(scaledXData);
                trainXParamData = xSSM.transform(scaledXData);
                ySSM = PCA(n_components=yNumComps);
                ySSM.fit(scaledYData);
                trainYParamData = ySSM.transform(scaledYData);

                # Linear regression from feature parameters to vertex parameters
                print("\t\t Linear regression from feature parameters to vertex parameters ...");
                model = MultiOutputRegressor(RidgeCV())
                model.fit(trainXParamData, trainYParamData)

                # Computing errors on the validating data
                print("\t\t Computing errors on the validating data ...");
                ## Scale and parameterize the validation data
                scaledValidXData = xScaler.transform(validPelvicFeatureData);
                validXParams = xSSM.transform(scaledValidXData);
                ## Try to predict the pelvic vertex params from the pelvic feature params
                predYParams = model.predict(validXParams);
                ## Inverse transform to the scaled Y data
                predYParams = predYParams.reshape(-1, yNumComps);
                predScaledYData = ySSM.inverse_transform(predYParams);
                predYData = yScaler.inverse_transform(predScaledYData);
                ## Computing validating errors
                avgP2PDists = [];
                for v, predY in enumerate(predYData):
                    # Getting the valiation data and predicted data
                    validY = validPelvicVertexData[v];
                    validPelvicStructureVertices = validY.reshape(-1, 3);
                    predPelvicStructureVertices = predY.reshape(-1, 3);

                    # Compute accuracy only on the bone vertices because we do not have ground truth of the muscle
                    validPelvisBoneVertices = validPelvicStructureVertices[pelvisBoneVertexIndices];
                    predPelvisBoneVertices = predPelvicStructureVertices[pelvisBoneVertexIndices];

                    # Compute points to points distances
                    avgP2PDist = sp.computeAveragePointsToPointsDistance(validPelvisBoneVertices, predPelvisBoneVertices);
                    avgP2PDists.append(avgP2PDist);
                avgP2PDists = np.array(avgP2PDists);

                # Save the computed errors
                print("\t\t Saving the computed errors ...", end=' ', flush=True);
                print("-> Grand Mean: ", avgP2PDists.mean());
                sp.saveMatrixToCSVFile(validationErrorFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}_{numComps}.csv", avgP2PDists);
def featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor_determineOptimalNumComponents():
    """
    Determines the optimal number of principal components for multi-output ridge regression across feature selection strategies.
    
    This function analyzes validation error data from pelvis bone-muscle reconstruction experiments to identify 
    the optimal number of PCA components that minimizes reconstruction errors for each feature selection strategy.
    The analysis is performed across multiple cross-validation folds to ensure robust optimization.
    
    Key Steps:
    1. Initialize file paths and load feature selection strategy configurations
    2. For each feature selection strategy (0 to max available strategies):
       - Load validation error data across all cross-validation folds (0-9)
       - Process errors for varying numbers of components (1-200)
       - Aggregate validation errors from CSV files for each fold and component count
       - Compute mean validation errors across all folds for each component number
       - Identify the optimal number of components that minimizes mean validation error
       - Store optimal component count for the current strategy
    3. Save all optimal component counts to CSV file for future use
    
    Data Processing:
    - Reads validation errors from CSV files organized by feature strategy, fold, and component count
    - Processes errors across extended component range (1-200) due to data augmentation
    - Computes statistical aggregation across 10-fold cross-validation
    - Identifies minimum error points for parameter optimization
    
    Optimization Strategy:
    - Grid search approach across component numbers to find global minimum
    - Cross-validation averaging to ensure robust parameter selection
    - Feature strategy-specific optimization for tailored performance
    
    Output:
    - CSV file containing optimal number of components for each feature selection strategy
    - Enables informed parameter selection for final model training and testing
    - Provides foundation for subsequent performance evaluation with optimal settings
    
    Note: This function serves as a critical preprocessing step for model optimization,
    ensuring that each feature selection strategy uses its optimal number of PCA components
    for maximum reconstruction accuracy in the final evaluation phase.
    """

    # Initializing
    print("Initializing ...");    
    mainFolder = disk + r"/SpinalPelvisPred";
    pelvisReconFolder = mainFolder + r"/Data/PelvisBoneRecon";
    crossValidationFolder = pelvisReconFolder + r"/CrossValidation/ShapeRelationStrategy/OptimalTrainValidTest";
    featureSelectionProtocolFolder = mainFolder + r"/Data/PelvisBoneRecon/CrossValidation/FeatureSelectionProtocol";
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");
    
    # Iterate for each feature selection strategy
    print("Iterating for each feature selection strategy ...");
    optimNumCompsBuffer = [];
    for featureSelIndex in range(len(featureSelectionStrategyDict)):
        # Debugging information
        print("/********************************************** FEATURE SELECTION INDEX: ", featureSelIndex);

        # Reading the validation errors for ten fold with 1 to 200 num of components
        print("\t Reading the validation errors for ten fold with 1 to 200 num of components ...");
        validationErrors = np.zeros((10, 200));
        for validIndex in range(10):
            for numComps in range(1, 201):
                # Reading the validation errors
                print("\t\t Validating index: ", validIndex, ", numComps: ", numComps);
                validErrors = sp.loadNumPYArrayFromNPY(crossValidationFolder + 
                                                       f"/ValidationErrors/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}_{numComps}.npy");
                validError = validErrors.mean();
                validationErrors[validIndex, numComps - 1] = validError;

        # Compute the mean errors for all folds of cross-validation
        print("\t Computing the mean errors for all folds of cross-validation ...");
        meanValidationErrors = np.mean(validationErrors, axis=0);

        # Estimate the optimal number of components
        print("\t Estimating the optimal number of components: ", end="", flush=True);
        optimNumComps = np.argmin(meanValidationErrors) + 1;  # +1 because numComps starts from 1
        print(optimNumComps);

        # Add optim numComps to buffer
        optimNumCompsBuffer.append(optimNumComps);

    # Save the optimal number of components
    print("Saving the optimal number of components ...");
    optimNumCompsBuffer = np.array(optimNumCompsBuffer);
    sp.saveMatrixToCSVFile(crossValidationFolder + "/OptimalNumComponents.csv", optimNumCompsBuffer);

    # Finished processing
    print("Finished processing.");
def featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor_testUsingOptimalNumComponents():
    """
    Performs comprehensive testing of multi-output ridge regression for pelvis bone-muscle reconstruction using optimal component configurations.
    
    This function evaluates the accuracy of pelvis bone reconstruction by testing pre-trained multi-output ridge regression models 
    on unseen test data using optimal numbers of principal components determined from validation experiments. The evaluation is 
    performed across multiple feature selection strategies and cross-validation folds to provide robust performance assessment.
    
    Key Steps:
    1. Parse command line arguments for processing range (feature strategies, validation folds)
    2. Initialize file paths and load template data from personalized pelvis structures
    3. Load feature selection strategies, template mesh information, and optimal component counts
    4. For each feature selection strategy and validation fold:
       - Load training and testing subject IDs from cross-validation splits
       - Extract pelvis bone-muscle vertices and feature points for each subject
       - Scale and normalize both feature and vertex data using StandardScaler
       - Apply PCA dimensionality reduction using optimal number of components
       - Train multi-output ridge regression model with cross-validation on training data
       - For each testing subject:
         * Transform features to PCA parameter space
         * Predict vertex parameters using trained regression model
         * Reconstruct pelvis bone-muscle mesh from predicted parameters
         * Save predicted meshes for visualization and analysis
         * Compute point-to-point distance errors (bone vertices only)
       - Save testing errors to CSV files
    
    Command Line Arguments:
    - StartFeatureIndex: Starting index for feature selection strategies
    - EndFeatureIndex: Ending index for feature selection strategies  
    - StartValidationIndex: Starting validation fold index
    - EndValidationIndex: Ending validation fold index
    
    Data Processing:
    - Uses personalized pelvis bone-muscle meshes as ground truth
    - Applies barycentric landmark reconstruction for feature point extraction
    - Implements standardized scaling for numerical stability
    - Uses PCA for dimensionality reduction with pre-determined optimal components
    
    Model Training and Testing:
    - Multi-output ridge regression with cross-validation (RidgeCV)
    - Automatic hyperparameter tuning through cross-validation
    - Tests on completely unseen data separate from training/validation sets
    - Prevents overfitting through L2 regularization and proper data splits
    
    Output:
    - CSV files containing average point-to-point distances for each configuration
    - Testing errors organized by feature strategy and validation fold
    - Predicted pelvis bone and bone-muscle meshes saved for visualization
    - Performance metrics for final model evaluation on test data
    
    Note: This function performs final testing using optimal parameters determined from validation,
    providing unbiased performance estimates on completely unseen test data. Accuracy is computed 
    only on bone vertices since muscle ground truth may not be available.
    """

    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 5):
        print("Please input the command as the following: [ProgramName] [StartFeatureIndex] [EndFeatureIndex] [StartValidationIndex] [EndValidationIndex]");
        return;
    startFeatureIndex = int(sys.argv[1]);
    endFeatureIndex = int(sys.argv[2]);
    startValidationIndex = int(sys.argv[3]);
    endValidationIndex = int(sys.argv[4]);
    mainFolder = disk + "/SpinalPelvisPred";
    pelvisReconFolder = mainFolder + "/Data/PelvisBoneRecon";
    femalePelvisFolder = pelvisReconFolder + "/FemalePelvisGeometries";
    personalizedFemalePelvisFolder = femalePelvisFolder + "/PersonalizedPelvisStructures";
    crossValidationFolder = mainFolder + "/Data/PelvisBoneRecon/CrossValidation/ShapeRelationStrategy/OptimalTrainValidTest";
    trainTestSplitFolder = crossValidationFolder + "/TrainTestSplits";
    testingErrorFolder = crossValidationFolder + "/TestingErrors";
    predictedTestingPelvicStructureFolder = crossValidationFolder + "/PredictedTestingPelvicStructures";    
    featureSelectionProtocolFolder = mainFolder + "/Data/PelvisBoneRecon/CrossValidation/FeatureSelectionProtocol";
    templateFolder = mainFolder + "/Data/Template";

    # Reading structural feature strategies
    print("Reading structural feature strategies ...");
    allPelvicFeatures = sp.read3DPointsFromPPFile(featureSelectionProtocolFolder + "/AllFeaturePoints.pp");
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");

    # Reading template information
    print("Reading template information ...");
    templatePelvisBoneMuscleMesh = sp.readMesh(templateFolder + "/PelvisBonesMuscles/TempPelvisBoneMuscles.ply");
    templatePelvisBoneMesh = sp.readMesh(templateFolder + "/PelvisBonesMuscles/TempPelvisBoneMesh.ply");
    pelvisBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(templatePelvisBoneMesh.vertices, templatePelvisBoneMuscleMesh.vertices);

    # Read optimal number of components
    print("Reading optimal number of components ...");
    optimNumComps = sp.readMatrixFromCSVFile(crossValidationFolder + "/OptimalNumComponents.csv");
    optimNumComps = optimNumComps.flatten();

    # Iterate for each feature selection strategy
    print("Iterating for each feature selection strategy ...");
    for featureSelIndex in range(startFeatureIndex, endFeatureIndex + 1):
        # Get the feature selection strategies
        print("\t Getting the feature selection strategy ...");
        featureSelectionIndices = list(featureSelectionStrategyDict.values())[featureSelIndex];
        tempPelvicFeatures = allPelvicFeatures[featureSelectionIndices];
        pelvicFeatureBaryIndices, pelvicFeatureBaryCoords = sp.computeBarycentricLandmarks(templatePelvisBoneMuscleMesh, tempPelvicFeatures);
        targetNumComps = optimNumComps[featureSelIndex];

        # Iterate for each validation index
        for validIndex in range(startValidationIndex, endValidationIndex + 1):
            # Debugging information
            print("/***************************** FEATURE SELECTION INDEX: ", featureSelIndex, ", VALIDATION INDEX: ", validIndex, ", NUM COMPS: ", targetNumComps);

            # Reading training and testing subject IDs
            print("\t Reading training and testing subject IDs ...");
            trainIDs = sp.readListOfStrings(trainTestSplitFolder + f"/TrainingIDs_{validIndex}.txt");
            testIDs = sp.readListOfStrings(trainTestSplitFolder + f"/TestingIDs_{validIndex}.txt");

            # Forming the training and testing data
            print("\t Forming the training and testing data ...");
            trainingPelvicVertexData = []; testPelvicVertexData = [];
            trainingPelvicFeatureData = []; testPelvicFeatureData = [];
            testingPelvisMeshes = [];
            for i, ID in enumerate(trainIDs):
                pelvicStructure = sp.readMesh(personalizedFemalePelvisFolder + f"/{ID}-PersonalizedPelvisBoneMuscleMesh.ply");
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvicStructure, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                trainingPelvicVertexData.append(pelvicStructure.vertices.flatten());
                trainingPelvicFeatureData.append(pelvicFeatures.flatten());
            for i, ID in enumerate(testIDs):
                pelvicStructure = sp.readMesh(personalizedFemalePelvisFolder + f"/{ID}-PersonalizedPelvisBoneMuscleMesh.ply");
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvicStructure, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                testPelvicVertexData.append(pelvicStructure.vertices.flatten());
                testPelvicFeatureData.append(pelvicFeatures.flatten());
            
            # Scale the pelvic feature and vertex data
            print("\t\t Scaling the pelvic feature and vertex data ...");
            yScaler = StandardScaler().fit(trainingPelvicVertexData);
            xScaler = StandardScaler().fit(trainingPelvicFeatureData);
            scaledYData = yScaler.transform(trainingPelvicVertexData);
            scaledXData = xScaler.transform(trainingPelvicFeatureData);

            # Control the number of components
            print("\t\t Controlling the number of components ...");
            xDims = scaledXData.shape[1];
            yDims = scaledYData.shape[1];
            xNumComps = min(xDims, targetNumComps);
            yNumComps = min(yDims, targetNumComps);

            # Parameterize the scaled pelvic vertex and feature data
            print("\t\t Parameterizing the scaled pelvic vertex and feature data ...");
            xSSM = PCA(n_components=xNumComps);
            xSSM.fit(scaledXData);
            trainXParamData = xSSM.transform(scaledXData);
            ySSM = PCA(n_components=yNumComps);
            ySSM.fit(scaledYData);
            trainYParamData = ySSM.transform(scaledYData);

            # Linear regression from feature parameters to vertex parameters
            print("\t\t Linear regression from feature parameters to vertex parameters ...");
            model = MultiOutputRegressor(RidgeCV())
            model.fit(trainXParamData, trainYParamData)

            # Computing errors on the testing data
            print("\t\t Computing errors on the testing data ...");
            ## Scale and parameterize the testing data
            scaledTestXData = xScaler.transform(testPelvicFeatureData);
            testXParams = xSSM.transform(scaledTestXData);
            ## Try to predict the pelvic vertex params from the pelvic feature params
            predYParams = model.predict(testXParams);
            ## Inverse transform to the scaled Y data
            predYParams = predYParams.reshape(-1, yNumComps);
            predScaledYData = ySSM.inverse_transform(predYParams);
            predYData = yScaler.inverse_transform(predScaledYData);
            ## Computing testing errors
            avgP2PDists = [];
            for t, predY in enumerate(predYData):
                # Getting the testing data and predicted data
                testY = testPelvicVertexData[t];
                testPelvicStructureVertices = testY.reshape(-1, 3);
                predPelvicStructureVertices = predY.reshape(-1, 3);

                # Compute accuracy only on the bone vertices because we do not have ground truth of the muscle
                testPelvisBoneVertices = testPelvicStructureVertices[pelvisBoneVertexIndices];
                predPelvisBoneVertices = predPelvicStructureVertices[pelvisBoneVertexIndices];

                # Save the predicted pelvis bone muscle mesh
                testPelvisMesh = sp.formMesh(predPelvisBoneVertices, templatePelvisBoneMesh.faces);
                testPelvisBoneMuscleMesh = sp.formMesh(predPelvicStructureVertices, templatePelvisBoneMuscleMesh.faces);
                testPelvisFilePath = predictedTestingPelvicStructureFolder + f"/PredictedPelvicStructure_{featureSelIndex}_{validIndex}_{t}.ply";
                testPelvisMuscleFilePath = predictedTestingPelvicStructureFolder + f"/PredictedPelvicMuscle_{featureSelIndex}_{validIndex}_{t}.ply";
                sp.saveMeshToPLY(testPelvisFilePath, testPelvisMesh);
                sp.saveMeshToPLY(testPelvisMuscleFilePath, testPelvisBoneMuscleMesh);

                # Compute points to points distances
                avgP2PDist = sp.computeAveragePointsToPointsDistance(testPelvisBoneVertices, predPelvisBoneVertices);
                avgP2PDists.append(avgP2PDist);
            avgP2PDists = np.array(avgP2PDists);

            # Save the computed errors
            print("\t\t Saving the computed errors ...", end=' ', flush=True);
            print("-> Grand Mean: ", avgP2PDists.mean());
            sp.saveMatrixToCSVFile(testingErrorFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}.csv", avgP2PDists);
def featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor_drawTestingErrors():
    """
    Analyzes and visualizes testing errors from multi-output ridge regression pelvis reconstruction across feature selection strategies.
    
    This function evaluates the final testing performance of the optimal pelvis bone-muscle reconstruction model
    by loading testing error data across different feature selection strategies and generating comprehensive
    statistical visualizations to assess model accuracy and feature importance relationships.
    
    Key Steps:
    1. Initialize file paths and load feature selection strategy configurations
    2. Calculate the number of features for each feature selection strategy
    3. Load testing error data across all feature strategies and validation folds
    4. Aggregate testing errors and compute statistical measures (mean, standard deviation)
    5. Generate bar chart visualization with error bars and trend analysis
    6. Perform polynomial fitting to identify error trends with feature count
    7. Calculate theoretical zero-error point for feature requirement estimation
    8. Save visualization results for publication and analysis
    
    Data Processing:
    - Reads testing errors from CSV files organized by feature strategy and validation fold
    - Processes errors across all available feature selection strategies (0 to max)
    - Computes mean and standard deviation across 10-fold cross-validation
    - Converts measurements from meters to millimeters for clinical presentation
    
    Statistical Analysis:
    - Mean testing error computation across cross-validation folds
    - Standard deviation calculation for error bar representation
    - Linear polynomial fitting to identify feature count vs error relationship
    - Zero-error point estimation to determine theoretical feature requirements
    
    Visualization Output:
    - Bar chart showing mean testing errors with standard deviation error bars
    - Precise error value annotations on each bar for quantitative assessment
    - Polynomial trend line with equation and zero-error point estimation
    - Professional formatting with bold fonts and clear axis labeling
    - Legend showing regression equation and zero-error point coordinates
    
    Feature Analysis:
    - Maps feature selection strategy indices to actual feature counts
    - Analyzes relationship between number of anatomical features and reconstruction accuracy
    - Provides insights into optimal feature set size for clinical applications
    - Enables determination of minimum feature requirements for acceptable accuracy
    
    Output Files:
    - High-resolution PNG chart saved for publication use
    - Statistical summary data for performance reporting
    - Trend analysis results for feature optimization guidance
    
    Note: This function provides final performance evaluation on completely unseen test data,
    offering unbiased assessment of model generalization capability across different feature
    selection strategies. The polynomial fitting helps identify the theoretical minimum number
    of anatomical features required for zero reconstruction error.
    """

    # Initializing
    print("Initializing ...");
    mainFolder = disk + "/SpinalPelvisPred";
    pelvisReconFolder = mainFolder + "/Data/PelvisBoneRecon";
    crossValidationFolder = pelvisReconFolder + "/CrossValidation/ShapeRelationStrategy/OptimalTrainValidTest";
    testingErrorFolder = crossValidationFolder + "/TestingErrors";
    featureSelectionProtocolFolder = mainFolder + "/Data/PelvisBoneRecon/CrossValidation/FeatureSelectionProtocol";
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");

    # Estimate the number of features in each feature selection strategy
    print("Estimating the number of features in each feature selection strategy ...");
    numOfStrategyFeatures = [];
    for featureSelIndex in range(len(featureSelectionStrategyDict)):
        featureSelectionIndices = list(featureSelectionStrategyDict.values())[featureSelIndex];
        numOfStrategyFeatures.append(len(featureSelectionIndices));
    numOfStrategyFeatures = np.array(numOfStrategyFeatures);

    # Reading testing errors
    print("Reading testing errors ...");
    drawingData = np.zeros((10, len(featureSelectionStrategyDict)));
    for featureSelIndex in range(len(featureSelectionStrategyDict)):
        for validIndex in range(10):
            # Reading the testing errors
            print(f"({featureSelIndex}, {validIndex})", end=' ', flush=True);
            testingErrors = sp.readMatrixFromCSVFile(testingErrorFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}.csv");
            testingError = testingErrors.mean();
            drawingData[validIndex, featureSelIndex] = testingError;
    drawingData = np.array(drawingData); print("");

    # Draw bar charts
    print("Drawing bar charts ...");
    ## Forming drawing data
    means = np.mean(drawingData, axis=0) * 1000;
    stds = np.std(drawingData, axis=0) * 1000;
    ## Define label as the number of features in each feature selection strategy
    labels = [f"{len(list(featureSelectionStrategyDict.values())[i])}" for i in range(len(featureSelectionStrategyDict))];
    ## Set x locations and bar width
    x_locs = np.arange(len(labels));
    bar_width = 0.75;
    ## Plotting the bar chart
    fig, ax = plt.subplots(figsize=(12, 6));
    bars = ax.bar(x_locs, means, width=bar_width, yerr=stds, capsize=5);
    ## Add value labels on top of each bar
    for bar, val in zip(bars, means):
        height = bar.get_height();
        ax.annotate(f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10); # not bold to avoid overlap with error bars
    ## Draw the poly fit line using linear regression line
    poly_fit = np.polyfit(x_locs, means, 1);
    poly_fit_line = np.polyval(poly_fit, x_locs);
    ## Draw the poly fit line
    line = ax.plot(x_locs, poly_fit_line, color='red', linestyle='--', linewidth=2, label='Poly Fit Line')[0];
    ## Find the zero error point by fitting another line with real number of feature points
    realPolyFit = np.polyfit(numOfStrategyFeatures[x_locs], means, 1);    
    ## Compute the zero error point when the mean error reach zero
    zero_error_x = -realPolyFit[1] / realPolyFit[0];  # x when y = 0
    zero_error_y = np.polyval(realPolyFit, zero_error_x);
    ## Show the zero error point in the legend
    ax.legend([bars, line],
          [f'Mean Testing Errors (mm)', f'Poly Fit Line (y = {realPolyFit[0]:.3f}x + {realPolyFit[1]:.3f})\nZero Error Point: ({zero_error_x:.2f}, {zero_error_y:.2f})'],
          loc='upper right', fontsize=12);
    ## Axis labels and ticks
    ax.set_ylabel('Testing Errors (mm)', fontsize=14, fontweight='bold');
    ax.set_xlabel('Number of Features in Feature Selection Strategies', fontsize=14, fontweight='bold');
    ax.set_xticks(x_locs);
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold', ha='center');
    ax.set_yticks(ax.get_yticks());
    ax.set_yticklabels([f'{tick:.1f}' for tick in ax.get_yticks()], fontsize=13, fontweight='bold');
    ## No legend needed (you have no groups/colors)
    ax.grid(False);
    ## Add a title
    ax.set_title('Testing Errors vs Number of Features in Feature Selection Strategies', fontsize=16, fontweight='bold');
    ## Layout
    plt.tight_layout();
    plt.show();
    ## Save the figure
    fig.savefig(crossValidationFolder + "/TestingErrorsBarChart.png", dpi=400, bbox_inches='tight');

    # Finished processing
    print("Finished processing.");
def featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor_computeVariousTestingErrors():
    """
    Computes comprehensive testing error metrics for multi-output ridge regression pelvis reconstruction across feature selection strategies.
    
    This function evaluates the final testing performance of the optimal pelvis bone-muscle reconstruction model
    by loading predicted and ground truth data to compute various error metrics including mesh accuracy, feature 
    accuracy, muscle attachment errors, and vertex-to-vertex distance measurements for comprehensive evaluation.
    
    Key Steps:
    1. Parse command line arguments for feature selection strategy range
    2. Initialize file paths and load template data for pelvis bone-muscle structures
    3. Load feature selection strategies and compute barycentric coordinates for feature extraction
    4. For each feature selection strategy and validation fold:
       - Load testing subject IDs from cross-validation splits
       - For each testing subject:
         * Load predicted pelvis bone-muscle mesh from reconstruction results
         * Load ground truth personalized pelvis structures and CT bone meshes
         * Perform rigid registration between predicted and ground truth meshes
         * Compute various error metrics:
           - Vertex-to-vertex distances on bone-muscle structures
           - Mesh-to-mesh distances for bone structures
           - Feature point reconstruction accuracy
           - Muscle attachment point accuracy
       - Aggregate error measurements across all testing subjects
    5. Save computed error metrics to files for analysis and visualization
    
    Command Line Arguments:
    - StartFeatureIndex: Starting index for feature selection strategies to process
    - EndFeatureIndex: Ending index for feature selection strategies to process
    
    Error Metrics Computed:
    - Mesh Testing Errors: Average point-to-point distances between predicted and CT bone meshes
    - Feature Testing Errors: Accuracy of anatomical feature point reconstruction
    - Muscle Attachment Errors: Accuracy at critical muscle attachment locations
    - Vertex-to-Vertex Distances: Comprehensive vertex-level accuracy on personalized meshes
    - Vertex-to-Mesh Distances: Accuracy against CT-reconstructed bone surfaces
    - Vertex-to-BoneMuscle Distances: Overall bone-muscle structure accuracy
    
    Data Processing:
    - Uses predicted pelvis structures from optimal multi-output ridge regression models
    - Compares against both personalized ground truth and CT-reconstructed references
    - Applies rigid registration for fair comparison between coordinate systems
    - Processes muscle attachment points within 6mm radius for clinical relevance
    
    Output Files:
    - CSV files containing aggregated error metrics for each feature strategy
    - NumPy arrays storing detailed vertex-level distance measurements
    - Organized by feature selection strategy for systematic analysis
    
    Note: This comprehensive error analysis provides multiple perspectives on reconstruction
    accuracy, enabling detailed assessment of model performance across different anatomical
    regions and clinical applications. The function focuses on clinically relevant metrics
    including muscle attachment accuracy and overall structural fidelity.
    """

    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 3):
        print("Please input the command as the following: [ProgramName] [StartFeatureIndex] [EndFeatureIndex]");
        return;
    startFeatureIndex = int(sys.argv[1]);
    endFeatureIndex = int(sys.argv[2]);
    mainFolder = disk + "/SpinalPelvisPred";
    pelvisReconFolder = mainFolder + "/Data/PelvisBoneRecon";
    templateFolder = mainFolder + "/Data/Template";
    femalePelvisFolder = pelvisReconFolder + "/FemalePelvisGeometries";
    crossValidationFolder = pelvisReconFolder + "/CrossValidation";
    featureSelectionProtocolFolder = mainFolder + "/Data/PelvisBoneRecon/CrossValidation/FeatureSelectionProtocol";
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");
    optimalTrainValidTestFolder = crossValidationFolder + "/ShapeRelationStrategy/OptimalTrainValidTest";
    predTestingPelvisStructureFolder = crossValidationFolder + "/ShapeRelationStrategy/OptimalTrainValidTest/PredictedTestingPelvicStructures";
    meshFeatureMuscleErrorFolder = crossValidationFolder + "/ShapeRelationStrategy/OptimalTrainValidTest/MeshFeatureMuscleErrors";
    numOfValids = 10; numOfTests = 30; numOfStrategies = len(featureSelectionStrategyDict);
    def estimateMuscleAttachmentIndices(pelvisBoneMesh, attachmentPoints, maxDist=0.006):
        # Define buffer as flatten vector
        outIndices = np.array([]);
    
        # Iterate for each attachment point
        for point in attachmentPoints:
            # Estimate the vertex indices within 6 mm
            indices = sp.estimateNearestIndicesWithinRadius(point, pelvisBoneMesh.vertices, inRadius=maxDist);
            indices = np.array(indices);
    
            # If indices are found, append vertical indices to the buffer
            if len(indices) > 0: 
                outIndices = np.concatenate([outIndices, indices.flatten()])

        # Return the unique indices
        return np.unique(outIndices).astype(np.int32);

    # Reading template information
    print("Reading template information ...");
    templatePelvisBoneMuscleMesh = sp.readMesh(templateFolder + "/PelvisBonesMuscles/TempPelvisBoneMuscles.ply");
    templatePelvisBoneMesh = sp.readMesh(templateFolder + "/PelvisBonesMuscles/TempPelvisBoneMesh.ply");
    templatePelvisFeaturePoints = sp.read3DPointsFromPPFile(templateFolder + "/PelvisBonesMuscles/TempPelvisBoneMesh_picked_points.pp");
    templatePelvisMuscleAttachmentPoints = sp.read3DPointsFromPPFile(templateFolder + "/PelvisBonesMuscles/TempPelvisBoneMesh_muscleAttachmentPoints.pp");
    pelvisBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(templatePelvisBoneMesh.vertices, templatePelvisBoneMuscleMesh.vertices);
    pelvisBoneFeatureBaryCoords, pelvisBoneFeatureBaryIndices = sp.computeBarycentricCoordinates(templatePelvisBoneMesh, templatePelvisFeaturePoints);
    muscleAttachmentIndices = estimateMuscleAttachmentIndices(templatePelvisBoneMesh, templatePelvisMuscleAttachmentPoints, 0.006);

    # Reading template information
    print("Reading template information ...");
    templatePelvisBoneMuscleMesh = sp.readMesh(templateFolder + "/PelvisBonesMuscles/TempPelvisBoneMuscles.ply");
    templatePelvisBoneMesh = sp.readMesh(templateFolder + "/PelvisBonesMuscles/TempPelvisBoneMesh.ply");

    # Iterate for each feature selection strategy
    print("Iterating for each feature selection strategy ...");
    for featureSelIndex in range(startFeatureIndex, endFeatureIndex + 1):
        # Generate buffer for saving testing errors
        print("Generate buffer for saving testing errors ...");
        meshTestingErrors = np.zeros((numOfValids, numOfTests));
        featureTestingErrors = np.zeros((numOfValids, numOfTests));
        muscleAttachmentTestingErrors = np.zeros((numOfValids, numOfTests));

        ## Iterate for each num of validation
        print("Iterating for each num of validation ...");
        vertex2VertexDistanceBuffer = [];
        vertex2VertexMeshDistanceBuffer = [];
        vertex2VertexBoneMuscleDistanceBuffer = [];
        for validIndex in range(numOfValids):
            # Reading testing IDs
            print("\t Reading testing IDs ...");
            testingIDFilePath = crossValidationFolder + "/ShapeRelationStrategy/OptimalTrainValidTest/TrainTestSplits/TestingIDs_" + str(validIndex) + ".txt";
            testingIDs = sp.readListOfStrings(testingIDFilePath);

            ## Iterate for each num of testing
            print("\t Iterating for each num of testing ...");
            for testIndex in range(numOfTests):
                # Debugging
                print(f"\t\t Comuting for the case: ({featureSelIndex}, {validIndex}, {testIndex})");

                # Reading the predicted testing pelvis structure
                pdPelvisBoneMuscleMesh = sp.readMesh(predTestingPelvisStructureFolder + f"/PredictedPelvicMuscle_{featureSelIndex}_{validIndex}_{testIndex}.ply");
                pdPelvisBoneMesh = sp.formMesh(pdPelvisBoneMuscleMesh.vertices[pelvisBoneVertexIndices], templatePelvisBoneMesh.faces);

                # Reading ground truth information
                gtPersonalizedBoneMuscleMesh = sp.readMesh(femalePelvisFolder + f"/AllPelvicStructures/{testingIDs[testIndex]}-PersonalizedPelvisBoneMuscleMesh.ply");
                gtPersonalizedBoneVertices = gtPersonalizedBoneMuscleMesh.vertices[pelvisBoneVertexIndices];
                gtPersonalizedBoneMesh = sp.formMesh(gtPersonalizedBoneVertices, templatePelvisBoneMesh.faces);
                gtBoneMesh = sp.readMesh(femalePelvisFolder + f"/AllPelvicStructures/{testingIDs[testIndex]}-PelvisBoneMesh.ply");
                gtPelvisBoneFeatures = sp.reconstructLandmarksFromBarycentric(gtPersonalizedBoneMesh, pelvisBoneFeatureBaryIndices, pelvisBoneFeatureBaryCoords);

                # Compute predicted information
                svdTransform = sp.estimateRigidSVDTransform(pdPelvisBoneMuscleMesh.vertices, gtPersonalizedBoneMuscleMesh.vertices);
                pdPelvisBoneMuscleMesh = sp.transformMesh(pdPelvisBoneMuscleMesh, svdTransform);
                pdPelvisBoneFeatures = sp.reconstructLandmarksFromBarycentric(pdPelvisBoneMesh, pelvisBoneFeatureBaryIndices, pelvisBoneFeatureBaryCoords);
                
                # Compute distances from predicted pelvis mesh to ground truth pelvis mesh
                averageMeshDistance = sp.computeAveragePointsToPointsDistance(pdPelvisBoneMesh.vertices, gtBoneMesh.vertices);

                # Compute distances from predicted pelvis features to ground truth pelvis features
                featureDistances = sp.computeCorrespondingDistancesPoints2Points(pdPelvisBoneFeatures, gtPelvisBoneFeatures);
                averageFeatureDistance = np.mean(featureDistances);

                # Compute distances in muscle attachment points
                nearestGtBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(pdPelvisBoneMesh.vertices, gtBoneMesh.vertices);
                nearestGtBoneVertices = gtBoneMesh.vertices[nearestGtBoneVertexIndices];
                correspondingDistances = sp.computeCorrespondingDistancesPoints2Points(pdPelvisBoneMesh.vertices, nearestGtBoneVertices);
                muscleAttachmentDistances = correspondingDistances[muscleAttachmentIndices];
                averageMuscleAttachmentDistance = np.mean(muscleAttachmentDistances);

                # Compute vertex 2 vertex distances
                nearestMeshVertices = sp.estimateNearestPointsFromPoints(pdPelvisBoneMesh.vertices, gtPersonalizedBoneVertices);
                vertex2VertexDistances = sp.computeCorrespondingDistancesPoints2Points(pdPelvisBoneMesh.vertices, nearestMeshVertices);

                # Compute vertex 2 vertex mesh distances
                nearestMeshVertices = sp.estimateNearestPointsFromPoints(pdPelvisBoneMesh.vertices, gtBoneMesh.vertices);
                vertex2VertexMeshDistances = sp.computeCorrespondingDistancesPoints2Points(pdPelvisBoneMesh.vertices, nearestMeshVertices);

                # Compute vertex 2 vertex bone muscle distances
                nearestBoneMuscleVertices = sp.estimateNearestPointsFromPoints(pdPelvisBoneMuscleMesh.vertices, gtPersonalizedBoneMuscleMesh.vertices);
                vertex2VertexBoneMuscleDistances = sp.computeCorrespondingDistancesPoints2Points(pdPelvisBoneMuscleMesh.vertices, nearestBoneMuscleVertices);

                # Save the computed errors to buffers
                meshTestingErrors[validIndex, testIndex] = averageMeshDistance;
                featureTestingErrors[validIndex, testIndex] = averageFeatureDistance;
                muscleAttachmentTestingErrors[validIndex, testIndex] = averageMuscleAttachmentDistance;
                vertex2VertexDistanceBuffer.append(vertex2VertexDistances);
                vertex2VertexMeshDistanceBuffer.append(vertex2VertexMeshDistances);
                vertex2VertexBoneMuscleDistanceBuffer.append(vertex2VertexBoneMuscleDistances);
        
        # Convert the buffer to numpy array
        print("Converting the buffer to numpy array ...");
        vertex2VertexDistanceBuffer = np.array(vertex2VertexDistanceBuffer);
        vertex2VertexMeshDistanceBuffer = np.array(vertex2VertexMeshDistanceBuffer);
        vertex2VertexBoneMuscleDistanceBuffer = np.array(vertex2VertexBoneMuscleDistanceBuffer);

        # Save the computed errors to files
        print("Saving the computed errors to files ...");
        sp.saveMatrixToCSVFile(meshFeatureMuscleErrorFolder + f"/MeshTestingErrors_{featureSelIndex}.csv", meshTestingErrors);
        sp.saveMatrixToCSVFile(meshFeatureMuscleErrorFolder + f"/FeatureTestingErrors_{featureSelIndex}.csv", featureTestingErrors);
        sp.saveMatrixToCSVFile(meshFeatureMuscleErrorFolder + f"/MuscleAttachmentTestingErrors_{featureSelIndex}.csv", muscleAttachmentTestingErrors);
        sp.saveNumPyArrayToNPY(optimalTrainValidTestFolder + f"/Vertex2VertexDistances_{featureSelIndex}.npy", vertex2VertexDistanceBuffer);
        sp.saveNumPyArrayToNPY(meshFeatureMuscleErrorFolder + f"/Vertex2VertexMeshDistances_{featureSelIndex}.npy", vertex2VertexMeshDistanceBuffer);
        sp.saveNumPyArrayToNPY(meshFeatureMuscleErrorFolder + f"/Vertex2VertexBoneMuscleDistances_{featureSelIndex}.npy", vertex2VertexBoneMuscleDistanceBuffer);

    # Finished processing
    print("Finished processing.");
def featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor_drawTestingErrorsInVariousMetrics():
    """
    Analyzes and visualizes testing errors from multi-output ridge regression pelvis reconstruction across various error metrics.
    
    This function evaluates the final testing performance of the optimal pelvis bone-muscle reconstruction model
    by loading testing error data across different metrics (mesh, feature, muscle attachment) and generating 
    comprehensive statistical visualizations to assess model accuracy and feature importance relationships.
    
    Key Steps:
    1. Initialize file paths and load feature selection strategy configurations
    2. Load testing error data for multiple error metrics across all feature strategies and validation folds
    3. Calculate the number of features for each feature selection strategy
    4. For each error metric type:
       - Aggregate testing errors and compute statistical measures (mean, standard deviation)
       - Generate bar chart visualization with error bars and trend analysis
       - Perform polynomial fitting to identify error trends with feature count
       - Calculate theoretical zero-error point for feature requirement estimation
       - Generate vertex-level color maps on 3D mesh models for spatial error visualization
    5. Save visualization results for publication and analysis
    
    Error Metrics Analyzed:
    - Mesh Testing Errors: Point-to-point distances between predicted and CT bone meshes
    - Feature Testing Errors: Accuracy of anatomical feature point reconstruction
    - Muscle Attachment Errors: Accuracy at critical muscle attachment locations
    - Vertex-to-Vertex Distances: Comprehensive vertex-level accuracy on personalized meshes
    - Vertex-to-Mesh Distances: Accuracy against CT-reconstructed bone surfaces
    - Vertex-to-BoneMuscle Distances: Overall bone-muscle structure accuracy
    
    Data Processing:
    - Reads testing errors from CSV and NPY files organized by feature strategy and validation fold
    - Processes errors across all available feature selection strategies (0 to max)
    - Computes mean and standard deviation across 10-fold cross-validation
    - Converts measurements from meters to millimeters for clinical presentation
    
    Statistical Analysis:
    - Mean testing error computation across cross-validation folds
    - Standard deviation calculation for error bar representation
    - Linear polynomial fitting to identify feature count vs error relationship
    - Zero-error point estimation to determine theoretical feature requirements
    
    Visualization Functions:
    - drawMeshTestingErrorsOnCTPelvisMeshes(): Bar chart analysis for CT mesh accuracy
    - drawFeatureTestingErrorsOnCTPelvisMeshes(): Bar chart analysis for feature point accuracy
    - drawMuscleAttachmentTestingErrorsOnCTPelvisMeshes(): Bar chart analysis for muscle attachment accuracy
    - drawVertex2VertexDistanceColorMapOnPersonalizedPelvisMeshes(): 3D color map on personalized meshes
    - drawVertex2VertexDistanceColorMapOnCTReconstructedPelvisMeshes(): 3D color map on CT meshes
    - drawVertex2VertexDistanceColorMapOnPelvisBoneMuscleMeshes(): 3D color map on bone-muscle meshes
    
    Visualization Outputs:
    - Bar charts showing mean testing errors with standard deviation error bars
    - Precise error value annotations on each bar for quantitative assessment
    - Polynomial trend lines with equations and zero-error point estimation
    - 3D color-coded mesh visualizations showing spatial error distribution
    - Color rulers and legends for error magnitude interpretation
    - Professional formatting with bold fonts and clear axis labeling
    
    Feature Analysis:
    - Maps feature selection strategy indices to actual feature counts
    - Analyzes relationship between number of anatomical features and reconstruction accuracy
    - Provides insights into optimal feature set size for clinical applications
    - Enables determination of minimum feature requirements for acceptable accuracy
    
    Output Files:
    - High-resolution PNG charts saved for publication use
    - 3D mesh visualizations with embedded error color maps
    - Statistical summary data for performance reporting
    - Trend analysis results for feature optimization guidance
    
    Note: This comprehensive function provides multiple perspectives on reconstruction accuracy,
    enabling detailed assessment of model performance across different anatomical regions and
    clinical applications. The combination of statistical charts and 3D visualizations offers
    both quantitative metrics and intuitive spatial understanding of prediction errors.
    """

    # Initializing
    print("Initializing ...");
    mainFolder = disk + "/SpinalPelvisPred";
    pelvisReconFolder = mainFolder + "/Data/PelvisBoneRecon";
    crossValidationFolder = pelvisReconFolder + "/CrossValidation";
    featureSelectionProtocolFolder = mainFolder + "/Data/PelvisBoneRecon/CrossValidation/FeatureSelectionProtocol";
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");
    meshFeatureMuscleErrorFolder = crossValidationFolder + "/ShapeRelationStrategy/OptimalTrainValidTest/MeshFeatureMuscleErrors";
    optimalTrainValidTestFolder = crossValidationFolder + "/ShapeRelationStrategy/OptimalTrainValidTest";
    numOfValids = 10; numOfTests = 30; numOfStrategies = len(featureSelectionStrategyDict);

    # Reading testing errors in various metrics
    print("Reading testing errors in various metrics ...");
    meshTestingErrors = np.zeros((numOfStrategies, numOfValids, numOfTests));
    featureTestingErrors = np.zeros((numOfStrategies, numOfValids, numOfTests));
    muscleAttachmentTestingErrors = np.zeros((numOfStrategies, numOfValids, numOfTests));
    for featureSelIndex in range(numOfStrategies):
        meshTestingErrors[featureSelIndex] = sp.readMatrixFromCSVFile(meshFeatureMuscleErrorFolder + f"/MeshTestingErrors_{featureSelIndex}.csv");
        featureTestingErrors[featureSelIndex] = sp.readMatrixFromCSVFile(meshFeatureMuscleErrorFolder + f"/FeatureTestingErrors_{featureSelIndex}.csv");
        muscleAttachmentTestingErrors[featureSelIndex] = sp.readMatrixFromCSVFile(meshFeatureMuscleErrorFolder + f"/MuscleAttachmentTestingErrors_{featureSelIndex}.csv");
    
    # Estimate the number of features in each feature selection strategy
    print("Estimating the number of features in each feature selection strategy ...");
    numOfStrategyFeatures = [];
    for featureSelIndex in range(numOfStrategies):
        featureSelectionIndices = list(featureSelectionStrategyDict.values())[featureSelIndex];
        numOfStrategyFeatures.append(len(featureSelectionIndices));
    numOfStrategyFeatures = np.array(numOfStrategyFeatures);
    
    # Draw the mesh testing errors.
    print("Draw the mesh testing errors ...");
    def drawMeshTestingErrorsOnCTPelvisMeshes():
        ## Forming drawing data
        meshMeans = np.mean(meshTestingErrors, axis=(1, 2)) * 1000;  # Convert to mm
        meshStds = np.std(meshTestingErrors, axis=(1, 2)) * 1000;  # Convert to mm
        ## Define label as the number of features in each feature selection strategy
        labels = [f"{len(list(featureSelectionStrategyDict.values())[i])}" for i in range(numOfStrategies)];
        ## Set x locations and bar width
        xLocations = np.arange(numOfStrategies);
        barWidth = 0.75;
        ## Plotting the bar chart
        fig, ax = plt.subplots(figsize=(12, 6));
        meshBars = ax.bar(xLocations, meshMeans, width=barWidth, yerr=meshStds, capsize=5, label='Mesh Errors');
        ## Add value labels on top of each bar
        for bar, val in zip(meshBars, meshMeans):
            height = bar.get_height();
            ax.annotate(f'{val:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom');
        ## Draw the poly fit line using linear regression line
        poly_fit = np.polyfit(xLocations, meshMeans, 1);
        poly_fit_line = np.polyval(poly_fit, xLocations);
        ## Draw the poly fit line
        line = ax.plot(xLocations, poly_fit_line, color='red', linestyle='--', linewidth=2, label='Poly Fit Line')[0];
        ## Find the zero error point by fitting another line with real number of feature points
        realPolyFit = np.polyfit(numOfStrategyFeatures[xLocations], meshMeans, 1);
        ## Compute the zero error point when the mean error reach zero
        zero_error_x = -realPolyFit[1] / realPolyFit[0];  # x when y = 0
        zero_error_y = np.polyval(realPolyFit, zero_error_x);
        ## Show the zero error point in the legend
        ax.legend([meshBars, line],
                [f'Mean Testing Errors (mm)', f'Poly Fit Line (y = {realPolyFit[0]:.3f}x + {realPolyFit[1]:.3f})\nZero Error Point: ({zero_error_x:.2f}, {zero_error_y:.2f})'],
                loc='upper right', fontsize=12);
        ## Axis labels and ticks
        ax.set_ylabel('Testing Errors (mm)', fontsize=14, fontweight='bold');
        ax.set_xlabel('Number of Features in Feature Selection Strategies', fontsize=14, fontweight='bold');
        ax.set_xticks(xLocations);
        ax.set_xticklabels(labels, fontsize=12, fontweight='bold', ha='center');
        ax.set_yticks(ax.get_yticks());
        ax.set_yticklabels([f'{tick:.1f}' for tick in ax.get_yticks()], fontsize=13, fontweight='bold');
        ## No grid
        ax.grid(False);
        ## Adding title
        ax.set_title('Testing Errors in CT-Reconstructed Pelvis Mesh', fontsize=16, fontweight='bold');
        ## Layout
        plt.tight_layout();
        plt.show();
        ## Save the figure
        fig.savefig(crossValidationFolder + "/ShapeRelationStrategy/OptimalTrainValidTest/TestingErrorsInPelvisMesh.png", dpi=400, bbox_inches='tight');
    def drawFeatureTestingErrorsOnCTPelvisMeshes():
        ## Forming drawing data
        featureMeans = np.mean(featureTestingErrors, axis=(1, 2)) * 1000;  # Convert to mm
        featureStds = np.std(featureTestingErrors, axis=(1, 2)) * 1000;  # Convert to mm
        ## Define label as the number of features in each feature selection strategy
        labels = [f"{len(list(featureSelectionStrategyDict.values())[i])}" for i in range(numOfStrategies)];
        ## Set x locations and bar width
        xLocations = np.arange(numOfStrategies);
        barWidth = 0.75;
        ## Plotting the bar chart
        fig, ax = plt.subplots(figsize=(12, 6));
        meshBars = ax.bar(xLocations, featureMeans, width=barWidth, yerr=featureStds, capsize=5, label='Mesh Errors');
        ## Add value labels on top of each bar
        for bar, val in zip(meshBars, featureMeans):
            height = bar.get_height();
            ax.annotate(f'{val:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom');
        ## Draw the poly fit line using linear regression line
        poly_fit = np.polyfit(xLocations, featureMeans, 1);
        poly_fit_line = np.polyval(poly_fit, xLocations);
        ## Draw the poly fit line
        line = ax.plot(xLocations, poly_fit_line, color='red', linestyle='--', linewidth=2, label='Poly Fit Line')[0];
        ## Find the zero error point by fitting another line with real number of feature points
        realPolyFit = np.polyfit(numOfStrategyFeatures[xLocations], featureMeans, 1);
        ## Compute the zero error point when the mean error reach zero
        zero_error_x = -realPolyFit[1] / realPolyFit[0];  # x when y = 0
        zero_error_y = np.polyval(realPolyFit, zero_error_x);
        ## Show the zero error point in the legend
        ax.legend([meshBars, line],
                [f'Mean Testing Errors (mm)', f'Poly Fit Line (y = {realPolyFit[0]:.3f}x + {realPolyFit[1]:.3f})\nZero Error Point: ({zero_error_x:.2f}, {zero_error_y:.2f})'],
                loc='upper right', fontsize=12);
        ## Axis labels and ticks
        ax.set_ylabel('Testing Errors (mm)', fontsize=14, fontweight='bold');
        ax.set_xlabel('Number of Features in Feature Selection Strategies', fontsize=14, fontweight='bold');
        ax.set_xticks(xLocations);
        ax.set_xticklabels(labels, fontsize=12, fontweight='bold', ha='center');
        ax.set_yticks(ax.get_yticks());
        ax.set_yticklabels([f'{tick:.1f}' for tick in ax.get_yticks()], fontsize=13, fontweight='bold');
        ## No grid
        ax.grid(False);
        ## Adding title
        ax.set_title('Testing Errors in Pelvis Features', fontsize=16, fontweight='bold');
        ## Layout
        plt.tight_layout();
        plt.show();
        ## Save the figure
        fig.savefig(crossValidationFolder + "/ShapeRelationStrategy/OptimalTrainValidTest/TestingErrorsInPelvisFeatures.png", dpi=400, bbox_inches='tight');
    def drawMuscleAttachmentTestingErrorsOnCTPelvisMeshes():
        ## Forming drawing data
        muscleMeans = np.mean(muscleAttachmentTestingErrors, axis=(1, 2)) * 1000;  # Convert to mm
        muscleStds = np.std(muscleAttachmentTestingErrors, axis=(1, 2)) * 1000;  # Convert to mm
        ## Define label as the number of features in each feature selection strategy
        labels = [f"{len(list(featureSelectionStrategyDict.values())[i])}" for i in range(numOfStrategies)];
        ## Set x locations and bar width
        xLocations = np.arange(numOfStrategies);
        barWidth = 0.75;
        ## Plotting the bar chart
        fig, ax = plt.subplots(figsize=(12, 6));
        meshBars = ax.bar(xLocations, muscleMeans, width=barWidth, yerr=muscleStds, capsize=5, label='Mesh Errors');
        ## Add value labels on top of each bar
        for bar, val in zip(meshBars, muscleMeans):
            height = bar.get_height();
            ax.annotate(f'{val:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom');
        ## Draw the poly fit line using linear regression line
        poly_fit = np.polyfit(xLocations, muscleMeans, 1);
        poly_fit_line = np.polyval(poly_fit, xLocations);
        ## Draw the poly fit line
        line = ax.plot(xLocations, poly_fit_line, color='red', linestyle='--', linewidth=2, label='Poly Fit Line')[0];
        ## Find the zero error point by fitting another line with real number of feature points
        realPolyFit = np.polyfit(numOfStrategyFeatures[xLocations], muscleMeans, 1);
        ## Compute the zero error point when the mean error reach zero
        zero_error_x = -realPolyFit[1] / realPolyFit[0];  # x when y = 0
        zero_error_y = np.polyval(realPolyFit, zero_error_x);
        ## Show the zero error point in the legend
        ax.legend([meshBars, line],
                [f'Mean Testing Errors (mm)', f'Poly Fit Line (y = {realPolyFit[0]:.3f}x + {realPolyFit[1]:.3f})\nZero Error Point: ({zero_error_x:.2f}, {zero_error_y:.2f})'],
                loc='upper right', fontsize=12);
        ## Axis labels and ticks
        ax.set_ylabel('Testing Errors (mm)', fontsize=14, fontweight='bold');
        ax.set_xlabel('Number of Features in Feature Selection Strategies', fontsize=14, fontweight='bold');
        ax.set_xticks(xLocations);
        ax.set_xticklabels(labels, fontsize=12, fontweight='bold', ha='center');
        ax.set_yticks(ax.get_yticks());
        ax.set_yticklabels([f'{tick:.1f}' for tick in ax.get_yticks()], fontsize=13, fontweight='bold');
        ## No grid
        ax.grid(False);
        ## Adding title
        ax.set_title('Testing Errors in Muscle Attachment Points', fontsize=16, fontweight='bold');
        ## Layout
        plt.tight_layout();
        plt.show();
        ## Save the figure
        fig.savefig(crossValidationFolder + "/ShapeRelationStrategy/OptimalTrainValidTest/TestingErrorsInMuscleAttachmentPoints.png", dpi=400, bbox_inches='tight');
    def drawVertex2VertexDistanceColorMapOnPersonalizedPelvisMeshes():
        # Forming drawing data
        print("Reading testing errors for vertex to vertex distances ...");
        vertex2VertexDistances = sp.loadNumPYArrayFromNPY(optimalTrainValidTestFolder + f"/Vertex2VertexDistances_{20}.npy");

        # Reading template pelvis bone mesh for drawing color map
        print("Reading template pelvis bone mesh for drawing color map ...");
        templatePelvisBoneMesh = sp.readMesh(mainFolder + "/Data/Template/PelvisBonesMuscles/TempPelvisBoneMesh.ply");

        # Compute the average vertex to vertex distances
        print("Computing the average vertex to vertex distances ...");
        meanTestingError = np.mean(vertex2VertexDistances, axis=0);

        # Draw the color map of the testing errors as red (lowest) to violet (highest) on the vertex of the skull shape
        print("Drawing the color map of the testing errors ...");
        ## Normalize the mean testing error to [0, 1] range
        normalizedMeanTestingError = (meanTestingError - np.min(meanTestingError)) / (np.max(meanTestingError) - np.min(meanTestingError));
        ## Create a custom colormap from red (low) to violet (high)
        from matplotlib.colors import LinearSegmentedColormap
        colors_list = ['red', 'orange', 'yellow', 'green', 'blue', 'violet']
        colormap = LinearSegmentedColormap.from_list('red_to_violet', colors_list)
        ## Map the normalized mean testing error to colors
        colors = colormap(normalizedMeanTestingError);
        ## Set the colors to the vertices of the template pelvis bone mesh using trimesh and visualize it
        templatePelvisBoneMesh.visual.vertex_colors = colors[:, :3] * 255;  # Convert to 0-255 range
        trimeshViewer = trimesh.Scene(templatePelvisBoneMesh);
        trimeshViewer.show();

        # Generate and display color ruler
        print("Generating color ruler ...");
        ## Create a figure for the color ruler
        fig, ax = plt.subplots(figsize=(8, 2))    
        ## Create a gradient for the colorbar
        gradient = np.linspace(0, 1, 256).reshape(1, -1)    
        ## Display the gradient
        im = ax.imshow(gradient, aspect='auto', cmap=colormap, extent=[0, 1, 0, 1])    
        ## Calculate the actual error values for min, middle, and max
        min_error = np.min(meanTestingError) * 1000;  # Convert to mm
        max_error = np.max(meanTestingError) * 1000;  # Convert to mm
        mid_error = (min_error + max_error) / 2;
        ## Set up the colorbar ticks and labels
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels([f'{min_error:.2f} mm\n(Min)', f'{mid_error:.2f} mm\n(Middle)', f'{max_error:.2f} mm\n(Max)'])
        ax.set_yticks([])
        ax.set_xlabel('Prediction Error', fontsize=12, fontweight='bold')
        ax.set_title('Testing Vertex-to-Vertex Distances on Personalized Pelvis Meshes\n(Mean ± Std = {:.2f} ± {:.2f} mm)'.format(np.mean(meanTestingError) * 1000, np.std(meanTestingError) * 1000), fontsize=14, fontweight='bold')    
        ## Add grid lines for better readability
        ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1)    
        ## Adjust layout and display
        plt.tight_layout()
        ## Save the color ruler figure
        plt.savefig(optimalTrainValidTestFolder + "/ColorMapLegend_PelvisPredictionErrors.png", bbox_inches='tight', dpi=300)
        ## Show the color ruler
        plt.show()

        # Finished processing
        print("Finished processing.");
    def drawVertex2VertexDistanceColorMapOnCTReconstructedPelvisMeshes():
        # Forming drawing data
        print("Reading testing errors for vertex to vertex distances ...");
        vertex2VertexDistances = sp.loadNumPYArrayFromNPY(optimalTrainValidTestFolder + f"/Vertex2VertexMeshDistances_{20}.npy");

        # Reading template pelvis bone mesh for drawing color map
        print("Reading template pelvis bone mesh for drawing color map ...");
        templatePelvisBoneMesh = sp.readMesh(mainFolder + "/Data/Template/PelvisBonesMuscles/TempPelvisBoneMesh.ply");

        # Compute the average vertex to vertex distances
        print("Computing the average vertex to vertex distances ...");
        meanTestingError = np.mean(vertex2VertexDistances, axis=0);

        # Draw the color map of the testing errors as red (lowest) to violet (highest) on the vertex of the skull shape
        print("Drawing the color map of the testing errors ...");
        ## Normalize the mean testing error to [0, 1] range
        normalizedMeanTestingError = (meanTestingError - np.min(meanTestingError)) / (np.max(meanTestingError) - np.min(meanTestingError));
        ## Create a custom colormap from red (low) to violet (high)
        from matplotlib.colors import LinearSegmentedColormap
        colors_list = ['red', 'orange', 'yellow', 'green', 'blue', 'violet']
        colormap = LinearSegmentedColormap.from_list('red_to_violet', colors_list)
        ## Map the normalized mean testing error to colors
        colors = colormap(normalizedMeanTestingError);
        ## Set the colors to the vertices of the template pelvis bone mesh using trimesh and visualize it
        templatePelvisBoneMesh.visual.vertex_colors = colors[:, :3] * 255;  # Convert to 0-255 range
        trimeshViewer = trimesh.Scene(templatePelvisBoneMesh);
        trimeshViewer.show();

        # Generate and display color ruler
        print("Generating color ruler ...");
        ## Create a figure for the color ruler
        fig, ax = plt.subplots(figsize=(8, 2))    
        ## Create a gradient for the colorbar
        gradient = np.linspace(0, 1, 256).reshape(1, -1)    
        ## Display the gradient
        im = ax.imshow(gradient, aspect='auto', cmap=colormap, extent=[0, 1, 0, 1])    
        ## Calculate the actual error values for min, middle, and max
        min_error = np.min(meanTestingError) * 1000;  # Convert to mm
        max_error = np.max(meanTestingError) * 1000;  # Convert to mm
        mid_error = (min_error + max_error) / 2;
        ## Set up the colorbar ticks and labels
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels([f'{min_error:.2f} mm\n(Min)', f'{mid_error:.2f} mm\n(Middle)', f'{max_error:.2f} mm\n(Max)'])
        ax.set_yticks([])
        ax.set_xlabel('Prediction Error', fontsize=12, fontweight='bold')
        ax.set_title('Testing Vertex-to-Vertex Distances on CT-reconstructed Pelvis Meshes\n(Mean ± Std = {:.2f} ± {:.2f} mm)'.format(np.mean(meanTestingError) * 1000, np.std(meanTestingError) * 1000), fontsize=14, fontweight='bold')    
        ## Add grid lines for better readability
        ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1)    
        ## Adjust layout and display
        plt.tight_layout()
        ## Save the color ruler figure
        plt.savefig(optimalTrainValidTestFolder + "/ColorMapLegend_PelvisPredictionErrors.png", bbox_inches='tight', dpi=300)
        ## Show the color ruler
        plt.show()

        # Finished processing
        print("Finished processing.");
    def drawVertex2VertexDistanceColorMapOnPelvisBoneMuscleMeshes():
        # Forming drawing data
        print("Reading testing errors for vertex to vertex distances ...");
        vertex2VertexDistances = sp.loadNumPYArrayFromNPY(optimalTrainValidTestFolder + f"/Vertex2VertexBoneMuscleDistances_{20}.npy");

        # Reading template pelvis bone mesh for drawing color map
        print("Reading template pelvis bone mesh for drawing color map ...");
        templatePelvisBoneMuscleMesh = sp.readMesh(mainFolder + "/Data/Template/PelvisBonesMuscles/TempPelvisBoneMuscles.ply");

        # Compute the average vertex to vertex distances
        print("Computing the average vertex to vertex distances ...");
        meanTestingError = np.mean(vertex2VertexDistances, axis=0);

        # Draw the color map of the testing errors as red (lowest) to violet (highest) on the vertex of the skull shape
        print("Drawing the color map of the testing errors ...");
        ## Normalize the mean testing error to [0, 1] range
        normalizedMeanTestingError = (meanTestingError - np.min(meanTestingError)) / (np.max(meanTestingError) - np.min(meanTestingError));
        ## Create a custom colormap from red (low) to violet (high)
        from matplotlib.colors import LinearSegmentedColormap
        colors_list = ['red', 'orange', 'yellow', 'green', 'blue', 'violet']
        colormap = LinearSegmentedColormap.from_list('red_to_violet', colors_list)
        ## Map the normalized mean testing error to colors
        colors = colormap(normalizedMeanTestingError);
        ## Set the colors to the vertices of the template pelvis bone mesh using trimesh and visualize it
        templatePelvisBoneMuscleMesh.visual.vertex_colors = colors[:, :3] * 255;  # Convert to 0-255 range
        trimeshViewer = trimesh.Scene(templatePelvisBoneMuscleMesh);
        trimeshViewer.show();

        # Generate and display color ruler
        print("Generating color ruler ...");
        ## Create a figure for the color ruler
        fig, ax = plt.subplots(figsize=(8, 2))    
        ## Create a gradient for the colorbar
        gradient = np.linspace(0, 1, 256).reshape(1, -1)    
        ## Display the gradient
        im = ax.imshow(gradient, aspect='auto', cmap=colormap, extent=[0, 1, 0, 1])    
        ## Calculate the actual error values for min, middle, and max
        min_error = np.min(meanTestingError) * 1000;  # Convert to mm
        max_error = np.max(meanTestingError) * 1000;  # Convert to mm
        mid_error = (min_error + max_error) / 2;
        ## Set up the colorbar ticks and labels
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels([f'{min_error:.2f} mm\n(Min)', f'{mid_error:.2f} mm\n(Middle)', f'{max_error:.2f} mm\n(Max)'])
        ax.set_yticks([])
        ax.set_xlabel('Prediction Error', fontsize=12, fontweight='bold')
        ax.set_title('Testing Vertex-to-Vertex Distances on Personalized Pelvis Bone Muscle Meshes\n(Mean ± Std = {:.2f} ± {:.2f} mm)'.format(np.mean(meanTestingError) * 1000, np.std(meanTestingError) * 1000), fontsize=14, fontweight='bold')    
        ## Add grid lines for better readability
        ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1)    
        ## Adjust layout and display
        plt.tight_layout()
        ## Save the color ruler figure
        plt.savefig(optimalTrainValidTestFolder + "/ColorMapLegend_PelvisPredictionErrors.png", bbox_inches='tight', dpi=300)
        ## Show the color ruler
        plt.show()

        # Finished processing
        print("Finished processing.");
    
    # Running the functions
    print("Running the functions ...");
    drawMuscleAttachmentTestingErrorsOnCTPelvisMeshes();

    # Finished processing
    print("Finished processing.");
def featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor_estimateBestAndWorstPredictedCases():
    """
    Identifies the best and worst performing test cases from multi-output ridge regression pelvis reconstruction results.
    
    This function analyzes testing error data across all validation folds to identify the specific test cases
    that achieved the lowest (best) and highest (worst) reconstruction errors. This analysis helps in understanding
    model performance extremes and enables targeted case study analysis for method improvement.
    
    Key Steps:
    1. Load testing error data from all validation folds using feature selection strategy 20 (38 features)
    2. Aggregate testing errors across multiple cross-validation folds into a single matrix
    3. Identify the global minimum and maximum error values across all test cases
    4. Determine the validation fold and subject indices corresponding to best/worst performance
    5. Map these indices to actual subject IDs using cross-validation split files
    6. Report the specific cases for detailed analysis and visualization
    
    Data Processing:
    - Reads testing errors from CSV files organized by validation fold
    - Processes 10 validation folds with 30 test subjects each (300 total test cases)
    - Uses feature selection strategy 20 which contains 38 anatomical feature points
    - Reshapes error matrix for efficient global minimum/maximum identification
    
    Analysis Strategy:
    - Global optimization across all test cases to find performance extremes
    - Cross-validation fold mapping to maintain data traceability
    - Subject ID resolution for case-specific analysis and visualization
    
    Output Information:
    - Best case: Validation fold index, subject index, error value, and subject ID
    - Worst case: Validation fold index, subject index, error value, and subject ID
    - Enables targeted analysis of model performance characteristics
    - Provides specific cases for detailed visualization and error analysis
    
    Statistical Summary:
    - Reports minimum and maximum reconstruction errors across all test cases
    - Identifies specific validation configurations that produced extreme results
    - Enables understanding of model performance distribution and outliers
    
    Use Cases:
    - Performance analysis: Understanding model capabilities and limitations
    - Case study selection: Identifying representative examples for detailed analysis
    - Method improvement: Analyzing failure modes and success patterns
    - Visualization preparation: Selecting cases for 3D mesh comparison and error mapping
    
    Note: This function focuses on feature selection strategy 20 (38 features) as it represents
    an optimal balance between feature richness and reconstruction accuracy based on validation results.
    The identified cases can be used for detailed visualization, error analysis, and method comparison.
    """

    # Initializing
    print("Initializing ...");
    testingErrorFolder = disk + r"\SpinalPelvisPred\Data\PelvisBoneRecon\CrossValidation\ShapeRelationStrategy\OptimalTrainValidTest\TestingErrors";
    trainTestSplitFolder = disk + r"\SpinalPelvisPred\Data\PelvisBoneRecon\CrossValidation\ShapeRelationStrategy\OptimalTrainValidTest\TrainTestSplits";
    numOfValids = 10;

    # Reading testing errors in various validation with the feature of 38
    print("Reading testing errors in various validation with the feature of 38 ...");
    testingErrors = [];
    for v in range(numOfValids):
        testingErrorFilePath = testingErrorFolder + f"/AveragePoint2PointDistances_20_{v}.csv";
        testingError = sp.readMatrixFromCSVFile(testingErrorFilePath);
        testingErrors.append(testingError);
    testingErrors = np.array(testingErrors);
    testingErrors = testingErrors.reshape((numOfValids, -1));
    print("\t The shape of testingErrors: ", testingErrors.shape);

    # Estimate the best and worst predicted cases
    print("Estimating the best and worst predicted cases ...");
    min_flat_index = np.argmin(testingErrors);
    max_flat_index = np.argmax(testingErrors);
    min_i, min_j = np.unravel_index(min_flat_index, testingErrors.shape)
    max_i, max_j = np.unravel_index(max_flat_index, testingErrors.shape)

    # Report the values for finding the indices
    print("Report the values ...");
    print(f"Minimum testing error: {testingErrors[min_i, min_j]} (Validation {min_i}, Subject Index {min_j})");
    print(f"Maximum testing error: {testingErrors[max_i, max_j]} (Validation {max_i}, Subject Index {max_j})");

    # Determing the best and worst ID
    print("Determinging the best and worst ID ...");
    minTestIDs = sp.readListOfStrings(trainTestSplitFolder + f"/TestingIDs_{min_i}.txt");
    maxTestIDs = sp.readListOfStrings(trainTestSplitFolder + f"/TestingIDs_{max_i}.txt");
    minTestID = minTestIDs[min_j];
    maxTestID = maxTestIDs[max_j];
    print("\t The best predicted ID: ", minTestID);
    print("\t The worst predicted ID: ", maxTestID);

    # Finished processing
    print("Finished processing.");
def featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor_visualizeBestWorstPredictedCases():
    """
    Visualizes and compares the best and worst performing pelvis reconstruction cases from multi-output ridge regression results.
    
    This function provides detailed visualization and error analysis of the extreme performance cases identified
    from pelvis bone-muscle reconstruction experiments. It loads predicted meshes, ground truth data, and performs
    comprehensive error calculations with 3D visualization to understand model performance characteristics.
    
    Key Steps:
    1. Initialize file paths for best and worst predicted case data storage
    2. Load template pelvis bone-muscle mesh information and compute vertex mappings
    3. Load predicted and ground truth meshes for both best and worst performing cases:
       - Best case: Validation fold 0, subject index 17 (Subject ID: 257)
       - Worst case: Validation fold 5, subject index 19 (Subject ID: 1091)
    4. Perform rigid registration between predicted and personalized ground truth meshes
    5. Extract bone-only meshes from bone-muscle structures for CT comparison
    6. Compute comprehensive error metrics for both cases:
       - Predicted vs personalized bone-muscle mesh distances
       - Predicted vs CT bone mesh distances
       - Statistical measures (mean, standard deviation) in millimeters
    7. Generate 3D visualizations with color-coded meshes for comparative analysis
    
    Data Processing:
    - Uses pre-identified best/worst cases from feature selection strategy 20 (38 features)
    - Applies rigid SVD transformation for fair coordinate system alignment
    - Extracts bone vertices from bone-muscle structures for targeted comparison
    - Computes nearest point correspondences for accurate distance measurements
    
    Error Analysis:
    - Point-to-point distance calculations between predicted and ground truth meshes
    - Separate analysis for personalized (bone-muscle) and CT (bone-only) comparisons
    - Statistical reporting with mean and standard deviation in clinically relevant units
    - Quantitative assessment of reconstruction quality extremes
    
    Visualization Features:
    - Color-coded mesh rendering with distinct colors for predicted vs ground truth
    - Red coloring for predicted meshes to highlight reconstruction results
    - Bone-colored rendering for ground truth meshes for anatomical realism
    - Interactive 3D scene viewer for detailed inspection and comparison
    - Side-by-side visualization capability for direct comparison
    
    Output Metrics (Example Results):
    - Best case performance: Lower reconstruction errors demonstrating model capability
    - Worst case performance: Higher reconstruction errors indicating model limitations
    - Statistical summaries enable quantitative performance range assessment
    - Visual inspection reveals spatial error patterns and anatomical accuracy
    
    Use Cases:
    - Performance validation: Understanding model capability range and limitations
    - Case study analysis: Detailed examination of success and failure patterns
    - Method improvement: Identifying areas for algorithmic enhancement
    - Clinical assessment: Evaluating reconstruction quality for medical applications
    - Publication visualization: Generating figures for research communication
    
    Note: This function focuses on the extreme cases from a comprehensive evaluation
    using 38 anatomical features, providing insights into the best achievable accuracy
    and potential failure modes of the multi-output ridge regression approach for
    pelvis reconstruction from anatomical landmarks.
    """

    # Initialize
    print("Initializing ...");
    bestWorstPredictedFolder = r"I:\SpinalPelvisPred\Data\PelvisBoneRecon\CrossValidation\ShapeRelationStrategy\OptimalTrainValidTest\BestWorstPredictedCases";
    bestPredictedFolder = bestWorstPredictedFolder + "/BestPredictedCase";
    worstPredictedFolder = bestWorstPredictedFolder + "/WorstPredictedCase";
    templateFolder = r"I:\SpinalPelvisPred\Data\Template\PelvisBonesMuscles";

    # Reading the template information
    print("Reading the template information ...");
    templatePelvisBoneMuscleMesh = sp.readMesh(templateFolder + "/TempPelvisBoneMuscles.ply");
    templatePelvisBoneMesh = sp.readMesh(templateFolder + "/TempPelvisBoneMesh.ply");
    pelvisBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(templatePelvisBoneMesh.vertices, templatePelvisBoneMuscleMesh.vertices);

    # Reading information
    print("Reading information ...");
    bestPredictedBoneMuscleMesh = sp.readMesh(bestPredictedFolder + "/PredictedPelvicMuscle_20_0_17.ply");
    bestPersonalizedBoneMuscleMesh = sp.readMesh(bestPredictedFolder + "/257-PersonalizedPelvisBoneMuscleMesh.ply");
    bestGroundTruthBoneMesh = sp.readMesh(bestPredictedFolder + "/257-PelvisBoneMesh.ply");
    worstPredictedBoneMuscleMesh = sp.readMesh(worstPredictedFolder + "/PredictedPelvicMuscle_20_5_19.ply");
    worstPersonalizedBoneMuscleMesh = sp.readMesh(worstPredictedFolder + "/1091-PersonalizedPelvisBoneMuscleMesh.ply");
    worstGroundTruthBoneMesh = sp.readMesh(worstPredictedFolder + "/1091-PelvisBoneMesh.ply");

    # Registration the predicted mesh to the personalized mesh
    print("Registration the predicted mesh to the personalized mesh ...");
    bestSVDTransform = sp.estimateRigidSVDTransform(bestPredictedBoneMuscleMesh.vertices, bestPersonalizedBoneMuscleMesh.vertices);
    worstSVDTransform = sp.estimateRigidSVDTransform(worstPredictedBoneMuscleMesh.vertices, worstPersonalizedBoneMuscleMesh.vertices);
    bestPredictedBoneMuscleMesh = sp.transformMesh(bestPredictedBoneMuscleMesh, bestSVDTransform);
    worstPredictedBoneMuscleMesh = sp.transformMesh(worstPredictedBoneMuscleMesh, worstSVDTransform);
    bestPredictedPelvisBoneMesh = sp.cloneMesh(templatePelvisBoneMesh);
    worstPPredictedPelvisBoneMesh = sp.cloneMesh(templatePelvisBoneMesh);
    bestPredictedPelvisBoneMesh.vertices = bestPredictedBoneMuscleMesh.vertices[pelvisBoneVertexIndices];
    worstPPredictedPelvisBoneMesh.vertices = worstPredictedBoneMuscleMesh.vertices[pelvisBoneVertexIndices];

    # Computing some errors for illustration
    print("Computing some errors for illustration ...");
    ## Compute best bone muscle errors
    bestPersonalizedNearestVertices = sp.estimateNearestPointsFromPoints(bestPredictedBoneMuscleMesh.vertices, bestPersonalizedBoneMuscleMesh.vertices);
    bestPersonalizedDistances = sp.computeCorrespondingDistancesPoints2Points(bestPersonalizedBoneMuscleMesh.vertices, bestPersonalizedNearestVertices);
    bestPersonalizedMeanDistances = np.mean(bestPersonalizedDistances)*1000;
    bestPersonalizedStdDistances = np.std(bestPersonalizedDistances)*1000;
    print("\t Best personalized Mean +- SD: ", f"{bestPersonalizedMeanDistances} +- {bestPersonalizedStdDistances}");
    ## Compute best bone errors
    bestGroundTruthNearestVertices = sp.estimateNearestPointsFromPoints(bestPredictedPelvisBoneMesh.vertices, bestGroundTruthBoneMesh.vertices);
    bestGroundTruthDistances = sp.computeCorrespondingDistancesPoints2Points(bestPredictedPelvisBoneMesh.vertices, bestGroundTruthNearestVertices);
    bestGroundTruthMeanDistances = np.mean(bestGroundTruthDistances)*1000;
    bestGroundTruthStdDistances = np.std(bestGroundTruthDistances)*1000;
    print("\t Best ground truth Mean +- SD: ", f"{bestGroundTruthMeanDistances} +- {bestGroundTruthStdDistances}");
    ## Compute worst bone muscle errors
    worstPersonalizedNearestVertices = sp.estimateNearestPointsFromPoints(worstPredictedBoneMuscleMesh.vertices, worstPersonalizedBoneMuscleMesh.vertices);
    worstPersonalizedDistances = sp.computeCorrespondingDistancesPoints2Points(worstPredictedBoneMuscleMesh.vertices, worstPersonalizedNearestVertices);
    worstPersonalizedMeanDistances = np.mean(worstPersonalizedDistances)*1000;
    worstPersonalizedStdDistances = np.std(worstPersonalizedDistances)*1000;
    print("\t Worst personalized Mean +- SD: ", f"{worstPersonalizedMeanDistances} +- {worstPersonalizedStdDistances}");
    ## Compute worst bone errors
    worstGroundTruthNearestVertices = sp.estimateNearestPointsFromPoints(worstPPredictedPelvisBoneMesh.vertices, worstGroundTruthBoneMesh.vertices);
    worstGroundTruthDistances = sp.computeCorrespondingDistancesPoints2Points(worstPPredictedPelvisBoneMesh.vertices, worstGroundTruthNearestVertices);
    worstGroundTruthMeanDistances = np.mean(worstGroundTruthDistances)*1000;
    worstGroundTruthStdDistances = np.std(worstGroundTruthDistances)*1000;
    print("\t Worst ground truth Mean +- SD: ", f"{worstGroundTruthMeanDistances} +- {worstGroundTruthStdDistances}");

    # Visualize the mesh using the trimesh libraries
    print("Visualize the mesh using the trimesh library ...");
    ## Fill color for the predicted meshes as red color
    bestPredictedBoneMuscleMesh.visual.vertex_colors = [255, 0, 0, 255];  # Red color
    worstPredictedBoneMuscleMesh.visual.vertex_colors = [255, 0, 0, 255];  # Red color
    ## Fill the color of the personalized pelvis bone mesh as the bone color
    bestPersonalizedBoneMuscleMesh.visual.vertex_colors = [227, 218, 201, 200];  # White color
    worstPersonalizedBoneMuscleMesh.visual.vertex_colors = [227, 218, 201, 200];  # White color
    ## Fill the color of predicted pelvis bone mesh as red color
    bestPredictedPelvisBoneMesh.visual.vertex_colors = [255, 0, 0, 255];  # Red color
    worstPPredictedPelvisBoneMesh.visual.vertex_colors = [255, 0, 0, 255];  # Red color
    ## Fill the ground truth pelvis bone mesh as the bone color
    bestGroundTruthBoneMesh.visual.vertex_colors = [227, 218, 201, 200];  # White color
    worstGroundTruthBoneMesh.visual.vertex_colors = [227, 218, 201, 200];  # White color
    ## Visualize the predicted bone muscle with the personalized bone muscles using the trimesh 
    scene = trimesh.Scene([worstPPredictedPelvisBoneMesh, worstGroundTruthBoneMesh]);
    scene.show();

    # Finished processing
    print("Finished processing.");
def featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor_visualizeBestWorstPredictedCasesWithCTImageScans():
    """
    Visualizes and compares pelvis reconstruction results with CT image scans for comprehensive clinical assessment.
    
    This function provides advanced visualization capabilities by overlaying predicted pelvis bone-muscle structures
    onto actual CT image scans, enabling direct comparison between computational predictions and medical imaging data.
    The visualization combines 3D mesh reconstruction results with volumetric CT data for clinical validation.
    
    Key Steps:
    1. Parse command line argument for feature selection strategy index
    2. Initialize file paths for DICOM illustration data and template information
    3. Load template pelvis bone-muscle mesh and feature selection strategy configurations
    4. Load predicted pelvis bone-muscle mesh and corresponding anatomical features
    5. Load CT-reconstructed spino-pelvic bone mesh and torso structures
    6. Perform rigid registration between predicted and CT coordinate systems
    7. Scale meshes to match DICOM image coordinate system (millimeter units)
    8. Extract structural feature points from predicted mesh using barycentric coordinates
    9. Generate 3D sphere markers for anatomical feature visualization
    10. Load and process DICOM image series for volumetric rendering
    11. Create interactive 3D visualization combining:
        - Volumetric CT image rendering with tissue-specific opacity settings
        - Predicted pelvis bone-muscle mesh overlays
        - Anatomical feature point markers
        - Ground truth CT bone structures for comparison
    
    Command Line Arguments:
    - FeatureSelIndex: Index specifying which feature selection strategy to visualize
    
    Data Processing:
    - Uses ITK for DICOM image series reading and processing
    - Applies rigid SVD transformation for coordinate system alignment
    - Scales all 3D structures from meters to millimeters for DICOM compatibility
    - Reconstructs anatomical features using barycentric landmark interpolation
    
    Registration and Alignment:
    - Registers predicted pelvis to CT-reconstructed reference using feature correspondences
    - Ensures spatial alignment between computational predictions and medical imaging
    - Maintains anatomical accuracy through rigid transformation constraints
    
    Visualization Features:
    - Volumetric CT rendering with tissue-differentiated opacity:
      * Background: Completely transparent (opacity = 0.0)
      * Soft tissue: Very low opacity (opacity = 0.05) for context
      * Bone tissue: Moderate opacity (opacity = 0.3) for structural visibility
    - Color-coded anatomical structures:
      * CT bone structures: Natural bone color (RGB: 227, 218, 201)
      * Predicted structures: Overlaid for direct comparison
      * Feature points: Red spherical markers for landmark identification
    - Interactive 3D navigation with trackball camera controls
    - Professional medical imaging color schemes and opacity settings
    
    Technical Implementation:
    - ITK integration for medical image processing and DICOM handling
    - VTK pipeline for advanced 3D visualization and volume rendering
    - Trimesh to VTK conversion for mesh structure compatibility
    - Smart volume mapping with tissue-specific transfer functions
    
    Clinical Applications:
    - Validation of computational pelvis reconstruction against medical imaging
    - Assessment of prediction accuracy in clinical coordinate systems
    - Visual verification of anatomical feature correspondence
    - Quality control for reconstruction algorithms in medical applications
    - Pre-surgical planning and biomechanical analysis support
    
    Output Capabilities:
    - Interactive 3D scene with real-time manipulation
    - Simultaneous visualization of predicted and actual anatomical structures
    - Anatomical landmark verification through feature point overlays
    - Clinical-grade visualization suitable for medical assessment
    
    Use Cases:
    - Clinical validation of reconstruction algorithms
    - Medical imaging research and algorithm development
    - Biomechanical modeling verification
    - Surgical planning assistance
    - Educational demonstration of computational anatomy methods
    
    Note: This function represents the culmination of the reconstruction pipeline,
    providing clinical-grade visualization that enables direct assessment of
    computational predictions against medical imaging gold standards. The integration
    of CT imaging with mesh reconstruction offers comprehensive validation capabilities
    essential for medical applications.
    """
    
    # Initialize
    print("Initializing ...");
    if (len(sys.argv) < 2):
        print("Please input the command as: [ProgramName] [FeatureSelIndex]");
        return
    featureSelIndex = int(sys.argv[1]);
    mainFolder = disk + r"\SpinalPelvisPred\Data\PelvisBoneRecon";
    dicomIllustrationFolder = mainFolder + r"\CrossValidation\ShapeRelationStrategy\OptimalTrainValidTest\DICOMIllustration";
    templateFolder = disk + r"\SpinalPelvisPred\Data\Template\PelvisBonesMuscles";
    featureSelectionProtocolFolder = disk + r"\SpinalPelvisPred\Data\PelvisBoneRecon\CrossValidation\FeatureSelectionProtocol";

    # Reading template data
    print("Reading template data ...");
    templateBoneMuscleMesh = sp.readMesh(templateFolder + "/TempPelvisBoneMuscles.ply");
    allPelvicFeatures = sp.read3DPointsFromPPFile(featureSelectionProtocolFolder + "/AllFeaturePoints.pp");
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");
    featureSelectionIndices = list(featureSelectionStrategyDict.values())[featureSelIndex];
    tempPelvicFeatures = allPelvicFeatures[featureSelectionIndices];
    pelvicFeatureBaryIndices, pelvicFeatureBaryCoords = sp.computeBarycentricLandmarks(templateBoneMuscleMesh, tempPelvicFeatures);

    # Reading source and target meshes
    print("Reading predicted case ...");
    predictedPelvicBoneMuscleMesh = sp.readMesh(dicomIllustrationFolder + "/PredictedPelvicMuscle_20_0_0.ply");
    predictedPelvicBoneMuscleFeatures = sp.read3DPointsFromPPFile(dicomIllustrationFolder + "/PredictedPelvicMuscle_20_0_0_picked_points.pp");
    ctSpinoPelvicBoneMesh = sp.readMesh(dicomIllustrationFolder + "/138645-SpinopelvicBoneMesh.ply");
    ctPelvicBoneFeatures = sp.read3DPointsFromPPFile(dicomIllustrationFolder + "/138645-SpinopelvicBoneMesh_picked_points.pp");
    ctTorsoMesh = sp.readMesh(dicomIllustrationFolder + "/138645-TorsoMesh_Simplified.ply");
    
    # Register the predicted pelvic bone muscle mesh to the target pelvic mesh
    print("Register the predicted to the target ...");
    ## Estimate the svd transform
    svdTransform = sp.estimateRigidSVDTransform(predictedPelvicBoneMuscleFeatures, ctPelvicBoneFeatures);
    ## Transform the predicted pelvis bone mesh to the 
    predictedPelvicBoneMuscleMesh = sp.transformMesh(predictedPelvicBoneMuscleMesh, svdTransform);
    ## Scale the meshes to the scale of the DICOM images
    ctSpinoPelvicBoneMesh.vertices *= 1000;
    predictedPelvicBoneMuscleMesh.vertices *= 1000;
    ctTorsoMesh.vertices *= 1000;
    ## Colorize the vertice of the ctSpinoPelvicBoneMesh as bone color rgb
    boneColorRGBA = np.array([227, 218, 201, 200], dtype=np.uint8);
    ctSpinoPelvicBoneMesh.visual.vertex_colors = np.tile(boneColorRGBA, (len(ctSpinoPelvicBoneMesh.vertices), 1));
    ## Compute the structural features from the predicted pelvic bone muscle mesh
    structuralFeatures = sp.reconstructPointsFromBaryCentric(predictedPelvicBoneMuscleMesh, pelvicFeatureBaryCoords, pelvicFeatureBaryIndices);
    ## Generate the structural feature points
    structuralFeatureSpheres = [];
    for i in range(len(structuralFeatures)):
        coord = structuralFeatures[i];
        sphere = sp.generateSphereMesh(inCenter=coord, inRadius=10, inResolution=2);
        structuralFeatureSpheres.append(sphere);

    # Visualize the dicom images of the CT
    print("Visualize the dicom image of the CT ...");
    dicomFolder = dicomIllustrationFolder + "/DICOM";
    ## Reading the dicom series
    print("\t\t Reading the dicom series ...");
    import itk;
    ImageType = itk.Image[itk.SS, 3]  # Signed short, 3D
    reader = itk.ImageSeriesReader[ImageType].New()
    names_generator = itk.GDCMSeriesFileNames.New()
    names_generator.SetDirectory(dicomFolder)
    dicom_names = names_generator.GetInputFileNames()
    reader.SetFileNames(dicom_names)
    reader.Update()
    image = reader.GetOutput()
    ## Convert itk image to vtk image
    print("\t\t Convert itk image to vtk image ...");
    vtk_image = itk.vtk_image_from_image(image)
    ## Visualize using vtk
    print("\t\t Visualize using vtk ...");
    import vtk;
    volume_mapper = vtk.vtkSmartVolumeMapper()
    volume_mapper.SetInputData(vtk_image)
    volume_property = vtk.vtkVolumeProperty()
    volume_property.ShadeOn()
    volume_property.SetInterpolationTypeToLinear()
    opacity = vtk.vtkPiecewiseFunction()
    opacity.AddPoint(0, 0.0)      # Background - completely transparent
    opacity.AddPoint(500, 0.05)   # Soft tissue - very low opacity (reduced from 0.15)
    opacity.AddPoint(1000, 0.3)   # Bone - reduced opacity (reduced from 0.85)
    color = vtk.vtkColorTransferFunction()
    color.AddRGBPoint(0, 0.0, 0.0, 0.0)
    color.AddRGBPoint(500, 1.0, 1.0, 0.9)
    color.AddRGBPoint(1000, 1.0, 1.0, 1.0)
    volume_property.SetScalarOpacity(opacity)
    volume_property.SetColor(color)
    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)
    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume)
    renderer.SetBackground(1.0, 1.0, 1.0)
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor_style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(interactor_style)
    ## Add more the structural feature points to the renderer
    for i, sphere_mesh in enumerate(structuralFeatureSpheres):
        # Convert trimesh to VTK polydata
        vtk_points = vtk.vtkPoints()
        for vertex in sphere_mesh.vertices:
            vtk_points.InsertNextPoint(vertex[0], vertex[1], vertex[2])        
        vtk_cells = vtk.vtkCellArray()
        for face in sphere_mesh.faces:
            vtk_cells.InsertNextCell(3)
            for vertex_id in face:
                vtk_cells.InsertCellPoint(vertex_id)        
        # Create VTK polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)
        polydata.SetPolys(vtk_cells)        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)        
        # Set sphere color (e.g., red for visibility)
        actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # Red color
        actor.GetProperty().SetOpacity(1.0)  # Fully opaque
        # Add actor to renderer
        renderer.AddActor(actor)
    ## Render the spheres
    interactor.Initialize()
    interactor.Start()

    # Finished processing
    print("Finished processing.");

#**************************************** MAIN FUNCTION
def main():
    os.system("cls");
    featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor_visualizeBestWorstPredictedCasesWithCTImageScans();
if __name__ == "__main__":
    main()
