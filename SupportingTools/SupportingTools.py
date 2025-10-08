#*************************************************************** SUPPORTING LIBRARIES
import trimesh;
import gc;
import os;
import time;
import pandas as pd;
import numpy as np;
import csv;
import warnings;
import json;
import glob;
import math;
import xml.etree.ElementTree as ET;
import open3d as o3d;
from scipy.spatial import KDTree;
import msvcrt;
from scipy.spatial import ConvexHull;
from scipy.spatial.transform import Rotation as R;
import cv2;
import vtk;
import pymeshlab;
from collections import defaultdict;
from multiprocessing import Pool, cpu_count;
import random
from typing import List, Tuple;
import shutil;
import zipfile;
from sklearn.model_selection import train_test_split;

#*************************************************************** PRE-PROCESSING FUNCTIONS
warnings.filterwarnings("ignore");

#*************************************************************** GENERAL PROCESSING FUNCTIONS
def pause():
    print("pause:: We are pausing. Press any key to terminate.");
    while(True):
        time.sleep(1);
def clearScreen():
    os.system("cls");
def readVertexIndicesWithRegionKeysFromJSONFile(inFilePath, inKeyList):
    try:
        with open(inFilePath, 'r') as json_file:
            data = json.load(json_file);
            numberList = [];
            for key in inKeyList:
                if key in data:
                    values = data[key];
                    numberList.append(values);
            flatNumbers = [item for sublist in numberList for item in sublist];
            return flatNumbers;
    except FileNotFoundError:
        print(f"File '{inFilePath}' not found.")
        return None
def readMatrixFromCSVFile(inFilePath, inDelimeter=','):
    dataBuffer = pd.read_csv(inFilePath, header=None, delimiter=inDelimeter);
    return dataBuffer.to_numpy();
def readIndicesFromCSVFile(inFilePath, inDelimeter=','):
    dataBuffer = readMatrixFromCSVFile(inFilePath);
    return dataBuffer.flatten();
def saveVectorXdToCSVFile(inFilePath, inVector):
    with open(inFilePath, 'w', newline='') as file:
        writer = csv.writer(file);
        for value in inVector:
            writer.writerow([value]);
def saveVectorXiToCSVFile(inFilePath, inVectorXi):
    # Ensure all values in the vector are integers
    intVector = [int(value) for value in inVectorXi]
    
    with open(inFilePath, mode='w', newline='') as file:
        writer = csv.writer(file);
        writer.writerow(intVector);
def saveMatrixToCSVFile(inFilePath, inMatrix):
    df = pd.DataFrame(inMatrix);
    df.to_csv(inFilePath, header=False, index=False);
def saveItegerArrayToCSV(array, filename):
    np.savetxt(filename, array, delimiter=",", fmt='%d')
    print(f"Array saved to {filename}")
def listAllFilesWithExtensionInsideAFolder(inFolder, inExtension):
    # Checking folder
    if (not os.path.isdir(inFolder)): return None;

    # List folder
    filePaths = glob.glob(os.path.join(inFolder, f'*{inExtension}'));

    # Return buffers
    return filePaths;
def getBaseName(inFilePath):
    return os.path.basename(inFilePath);
def extractFileName(inFilePath):
    return os.path.basename(os.path.splitext(inFilePath)[0]);
def zeros(inShape):
    return np.zeros(inShape);
def randoms(inShape):
    return np.array(np.random.randn(inShape[0], inShape[1]));
def boundRandoms(inShape, inMin, inMax):
    """
    Generate random numbers with a given shape and within a specified range.

    Parameters:
        inShape (tuple): Shape of the output array (e.g., (3,), (72,), (10, 3), etc.)
        inMin (float or int): Minimum value of the random numbers.
        inMax (float or int): Maximum value of the random numbers.

    Returns:
        np.ndarray: Array of random numbers within the specified range and shape.
    """
    return np.random.uniform(low=inMin, high=inMax, size=inShape)
def read3DPointsFromPPFile(inFilePath):
    tree = ET.parse(inFilePath);
    root = tree.getroot();

    coordinates = [];
    for point_elem in root.findall(".//point"):
        x = float(point_elem.get("x"));
        y = float(point_elem.get("y"));
        z = float(point_elem.get("z"));
        coordinates.append((x, y, z));
    
    coordinates = np.array(coordinates);
    return coordinates
def read3DPointsFromOFFFile(inFilePath):
    """
    Reads 3D points from an OFF file using trimesh.

    Args:
        file_path (str): Path to the OFF file.

    Returns:
        numpy.ndarray: Array of 3D points.
    """
    mesh = trimesh.load_mesh(inFilePath, process=False)
    return np.array(mesh.vertices)
def waitForKey():
    print("Press any key to continue...")
    msvcrt.getch()
    print("Continuing execution...")
def wait():
    print("Press any key to continue...")
    msvcrt.getch()
    exit();
def makeDirectory(path):
    """
    Create a directory if it doesn't already exist.
    
    Parameters:
    path (str): The path of the directory to create.
    """
    try:
        os.makedirs(path, exist_ok=True);
    except Exception as e:
        print(f"Error creating directory '{path}': {e}")
def delayInSeconds(seconds):
    if seconds < 0:
        raise ValueError("Delay time cannot be negative.")
    time.sleep(seconds)
def isFileExist(filePath):
    """
    Check if a given file path exists.
    
    Parameters:
        filePath (str): The path to the file.
    
    Returns:
        bool: True if the file exists, False otherwise.
    """
    return os.path.isfile(filePath)
def saveListOfStrings(filePath, stringList):
    """
    Save a list of strings to a text file, each string on a new line.
    
    Parameters:
        stringList (list): List of strings to save.
        filePath (str): Path to the text file.
    """
    try:
        with open(filePath, 'w', encoding='utf-8') as file:
            for string in stringList:
                file.write(string + '\n')  # Write each string followed by a newline
    except Exception as e:
        print(f"Error saving list to file: {e}")
def readListOfStrings(filePath):
    try:
        with open(filePath, 'r', encoding='utf-16') as file:
            return [line.strip() for line in file.readlines()]
    except UnicodeError:
        # Try reading with a fallback encoding
        try:
            with open(filePath, 'r', encoding='latin1') as file:
                return [line.strip() for line in file.readlines()]
        except Exception as e:
            print(f"Error reading file with fallback encoding: {e}")
            return []
    except FileNotFoundError:
        print(f"Error: The file '{filePath}' does not exist.")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []
def getAllFileNames(folderPath):
    try:
        # Get all files in the folder
        allFiles = os.listdir(folderPath)
        # Filter out directories and remove file extensions
        fileNames = [
            os.path.splitext(file)[0] for file in allFiles if os.path.isfile(os.path.join(folderPath, file))
        ]
        return fileNames
    except FileNotFoundError:
        print(f"Error: The folder '{folderPath}' does not exist.")
        return []
    except Exception as e:
        print(f"Error reading folder: {e}")
        return []
def getAllFileNamesWithExtension(folderPath):
    try:
        # Get all files in the folder
        allFiles = os.listdir(folderPath)
        # Filter out directories and include file extensions
        fileNames = [
            file for file in allFiles if os.path.isfile(os.path.join(folderPath, file))
        ]
        return fileNames
    except FileNotFoundError:
        print(f"Error: The folder '{folderPath}' does not exist.")
        return []
    except Exception as e:
        print(f"Error reading folder: {e}")
        return []
def saveNumPyArrayToNPY(filename, array):
    """
    Saves a NumPy array to an .npy file using a lower precision to reduce memory usage.

    Parameters:
    - filename (str): The file path to save the array.
    - array (numpy.ndarray): The NumPy array to save.
    - dtype (numpy dtype, optional): The target data type (default: np.float16 for lowest memory).
    """
    np.save(filename, array)
def loadNumPYArrayFromNPY(filePath):
    """
    Load a NumPy array from an .npy file.
    
    Parameters:
        filePath (str): The path to the .npy file.
    
    Returns:
        np.ndarray: The loaded NumPy array.
    """
    return np.load(filePath)
def copyFileToFolder(inSourceFile, inTargetFolder):
    try:
        if not os.path.exists(inTargetFolder):
            os.makedirs(inTargetFolder)  # Create the folder if it doesn't exist
        
        target_path = os.path.join(inTargetFolder, os.path.basename(inSourceFile))
        shutil.copy2(inSourceFile, target_path)  # Copy with metadata
    except FileNotFoundError:
        print("Source file not found.")
    except PermissionError:
        print("Permission denied.")
    except Exception as e:
        print(f"Error: {e}")
def save3DPointsToPPFile(filename, points, username="user", data_filename="points.pp"):
    """
    Save an Mx3 numpy array of 3D points to a .pp file.
    
    Parameters:
        filename (str): Output file name ending with .pp
        points (numpy.ndarray): Mx3 array with each row as [x, y, z]
        username (str): Name of the user (default: "user")
        data_filename (str): Name of the associated data file (default: "points.ply")
    """
    if not isinstance(points, np.ndarray) or points.shape[1] != 3:
        raise ValueError("points must be an Mx3 numpy array")
    
    header = f"""<!DOCTYPE PickedPoints>
<PickedPoints>
 <DocumentData>
  <DateTime date="2025-02-25" time="00:48:51"/>
  <User name="{username}"/>
  <DataFileName name="{data_filename}"/>
  <templateName name=""/>
 </DocumentData>
"""
    
    point_entries = "".join(
        f' <point name="{i}" x="{x:.9f}" y="{y:.9f}" z="{z:.9f}" active="1"/>\n' 
        for i, (x, y, z) in enumerate(points)
    )
    
    footer = "</PickedPoints>"
    
    with open(filename, "w") as f:
        f.write(header + point_entries + footer)
def save3DPointsToPLY(filePath, points3D, color=None):
    """
    Save a set of 3D points to a PLY file using trimesh.

    Parameters:
        filePath (str): Path to save the .ply file.
        points3D (np.ndarray): A (N, 3) array of 3D points.
        color (np.ndarray or list, optional): A (N, 3) or (3,) array of RGB values (0–255).
                                              Can also be None for no color.
    """
    points3D = np.asarray(points3D)

    if color is not None:
        color = np.asarray(color, dtype=np.uint8)
        if color.ndim == 1:
            color = np.tile(color, (points3D.shape[0], 1))
        elif color.shape[0] != points3D.shape[0]:
            raise ValueError("Color array must have same number of rows as points3D")
    else:
        color = None

    cloud = trimesh.points.PointCloud(vertices=points3D, colors=color)
    cloud.export(filePath)
def listAllFolderNames(path):
    try:
        # List all directories in the given path
        folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
        return folders
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
def moveFiles(source_folder, destination_folder):
    try:
        # Create destination folder if it doesn't exist
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # Iterate through all files in the source folder
        for file_name in os.listdir(source_folder):
            source_file = os.path.join(source_folder, file_name)
            destination_file = os.path.join(destination_folder, file_name)

            # Move the file
            if os.path.isfile(source_file):
                shutil.move(source_file, destination_file)
                print(f"Moved: {file_name}")

        print("All files have been moved successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
def removeFolder(path):
    try:
        # Check if the folder exists
        if os.path.exists(path):
            # Remove the folder and its contents
            shutil.rmtree(path)
            print(f"Folder '{path}' has been removed.")
        else:
            print(f"Folder '{path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")
def isFolderEmpty(path):
    try:
        # Check if the folder exists
        if os.path.exists(path):
            # Check if the folder is empty
            if not os.listdir(path):
                return True
            else:
                return False
        else:
            print(f"Folder '{path}' does not exist.")
            return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
def zipFolder(sourceFolder, targetFolder):
    # Get the folder name
    folder_name = os.path.basename(sourceFolder.rstrip('/'))

    # Create the zip file path inside the target folder
    zip_path = os.path.join(targetFolder, f"{folder_name}.zip")

    # Create the zip file
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(sourceFolder):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, sourceFolder))

    print(f"Folder '{folder_name}' zipped successfully as '{zip_path}'")
def replaceFileName(folderPath, oldFileName, newFileName):
    try:
        # Build the full paths for the old and new file names
        oldFilePath = os.path.join(folderPath, oldFileName)
        newFilePath = os.path.join(folderPath, newFileName)
        
        # Check if the old file exists
        if not os.path.exists(oldFilePath):
            print(f"Error: '{oldFileName}' does not exist in the folder '{folderPath}'.")
            return False
        
        # Rename the file
        os.rename(oldFilePath, newFilePath)
        print(f"File '{oldFileName}' successfully renamed to '{newFileName}'.")
        return True
    except Exception as e:
        print(f"Error renaming file: {e}")
        return False
def removeFilesWithExtension(folderPath, extension):
    try:
        # List all files in the folder
        allFiles = os.listdir(folderPath)
        
        # Loop through the files and remove those with the specified extension
        for file in allFiles:
            filePath = os.path.join(folderPath, file)
            if os.path.isfile(filePath) and file.endswith(extension):
                os.remove(filePath)
                print(f"Removed: {file}")
        
        print(f"All files with the '{extension}' extension have been removed.")
    except FileNotFoundError:
        print(f"Error: The folder '{folderPath}' does not exist.")
    except Exception as e:
        print(f"Error: {e}")
def extractFileExtension(inFileNameWithExt):
    return os.path.splitext(inFileNameWithExt)[1];
def removeFile(inFilePath):
    try:
        os.remove(inFilePath)
        print(f"File '{inFilePath}' has been removed.")
    except FileNotFoundError:
        print(f"File '{inFilePath}' not found.")
    except Exception as e:
        print(f"Error: {e}")
def removeAllFilesInFolder(folderPath):
    # Check if the folder exists
    if os.path.exists(folderPath):
        # Iterate over all the files and directories in the folder
        for filename in os.listdir(folderPath):
            file_path = os.path.join(folderPath, filename)
            try:
                # Check if it is a file or directory and remove accordingly
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        print(f'The folder {folderPath} does not exist.')
def getFileNameFromPath(inFilePath):
    """
    Get the filename from a file path.

    :param file_path: The full file path.
    :return: The filename.
    """
    return os.path.basename(inFilePath)
def renameFile(inOldFilePath, inNewFilePath):
    try:
        os.rename(inOldFilePath, inNewFilePath)
        print(f"File renamed from '{inOldFilePath}' to '{inNewFilePath}'")
    except FileNotFoundError:
        print(f"Error: '{inOldFilePath}' does not exist.")
    except FileExistsError:
        print(f"Error: '{inNewFilePath}' already exists.")
    except Exception as e:
        print(f"Unexpected error: {e}")
def moveFileToFolder(filePath, targetFolder):
    """
    Moves a file to the specified target folder.

    Parameters:
    - filePath (str): Full path of the file to move.
    - targetFolder (str): Path to the folder where the file should be moved.

    Returns:
    - str: New file path after moving.
    """
    if not os.path.isfile(filePath):
        raise FileNotFoundError(f"File not found: {filePath}")
    
    if not os.path.exists(targetFolder):
        os.makedirs(targetFolder)  # Create target folder if it doesn't exist
    
    fileName = os.path.basename(filePath)
    targetPath = os.path.join(targetFolder, fileName)

    shutil.move(filePath, targetPath)
    return targetPath
def vPrint(*args, sep=' ', end='\n', file=None, flush=False, visualize=True):
    """
    A print-like function with an additional 'visualize' parameter.

    Parameters:
    - *args: Values to print (same as print).
    - sep (str): Separator between values (default: space).
    - end (str): End character (default: newline).
    - file: Output stream (default: sys.stdout).
    - flush (bool): Whether to forcibly flush the stream.
    - visualize (bool): Whether to actually print or not.
    """
    if visualize:
        print(*args, sep=sep, end=end, file=file, flush=flush)
def readVectorsFromFile(inFilePath):
    """
    Reads a text file where each line represents a vector of integers,
    stores them in a dictionary, and returns the dictionary.

    :param file_path: Path to the text file
    :return: Dictionary containing vectors with indexed keys
    """
    data_dict = {}

    with open(inFilePath, "r") as file:
        for i, line in enumerate(file):
            vector = np.array(list(map(int, line.strip().split(", "))))
            data_dict[f"vector_{i}"] = vector

    return data_dict

#*************************************************************** MACHINE LEARNING SUPPORTING FUNCTIONS
def generateTrainTestSplits(subjectIds: List[str], trainPct: float, testPct: float, numValidations: int, seed: int = 42) -> List[Tuple[List[str], List[str]]]:
    """
    Generate training and testing splits for given subject IDs.
    
    Parameters:
    - subjectIds (List[str]): List of subject IDs.
    - trainPct (float): Percentage of data for training (0-1).
    - testPct (float): Percentage of data for testing (0-1).
    - numValidations (int): Number of validation folds.
    - seed (int): Random seed for reproducibility.
    
    Returns:
    - List[Tuple[List[str], List[str]]]: A list of (trainIds, testIds) for each validation fold.
    """
    assert 0 < trainPct < 1, "Training percentage should be between 0 and 1."
    assert 0 < testPct < 1, "Testing percentage should be between 0 and 1."
    assert trainPct + testPct <= 1, "Sum of training and testing percentages should be <= 1."
    
    random.seed(seed)
    totalSubjects = len(subjectIds)
    splits = []
    
    for _ in range(numValidations):
        shuffledIds = subjectIds[:]
        random.shuffle(shuffledIds)
        
        trainSize = int(totalSubjects * trainPct)
        testSize = int(totalSubjects * testPct)
        
        trainIds = shuffledIds[:trainSize]
        testIds = shuffledIds[trainSize:trainSize + testSize]
        
        splits.append((trainIds, testIds))
    
    return splits
def trainValidSplit(ids, valid_size=0.2, random_state=42):
    """
    Split a list of string IDs into training and validation sets.
    
    Parameters:
    - ids (list): List of string IDs.
    - valid_size (float): Fraction of IDs for validation (default: 20%).
    - random_state (int): Seed for reproducibility.
    
    Returns:
    - train_ids (list): Training set.
    - valid_ids (list): Validation set.
    """
    train_ids, valid_ids = train_test_split(ids, test_size=valid_size, random_state=random_state)
    return train_ids, valid_ids

#*************************************************************** MESH PROCESSING FUNCTIONS
def readMesh(filePath, isProcessedMesh = False):
    mesh = trimesh.load_mesh(filePath, process=isProcessedMesh);
    return mesh;
def saveMesh(filePath, mesh, fileType = "ply"):
    # Checking mesh
    if (mesh.is_empty): 
        print("saveTriMeshToFile:: The inMesh is empty.");
        return None;

    # Save the mesh
    mesh.export(filePath, file_type=fileType);
def saveMeshToOFF(offFilePath, mesh):
     # Checking mesh
    if (mesh.is_empty): 
        print("saveTriMeshToOFFFile:: The inMesh is empty.");
        return None;

    # Save the mesh
    mesh.export(offFilePath, file_type="off");
def saveMeshToSTL(filePath, mesh, isASCIIFormat=False):
    """
    Save a trimesh object to an STL file.

    Parameters:
    - mesh (trimesh.Trimesh): The mesh object to save.
    - file_path (str): The path to save the STL file.
    - ascii_format (bool): If True, saves the STL in ASCII format. Default is binary.
    """
    try:
        # Determine file type based on ascii_format
        file_type = 'stl_ascii' if isASCIIFormat else 'stl'
        # Export the mesh to the specified file
        mesh.export(filePath, file_type=file_type)
        print(f"Mesh successfully saved to {filePath}")
    except Exception as e:
        print(f"An error occurred while saving the mesh: {e}")
def saveMeshToPLY(filePath, mesh):
    """
    Save a Trimesh object to a PLY file.
    
    Parameters:
        mesh (trimesh.Trimesh): The mesh object to save.
        filePath (str): Path to save the PLY file.
    """
    try:
        mesh.export(filePath, file_type='ply')  # Export the mesh as PLY
    except Exception as e:
        print(f"Error saving mesh to PLY file: {e}")
def readMeshFromPLY(filePath):
    """
    Read a PLY file and load it as a Trimesh object.
    
    Parameters:
        filePath (str): Path to the PLY file.
    
    Returns:
        trimesh.Trimesh: The loaded mesh object, or None if an error occurs.
    """
    try:
        mesh = trimesh.load(filePath, file_type='ply')  # Load the PLY file
        return mesh
    except FileNotFoundError:
        print(f"Error: The file '{filePath}' does not exist.")
        return None
    except Exception as e:
        print(f"Error reading mesh from PLY file: {e}")
        return None
def trimeshToMeshSet(meshInTrimesh):
    # Getting the vertices and faces
    vertices = np.asarray(meshInTrimesh.vertices);
    faces = np.asarray(meshInTrimesh.faces);

    # Create a pymeshlab Mesh
    mesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces);

    # Create a MeshSet and add the mesh
    ms = pymeshlab.MeshSet();
    ms.add_mesh(mesh);
    
    return ms;
def meshSetToTrimesh(meshInPyMeshLab):
   mesh = meshInPyMeshLab.current_mesh();
   vertices = np.asarray(mesh.vertex_matrix());
   faces = np.asarray(mesh.face_matrix());    
   outTrimesh = trimesh.Trimesh(vertices=vertices, faces=faces);    
   return outTrimesh;
def trimeshToPymeshlabMesh(triMesh):
    # Convert Trimesh vertices and faces to PyMeshLab compatible format
    vertices = triMesh.vertices
    faces = triMesh.faces

    # Create a PyMeshLab Mesh from vertices and faces
    mesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)
    
    return mesh
def pymeshlabMeshToTrimesh(pmlMesh):
    # Extract vertices and faces from PyMeshLab Mesh
    vertices = pmlMesh.vertex_matrix()
    faces = pmlMesh.face_matrix()
    
    # Create a Trimesh object using the extracted vertices and faces
    triMesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    return triMesh
def isotropicRemeshToTargetEdgeLength(mesh, targetLen = 0.01):
    bbox = mesh.bounding_box;
    bbox_diagonal = np.linalg.norm(bbox.extents);
    targetLengthPercentage = (targetLen / bbox_diagonal) * 100;
    ms = trimeshToMeshSet(mesh);
    ms.meshing_isotropic_explicit_remeshing(targetlen = pymeshlab.PercentageValue(targetLengthPercentage));
    outMesh = meshSetToTrimesh(ms);
    return outMesh;
def isotropicRemeshToTargetDiagonalPercentage(mesh, iterations = 10, targetPercentage = 1, adaptive = False):
    ms = trimeshToMeshSet(mesh);
    ms.meshing_isotropic_explicit_remeshing(targetlen = pymeshlab.PercentageValue(targetPercentage),
                                            iterations = iterations,
                                            adaptive = adaptive);
    outMesh = meshSetToTrimesh(ms);
    return outMesh;
def formMesh(inVertices, inFacets, inProcess = False):
    # Checking the vertices and faces
    if (len(inVertices.flatten()) == 0):
        print("formTriMesh:: The inVertices are empty."); return None;
    if (len(inFacets.flatten()) == 0):
        print("formTriMesh:: The inFaces are empty."); return None;
    
    # Forming the mesh
    mesh = trimesh.Trimesh(inVertices, inFacets, process=inProcess);
    return mesh;
def estimateMeshNearestPoints(inFromPoints, inTargetMesh):
    # Checking points
    if (len(inFromPoints) == 0): return None;
    
    # Creating buffers
    fromPoints = np.array(inFromPoints);

    # Finding the nearest indices using kdtree
    output = trimesh.proximity.closest_point(inTargetMesh, fromPoints);
    
    # Returning buffer
    return output[0];
def estimateMeshBoundingBox(inMesh):
    # Checking mesh
    if (inMesh.is_empty): return None;

    # Estimate bounding box
    boundingBox = inMesh.bounding_box;

    # Return buffer
    return boundingBox;
def booleanMeshManifolds(inMesh1, inMesh2, inOperation):
    # Checking input
    if (inMesh1.is_empty): return None;
    if (inMesh2.is_empty): return None;
    myValidOperations = ['difference', 'intersection', 'union'];
    if (not inOperation in myValidOperations): return None;
    
    # Boolean the mesh
    if inOperation == 'union':
        resultMesh = trimesh.boolean.union([inMesh1, inMesh2])
    elif inOperation == 'intersection':
        resultMesh = trimesh.boolean.intersection([inMesh1, inMesh2], engine='manifold')
    elif inOperation == 'difference':
        resultMesh = trimesh.boolean.difference([inMesh1, inMesh2])
    else:
        raise ValueError(f"Unsupported boolean operation: {inOperation}")
        
    # Return output
    return resultMesh;
def scaleMesh(inMesh, inScaleFactor):
    # Checking mesh
    if (inMesh.is_empty):
        print("scaleTriMesh:: The mesh is empty.");
        return None;

    # Estimate the transform matrix
    scaleMatrix = trimesh.transformations.scale_matrix(inScaleFactor);

    # Apply the transform
    meshBuffer = inMesh.apply_transform(scaleMatrix);

    # Return outputs
    return meshBuffer;
def estimateTranslationMatrixFromSourceToTargetPoint(a, b):
    """
    Generate a 4x4 translation matrix from point A to point B.
    
    Parameters:
        a (tuple or list): Coordinates of point A (xA, yA, zA).
        b (tuple or list): Coordinates of point B (xB, yB, zB).
        
    Returns:
        numpy.ndarray: 4x4 translation matrix.
    """
    a = np.array(a)
    b = np.array(b)
    translation = b - a
    
    matrix = np.eye(4)
    matrix[:3, 3] = translation
    
    return matrix
def scaleMeshInPlace(inMesh, inScaleFactor):
    # Checking mesh
    if (inMesh.is_empty):
        print("scaleTriMesh:: The mesh is empty.");
        return None;

    # Original centroid
    originalCentroid = computeCentroidPoint(inMesh.vertices);

    # Estimate the transform matrix
    scaleMatrix = trimesh.transformations.scale_matrix(inScaleFactor);

    # Apply the transform
    meshBuffer = inMesh.apply_transform(scaleMatrix);

    # Transform the mesh to original position
    postCentroid = computeCentroidPoint(meshBuffer.vertices);

    # Estimate the translation matrix
    transMatrix = estimateTranslationMatrixFromSourceToTargetPoint(postCentroid, originalCentroid);

    # Apply transform
    meshBuffer = meshBuffer.apply_transform(transMatrix);

    # Return outputs
    return meshBuffer;
def formMeshLines(inStartingPoints, inNormals, inScale):
    # Computing starting points
    startPoints = np.array(inStartingPoints);
    normals = inNormals*inScale;
    endPoints = startPoints + normals;

    # Forming the lines
    lines = np.hstack([startPoints, endPoints]).reshape(-1, 2, 3);
    triMeshPaths = trimesh.load_path(lines);
    
    # Return
    return triMeshPaths;
def generateCuttingBox(inCentroidPoint, inThicknessNormal, inTopPoint, inHeight, inWidth, inThickness):
    # Computing points
    thicknessNormal = np.array(inThicknessNormal);
    width = inWidth; height = inHeight; thickness = inThickness;
    lowerCentroid = np.array(inCentroidPoint);
    centerLeftVector = inTopPoint - lowerCentroid;
    centerLeftVector = centerLeftVector / np.linalg.norm(centerLeftVector);
    centerUpVector = np.cross(thicknessNormal, centerLeftVector);
    centerUpVector = centerUpVector / np.linalg.norm(centerUpVector);
    lowerRightUpPoint = lowerCentroid + centerLeftVector*width/2.0 + centerUpVector*height/2.0;
    lowerLeftUpPoint = lowerCentroid - centerLeftVector*width/2.0 + centerUpVector*height/2.0;
    lowerRightDownPoint = lowerCentroid + centerLeftVector*width/2.0 - centerUpVector*height/2.0;
    lowerLeftDownPoint = lowerCentroid - centerLeftVector*width/2.0 - centerUpVector*height/2.0;
    upperRightUpPoint = lowerRightUpPoint + thicknessNormal*thickness;
    upperLeftUpPoint = lowerLeftUpPoint + thicknessNormal*thickness;
    upperRightDownPoint = lowerRightDownPoint + thicknessNormal*thickness;
    upperLeftDownPoint = lowerLeftDownPoint + thicknessNormal*thickness;
    
    lowerRightUpPoint = lowerRightUpPoint - thicknessNormal*thickness/2.0;
    lowerLeftUpPoint = lowerLeftUpPoint - thicknessNormal*thickness/2.0;
    lowerRightDownPoint = lowerRightDownPoint - thicknessNormal*thickness/2.0;
    lowerLeftDownPoint = lowerLeftDownPoint - thicknessNormal*thickness/2.0;
    upperRightUpPoint = upperRightUpPoint - thicknessNormal*thickness/2.0;
    upperLeftUpPoint = upperLeftUpPoint - thicknessNormal*thickness/2.0;
    upperRightDownPoint = upperRightDownPoint - thicknessNormal*thickness/2.0;
    upperLeftDownPoint = upperLeftDownPoint - thicknessNormal*thickness/2.0;
        
    # Forming box mesh
    vertices = np.array([
        upperRightUpPoint,
        upperRightDownPoint,
        upperLeftDownPoint,
        upperLeftUpPoint,
        lowerRightUpPoint,
        lowerRightDownPoint,
        lowerLeftDownPoint,
        lowerLeftUpPoint,
    ]);
    faces = [
        [0, 1, 2],
        [0, 2, 3],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [3, 2, 6],
        [3, 6, 7],
        [0, 4, 7],
        [0, 7, 3],
        [1, 5, 6],
        [1, 6, 2],
    ];
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False);
    mesh.fix_normals();

    # Return the mesh
    return mesh;
def generateCuttingBoxWithNormalCentroid(inCentroidPoint, inThicknessNormal, inHeight, inWidth, inThickness):
    """
    Generates a cutting box mesh with a specified centroid, normal vector, height, width, and thickness.

    Parameters:
        inCentroidPoint (array-like): The centroid point of the box's base.
        inThicknessNormal (array-like): The normal vector defining the thickness direction.
        inHeight (float): The height of the box.
        inWidth (float): The width of the box.
        inThickness (float): The thickness of the box along the normal vector.

    Returns:
        mesh (trimesh.Trimesh): The generated cutting box mesh.
    """
    # Normalize the thickness normal vector
    thicknessNormal = np.array(inThicknessNormal) / np.linalg.norm(inThicknessNormal);
    
    # Generate an arbitrary vector not aligned with the thicknessNormal
    arbitrary_vector = np.array([1, 0, 0]) if abs(thicknessNormal[0]) < 0.9 else np.array([0, 1, 0]);
    
    # Compute orthogonal vectors for width and height
    centerLeftVector = np.cross(thicknessNormal, arbitrary_vector);
    centerLeftVector = centerLeftVector / np.linalg.norm(centerLeftVector);
    centerUpVector = np.cross(thicknessNormal, centerLeftVector);
    centerUpVector = centerUpVector / np.linalg.norm(centerUpVector);

    # Box dimensions
    width = inWidth;
    height = inHeight;
    thickness = inThickness;
    
    # Compute the box corner points based on the centroid
    lowerCentroid = np.array(inCentroidPoint);
    lowerRightUpPoint = lowerCentroid + centerLeftVector * width / 2.0 + centerUpVector * height / 2.0;
    lowerLeftUpPoint = lowerCentroid - centerLeftVector * width / 2.0 + centerUpVector * height / 2.0;
    lowerRightDownPoint = lowerCentroid + centerLeftVector * width / 2.0 - centerUpVector * height / 2.0;
    lowerLeftDownPoint = lowerCentroid - centerLeftVector * width / 2.0 - centerUpVector * height / 2.0;
    
    # Compute the upper points by moving along the thickness direction
    upperRightUpPoint = lowerRightUpPoint + thicknessNormal * thickness;
    upperLeftUpPoint = lowerLeftUpPoint + thicknessNormal * thickness;
    upperRightDownPoint = lowerRightDownPoint + thicknessNormal * thickness;
    upperLeftDownPoint = lowerLeftDownPoint + thicknessNormal * thickness;
    
    # Adjust all points to center the box around the centroid
    offset = thicknessNormal * thickness / 2.0;
    lowerRightUpPoint -= offset;
    lowerLeftUpPoint -= offset;
    lowerRightDownPoint -= offset;
    lowerLeftDownPoint -= offset;
    upperRightUpPoint -= offset;
    upperLeftUpPoint -= offset;
    upperRightDownPoint -= offset;
    upperLeftDownPoint -= offset;
    
    # Define vertices and faces for the mesh
    vertices = np.array([
        upperRightUpPoint,
        upperRightDownPoint,
        upperLeftDownPoint,
        upperLeftUpPoint,
        lowerRightUpPoint,
        lowerRightDownPoint,
        lowerLeftDownPoint,
        lowerLeftUpPoint,
    ]);
    faces = [
        [0, 1, 2],
        [0, 2, 3],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [3, 2, 6],
        [3, 6, 7],
        [0, 4, 7],
        [0, 7, 3],
        [1, 5, 6],
        [1, 6, 2],
    ];
    
    # Create the mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False);
    mesh.fix_normals();

    return mesh;
def computeBarycentricCoordinates(inTriMesh, inPoints):
    """
    Compute the barycentric coordinates of points with respect to the triangles in a 3D mesh.

    Parameters:
    mesh (trimesh.Trimesh): The input 3D mesh.
    points (np.ndarray): An array of shape (n, 3) representing the 3D points.

    Returns:
    list of np.ndarray: A list of barycentric coordinates for each point.
    """
    barycentricCoords = [];

    # Find the nearest points on the surface and the corresponding face indices
    nearestPoints, nearestDistances, nearestFaceIndices = inTriMesh.nearest.on_surface(inPoints);

    for point, face_index in zip(inPoints, nearestFaceIndices):
        face_vertices = inTriMesh.vertices[inTriMesh.faces[face_index]];

        # Compute barycentric coordinates for the point with respect to the closest face
        A, B, C = face_vertices;
        AB = B - A;
        AC = C - A;
        AP = point - A;

        areaABC = np.linalg.norm(np.cross(AB, AC));
        areaPBC = np.linalg.norm(np.cross(B - point, C - point));
        areaPCA = np.linalg.norm(np.cross(C - point, A - point));

        u = areaPBC / areaABC;
        v = areaPCA / areaABC;
        w = 1 - u - v;

        barycentricCoords.append(np.array([u, v, w]));

    return np.array(barycentricCoords), np.array(nearestFaceIndices);
def reconstructPointsFromBaryCentric(inTriMesh, inBarycentricCoords, inFaceIndices):
    """
    Reconstructs 3D points from barycentric coordinates.

    Parameters:
    mesh (trimesh.Trimesh): The input 3D mesh.
    barycentric_coords (np.ndarray): An array of shape (n, 3) representing the barycentric coordinates.
    face_indices (np.ndarray): An array of shape (n,) representing the indices of the faces.

    Returns:
    np.ndarray: An array of shape (n, 3) representing the reconstructed 3D points.
    """
    
    reconstructedPoints = [];

    for baryCoords, faceIndex in zip(inBarycentricCoords, inFaceIndices):
        face_vertices = inTriMesh.vertices[inTriMesh.faces[faceIndex]];
        A, B, C = face_vertices;
        u, v, w = baryCoords;
        point = u * A + v * B + w * C;
        reconstructedPoints.append(point);

    return np.array(reconstructedPoints)
def concatenateMeshes(inTrimeshes):
    return trimesh.util.concatenate(inTrimeshes);
def transformMesh(inTriMesh, inTransformMatrix):
    inVertices = inTriMesh.vertices;
    transform = np.array(inTransformMatrix);
    transVertices = transform3DPoints(inVertices, transform);
    transFaces = inTriMesh.faces;
    transMesh = trimesh.Trimesh(vertices=transVertices, faces=transFaces, process=False);
    return transMesh;
def transformMeshInPlace(inTriMesh, inTransformMatrix):
    return inTriMesh.apply_transform(inTransformMatrix);
def selectMeshWithMinNumVertices(inTriMeshes):
    if not inTriMeshes:
        raise ValueError("selectMeshWithMinNumVertices:: The list of meshes is empty.");    
    minNumVerMesh = min(inTriMeshes, key=lambda mesh: len(mesh.vertices));
    return minNumVerMesh;
def sampleMesh(inTriMesh, inNumSamples):
    """
    Sample a mesh with a predefined number of points.
    
    Parameters:
    mesh (trimesh.Trimesh): The mesh to sample.
    num_samples (int): The number of points to sample from the mesh.
    
    Returns:
    numpy.ndarray: Array of sampled points.
    """
    samples, _ = trimesh.sample.sample_surface(inTriMesh, inNumSamples);
    return samples;
def generateSphereMesh(inCenter, inRadius, inResolution):
    # Checking the inCenter
    if inCenter.shape != (3,):
        raise ValueError("Center must be a 1x3 numpy array")
    
    # Create an icosphere centered at the origin
    sphere = trimesh.creation.icosphere(subdivisions=inResolution, radius=inRadius)
    
    # Translate the sphere to the specified center
    sphere.apply_translation(inCenter);
    
    return sphere;
def generateSphereMeshCoveringTarget(targetMesh, resolution):
    """
    Generate a sphere mesh that fully covers a target mesh.
    
    Parameters:
        targetMesh (trimesh.Trimesh): The target mesh to cover.
    
    Returns:
        trimesh.Trimesh: A sphere mesh covering the target mesh.
    """
    # Compute the centroid of the target mesh
    centroid = targetMesh.centroid  # Centroid of the mesh
    
    # Compute the radius as the maximum distance from the centroid to the vertices
    vertices = targetMesh.vertices
    distances = np.linalg.norm(vertices - centroid, axis=1)
    radius = np.max(distances)
    
    # Scale the radius slightly to ensure full coverage
    radius *= 1;  # Add 10% padding to the radius

    # Generate a sphere mesh with the calculated radius
    sphereMesh = trimesh.creation.icosphere(subdivisions=resolution, radius=radius)
    sphereMesh.apply_translation(centroid)
    
    return sphereMesh
def cloneMesh(inTrimesh):
    return trimesh.Trimesh(inTrimesh.vertices, inTrimesh.faces, process=False);
def subtractMeshFromMesh(inFullMesh, inMesh):
    # Getting the buffers
    tree = KDTree(inFullMesh.vertices);
    distances, closestIndices = tree.query(inMesh.vertices);
    meshFaces = np.array([[closestIndices[vertex] for vertex in face] for face in inMesh.faces]);
    mask = np.ones(len(inFullMesh.vertices), dtype=bool)
    mask[closestIndices] = False

    # Getting the out mesh
    outVertices = inFullMesh.vertices[mask];
    
    # Mapping faces   
    oldToNewIndex = np.full(len(inFullMesh.vertices), -1, dtype=int);
    oldToNewIndex[mask] = np.arange(len(outVertices)); outFaces = [];
    for face in inFullMesh.faces:
        if all(mask[vertex] for vertex in face):
            newFace = [oldToNewIndex[vertex] for vertex in face];
            outFaces.append(newFace);
    outFaces = np.array(outFaces);

    # Return buffers
    outMesh = trimesh.Trimesh(vertices=outVertices, faces=outFaces);
    return outMesh;
def computeHausdorffDistance(inMeshA, inMeshB, inNumSamples):
    # Sample points on the surface of the meshes
    points1 = inMeshA.sample(inNumSamples);
    points2 = inMeshB.sample(inNumSamples);
    
    # Compute the distances from points in mesh1 to mesh2
    distances1 = trimesh.proximity.signed_distance(inMeshB, points1);
    
    # Compute the distances from points in mesh2 to mesh1
    distances2 = trimesh.proximity.signed_distance(inMeshA, points2);
    
    # Hausdorff distance is the maximum of these distances
    hausdorffDistance = max(np.max(np.abs(distances1)), np.max(np.abs(distances2)));
    
    return hausdorffDistance;
def computeHaflSideHausdorffDistances(inMeshA, inMeshB, inNumSamples):
    # Sample points on the surface of the meshes
    points1 = inMeshA.sample(inNumSamples);
    
    # Compute the distances from points in mesh1 to mesh2
    distances1 = trimesh.proximity.signed_distance(inMeshB, points1);
    
    # Return distances
    return distances1;
def estimateTriMeshSectionPoints(inTrimesh, inPlaneCenter, inPlaneNormal):
    section = inTrimesh.section(plane_origin=inPlaneCenter, plane_normal=inPlaneNormal);
    if section is None:
        return None;
    else:
        return section.vertices;
def estimate3DConvexHull(inPointCloud):
    cloudMesh = trimesh.Trimesh(vertices=inPointCloud, faces=[]);
    return cloudMesh.convex_hull;
def subdivideMeshToEdgeLength(inTrimesh, inMaxEdgeLength):
    # Initializing
    newVertices, newFaces = trimesh.remesh.subdivide_to_size(inTrimesh.vertices, inTrimesh.faces, inMaxEdgeLength);

    # Forming trimesh
    newMesh = trimesh.Trimesh(vertices=newVertices, faces=newFaces);

    # Return buffer
    return newMesh;
def computeAverageEdgeLength(inTrimesh):
    edges = inTrimesh.edges_unique_length;
    averageEdgeLength = np.mean(edges);
    return averageEdgeLength;
def estimateMeshShape(mesh, firstEdgeLength, secondEdgeLength, firstIterations = 10, secondIterations = 10, firstAdaptive = False, secondAdaptive = False):
    # Estimate the convex hul
    convexHull = estimate3DConvexHull(mesh.vertices);

    # Remesh the convex hull
    remeshedConvexHull = isotropicRemeshToTargetEdgeLength(convexHull, firstIterations, firstEdgeLength, firstAdaptive);

    # Estimate the nearset points on the trimesh
    nearestPoints = estimateMeshNearestPoints(remeshedConvexHull.vertices, mesh);
        
    # Form out mesh
    outMesh = trimesh.Trimesh(vertices=nearestPoints, faces=remeshedConvexHull.faces);

    # Remeshing out mesh
    remeshedOutMesh = isotropicRemeshToTargetEdgeLength(outMesh, secondIterations, secondEdgeLength, secondAdaptive);

    # Finished processing
    return remeshedOutMesh;
def getFacesFromIndices(inMesh, inIndices):
    """
    Get the faces of the mesh that correspond to the vertices in headBodyIndices, remapped to the indices in headBodyIndices.

    Parameters:
    - mesh: A trimesh object representing the 3D mesh.
    - headBodyIndices: A list or array of vertex indices (subset of mesh vertices).

    Returns:
    - remapped_faces: An array of faces where all vertices are remapped to the indices in headBodyIndices.
    """
    
    # Convert headBodyIndices to a set for fast lookup
    headBodyIndicesSet = set(inIndices)
    
    # Get the faces of the mesh
    all_faces = inMesh.faces
    
    # Create a mapping from original vertex indices to headBodyIndices positions
    index_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(inIndices)}
    
    # Filter the faces where all vertices are in headBodyIndices and remap them
    remapped_faces = []
    for face in all_faces:
        if set(face).issubset(headBodyIndicesSet):
            # Remap the face indices to headBodyIndices
            remapped_face = [index_map[vert_idx] for vert_idx in face]
            remapped_faces.append(remapped_face)
    
    # Convert the remapped faces to a numpy array
    remapped_faces = np.array(remapped_faces)
    
    return remapped_faces
def fixMesh(inMesh):
    # Create a processed mesh
    outMesh = trimesh.Trimesh(vertices=inMesh.vertices, faces=inMesh.faces, process=True)

    # Remove degenerate faces
    outMesh.remove_degenerate_faces()

    # Fill any holes
    outMesh.fill_holes()

    # Ensure the normals are computed correctly
    outMesh.fix_normals()

    # Merge close vertices
    outMesh.merge_vertices()

    return outMesh
def closeHolesWithTrimesh(inMesh):
    inMesh.fill_holes();
    return inMesh;
def generateBox(length, width, height, center=(0, 0, 0)):
    """
    Generate a 3D triangular mesh box with a given center position.

    Parameters:
        length (float): Length of the box along the x-axis.
        width (float): Width of the box along the y-axis.
        height (float): Height of the box along the z-axis.
        center (tuple): (x, y, z) coordinates for the center of the box.

    Returns:
        trimesh.Trimesh: The generated 3D box as a triangular mesh.
    """
    # Define the offset for the box center
    cx, cy, cz = center

    # Define corner vertices of the box, centered at the origin
    vertices = np.array([
        [-length / 2, -width / 2, -height / 2],  # Vertex 0
        [ length / 2, -width / 2, -height / 2],  # Vertex 1
        [ length / 2,  width / 2, -height / 2],  # Vertex 2
        [-length / 2,  width / 2, -height / 2],  # Vertex 3
        [-length / 2, -width / 2,  height / 2],  # Vertex 4
        [ length / 2, -width / 2,  height / 2],  # Vertex 5
        [ length / 2,  width / 2,  height / 2],  # Vertex 6
        [-length / 2,  width / 2,  height / 2],  # Vertex 7
    ])

    # Translate the vertices to the specified center
    vertices += np.array([cx, cy, cz])

    # Define triangular faces (two triangles per face)
    faces = np.array([
        [0, 1, 3], [1, 2, 3],  # Bottom face
        [4, 5, 7], [5, 6, 7],  # Top face
        [0, 1, 4], [1, 5, 4],  # Front face
        [1, 2, 5], [2, 6, 5],  # Right face
        [2, 3, 6], [3, 7, 6],  # Back face
        [3, 0, 7], [0, 4, 7],  # Left face
    ])

    # Create the trimesh object
    box_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return box_mesh
def generateCylinderMesh(radius=1.0, height=2.0, resolution=32, centroid=(0, 0, 0)):
    """
    Generates a cylindrical mesh aligned along the Z-axis and positions its centroid at a given location.

    Parameters:
        radius (float): The radius of the cylinder.
        height (float): The height of the cylinder along the Z-axis.
        resolution (int): Number of segments around the circumference (smoothness).
        centroid (tuple): The desired centroid of the cylinder (x, y, z).
        
    Returns:
        trimesh.Trimesh: The cylindrical mesh object.
    """
    # Create the cylinder centered at (0, 0, 0), with height along the Z-axis
    cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=resolution)
    
    # The cylinder's default centroid is at (0, 0, 0), adjust it to match the desired centroid
    translation = np.array(centroid)  # Desired centroid (x, y, z)
    
    # Apply the translation to move the cylinder's centroid
    cylinder.apply_translation(translation)
    
    return cylinder
def createChessboardMesh(rows, cols, squareSize):
    meshes = []
    for i in range(rows):
        for j in range(cols):
            x_start = j * squareSize
            y_start = i * squareSize
            
            # Define vertices and faces for each square
            vertices = np.array([
                [x_start, y_start, 0],
                [x_start + squareSize, y_start, 0],
                [x_start + squareSize, y_start + squareSize, 0],
                [x_start, y_start + squareSize, 0]
            ])
            
            faces = np.array([
                [0, 1, 2],
                [0, 2, 3]
            ])
            
            squareMesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            color = [0, 0, 0, 255] if (i + j) % 2 == 0 else [255, 255, 255, 255]
            squareMesh.visual.face_colors = np.tile(color, (len(faces), 1))
            
            meshes.append(squareMesh)
    
    return trimesh.util.concatenate(meshes)
def closeHolesWithMeshLab(inMesh, inMaxHoleSize = 10):
    meshSet = trimeshToMeshSet(inMesh);
    meshSet.meshing_close_holes(maxholesize = inMaxHoleSize, newfaceselected = False, selfintersection = False)
    return meshSetToTrimesh(meshSet);
def isMeshWaterTight(inMesh):
    return inMesh.is_watertight;
def checkAndFixMeshWatertight(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Checks whether the mesh is watertight. If not, attempts to repair it by filling holes and cleaning up geometry.
    
    Parameters:
      mesh (trimesh.Trimesh): The input mesh.
    
    Returns:
      trimesh.Trimesh: A (hopefully) watertight version of the input mesh.
    """
    # Work on a copy of the mesh so the original is not modified.
    mesh = mesh.copy()
    
    if mesh.is_watertight:
        print("Mesh is already watertight.")
        return mesh
        
    # Try to fill holes using the built-in method if available.
    try:
        mesh.fill_holes()
    except AttributeError:
        # Fallback to the repair function if fill_holes() is not available.
        trimesh.repair.fill_holes(mesh)
    
    # Clean up the mesh.
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    
    # Return mesh
    return mesh
def computeAmbientOcclusion(triMesh: trimesh.Trimesh,
                               reqViews: int = 128,
                               dirBias: float = 0.0,
                               coneDir: np.ndarray = np.array([0.0, 1.0, 0.0]),
                               coneAngle: float = 30.0,
                               depthTexSize: int = 512,
                               useGPU: bool = False) -> np.ndarray:
    """
    Compute per-vertex ambient occlusion using pymeshlab with GPU acceleration.
    
    Args:
        triMesh: A trimesh.Trimesh object.
        reqViews: Number of view directions (higher = better quality).
        dirBias: Directional bias (0=uniform, 1=fully directional).
        coneDir: Direction vector of light source bias.
        coneAngle: Cone angle in degrees.
        depthTexSize: Size of depth texture (power of 2, e.g., 512).
        useGPU: Whether to use GPU acceleration.
    
    Returns:
        A numpy array containing ambient occlusion value per vertex.
    """
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=triMesh.vertices, face_matrix=triMesh.faces))
    
    ms.compute_scalar_ambient_occlusion_gpu(
        occmode='per-Vertex',
        dirbias=dirBias,
        reqviews=reqViews,
        conedir=coneDir,
        coneangle=coneAngle,
        usegpu=useGPU,
        depthtexsize=depthTexSize
    )
    
    ao_values = ms.current_mesh().vertex_scalar_array()
    return np.array(ao_values)
def removeOccludedGeometries(mesh: trimesh.Trimesh,
                            num_rays: int = 64,
                            ao_threshold: float = 0.2) -> trimesh.Trimesh:
    """
    Remove occluded parts of a mesh based on ambient occlusion values.

    Args:
        mesh: The input trimesh.Trimesh object.
        num_rays: Number of rays per vertex for AO computation (reqviews parameter).
        ao_threshold: Threshold for ambient occlusion (vertices with AO > threshold are kept).

    Returns:
        A cleaned trimesh.Trimesh object with occluded parts removed.
    """
    # Compute AO values using our earlier GPU-based function
    ao_values = computeAmbientOcclusion(mesh, reqViews=num_rays)

    # Select vertices that are more exposed (higher AO values)
    visible_vertices_mask = ao_values > ao_threshold
    visible_vertex_indices = np.nonzero(visible_vertices_mask)[0]

    if len(visible_vertex_indices) == 0:
        raise ValueError("No visible vertices found. Try lowering the 'ao_threshold'.")

    # Create a mapping from old vertex indices to new ones
    index_map = -np.ones(len(mesh.vertices), dtype=int)
    index_map[visible_vertex_indices] = np.arange(len(visible_vertex_indices))

    # Filter faces: keep faces where all three vertices are visible
    valid_faces_mask = np.all(np.isin(mesh.faces, visible_vertex_indices), axis=1)
    valid_faces = mesh.faces[valid_faces_mask]

    if len(valid_faces) == 0:
        raise ValueError("No valid faces remain after occlusion removal.")

    # Remap face indices
    cleaned_faces = index_map[valid_faces]

    # Create the cleaned mesh
    cleaned_mesh = trimesh.Trimesh(vertices=mesh.vertices[visible_vertices_mask],
                                   faces=cleaned_faces,
                                   process=False)

    return cleaned_mesh
def repairNonManifoldEdges(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Repairs a mesh by removing faces that contribute to non-manifold edges.
    
    In a well-formed closed manifold mesh, every edge should be shared by exactly 2 faces.
    This function identifies edges that are not shared by 2 faces and removes all faces
    incident to those edges.
    
    Parameters:
        mesh (trimesh.Trimesh): The input mesh that may have non-manifold edges.
    
    Returns:
        trimesh.Trimesh: A new mesh with faces causing non-manifold edges removed.
    """
    # Copy the faces for processing
    faces = mesh.faces.copy()
    
    # Build a mapping from each edge (as a sorted tuple of vertex indices) to a list of face indices.
    edge_face_map = defaultdict(list)
    for face_index, face in enumerate(faces):
        # Each triangular face has 3 edges.
        for i in range(3):
            # Get the edge as a sorted tuple (so that the edge (a, b) is the same as (b, a))
            a = face[i]
            b = face[(i + 1) % 3]
            edge = tuple(sorted((a, b)))
            edge_face_map[edge].append(face_index)
    
    # Identify faces that are incident to any non-manifold edge.
    problematic_faces = set()
    for edge, incident_faces in edge_face_map.items():
        # For a proper closed manifold, an edge should be incident to exactly 2 faces.
        if len(incident_faces) != 2:
            problematic_faces.update(incident_faces)
    
    # Determine indices of faces to keep.
    keep_faces = [i for i in range(len(faces)) if i not in problematic_faces]
    
    if not keep_faces:
        raise ValueError("All faces have been removed. The mesh is too non-manifold to repair using this method.")
    
    # Create a new face array for the repaired mesh.
    new_faces = faces[keep_faces]
    
    # Create a new mesh with the original vertices and the filtered faces.
    repaired_mesh = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=new_faces, process=False)
    
    # Optional cleanup: remove duplicate faces, remove unreferenced vertices, and fix normals.
    repaired_mesh.remove_duplicate_faces()
    repaired_mesh.remove_unreferenced_vertices()
    repaired_mesh.fix_normals()
    
    return repaired_mesh
def poissonDiskSampleMesh(input_mesh, sample_num=100000):
    # Create a MeshSet
    ms = pymeshlab.MeshSet()

    # Load the mesh from the trimesh object
    ms.add_mesh(pymeshlab.Mesh(input_mesh.vertices, input_mesh.faces))

    # Perform Poisson disk sampling
    ms.generate_sampling_poisson_disk(samplenum=sample_num)

    # Retrieve the sampled mesh
    sampled_mesh = ms.current_mesh()

    # Convert the sampled mesh vertices to a Nx3 numpy array
    sampled_vertices = sampled_mesh.vertex_matrix()

    return sampled_vertices
def surfaceReconstructionBallPivoting(points):
    # Create a MeshSet
    ms = pymeshlab.MeshSet()

    # Convert points to a PyMeshLab mesh
    ms.add_mesh(pymeshlab.Mesh(points))

    # Perform surface reconstruction using Ball Pivoting algorithm
    ms.generate_surface_reconstruction_ball_pivoting()

    # Retrieve the reconstructed mesh
    reconstructedMesh = ms.current_mesh()

    # Convert the reconstructed mesh back to a trimesh object
    reconstructedVertices = reconstructedMesh.vertex_matrix()
    reconstructedFaces = reconstructedMesh.face_matrix()
    reconstructedTrimesh = trimesh.Trimesh(vertices=reconstructedVertices, faces=reconstructedFaces)

    return reconstructedTrimesh
def isMeshInsideVerticesOnly(checkingMesh, coveringMesh):
    # Check if all vertices of checkingMesh are inside coveringMesh
    verticesInside = coveringMesh.contains(checkingMesh.vertices).all()
    return verticesInside
def separateMeshes(inMesh):
    return inMesh.split(only_watertight=False);
def mergeMeshes(meshList):
    return trimesh.util.concatenate(meshList)
def processVertex(args):
    k, V_mesh, V_cage, F_cage, num_vertices_cage, num_faces_cage, epsilon = args
    weightsK = np.zeros(num_vertices_cage)
    totalW = 0.0
    x = V_mesh[k]
    
    matrix_U = np.zeros((num_vertices_cage, 3))
    vector_D = np.zeros(num_vertices_cage)
    
    for j in range(num_vertices_cage):
        pJ = V_cage[j]
        dJ = np.linalg.norm(pJ - x)
        vector_D[j] = dJ
        matrix_U[j] = (pJ - x) / dJ
    
    num_planar_cases1, num_planar_cases2 = 0, 0
    
    for f in range(num_faces_cage):
        indices = F_cage[f]
        p1, p2, p3 = V_cage[indices]
        
        thetaIs = np.zeros(3)
        cIs = np.zeros(3)
        sIs = np.zeros(3)
        h = 0.0
        u123 = np.zeros((3, 3))
        
        for i in range(3):
            iMinus = (i - 1) % 3
            uIPlus = matrix_U[indices[(i + 1) % 3]]
            uIMinus = matrix_U[indices[iMinus]]
            lI = np.linalg.norm(uIPlus - uIMinus)
            thetaIs[i] = 2.0 * np.arcsin(lI / 2.0)
            h += thetaIs[i] / 2.0
            u123[i] = matrix_U[indices[i]]
        
        if np.pi - h < epsilon:
            weightsK.fill(0)
            totalW = 0.0
            for i in range(3):
                iMinus = (i - 1) % 3
                wI = np.sin(thetaIs[i]) * vector_D[indices[(i + 1) % 3]] * vector_D[indices[iMinus]]
                weightsK[indices[i]] = wI
                totalW += wI
            num_planar_cases1 += 1
            break
        
        signDet = np.linalg.det(u123)
        if signDet == 0:
            print("Singular matrix encountered in determinant calculation.")
        
        signDet = np.sign(signDet)
        discardTriangle = False
        
        for i in range(3):
            iMinus = (i - 1) % 3
            cI = -1.0 + 2.0 * np.sin(h) * np.sin(h - thetaIs[i]) / (np.sin(thetaIs[(i + 1) % 3]) * np.sin(thetaIs[iMinus]))
            cIs[i] = np.clip(cI, -1.0, 1.0)
            sI = signDet * np.sqrt(1.0 - cIs[i] ** 2)
            if np.abs(sI) < epsilon:
                discardTriangle = True
                num_planar_cases2 += 1
                break
            else:
                sIs[i] = sI
        
        if not discardTriangle:
            for i in range(3):
                iPlus = (i + 1) % 3
                iMinus = (i - 1) % 3
                dI = vector_D[indices[i]]
                wI = (thetaIs[i] - cIs[iPlus] * thetaIs[iMinus] - cIs[iMinus] * thetaIs[iPlus]) / (dI * np.sin(thetaIs[iPlus]) * sIs[iMinus])
                weightsK[indices[i]] += wI
                totalW += wI
    
    weightsK /= totalW
    return weightsK, num_planar_cases1, num_planar_cases2
def computeMeanValueWeights(mesh: trimesh.Trimesh, cage: trimesh.Trimesh, numThreads=None):
    """
    Computes mean value weights for a mesh relative to a cage using multiprocessing.

    Parameters:
    - mesh (trimesh.Trimesh): The input mesh.
    - cage (trimesh.Trimesh): The cage mesh.
    - epsilon (float, optional): Small value to prevent numerical issues (default: 1e-8).
    - numThreads (int, optional): Number of threads to use (default: all available CPUs).

    Returns:
    - mV_weights (np.ndarray): Mean value weights for each vertex in the mesh.
    """
    num_vertices_mesh = len(mesh.vertices)
    num_vertices_cage = len(cage.vertices)
    num_faces_cage = len(cage.faces)
    epsilon=1e-8;
    
    V_mesh = np.array(mesh.vertices)
    V_cage = np.array(cage.vertices)
    F_cage = np.array(cage.faces)
    
    if numThreads is None:
        numThreads = cpu_count()  # Use all available CPUs if not specified
    
    with Pool(numThreads) as pool:
        results = pool.map(processVertex, [(k, V_mesh, V_cage, F_cage, num_vertices_cage, num_faces_cage, epsilon) 
                                           for k in range(num_vertices_mesh)])
    
    mV_weights = np.array([res[0] for res in results])
    num_planar_cases1 = sum(res[1] for res in results)
    num_planar_cases2 = sum(res[2] for res in results)
    
    print(f"[Mean Value Coordinates] number of planar cases: {num_planar_cases1} and {num_planar_cases2}")
    return mV_weights
def optimizeCageVerticesToDeformToTargetVertices(targetVerticesOfMesh: np.ndarray, cageBasedWeights: np.ndarray):
    """
    Optimizes cage vertices to minimize the point-to-point distance between 
    deformed vertices and target vertices.

    Parameters:
        tarVertices (np.ndarray): (N, 3) target points.
        cageVertices (np.ndarray): (M, 3) cage vertices (to be optimized).
        weights (np.ndarray): (N, M) mean value coordinate weights.

    Returns:
        np.ndarray: (M, 3) optimized cage vertices.
    """
    # Solve the least squares problem: W * cageVertices' ≈ tarVertices
    optimizedCageVertices, _, _, _ = np.linalg.lstsq(cageBasedWeights, targetVerticesOfMesh, rcond=None)    
    return optimizedCageVertices;
def findFilePath(file_name, searchingFolder="."):
    for root, _, files in os.walk(searchingFolder):
        for file in files:
            if file.startswith(file_name.split('.')[0]):  # Matching filename without extension
                return os.path.join(root, file)  # Return the first match
    return None
def computeCenterToVertexNormals(mesh):
    """
    Computes direction vectors from mesh center to each vertex.
    
    Parameters:
    - mesh (trimesh.Trimesh): The input mesh.

    Returns:
    - normals (np.ndarray): Array of shape (n_vertices, 3) with unit vectors.
    """
    center = mesh.vertices.mean(axis=0)  # mesh center
    directions = mesh.vertices - center  # direction vectors to each vertex
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    normals = directions / norms  # normalize to unit vectors
    return normals
def estimatePelvisShape(inPelvisMesh, numOfRemesh = 4, targetEdgeLength = 0.001):   
    # Debugging
    debugFolder = r"H:\Data\PelvisBoneRecon\Debugs";
    # Estimate the convex hull of the pelvis
    print("\t\t Estimate convex hull ...");
    pelvisConvexHull = estimate3DConvexHull(inPelvisMesh.vertices);
    saveMeshToPLY(debugFolder + "/pelvisConvexHull.ply", pelvisConvexHull);

    # Resample the convex hull
    print("\t\t Resample the convex hull ...");
    remeshedPelvisConvexHull = isotropicRemeshToTargetEdgeLength(pelvisConvexHull, targetEdgeLength);
    saveMeshToPLY(debugFolder + "/remeshedPelvisConvexHull.ply", remeshedPelvisConvexHull);

    # Project remeshed pelvis convex hull onto the pelvis mesh
    print("\t\t Project remeshed pelvis convex hull onto the pelvis mesh ...");
    projectedPelvisConvexHullVertices = estimateNearestPointsFromPoints(remeshedPelvisConvexHull.vertices, inPelvisMesh.vertices);
    projectedPelvisConvexHull = formMesh(projectedPelvisConvexHullVertices, remeshedPelvisConvexHull.faces);
    saveMeshToPLY(debugFolder + "/projectedPelvisConvexHull.ply", projectedPelvisConvexHull);

    # Repeat of remeshing and projecting
    for i in range(numOfRemesh):
        # Isotropic remesh and project
        print("\t\t Resample the convex hull ...");
        remeshedProjectedPelvisConvexHull = isotropicRemeshToTargetEdgeLength(projectedPelvisConvexHull, targetEdgeLength);
        saveMeshToPLY(debugFolder + "/remeshedProjectedPelvisConvexHull.ply", remeshedProjectedPelvisConvexHull);
        projectedPelvisConvexHullVertices = estimateNearestPointsFromPoints(remeshedPelvisConvexHull.vertices, remeshedProjectedPelvisConvexHull.vertices);
        projectedPelvisConvexHull = formMesh(projectedPelvisConvexHullVertices, remeshedPelvisConvexHull.faces);
        saveMeshToPLY(debugFolder + "/projectedPelvisConvexHull.ply", projectedPelvisConvexHull);

    # Isotropic remesh the final time
    print("\t\t Isotropic remesh the final time ...");
    pelvisShape = isotropicRemeshToTargetEdgeLength(projectedPelvisConvexHull, targetEdgeLength);
    saveMeshToPLY(debugFolder + "/pelvisShape.ply", pelvisShape);

    # Sample the pelvis shape
    print("\t\t Sample the pelvis shape ...");
    pelvisShapeSamples = poissonDiskSampleMesh(pelvisShape, 100000);
    save3DPointsToOFFFile(debugFolder + "/pelvisShapeSamples.off", pelvisShapeSamples);

    # Surface reconstruct the pelvis shape
    print("\t\t Surface reconstruct the pelvis shape ...");
    reconPelvisShape = surfaceReconstructionBallPivoting(pelvisShapeSamples);
    saveMeshToPLY(debugFolder + "/reconPelvisShape.ply", reconPelvisShape);

    # Fix the normals of the pelvis shape
    print("\t\t Fix the normals of the pelvis shape ...");
    newNormals = computeCenterToVertexNormals(reconPelvisShape);
    reconPelvisShape.vertex_normals = newNormals;

    # Close holes the mesh
    print("\t\t Close holes the mesh ...");
    reconPelvisShape = closeHolesWithMeshLab(reconPelvisShape, 30000);

    # Return the shape
    print("\t\t Return the shape ...");
    return reconPelvisShape;
def findRayMeshIntersections(inMesh, inStartPoints, inEndPoints):
    """
    Finds intersections of rays with the given mesh.

    :param mesh: Trimesh mesh object
    :param start_points: List or array of starting points
    :param end_points: List or array of ending points
    :return: Array of intersection points
    """
    # Compute ray directions
    directions = np.array(inEndPoints) - np.array(inStartPoints)

    # Perform ray-mesh intersection
    locations, index_ray, index_tri = inMesh.ray.intersects_location(
        ray_origins=inStartPoints, ray_directions=directions
    )

    return locations
def findMeshToMeshIntersections(inSamplingMesh, inTargetMesh):
    """
    Finds intersections of rays from the centroid of the sampling mesh to its vertices with the target mesh.

    :param mesh: Trimesh target mesh object
    :param sampling_mesh: Trimesh sampling mesh object
    :return: Array of intersection points
    """
    # Compute the centroid of the sampling mesh
    centroid = inSamplingMesh.centroid

    # Get the vertices of the sampling mesh
    vertices = inSamplingMesh.vertices

    # Compute ray directions from centroid to vertices
    directions = vertices - centroid

    # Perform ray-mesh intersection
    locations, index_ray, index_tri = inTargetMesh.ray.intersects_location(
        ray_origins=np.array([centroid] * len(vertices)), ray_directions=directions
    )

    return locations
def findMeshToMeshIntersectionUsingNormals(inSamplingMesh, inTargetMesh):
    """
    Finds intersections of rays cast from the vertices of the sampling mesh along their normals.

    :param target_mesh: Trimesh target mesh object
    :param sampling_mesh: Trimesh sampling mesh object
    :return: Array of intersection points
    """
    # Get the vertices and vertex normals of the sampling mesh
    vertices = inSamplingMesh.vertices
    normals = inSamplingMesh.vertex_normals

    # Perform ray-mesh intersection using normals as directions
    locations, index_ray, index_tri = inTargetMesh.ray.intersects_location(
        ray_origins=vertices, ray_directions=normals
    )

    return locations
def projectMeshOntoMesh(sourceMesh, targetMesh):
    """
    Projects vertices of sourceMesh onto targetMesh using normal-based ray intersections.
    If no intersection is found, the nearest point is used.

    Args:
        sourceMesh (trimesh.Trimesh): The source mesh (Mesh A) to be projected.
        targetMesh (trimesh.Trimesh): The target mesh (Mesh B) onto which projection occurs.

    Returns:
        trimesh.Trimesh: A new mesh with projected vertices.
    """
    # Extract vertices and normal vectors
    verticesA = sourceMesh.vertices
    normalsA = sourceMesh.vertex_normals
    verticesB = targetMesh.vertices

    # Initialize KDTree for nearest-neighbor search
    kd_tree = KDTree(verticesB)

    def project_vertex(vertex, normal):
        """Projects a single vertex along its normal direction onto the target mesh."""
        ray_origins = vertex.reshape(1, 3)
        ray_directions = normal.reshape(1, 3)

        # Perform ray-mesh intersection
        locations, index_ray, index_tri = targetMesh.ray.intersects_location(
            ray_origins, ray_directions
        )

        # If intersection exists, return the first intersection point
        if locations.shape[0] > 0:
            return locations[0]

        # If no intersection, fallback to nearest vertex in targetMesh
        _, nearest_index = kd_tree.query(vertex)
        return verticesB[nearest_index]

    # Project all vertices
    projected_vertices = np.array([
        project_vertex(vertex, normal)
        for vertex, normal in zip(verticesA, normalsA)
    ])

    # Create new mesh with projected vertices
    projectedMesh = sourceMesh.copy()
    projectedMesh.vertices = projected_vertices.copy();

    # Free up memory
    del(verticesA, normalsA, verticesB, kd_tree, projected_vertices); gc.collect();

    # Return buffer
    return projectedMesh

#*************************************************************** POINT CLOUD PROCESSING FUNCTIONS
def createTransformMatrix(inTrans, inRot):
    # Create a 4x4 identity matrix
    transform = np.eye(4);

    # Use scipy's Rotation class to create the rotation matrix from Euler angles
    rotation_matrix = R.from_euler('xyz', inRot).as_matrix();

    # Assign the rotation matrix to the top-left part of the 4x4 matrix
    transform[0:3, 0:3] = rotation_matrix;

    # Add translation (inTrans) to the last column
    transform[0:3, 3] = inTrans;

    # Return transform
    return transform;
def createTransformMatrixInPoint(inTrans, inRot, inPoint):
    rotationTransform = np.eye(4);
    rotationMatrix = R.from_euler('xyz', inRot).as_matrix();
    rotationTransform[0:3, 0:3] = rotationMatrix;

    translationTransform = np.eye(4);
    translationTransform[0:3, 3] = inTrans;

    toOriginTransform = np.eye(4);
    toOriginTransform[:3, 3] = -inPoint.flatten();
    
    fromOriginTransform = np.eye(4);
    fromOriginTransform[:3, 3] = inPoint.flatten();

    transform = translationTransform@fromOriginTransform@rotationTransform@toOriginTransform;

    return transform;
def createRotationTransform(inParams, inPoint):
    """
    Create a transformation matrix that rotates around a point using scipy's spatial library.

    Args:
        inParams (list): A list of three rotation angles [rx, ry, rz] in radians (Euler angles).
        inPoint (list): A list of three coordinates [px, py, pz] representing the point to rotate around.

    Returns:
        np.ndarray: The final 4x4 transformation matrix.
    """
    
    # Extract the rotation angles
    rx, ry, rz = inParams
    
    # Create a rotation matrix using scipy's Rotation from Euler angles (in 'xyz' order)
    rotation = R.from_euler('xyz', [rx, ry, rz])
    
    # Convert the rotation matrix to 4x4 homogeneous matrix
    rotation_matrix_4x4 = np.eye(4)
    rotation_matrix_4x4[:3, :3] = rotation.as_matrix()
    
    # Translation to the point of rotation (inPoint)
    translation_to_origin = np.eye(4)
    translation_to_origin[:3, 3] = -np.array(inPoint)
    
    # Translation back from the point of rotation
    translation_back = np.eye(4)
    translation_back[:3, 3] = np.array(inPoint)
    
    # Combine transformations: T_back * R * T_origin
    transformMatrix = translation_back @ rotation_matrix_4x4 @ translation_to_origin
    
    return transformMatrix;
def computeLinearBlendShapeWeights(inVertices, inPoses):
    # Preparing buffers
    vertices = np.array(inVertices);
    bones = np.array(inPoses);

    # Compute the weights
    weights = np.zeros((len(vertices), len(bones)));
    for i, vertex in enumerate(vertices):
        for j, bone in enumerate(bones):
            distance = np.linalg.norm(vertex - bone);
            weights[i, j] = 1/ (distance + 1e-5); # Avoid division by zero
        
    # Normalizing the weights
    weights /= weights.sum(axis=1, keepdims=True);

    # Returning the weights
    return weights;
def computeGaussianBlendShapeWeights(inVertices, inPoses, sigma=0.1):
    # Preparing buffers
    vertices = np.array(inVertices)
    bones = np.array(inPoses)

    # Compute the weights using a Gaussian function
    weights = np.zeros((len(vertices), len(bones)))
    for i, vertex in enumerate(vertices):
        for j, bone in enumerate(bones):
            distance = np.linalg.norm(vertex - bone)
            weights[i, j] = np.exp(-0.5 * (distance / sigma) ** 2)  # Gaussian function
        
    # Normalizing the weights
    weights /= weights.sum(axis=1, keepdims=True)

    # Returning the weights
    return weights
def applyBlendShapeTransform(inVertices, inBlendShapeWeights, inPoseTransforms):
    transformedVertices = np.zeros_like(inVertices);
    for i, vertex in enumerate(inVertices):
        transformed_vertex = np.zeros(4)
        for j, weight in enumerate(inBlendShapeWeights[i]):
            if weight > 0:
                # Convert vertex to homogeneous coordinates
                vertexHomogeneous = np.append(vertex, 1);
                # Apply transformation
                transformed_vertex += weight * (inPoseTransforms[j] @ vertexHomogeneous);
        # Convert back to 3D coordinates
        w = transformed_vertex[3];
        transformedVertices[i] = transformed_vertex[:3] / w;
    return transformedVertices;
def computeRadiusGaussianBlendShapeWeights(vertices, featurePoints, radius):
    """
    Compute Gaussian weights for each vertex based on feature points.
    :param vertices: (N, 3) array of skull mesh vertices
    :param featurePoints: (M, 3) array of feature point positions
    :param radius: Radius of effect for Gaussian weighting
    :return: (N, M) array of weights for each vertex with respect to feature points
    """
    numVertices = vertices.shape[0]
    numFeatures = featurePoints.shape[0]
    weights = np.zeros((numVertices, numFeatures))
    
    for i in range(numFeatures):
        distances = np.linalg.norm(vertices - featurePoints[i], axis=1)
        weights[:, i] = np.exp(- (distances ** 2) / (2 * (radius / 3) ** 2))
    
    # Normalize weights so they sum to 1 for each vertex
    weights /= np.sum(weights, axis=1, keepdims=True)
    return weights
def deformMeshWithBlendShapeWeights(vertices, featurePoints, displacedFeaturePoints, weights):
    """
    Deform the mesh vertices using the motion of feature points.
    :param vertices: (N, 3) array of skull mesh vertices
    :param featurePoints: (M, 3) array of initial feature point positions
    :param displacedFeaturePoints: (M, 3) array of feature points after displacement
    :param weights: (N, M) array of weights computed using Gaussian blending
    :return: (N, 3) array of deformed vertices
    """
    # Compute feature point displacements
    displacements = displacedFeaturePoints - featurePoints
    
    # Apply weighted sum of displacements to each vertex
    deformedVertices = vertices + np.dot(weights, displacements)
    return deformedVertices
def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v);
def rotationMatrixFromVectors(inVec1, inVec2):
    """Find the rotation matrix that aligns vec1 to vec2."""
    a = normalize(inVec1)
    b = normalize(inVec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    # Skew-symmetric cross-product matrix of v
    k_mat = np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])

    # Rotation matrix formula
    rotation_matrix = np.eye(3) + k_mat + k_mat @ k_mat * ((1 - c) / (s ** 2))

    return rotation_matrix
def rotationMatrixAroundVector(inLinePoint, inVector, inTheta):
    # Normalize the vector (axis of rotation)
    vector = np.array(inVector);
    vector = vector / np.linalg.norm(vector);
    
    # Components of the normalized vector
    vx, vy, vz = vector;
    
    # Create the cross-product matrix for the vector (K)
    K = np.array([[0, -vz, vy],
                  [vz, 0, -vx],
                  [-vy, vx, 0]]);
    
    # Identity matrix
    I = np.eye(3);
    
    # Rodrigues' rotation matrix formula
    R = I + np.sin(inTheta) * K + (1 - np.cos(inTheta)) * np.dot(K, K);
    
    # Translation matrix for moving the origin to point_on_line
    T1 = np.eye(4);
    T1[:3, 3] = -np.array(inLinePoint);
    
    # Translation matrix for moving the origin back from point_on_line
    T2 = np.eye(4);
    T2[:3, 3] = np.array(inLinePoint);
    
    # Augmented rotation matrix (4x4) to include translation
    R_augmented = np.eye(4);
    R_augmented[:3, :3] = R;
    
    # Final transformation matrix including translation and rotation
    transformMatrix = np.dot(T2, np.dot(R_augmented, T1));
    
    return transformMatrix;
def rotationMatrixAroundPoint(inXYZAngles, inPoint):
    """
    Generate a 4x4 transformation matrix that rotates around the x, y, and z axes 
    around a specified center point, without scaling (scaling factor is 1).

    Parameters:
        inXYZAngles (tuple of floats): The rotation angles in radians for the x, y, and z axes.
        inPoint (np.ndarray): The center of rotation (x, y, z) as a NumPy array.

    Returns:
        np.ndarray: A 4x4 transformation matrix.
    """
    angle_x, angle_y, angle_z = inXYZAngles
    
    # Rotation matrix for the x-axis
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle_x), -np.sin(angle_x)],
                   [0, np.sin(angle_x), np.cos(angle_x)]])
    
    # Rotation matrix for the y-axis
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                   [0, 1, 0],
                   [-np.sin(angle_y), 0, np.cos(angle_y)]])
    
    # Rotation matrix for the z-axis
    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                   [np.sin(angle_z), np.cos(angle_z), 0],
                   [0, 0, 1]])
    
    # Combine the rotations: first x, then y, then z (order matters)
    R = Rz @ Ry @ Rx

    # Create the full 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = R

    # Handle the translation to rotate around the center point
    translation_to_center = np.eye(4)
    translation_to_center[:3, 3] = -inPoint

    translation_back = np.eye(4)
    translation_back[:3, 3] = inPoint

    # Full transformation: translate to center, rotate, then translate back
    full_transform = translation_back @ transform @ translation_to_center

    return full_transform;
def estimateRigidSVDTransform(inSourcePoints, inTargetPoints):
    # Buffer
    sourcePoints = np.array(inSourcePoints);
    targetPoints = np.array(inTargetPoints);
    
    # Compute centroids of source and target points
    sourceCentroid = np.mean(sourcePoints, axis=0);
    targetCentroid = np.mean(targetPoints, axis=0);

    # Center the points around the centroids
    sourceCenter = sourcePoints - sourceCentroid;
    targetCenter = targetPoints - targetCentroid;

    # Compute the cross-covariance matrix
    H = np.dot(sourceCenter.T, targetCenter);

    # Compute the Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(H);

    # Compute rotation matrix R
    R = np.dot(Vt.T, U.T);

    # Ensure a right-handed coordinate system
    if np.linalg.det(R) < 0:
       Vt[2,:] *= -1;
       R = np.dot(Vt.T, U.T);

    # Compute translation vector t
    t = targetCentroid.T - np.dot(R, sourceCentroid.T);

    # Construct the transformation matrix
    transformMatrix = np.identity(4);
    transformMatrix[:3, :3] = R;
    transformMatrix[:3, 3] = t;
    return transformMatrix;
def transform3DPoints(inPoints, inTransformMatrix):
    # Convert points to homogeneous coordinates
    numPoints = inPoints.shape[0];
    homogeneousPoints = np.hstack((inPoints, np.ones((numPoints, 1))));
    
    # Apply the transformation matrix to the points
    transformedHomogeneousPoints = np.dot(homogeneousPoints, inTransformMatrix.T);
    
    # Convert back to 3-D coordinates
    transformedPoints = transformedHomogeneousPoints[:, :3] / transformedHomogeneousPoints[:, 3][:, np.newaxis];
    return transformedPoints;
def estimateTranslationTransformFromPointToPoint(A, B):
    """
    Estimate the translation vector that maps 3D points from set A to set B.
    
    Args:
        A: A numpy array of shape (N, 3) representing N points in 3D space.
        B: A numpy array of shape (N, 3) representing N points in 3D space.
    
    Returns:
        A 4x4 numpy array representing the translation matrix in homogeneous coordinates.
    """
    # Calculate the translation vector as the mean difference between A and B
    translation_vector = np.mean(B - A, axis=0);
    
    # Create a 4x4 identity matrix and insert the translation vector
    translation_matrix = np.eye(4);
    translation_matrix[:3, 3] = translation_vector;
    
    return translation_matrix;
def estimateTranslationAlongPlane(translation_vector, plane_normal, plane_point):
    """
    Estimate the translation matrix for a given translation vector, constrained to move parallel to a plane.
    
    Args:
        translation_vector: A numpy array of shape (3,) representing the translation vector in 3D space.
        plane_normal: A numpy array of shape (3,) representing the normal vector of the plane.
        plane_point: A numpy array of shape (3,) representing a point on the plane (not used for calculation).
    
    Returns:
        A 4x4 numpy array representing the translation matrix in homogeneous coordinates.
    """
    # Normalize the plane normal vector
    plane_normal = plane_normal / np.linalg.norm(plane_normal);
    
    # Project the translation vector onto the plane (remove the component along the plane normal)
    translation_along_plane = translation_vector - np.dot(translation_vector, plane_normal) * plane_normal;
    
    # Create a 4x4 identity matrix and insert the translation along the plane
    translation_matrix = np.eye(4);
    translation_matrix[:3, 3] = translation_along_plane;
    
    return translation_matrix;
def select3DPointsWithIndices(inPoints, inIndices):
    indices = np.array(inIndices).reshape((1, len(inIndices.flatten())));
    selPoints = inPoints[indices].squeeze(0);
    return selPoints;
def estimateNearestIndicesMinimumBased(inSourcePoints, inTargetPoints):
    # Create two arrays of 3D points (replace with your data)
    X = np.array(inSourcePoints);
    Y = np.array(inTargetPoints);

    # Calculate pairwise distances
    D = np.sqrt(np.sum((X[:, None] - Y[None, :])**2, axis=-1));

    # Get the indices of the nearest neighbors
    nearestIndices = np.argmin(D, axis=1);

    # Return
    return nearestIndices;
def estimateNearestIndicesKDTreeBased(inSourcePoints, inTargetPoints, inThreshold=1e-6):
    # Prepare buffers
    sourcePoints = np.array(inSourcePoints);
    targetPoints = np.array(inTargetPoints);

    # Create a KD-tree for the body vertices
    targetPointTree = KDTree(targetPoints);
    
    # Find the distances from each head vertex to the nearest body vertex
    distances, indices = targetPointTree.query(sourcePoints);
    
    # Return buffer
    return indices;
def estimateNearestIndicesWithinRadius(inSingleSourcePoint, inTargetPoints, inRadius=0.01):
    # Prepare buffers
    targetPoints = np.array(inTargetPoints);

    # Create a KD-tree for the target points
    targetPointTree = KDTree(targetPoints);

    # Find the indices of points within the specified radius
    indices = targetPointTree.query_ball_point(inSingleSourcePoint, r=inRadius);

    # If no points are found, return an empty array
    return indices;
def save3DPointsToOFFFile(inFilePath, inPoints):
    # Forming the trimesh buffer
    pointMesh = trimesh.Trimesh(vertices=inPoints, faces=[], process=False);
    pointMesh.export(inFilePath, file_type="off");
def estimateAffineTransformCPD(inSourcePoints, inTargetPoints):
    # Checking inputs
    if (len(inSourcePoints.flatten()) == 0):
        print("The inFromPoints are empty."); return None;
    if (len(inTargetPoints.flatten()) == 0):
        print("The inToPoints are empty."); return None;

    # Find the affine transform
    targetPoints = np.array(inTargetPoints);
    sourcePoints = np.array(inSourcePoints);
    source_homogeneous = np.hstack((sourcePoints, np.ones((len(sourcePoints), 1))));
    target_homogeneous = np.hstack((targetPoints, np.ones((len(targetPoints), 1))));
    transformMatrix, _ = np.linalg.lstsq(source_homogeneous, target_homogeneous, rcond=None)[:2];
    
    # Return the affine transform
    return transformMatrix;
def estimateICPRigidTransform(inSourcePoints, inTargetPoints, inThresHold=0.02):
    source = o3d.geometry.PointCloud();
    target = o3d.geometry.PointCloud();
    source.points = o3d.utility.Vector3dVector(inSourcePoints);
    target.points = o3d.utility.Vector3dVector(inTargetPoints);
    registrationPoint2Point = o3d.pipelines.registration.registration_icp(source, target, max_correspondence_distance=inThresHold, 
                                                          estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint());
    return registrationPoint2Point.transformation;
def estimateRigidTransformCPD_Slow(inSourcePoints, inTargetPoints, inMaxIteration=100, inTolerance=1e-5):

    X = np.array(inSourcePoints);
    Y = np.array(inTargetPoints);

    N, D = X.shape
    M, _ = Y.shape

    # Initialize parameters
    R = np.eye(D)
    t = np.zeros((D,))
    sigma2 = np.var(X - Y.mean(axis=0))

    for iteration in range(inMaxIteration):
        # E-step: Compute the probability matrix
        P = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                P[i, j] = np.exp(-np.linalg.norm(X[i] - (R @ Y[j] + t))**2 / (2 * sigma2))
        P /= P.sum(axis=1, keepdims=True)

        # M-step: Update the transformation parameters
        mu_X = np.sum(P @ X, axis=0) / P.sum()
        mu_Y = np.sum(P.T @ Y, axis=0) / P.sum()

        X_hat = X - mu_X
        Y_hat = Y - mu_Y

        A = X_hat.T @ P @ Y_hat
        U, _, Vt = np.linalg.svd(A)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt

        t = mu_X - R @ mu_Y

        # Check for convergence
        X_transformed = R @ Y.T + t[:, np.newaxis]
        error = np.linalg.norm(X - X_transformed.T)
        if error < inTolerance:
            break

    # Create the 4x4 transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R
    transform_matrix[:3, 3] = t

    return transform_matrix;
def estimateNormalsForPointCloud(inSourcePoints):
    # Checking inSourcePoints
    if (len(inSourcePoints) == 0):
        print("estimateNormalsForPointClouds:: The inSourcePoints are empty."); return [];

    # Forming the open3d point cloud
    sourcePoints = np.array(inSourcePoints);
    pointCloud = o3d.geometry.PointCloud();
    pointCloud.points = o3d.utility.Vector3dVector(sourcePoints);

    # Compute the normal vector
    pointCloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30));

    # Return the normal
    return np.asarray(pointCloud.normals);
def estimateSingleNormalForPointCloud(inSourcePoints):
    # Checking input
    if (len(inSourcePoints) == 0):
        print("estimateSingleNormalForPointCloud:: The inSourcePoints are empty."); return [];

    # Compute centroids
    sourcePoints = np.array(inSourcePoints);
    centroid = np.mean(sourcePoints, axis=0);
    centeredPoints = sourcePoints - centroid;
    _, _, vh = np.linalg.svd(centeredPoints);
    normalVector = vh[-1];

    # Finish processing
    return normalVector;
def computeCentroidPoint(in3DPoints):
    if in3DPoints.shape[1] != 3:
        raise ValueError("computeCentroidPoint:: Input array must have shape Nx3");
    
    centroid = np.mean(in3DPoints, axis=0);
    return centroid;
def scale3DPoints(in3DPoints, inScaleFactor):
    scaledPoints = in3DPoints * inScaleFactor;
    return scaledPoints;
def computeAveragePointsToPointsDistance(inFromPoints, inToPoints):
    # Initialize buffer
    fromPoints = np.array(inFromPoints);
    toPoints = np.array(inToPoints);

    # Create a KD-tree for the body vertices
    toPointTree = KDTree(toPoints);
    
    # Find the distances from each head vertex to the nearest body vertex
    distances, indices = toPointTree.query(fromPoints);

    # Return the average distance
    return np.mean(distances);
def computeCorrespondingDistancesPoints2Points(inFromPoints, inToPoints):
    # Initialize buffer
    fromPoints = np.array(inFromPoints);
    toPoints = np.array(inToPoints);

    # Compute euclidean distances between corresponding points
    distances = np.linalg.norm(fromPoints - toPoints, axis=1);

    # Return the average distance
    return distances;
def estimateNearestPointsFromPoints(inSourcePoints, inTargetPoints):
    tree = KDTree(inTargetPoints);
    distances, indices = tree.query(inSourcePoints);
    nearestPoints = inTargetPoints[indices];
    return nearestPoints;
def computeChamferDistance(inPointA, inPointB):
    # Initialize KDTree
    treeA = KDTree(inPointA);
    treeB = KDTree(inPointB);

    # Compute distances
    AtoBDistances, _ = treeA.query(inPointB);
    BtoADistances, _ = treeB.query(inPointA);

    # Compute the sum of average distance
    chamferDistance = np.mean(AtoBDistances) + np.mean(BtoADistances);
    return chamferDistance;
def transform3DPoint(inPoint, inTransform):
    pointHomo = np.append(inPoint, 1);
    transformedPointHomo = np.dot(inTransform, pointHomo);
    transformedPoint = transformedPointHomo[:3];
    return transformedPoint;
def select3DPoints(inSourcePoints, inIndices):
    return inSourcePoints[inIndices].copy();
def project3DTo2DPoints(inPoints3D, inNormal, inPointOnPlane):
    projected_points = []
    inNormal = np.array(inNormal)
    inPointOnPlane = np.array(inPointOnPlane)
    for point in inPoints3D:
        point = np.array(point)
        vector = point - inPointOnPlane
        distance = np.dot(vector, inNormal)
        projected_point = point - distance * inNormal
        projected_points.append(tuple(projected_point[:2]))  # Map to 2D
    return np.array(projected_points);
def reProject2DTo3DPoints(inPoints2D, inNormal, inPointOnPlane):
    reprojected_points = []
    inNormal = np.array(inNormal)
    inPointOnPlane = np.array(inPointOnPlane)

    # Ensure the normal vector is normalized
    inNormal = inNormal / np.linalg.norm(inNormal)

    for point in inPoints2D:
        point_2d = np.array([point[0], point[1], 0])  # Assuming initial z = 0
        vector = point_2d - inPointOnPlane

        # Calculate the distance from the point to the plane along the normal
        distance = -np.dot(vector, inNormal)
        
        # Calculate the reprojected 3D point
        reprojected_point = point_2d + distance * inNormal
        reprojected_points.append(tuple(reprojected_point))

    return np.array(reprojected_points);
def estimate2DConvexHull(in3DPoints):
    in3DPoints = np.array(in3DPoints)
    hull = ConvexHull(in3DPoints)
    hull_points = in3DPoints[hull.vertices]
    return hull_points
def isometricallyDivideLine(start_point, end_point, num_points):
    if num_points == 0:
        return []
    return np.linspace(start_point, end_point, num_points, endpoint=False)[1:]
def divide2DPointsIsometrically(in2DPoints, inNumPoints):
    num_hull_points = len(in2DPoints)
    if num_hull_points == 0:
        return np.array([])
    
    perimeter = sum(np.linalg.norm(in2DPoints[(i + 1) % num_hull_points] - in2DPoints[i]) for i in range(num_hull_points))
    total_points = num_hull_points * inNumPoints
    if total_points == 0:
        return np.array([])
    
    point_spacing = perimeter / total_points
    divided_points = []
    
    for i in range(num_hull_points):
        start_point = in2DPoints[i]
        end_point = in2DPoints[(i + 1) % num_hull_points]
        segment_length = np.linalg.norm(end_point - start_point)
        if point_spacing == 0:
            num_divide_points = 0
        else:
            num_divide_points = int(segment_length // point_spacing)
        divided_points.extend(isometricallyDivideLine(start_point, end_point, num_divide_points));
    
    return np.array(divided_points);
def divide3DPointsInPlaneIsometrically(in3DPoints, inNumPoints):
    # Estimate the in3DPoints
    point3Ds = in3DPoints.copy();

    # Estimate normals and centroid
    centroid = np.mean(point3Ds, axis=0);
    normal = estimateSingleNormalForPointCloud(point3Ds);

    # Project 2-D points
    pro2DPoints = project3DTo2DPoints(point3Ds, normal, centroid);

    # Isometrically divide 2-D points
    div2DPoints = divide2DPointsIsometrically(pro2DPoints, inNumPoints);

    # Back project to 3-D points
    div3DPoints = reProject2DTo3DPoints(div2DPoints, normal, centroid);

    # Return output
    return np.array(div3DPoints);
def estimateIsometricalConvexHullPointsInPlane(inPlanePoints, inNumPoints):
    planePoints = inPlanePoints.copy();
    
    centroid = np.mean(planePoints, axis=0);
    normal = estimateSingleNormalForPointCloud(planePoints);
    points2D = project3DTo2DPoints(planePoints, normal, centroid);

    hull2DPoints = estimate2DConvexHull(points2D); hull2DPoints = np.array(hull2DPoints);
    dividedHullPoints = divide2DPointsIsometrically(hull2DPoints, inNumPoints);

    hull3DPoints = reProject2DTo3DPoints(dividedHullPoints, normal, centroid);
    hull3DPoints = np.array(hull3DPoints);

    return hull3DPoints;
def project3DPointsTo3DPoints(inFromPoints, inToPoints):
    tree = KDTree(inToPoints);
    distances, indices = tree.query(inFromPoints);
    nearest_points = [inToPoints[i] for i in indices];
    return np.array(nearest_points);
def normalizedVector(inVector):
    """
    Normalize a 3D vector.

    Parameters:
    vector (np.ndarray): A 1x3 numpy array representing the vector.

    Returns:
    np.ndarray: The normalized 3D vector.
    """
    magnitude = np.linalg.norm(inVector)
    if magnitude == 0:
        raise ValueError("Cannot normalize a zero vector")
    return inVector / magnitude
def anlgeBetweenVectors(inVector1, inVector2):
    """
    Compute the angle between two vectors in 3-D space.

    Parameters:
    v1 (ndarray): The first vector.
    v2 (ndarray): The second vector.

    Returns:
    float: The angle between the vectors in radians.
    """
    # Getting buffers
    v1 = inVector1.copy();
    v2 = inVector2.copy();

    # Ensure the vectors are NumPy arrays
    v1 = np.array(v1);
    v2 = np.array(v2);

    # Compute the dot product
    dot_product = np.dot(v1, v2);

    # Compute the magnitudes (norms) of the vectors
    norm_v1 = np.linalg.norm(v1);
    norm_v2 = np.linalg.norm(v2);

    # Compute the cosine of the angle
    cos_theta = dot_product / (norm_v1 * norm_v2);

    # Compute the angle in radians
    angle = np.arccos(cos_theta);

    # Return angle
    return angle;
def estimatePlaneNormalCentroid(inPoint3Ds):
    """
    Estimates the normal vector and centroid of a plane from 3D points.

    Parameters:
        points (np.ndarray): An Nx3 array where each row is a 3D point [x, y, z].

    Returns:
        (tuple): A tuple containing:
                 - centroid (np.ndarray): The centroid of the points as a 1x3 array.
                 - normal (np.ndarray): The normal vector of the plane as a 1x3 array.
    """
    # Convert points to a numpy array if they aren't already
    inPoint3Ds = np.array(inPoint3Ds)
    
    # Calculate the centroid of the points
    centroid = inPoint3Ds.mean(axis=0)
    
    # Center the points by subtracting the centroid
    centered_points = inPoint3Ds - centroid
    
    # Perform Singular Value Decomposition (SVD) to find the normal
    _, _, vh = np.linalg.svd(centered_points)
    
    # The normal vector is the last row of vh (smallest singular value direction)
    normal = vh[-1]
    
    return centroid, normal
def projectToPlaneOfPoints(in3DPoints):
    """
    Projects 3D points onto a best-fit plane and returns their 2D coordinates and reprojected 3D points.
    
    Parameters:
        points (np.ndarray): An Nx3 array where each row is a 3D point [x, y, z].

    Returns:
        tuple: (projected_2D, reprojected_3D) where:
            - projected_2D is an Nx2 array of 2D coordinates on the plane.
            - reprojected_3D is an Nx3 array of 3D coordinates, back on the original plane.
    """
    # Ensure points is an array
    in3DPoints = np.array(in3DPoints)
    
    # Step 1: Calculate the centroid and normal vector of the plane
    centroid = in3DPoints.mean(axis=0)
    centered_points = in3DPoints - centroid
    _, _, vh = np.linalg.svd(centered_points)
    normal = vh[-1]

    # Step 2: Create a local coordinate system on the plane
    # First axis (u) can be any vector orthogonal to the normal
    u = np.cross([1, 0, 0], normal)
    if np.linalg.norm(u) < 1e-6:  # In case normal is collinear with x-axis
        u = np.cross([0, 1, 0], normal)
    u = u / np.linalg.norm(u)  # Normalize

    # Second axis (v) is orthogonal to both the normal and u
    v = np.cross(normal, u)
    
    # Step 3: Project each 3D point to the plane in 2D
    projected_2D = []
    for point in centered_points:
        x = np.dot(point, u)
        y = np.dot(point, v)
        projected_2D.append([x, y])
    projected_2D = np.array(projected_2D)

    # Step 4: Reproject 2D points back to 3D
    reprojected_3D = []
    for x, y in projected_2D:
        reprojected_point = centroid + x * u + y * v
        reprojected_3D.append(reprojected_point)
    reprojected_3D = np.array(reprojected_3D)
    
    return projected_2D, reprojected_3D
def estimatePlaneCoordinateSystem(in3DPoints):
    """
    Projects 3D points onto a best-fit plane and returns their 2D coordinates and reprojected 3D points.
    
    Parameters:
        points (np.ndarray): An Nx3 array where each row is a 3D point [x, y, z].

    Returns:
        tuple: (projected_2D, reprojected_3D) where:
            - projected_2D is an Nx2 array of 2D coordinates on the plane.
            - reprojected_3D is an Nx3 array of 3D coordinates, back on the original plane.
    """
    # Ensure points is an array
    in3DPoints = np.array(in3DPoints)
    
    # Step 1: Calculate the centroid and normal vector of the plane
    origin = in3DPoints.mean(axis=0)
    centered_points = in3DPoints - origin
    _, _, vh = np.linalg.svd(centered_points)
    normal = vh[-1]

    # Step 2: Create a local coordinate system on the plane
    # First axis (u) can be any vector orthogonal to the normal
    u = np.cross([1, 0, 0], normal)
    if np.linalg.norm(u) < 1e-6:  # In case normal is collinear with x-axis
        u = np.cross([0, 1, 0], normal)
    u = u / np.linalg.norm(u)  # Normalize

    # Second axis (v) is orthogonal to both the normal and u
    v = np.cross(normal, u);

    return origin, u, v;
def project3DPointsToPlaneCoordinateSystem(in3DPoints, origin, u, v):
    centered_points = in3DPoints - origin;        
    projected2DPoints = [];
    for point in centered_points:
        x = np.dot(point, u);
        y = np.dot(point, v);
        projected2DPoints.append([x, y]);
    projected2DPoints = np.array(projected2DPoints);
    return projected2DPoints;
def reProject2DPointsFromPlaneCoordinateSystem(in2DPoints, origin, u, v):
    reprojected_3D = [];
    for x, y in in2DPoints:
        reprojected_point = origin + x * u + y * v;
        reprojected_3D.append(reprojected_point);
    reprojected_3D = np.array(reprojected_3D);
    return reprojected_3D;
def computeVectorAngle(v1, v2, inDegrees=False):
    """
    Calculate the angle between two vectors in 3D space.

    Parameters:
        v1 (array-like): The first vector [x1, y1, z1].
        v2 (array-like): The second vector [x2, y2, z2].
        in_degrees (bool): If True, return the angle in degrees; otherwise, in radians.

    Returns:
        float: The angle between the two vectors in radians (default) or degrees.
    """
    # Convert inputs to numpy arrays
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    # Compute dot product and magnitudes
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    
    # Prevent division by zero
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        raise ValueError("One or both vectors have zero magnitude.")
    
    # Compute the cosine of the angle
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    
    # Clamp the cosine value to the valid range [-1, 1] to avoid numerical issues
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Calculate the angle in radians
    angle = np.arccos(cos_theta)
    
    # Convert to degrees if requested
    if inDegrees:
        angle = np.degrees(angle)
    
    return angle
def computeSignedAngle(v1, v2, normal, in_degrees=False):
    """
    Calculate the signed angle between two vectors in 3D space.

    Parameters:
        v1 (array-like): The first vector [x1, y1, z1].
        v2 (array-like): The second vector [x2, y2, z2].
        normal (array-like): The normal vector defining the plane for the signed angle.
        in_degrees (bool): If True, return the angle in degrees; otherwise, in radians.

    Returns:
        float: The signed angle between the two vectors in radians (default) or degrees.
    """
    # Convert inputs to numpy arrays
    v1 = np.array(v1)
    v2 = np.array(v2)
    normal = np.array(normal)
    
    # Compute dot product and cross product
    dot_product = np.dot(v1, v2)
    cross_product = np.cross(v1, v2)
    
    # Compute magnitudes
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    
    # Prevent division by zero
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        raise ValueError("One or both vectors have zero magnitude.")
    
    # Compute the cosine of the angle
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Clamp to avoid numerical errors
    
    # Compute the angle in radians
    angle = np.arccos(cos_theta)
    
    # Determine the sign using the dot product of the cross product with the normal vector
    sign = np.sign(np.dot(cross_product, normal))
    signed_angle = sign * angle
    
    # Convert to degrees if requested
    if in_degrees:
        signed_angle = np.degrees(signed_angle)
    
    return signed_angle
def projectPointToPlane(point, planePoint, normal_vector):
    """
    Project a 3D point onto a plane defined by a point and normal vector.

    Args:
    - point: np.array, the point to be projected (x, y, z).
    - plane_point: np.array, a point on the plane (x0, y0, z0).
    - normal_vector: np.array, the normal vector to the plane (a, b, c).

    Returns:
    - np.array, the projected point on the plane (x', y', z').
    """
    point = np.array(point)
    planePoint = np.array(planePoint)
    normal_vector = np.array(normal_vector)
    
    # Calculate the vector from plane_point to the point
    v = point - planePoint
    
    # Calculate the projection scalar
    t = np.dot(v, normal_vector) / np.dot(normal_vector, normal_vector)
    
    # Subtract the projection vector from the original point
    projection = point - t * normal_vector
    
    return projection
def projectPointOntoVector3D(point, vector):
    """
    Projects a 3D point onto a 3D vector.

    Parameters:
        point (array-like): A 3D point (x, y, z).
        vector (array-like): A 3D vector (vx, vy, vz).

    Returns:
        numpy.ndarray: The projected point on the vector.
    """
    point = np.array(point)
    vector = np.array(vector)
    
    # Normalize the vector
    vectorNorm = vector / np.linalg.norm(vector)
    
    # Compute the projection scalar
    projectionScalar = np.dot(point, vectorNorm)
    
    # Compute the projected point
    projectedPoint = projectionScalar * vectorNorm
    
    return projectedPoint
def computeRotationMatrixToAlignVector(sourceVector, targetVector):
    """
    Computes a 4x4 rotation matrix to rotate a source vector to align with a target vector.

    Parameters:
        sourceVector (array-like): The initial vector (e.g., [x, y, z]).
        targetVector (array-like): The target vector (e.g., [x, y, z]).

    Returns:
        numpy.ndarray: A 4x4 rotation matrix.
    """
    # Normalize the input vectors
    sourceVector = np.array(sourceVector) / np.linalg.norm(sourceVector)
    targetVector = np.array(targetVector) / np.linalg.norm(targetVector)

    # Compute the cross product and the angle
    crossProduct = np.cross(sourceVector, targetVector)
    dotProduct = np.dot(sourceVector, targetVector)
    angle = np.arccos(np.clip(dotProduct, -1.0, 1.0))  # Clip to avoid numerical issues

    # If the vectors are nearly identical, return the identity matrix
    if np.isclose(angle, 0):
        return np.eye(4)

    # If the vectors are opposite, we need a special case
    if np.isclose(angle, np.pi):
        # Find an orthogonal vector to use as the axis
        orthogonal = np.array([1, 0, 0]) if abs(sourceVector[0]) < 0.9 else np.array([0, 1, 0])
        axis = np.cross(sourceVector, orthogonal)
        axis /= np.linalg.norm(axis)
        angle = np.pi
    else:
        # Normalize the cross product to get the rotation axis
        axis = crossProduct / np.linalg.norm(crossProduct)

    # Compute the rotation matrix using the axis-angle representation
    ux, uy, uz = axis
    cosAngle = np.cos(angle)
    sinAngle = np.sin(angle)

    rotationMatrix3x3 = np.array([
        [cosAngle + ux**2 * (1 - cosAngle), ux * uy * (1 - cosAngle) - uz * sinAngle, ux * uz * (1 - cosAngle) + uy * sinAngle],
        [uy * ux * (1 - cosAngle) + uz * sinAngle, cosAngle + uy**2 * (1 - cosAngle), uy * uz * (1 - cosAngle) - ux * sinAngle],
        [uz * ux * (1 - cosAngle) - uy * sinAngle, uz * uy * (1 - cosAngle) + ux * sinAngle, cosAngle + uz**2 * (1 - cosAngle)]
    ])

    # Convert the 3x3 matrix to a 4x4 matrix
    rotationMatrix4x4 = np.eye(4)
    rotationMatrix4x4[:3, :3] = rotationMatrix3x3

    return rotationMatrix4x4
def computeRotationMatrixToAlignVectors(sourceDirection, sourceViewUp, targetDirection, targetViewUp):
    """
    Computes a 4x4 rotation matrix to align a source direction and view-up with a target direction and view-up.

    Parameters:
        sourceDirection (array-like): The initial direction vector (e.g., [dx, dy, dz]).
        sourceViewUp (array-like): The initial view-up vector (e.g., [vx, vy, vz]).
        targetDirection (array-like): The target direction vector (e.g., [dx, dy, dz]).
        targetViewUp (array-like): The target view-up vector (e.g., [vx, vy, vz]).

    Returns:
        numpy.ndarray: A 4x4 rotation matrix.
    """
    # Normalize the input vectors
    sourceDirection = np.array(sourceDirection) / np.linalg.norm(sourceDirection)
    sourceViewUp = np.array(sourceViewUp) / np.linalg.norm(sourceViewUp)
    targetDirection = np.array(targetDirection) / np.linalg.norm(targetDirection)
    targetViewUp = np.array(targetViewUp) / np.linalg.norm(targetViewUp)

    # Compute the right vector for both source and target coordinate systems
    sourceRight = np.cross(sourceDirection, sourceViewUp)
    sourceRight /= np.linalg.norm(sourceRight)

    targetRight = np.cross(targetDirection, targetViewUp)
    targetRight /= np.linalg.norm(targetRight)

    # Create the rotation matrix for aligning the direction and up vectors
    rotationMatrix3x3 = np.zeros((3, 3))
    rotationMatrix3x3[:, 0] = targetRight
    rotationMatrix3x3[:, 1] = targetViewUp
    rotationMatrix3x3[:, 2] = -targetDirection

    # Compute the transformation matrix that converts the source basis to the target basis
    sourceBasis = np.column_stack((sourceRight, sourceViewUp, -sourceDirection))
    rotationMatrix = np.dot(rotationMatrix3x3, np.linalg.inv(sourceBasis))

    # Convert the 3x3 matrix to a 4x4 matrix
    rotationMatrix4x4 = np.eye(4)
    rotationMatrix4x4[:3, :3] = rotationMatrix

    return rotationMatrix4x4
def computeTransformFromCentroidDirections(fromCentroid, fromForwardDirection, fromUpDirection,
                           toCentroid, toForwardDirection, toUpDirection):
    # Normalize the direction vectors
    fromForwardDirection = fromForwardDirection / np.linalg.norm(fromForwardDirection)
    fromUpDirection = fromUpDirection / np.linalg.norm(fromUpDirection)
    toForwardDirection = toForwardDirection / np.linalg.norm(toForwardDirection)
    toUpDirection = toUpDirection / np.linalg.norm(toUpDirection)
    
    # Calculate right vectors
    trapRightDirection = np.cross(fromForwardDirection, fromUpDirection)
    targetRightDirection = np.cross(toForwardDirection, toUpDirection)
    
    # Form rotation matrices
    trapRotationMatrix = np.vstack([trapRightDirection, fromUpDirection, fromForwardDirection]).T
    targetRotationMatrix = np.vstack([targetRightDirection, toUpDirection, toForwardDirection]).T
    
    # Compute the rotation from trap to target
    rotationMatrix = targetRotationMatrix @ np.linalg.inv(trapRotationMatrix)
    
    # Compute the translation from trap to target
    translationVector = toCentroid - rotationMatrix @ fromCentroid
    
    # Form the transformation matrix
    transformationMatrix = np.eye(4)
    transformationMatrix[:3, :3] = rotationMatrix
    transformationMatrix[:3, 3] = translationVector
    
    return transformationMatrix
def normalize3DVector(vector):
    vector = vector / np.linalg.norm(vector);
    return vector;
def decomposeRigidTransform(inTransformMatrix):
    """
    Decomposes a 4x4 rigid transformation matrix into translation and rotation (Euler angles in XYZ order).
    
    Args:
        matrix4x4 (np.ndarray): A 4x4 homogeneous transformation matrix.
        
    Returns:
        translation (np.ndarray): A 3-element array [tx, ty, tz].
        euler_angles (np.ndarray): A 3-element array [rx, ry, rz] in radians.
    """
    assert inTransformMatrix.shape == (4, 4), "Input must be a 4x4 matrix."

    # Extract translation
    translation = inTransformMatrix[:3, 3]

    # Extract rotation matrix
    rotation_matrix = inTransformMatrix[:3, :3]

    # Convert to Euler angles (XYZ order)
    rotation = R.from_matrix(rotation_matrix)
    euler_angles = rotation.as_euler('xyz', degrees=False)

    return translation, euler_angles

#*************************************************************** 2-D IMAGE PROCESSING
def saveImage(filePath, image):
    cv2.imwrite(filePath, image);
def VTKToNumpyImage(imageVTK, imageWidth, imageHeight):
    # Convert the image VTK to numpy array
    numComponents = imageVTK.GetNumberOfScalarComponents();
    vtkImageArray = imageVTK.GetPointData().GetScalars();
    numValues = imageWidth * imageHeight * numComponents;
    imageArray = np.zeros(numValues, dtype=np.uint8)
    vtkImageArray.ExportToVoidPointer(imageArray);
    imageArray = imageArray.reshape(imageHeight, imageWidth, numComponents);
    imageArray = np.flipud(imageArray);

    # Convert numpy array to opencv image
    if numComponents == 3:
        # Convert the NumPy array to an OpenCV image (BGR format)
        opencvImage = cv2.cvtColor(imageArray, cv2.COLOR_RGB2BGR);
    elif numComponents == 4:
        # Convert the NumPy array to an OpenCV image (BGRA format)
        opencvImage = cv2.cvtColor(imageArray, cv2.COLOR_RGBA2BGRA);
    else:
        raise ValueError("Unsupported number of components");

    # Return buffer
    return opencvImage;
def numpyToVTKImage(numpyImage):
    # Convert the numpy array back to VTK image format
    vtkImage = vtk.vtkImageData()
    vtkImage.SetDimensions(numpyImage.shape[1], numpyImage.shape[0], 1)
    vtkImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)  # RGB image

    for i in range(numpyImage.shape[0]):
        for j in range(numpyImage.shape[1]):
            pixel = numpyImage[i, j]
            vtkImage.SetScalarComponentFromDouble(j, i, 0, 0, pixel[0])
            vtkImage.SetScalarComponentFromDouble(j, i, 0, 1, pixel[1])
            vtkImage.SetScalarComponentFromDouble(j, i, 0, 2, pixel[2])

    return vtkImage
def formCameraMatrixDistCoeffs(focalLength, sensorWidth, sensorHeight, imageWidth, imageHeight, cameraK1, cameraK2, cameraP1, cameraP2, cameraK3):
    # Compute focal lengths in pixels (fx, fy)
    fx = (focalLength * imageWidth) / sensorWidth;
    fy = (focalLength * imageHeight) / sensorHeight;
    
    # Camera matrix (K)
    K = np.array([[fx, 0, imageWidth / 2],
                  [0, fy, imageHeight / 2],
                  [0, 0, 1]], dtype=np.float32);
    
    # Distortion coefficients
    distCoeffs = np.array([cameraK1, cameraK2, cameraP1, cameraP2, cameraK3]);

    # Return camera matrix and coeffs
    return K, distCoeffs;
def readImage(filePath):
    """
    Reads an image from the specified file path using OpenCV.

    Parameters:
        filePath (str): The path to the image file.

    Returns:
        image (numpy.ndarray): The loaded image, or None if the file couldn't be read.
    """
    image = cv2.imread(filePath)
    if image is None:
        raise FileNotFoundError(f"Image file not found at: {filePath}")
    return image
def computeDistortionMaps(cameraMatrix, distCoeffs, imageWidth, imageHeight, mapType=cv2.CV_32FC1):
    # Ensure imageSize is a tuple of integers
    imageSize = (int(imageWidth), int(imageHeight))  # Width x Height

    # Ensure newCameraMatrix is a copy of the original cameraMatrix
    newCameraMatrix = cameraMatrix.copy()

    # Compute the distortion maps
    map1, map2 = cv2.initUndistortRectifyMap(
        cameraMatrix, distCoeffs, None, newCameraMatrix, imageSize, mapType
    )

    return map1, map2;
def applyDistortionIntoImage(image, map1, map2, interpolation=cv2.INTER_LINEAR):
    """
    Applies distortion to an image using precomputed distortion maps.

    Parameters:
        image (numpy.ndarray): The input image to distort.
        map1 (numpy.ndarray): The first distortion map (x-coordinates).
        map2 (numpy.ndarray): The second distortion map (y-coordinates or interpolation indices).
        interpolation (int): Interpolation method (default is cv2.INTER_LINEAR).

    Returns:
        numpy.ndarray: The distorted image.
    """
    if image is None:
        raise ValueError("Input image is None. Please provide a valid image.")
    
    distortedImage = cv2.remap(image, map1, map2, interpolation)
    return distortedImage
def computeProjectionmatrixFromKandDistorts(cameraMatrix, distortionCoeffs):
    fx, fy, cx, cy = cameraMatrix[0, 0], cameraMatrix[1, 1], cameraMatrix[0, 2], cameraMatrix[1, 2]
    k1, k2, p1, p2, k3 = distortionCoeffs

    # Create a new projection matrix with distortion
    projectionMatrix = np.eye(4)

    # Apply intrinsic parameters
    projectionMatrix[0, 0] = fx
    projectionMatrix[1, 1] = fy
    projectionMatrix[0, 2] = cx
    projectionMatrix[1, 2] = cy

    # Incorporate distortion in a custom way (e.g., modifying the third column)
    projectionMatrix[0, 3] = k1
    projectionMatrix[1, 3] = k2
    projectionMatrix[2, 3] = p1 + p2  # Simplified, for demonstration

    return projectionMatrix
def numpyToVKMatrix(numpy_matrix):
    vtkMatrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            vtkMatrix.SetElement(i, j, numpy_matrix[i, j])
    return vtkMatrix
def calibrateMonoCamera(imageFolder, chessboardSideLength=0.1, numWidthCorners=9, numHeightCorners=7, imageType="png"):
    # Termination criteria for corner sub-pixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.00001);
    
    # Prepare object points based on the chessboard dimensions
    objp = np.zeros((numWidthCorners * numHeightCorners, 3), np.float32);
    objp[:, :2] = np.mgrid[0:numWidthCorners, 0:numHeightCorners].T.reshape(-1, 2);
    objp *= chessboardSideLength;
    
    # Arrays to store object points and image points from all the images
    objpoints = [];  # 3d point in real world space
    imgpoints = [];  # 2d points in image plane
    imageNames = getAllFileNames(imageFolder);
    numOfImages = len(imageNames);
    
    # Get all images     
    for i in range(0, numOfImages):
        # Debugging
        print("Processing image: ", i, end="", flush=True);

        # Reading image
        imageName = imageNames[i];
        img = cv2.imread(imageFolder + f"/{imageName}.{imageType}");
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
        
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (numWidthCorners, numHeightCorners), None);
        
        # Checking the returning image
        if (not ret):
            print(" -> Cannot detect chessboard.");
            continue;

        # Checking corners
        first_point = corners[0][0]
        mean_x = np.mean(corners[:, 0, 0])
        mean_y = np.mean(corners[:, 0, 1])
        if not (first_point[0] < mean_x and first_point[1] < mean_y):
            print(" -> Direction of chessboard is wrong.");
            continue;
        print("");

        # Get object points and image points
        objpoints.append(objp);
        imagePoints = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria);
        imgpoints.append(imagePoints);
        
    # Calibrate the camera
    print("Calibrating camera ...");
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    # Return the camera intrinsic parameters
    print("Returning parameters ...");
    return mtx, dist, ret;
def saveMonoCalibResultsToCSV(csvFilePath, cameraMatrix, cameraDistCoeffs, calibrationErrors):
    """
    Save camera matrix, distortion coefficients, and calibration errors to a CSV file.

    :param file_path: Path to the CSV file.
    :param cam_matrix: Camera matrix (3x3).
    :param dist_coeffs: Distortion coefficients (1x5 or 1x8).
    :param calib_errors: Calibration errors.
    """
    with open(csvFilePath, mode='w', newline='') as file:
        writer = csv.writer(file);
        
        # Write camera matrix
        writer.writerow(['Camera Matrix']);
        for row in cameraMatrix:
            writer.writerow(row);
        
        # Write distortion coefficients
        writer.writerow(['Distortion Coefficients']);
        writer.writerow(cameraDistCoeffs);
        
        # Write calibration errors
        writer.writerow(['Calibration Errors']);
        writer.writerow([calibrationErrors]);
def loadMonoCalibResultsFromCSV(csvFilePath):
    """
    Load camera matrix, distortion coefficients, and calibration errors from a CSV file.

    :param csvFilePath: Path to the CSV file.
    :return: Tuple containing camera matrix, distortion coefficients, and calibration errors.
    """
    cameraMatrix = []
    cameraDistCoeffs = []
    calibrationErrors = None

    with open(csvFilePath, mode='r') as file:
        reader = csv.reader(file)
        section = None
        
        for row in reader:
            if row[0] == 'Camera Matrix':
                section = 'cam_matrix'
                continue
            elif row[0] == 'Distortion Coefficients':
                section = 'dist_coeffs'
                continue
            elif row[0] == 'Calibration Errors':
                section = 'calib_errors'
                continue
            
            if section == 'cam_matrix':
                cameraMatrix.append([float(val) for val in row])
            elif section == 'dist_coeffs':
                # Handle space-separated coefficients
                row_data = row[0].replace('  ', ' ')  # Replace double spaces with single space
                row_data = row_data.replace('[', '').replace(']', '')  # Remove brackets
                cameraDistCoeffs = np.array([float(val) for val in row_data.split()], dtype=np.float32)
            elif section == 'calib_errors':
                calibrationErrors = float(row[0])
    
    cameraMatrix = np.array(cameraMatrix, dtype=np.float32)

    return cameraMatrix, cameraDistCoeffs, calibrationErrors
def saveMonoCalibResultsToXML(xmlFilePath, cameraMatrix, cameraDistCoeffs, calibrationErrors):
    """
    Save camera matrix, distortion coefficients, and calibration errors to an XML file using OpenCV's FileStorage.

    Parameters:
        xmlFilePath (str): Path to the XML file.
        cameraMatrix (np.ndarray): Camera matrix (3x3).
        cameraDistCoeffs (np.ndarray): Distortion coefficients (1x5 or 1x8).
        calibrationErrors (float): Calibration errors.
    """
    fs = cv2.FileStorage(xmlFilePath, cv2.FILE_STORAGE_WRITE)

    # Write camera matrix
    fs.write("CameraMatrix", cameraMatrix)

    # Write distortion coefficients
    fs.write("DistortionCoefficients", cameraDistCoeffs)

    # Write calibration errors
    fs.write("CalibrationErrors", calibrationErrors)

    # Release file
    fs.release()
def loadMonoCalibResultsFromXML(xmlFilePath):
    """
    Load camera matrix, distortion coefficients, and calibration errors from an XML file using OpenCV's FileStorage.

    Parameters:
        xmlFilePath (str): Path to the XML file.

    Returns:
        tuple: Containing camera matrix, distortion coefficients, and calibration errors.
    """
    fs = cv2.FileStorage(xmlFilePath, cv2.FILE_STORAGE_READ)

    # Read camera matrix
    cameraMatrix = fs.getNode("CameraMatrix").mat()

    # Read distortion coefficients
    cameraDistCoeffs = fs.getNode("DistortionCoefficients").mat()

    # Read calibration errors
    calibrationErrors = fs.getNode("CalibrationErrors").real()

    # Release file
    fs.release()

    return cameraMatrix, cameraDistCoeffs, calibrationErrors
def detectChessboardCorners(image, boardSize=(9, 7)):
    """
    Detect chessboard corners in an image and return the list of 2-D points.

    Parameters:
        image (np.ndarray): The input image.
        boardSize (tuple): Number of inner corners per a chessboard row and column (rows, columns).

    Returns:
        np.ndarray: Nx2 array of detected 2-D points, where N is the number of corners.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, boardSize, None)

    if not ret:
        raise ValueError("Chessboard corners not found in the image.")

    # Refine the corner locations
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

    # Reshape the corners to Nx2 format
    corners = corners.reshape(-1, 2)
    corners = np.asarray(corners);

    return corners
def calibrateStereoCameras(folderPath1, folderPath2, firstImagePrefix, secondImagePrefix, startIndex, endIndex, boardSize, squareSize,
                          firstCamMatrix, firstCamDistCoeffs, secondCamMatrix, secondCamDistCoeffs, imageType="png", imageSize = (1920, 1080), concatType="horizontal"):
    # Define the criteria for corner refinement
    print("stereoCalibrateCameras:: Define the criteria for corner refinement ...")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.000001)

    # Initialize object points based on the chessboard pattern
    print("stereoCalibrateCameras:: Initialize object points based on the chessboard pattern ...")
    objp = np.zeros((boardSize[0] * boardSize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:boardSize[0], 0:boardSize[1]].T.reshape(-1, 2)
    objp *= squareSize

    # Arrays to store object points and image points from all images
    print("stereoCalibrateCameras:: Arrays to store object points and image points from all images ...")
    objPointSets = []  # 3d points in real world space
    firstImagePointSets = []  # 2d points in image plane for camera 1
    secondImagePointSets = []  # 2d points in image plane for camera 2

    # Iterate through the image pairs
    print("stereoCalibrateCameras:: Iterate through the image pairs ...")    
    for i in range(startIndex, endIndex + 1):
        # Debugging
        print(i, end=" ", flush=True);

        # Reading images
        img1_path = os.path.join(folderPath1, f"{firstImagePrefix}_{i}.{imageType}")
        img2_path = os.path.join(folderPath2, f"{secondImagePrefix}_{i}.{imageType}")
        img1 = cv2.imread(img1_path)
        height, width = img1.shape[:2]
        imageSize = (width, height)
        img2 = cv2.imread(img2_path)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        firstRet, firstImgPoints = cv2.findChessboardCorners(gray1, boardSize, None)
        secondRet, secondImgPoints = cv2.findChessboardCorners(gray2, boardSize, None)
    
        # If found, draw and display the corners
        if firstRet and secondRet:
            # Get corner subpixel
            firstImgPoints = cv2.cornerSubPix(gray1, firstImgPoints, (11, 11), (-1, -1), criteria)
            secondImgPoints = cv2.cornerSubPix(gray2, secondImgPoints, (11, 11), (-1, -1), criteria)
            
            # Checking the corner points
            if not (firstImgPoints[0][0][0] < np.mean(firstImgPoints[:, 0, 0]) and firstImgPoints[0][0][1] < np.mean(firstImgPoints[:, 0, 1]) and
                    secondImgPoints[0][0][0] < np.mean(secondImgPoints[:, 0, 0]) and secondImgPoints[0][0][1] < np.mean(secondImgPoints[:, 0, 1])):
                print("\t\t First points are not to the left and above relative to the other points. Skipping this pair.")
                continue;

            # Append the object points and image points
            objPointSets.append(objp); firstImagePointSets.append(firstImgPoints); secondImagePointSets.append(secondImgPoints)
    print("");

    # Stereo calibration
    print("stereoCalibrateCameras:: Calibrating ...");
    retval, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(objPointSets, firstImagePointSets, secondImagePointSets, firstCamMatrix, firstCamDistCoeffs,
                                                             secondCamMatrix, secondCamDistCoeffs, gray1.shape[::-1], criteria=criteria, 
                                                             flags=cv2.CALIB_USE_INTRINSIC_GUESS);
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, imageSize, R, T);

    # 3D reconstruction and error calculation
    print("stereoCalibrateCameras:: Compute testing error ...");
    meanErrors = [];
    for i in range(0, len(firstImagePointSets)):
        # Project points to 3D
        recon3DPoints = stereoReconstruct3DPoints(firstImagePointSets[i], secondImagePointSets[i], K1, D1, K2, D2, P1, P2);
                
        # Compute the svd registration
        gt3DPoints = objPointSets[i];
        svdTransform = estimateRigidSVDTransform(gt3DPoints, recon3DPoints);
        gt3DPoints = transform3DPoints(gt3DPoints, svdTransform);
        meanP2PDistance = computeAveragePointsToPointsDistance(recon3DPoints, gt3DPoints);
        meanErrors.append(meanP2PDistance);
    meanErrors = np.asarray(meanErrors);
    meanError = np.mean(meanErrors);

    # Output the extrinsic parameters and calibration error
    extrinsic_params = {
        "K1": K1,
        "D1": D1,
        "K2": K2,
        "D2": D2,
        "P1": P1,
        "P2": P2,
        "R": R,
        "T": T,
        "E": E,
        "F": F,
        "MeanReconErrors": meanError
    }

    return extrinsic_params
def saveStereoCalibrationResultsToXML(xmlFilePath, extrinsicParams):
    """
    Save stereo calibration results to an XML file using OpenCV's FileStorage.

    Parameters:
        filePath (str): Path to the XML file.
        extrinsicParams (dict): Dictionary containing the stereo calibration parameters.
    """
    fs = cv2.FileStorage(xmlFilePath, cv2.FILE_STORAGE_WRITE)

    # Write matrices
    fs.write("K1", extrinsicParams["K1"])
    fs.write("D1", extrinsicParams["D1"])
    fs.write("K2", extrinsicParams["K2"])
    fs.write("D2", extrinsicParams["D2"])
    fs.write("P1", extrinsicParams["P1"])
    fs.write("P2", extrinsicParams["P2"])
    fs.write("R", extrinsicParams["R"])
    fs.write("T", extrinsicParams["T"])
    fs.write("E", extrinsicParams["E"])
    fs.write("F", extrinsicParams["F"])
    fs.write("MeanReconErrors", extrinsicParams["MeanReconErrors"])

    # Release file
    fs.release()
def loadStereoCalibrationResultsFromXML(xmlFilePath):
    """
    Load stereo calibration results from an XML file using OpenCV's FileStorage.

    Parameters:
        filePath (str): Path to the XML file.

    Returns:
        dict: Dictionary containing the stereo calibration parameters.
    """
    fs = cv2.FileStorage(xmlFilePath, cv2.FILE_STORAGE_READ)

    # Read matrices
    K1 = fs.getNode("K1").mat()
    D1 = fs.getNode("D1").mat()
    K2 = fs.getNode("K2").mat()
    D2 = fs.getNode("D2").mat()
    P1 = fs.getNode("P1").mat()
    P2 = fs.getNode("P2").mat()
    R = fs.getNode("R").mat()
    T = fs.getNode("T").mat()
    E = fs.getNode("E").mat()
    F = fs.getNode("F").mat()
    MeanReconErrors = fs.getNode("MeanReconErrors").real()

    # Release file
    fs.release()

    # Return the parameters as a dictionary
    extrinsicParams = {
        "K1": K1,
        "D1": D1,
        "K2": K2,
        "D2": D2,
        "P1": P1,
        "P2": P2,
        "R": R,
        "T": T,
        "E": E,
        "F": F,
        "MeanReconErrors": MeanReconErrors
    }

    return extrinsicParams
def stereoReconstruct3DPoints(imagePoints1, imagePoints2, K1, D1, K2, D2, P1, P2):
    # Reshape the image points
    numOfPoints = len(imagePoints1);
    firstImagePoints = imagePoints1.reshape(numOfPoints, 1, 2);
    secondImagePoints = imagePoints2.reshape(numOfPoints, 1, 2);

    # Undistort corners
    undistCorners1 = cv2.undistortPoints(firstImagePoints, K1, D1, P=K1)
    undistCorners2 = cv2.undistortPoints(secondImagePoints, K2, D2, P=K2)

    # Triangulate the points
    undistPoints1 = undistCorners1.reshape(-1, 2)
    undistPoints2 = undistCorners2.reshape(-1, 2)
    points3D = cv2.triangulatePoints(P1, P2, undistPoints1.T, undistPoints2.T)
    points3D /= points3D[len(points3D) - 1]
    points3D = points3D[:(len(points3D) - 1)].T

    return points3D;
def sfmReconstruct3DPoints(firstImgPoints, secondImgPoints, K, D):
    # Reshape points
    numOfPoints = len(firstImgPoints);
    firstViewPoints = firstImgPoints.reshape(numOfPoints, 1, 2);
    secondViewPoints = secondImgPoints.reshape(numOfPoints, 1, 2);

    # Undistort points
    undistFirstViewPoints = cv2.undistortPoints(firstViewPoints, K, D, P=K);
    undistSecondViewPoints = cv2.undistortPoints(secondViewPoints, K, D, P=K);

    # Reconstruct the 3D points using the structure from motion
    E, _ = cv2.findEssentialMat(undistFirstViewPoints, undistSecondViewPoints, K, method=cv2.RANSAC, threshold=1.0)
    U, _, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], 
                  [1,  0, 0], 
                  [0,  0, 1]])
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t1 = U[:, 2]
    t2 = -U[:, 2]

    if np.linalg.det(R1) < 0: R1 = -R1
    if np.linalg.det(R2) < 0: R2 = -R2

    # Four possible solutions for (R, t)
    solutions = [(R1, t1), (R1, t2), (R2, t1), (R2, t2)]

    def triangulate(R, t):
        """ Triangulates 3D points given a rotation R and translation t """
        P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # Projection matrix for first camera
        P2 = K @ np.hstack((R, t.reshape(3, 1)))  # Projection matrix for second camera

        # Convert points to homogeneous format
        hFirstPoints = undistFirstViewPoints.reshape(-1, 2).T;
        hSecondPoints = undistSecondViewPoints.reshape(-1, 2).T;

        # Triangulate 3D points
        h4DPoints = cv2.triangulatePoints(P1, P2, hFirstPoints, hSecondPoints)
        recon3DPoints = (h4DPoints[:3] / h4DPoints[3]).T  # Convert to 3D

        # Count valid points (Z > 0 in both cameras)
        validCount = np.sum(recon3DPoints[:, 2] > 0)
        return validCount, recon3DPoints

    # Select the best (R, t) by maximizing valid 3D points
    bestR, bestT, best3DPoints = max(
        [(R, t, triangulate(R, t)[1]) for R, t in solutions], 
        key=lambda item: np.sum(item[2][:, 2] > 0)
    )
    distances = np.linalg.norm(np.diff(best3DPoints, axis=0), axis=1)
    scaleFactor = np.mean(distances)
    if scaleFactor > 0:
        best3DPoints /= scaleFactor

    # Return the 3-D points
    return best3DPoints;
def computeBarycentricLandmarks(mesh: trimesh.Trimesh, landmarkPoints: np.ndarray):
        """
        Given 3D landmark points and a mesh, compute the triangle indices and
        barycentric coordinates for use in trimesh.registration.nricp_amberg.

        Args:
            mesh (trimesh.Trimesh): The mesh where landmarks will be localized.
            landmarkPoints (np.ndarray): (n, 3) array of 3D landmark points.

        Returns:
            tuple:
                triIndices (np.ndarray): (n,) int array of triangle indices.
                baryCoords (np.ndarray): (n, 3) float array of barycentric coordinates.
        """
        closest = trimesh.proximity.closest_point(mesh, landmarkPoints)
        triIndices = closest[2]      # (n,) array of triangle indices
        
        baryCoords = []
        for point, triIndex in zip(landmarkPoints, triIndices):
            triVerts = mesh.triangles[triIndex]  # shape (3, 3)
            v0, v1, v2 = triVerts

            # Compute barycentric coordinates
            T = np.column_stack([v1 - v0, v2 - v0])
            v = point - v0
            A = np.dot(T.T, T)
            b = np.dot(T.T, v)

            try:
                x = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                x = np.zeros(2)

            w1 = 1.0 - x[0] - x[1]
            w2 = x[0]
            w3 = x[1]
            baryCoords.append([w1, w2, w3])

        return np.array(triIndices), np.array(baryCoords)
def reconstructLandmarksFromBarycentric(mesh: trimesh.Trimesh, baryIndices: np.ndarray, baryCoords: np.ndarray) -> np.ndarray:
    """
    Reconstruct 3D points from barycentric coordinates and triangle indices.

    Args:
        mesh (trimesh.Trimesh): The source mesh.
        triIndices (np.ndarray): (n,) array of triangle indices.
        baryCoords (np.ndarray): (n, 3) array of barycentric coordinates.

    Returns:
        np.ndarray: (n, 3) array of reconstructed 3D points.
    """
    if baryIndices.shape[0] != baryCoords.shape[0]:
        raise ValueError("triIndices and baryCoords must have the same number of entries.")

    # Get the triangles from the mesh
    tris = mesh.triangles[baryIndices]  # shape (n, 3, 3)

    # Unpack vertices: V = w1*V0 + w2*V1 + w3*V2
    v0 = tris[:, 0, :]  # (n, 3)
    v1 = tris[:, 1, :]
    v2 = tris[:, 2, :]

    w1 = baryCoords[:, 0][:, np.newaxis]
    w2 = baryCoords[:, 1][:, np.newaxis]
    w3 = baryCoords[:, 2][:, np.newaxis]

    points = w1 * v0 + w2 * v1 + w3 * v2  # (n, 3)
    return points
def transferFaceIndicesToOtherMesh(sourceFaceIndices, sourceMesh, targetMesh):
    # Compute the vertex mapping
    sourceVertexIndicesOnTargetMesh = estimateNearestIndicesKDTreeBased(sourceMesh.vertices, targetMesh.vertices);

    # Replace vertex indices on from mesh triangles
    sourceFaces = sourceMesh.faces[sourceFaceIndices];
    for i, face in enumerate(sourceFaces):
        face[0] = sourceVertexIndicesOnTargetMesh[face[0]];
        face[1] = sourceVertexIndicesOnTargetMesh[face[1]];
        face[2] = sourceVertexIndicesOnTargetMesh[face[2]];
        sourceFaces[i] = face;

    # Build the kdtree of the target faces
    targetFaceKDTree = KDTree(targetMesh.faces);

    # Query the point indices
    distanceBuffer, outFaceIndices = targetFaceKDTree.query(sourceFaces);

    # Return the outFaceIndices
    return outFaceIndices;
def generateRainBowColor(n, maxNumColors):
    """
    Generate a smoothly transitioning RGB color based on an infinite input n.
    The color transitions linearly and adapts based on the max number of colors.
    
    Parameters:
        n (int): Input number for color calculation.
        max_colors (int): Maximum number of distinct color variations.
        
    Returns:
        tuple: (r, g, b) values scaled between 0 and 1.
    """
    scale = 2 * math.pi / maxNumColors  # Auto computed scale based on max colors
    n *= scale  # Normalize input to smoothly transition within the set max range
    
    r = 0.5 * (math.sin(n) + 1)  # Oscillates between 0 and 1
    g = 0.5 * (math.sin(n + 2 * math.pi / 3) + 1)
    b = 0.5 * (math.sin(n + 4 * math.pi / 3) + 1)

    return (r, g, b)
