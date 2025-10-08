# Supporting libraries
import trimesh.creation
import vtk;
from vtk.util.numpy_support import vtk_to_numpy;
from vtk.util.numpy_support import numpy_to_vtk;
import SupportingTools.SupportingTools as sp;
import trimesh;
import colorsys;
import numpy as np;
import cv2;

# Supporting class
class VisualInterface:
    def __init__(self):
        # For multiple rendering
        self.renderers = [];

        # For single window rendering
        self.mainRenderer = None;
        self.renderWindow = None;
        self.renderWindowInteractor = None;
        self.interactorStyle = None;

        # For rendered objects
        self.CoordinateSystems = {};
        self.MultiRenderCoordinateSystems = {};
        self.SurfaceMeshes = {};
        self.MultiRenderSurfaceMeshes = {};
        self.SphereMeshes = {};
        self.MultiRenderSphereMeshes = {};
        self.PointSets = {};
        self.DICOMImages = {};
        self.DICOMVolumes = {};
        self.MultiRenderPointSets = {};
        self.ChessBoards = {};

        # For cameras
        self.subCameras = {};

        # For supporting buffers
        self.colorDict = {
            "white": (1.0, 1.0, 1.0), 
            "lightgray": (0.9, 0.9, 0.9), 
            "lightblue": (0.68, 0.85, 0.9), 
            "lightgreen": (0.56, 0.93, 0.56), 
            "lightyellow": (1.0, 1.0, 0.88), 
            "lightpink": (1.0, 0.71, 0.76), 
            "lightcyan": (0.88, 1.0, 1.0), 
            "lightcoral": (0.94, 0.5, 0.5), 
            "lightsalmon": (1.0, 0.63, 0.48), 
            "lightlavender": (0.9, 0.9, 0.98), 
            "lightpeach": (1.0, 0.85, 0.73), 
            "lightmint": (0.6, 1.0, 0.6),

            # General skin tone
            "fair": (1.0, 0.87, 0.77),          # Light skin tone
            "light_olive": (0.91, 0.76, 0.65),  # Light olive skin
            "medium_olive": (0.82, 0.67, 0.54), # Medium olive skin
            "tan": (0.76, 0.60, 0.42),          # Tanned skin tone
            "light_brown": (0.72, 0.52, 0.40),  # Light brown skin tone
            "medium_brown": (0.62, 0.42, 0.30), # Medium brown skin tone
            "dark_brown": (0.50, 0.35, 0.25),   # Dark brown skin tone
            "ebony": (0.35, 0.24, 0.19),        # Darker skin tone (ebony)
            "mahogany": (0.41, 0.26, 0.23),     # Rich dark tone
            "deep_brown": (0.27, 0.18, 0.13),   # Deep brown/black skin tone 

            # Red tone skin color
            "fair_red": (1.0, 0.82, 0.72),          # Light skin tone with reddish hue
            "light_olive_red": (0.95, 0.72, 0.62), # Light olive skin with reddish hue
            "medium_olive_red": (0.86, 0.62, 0.50),# Medium olive skin with reddish hue
            "tan_red": (0.80, 0.54, 0.38),         # Tanned skin tone with reddish hue
            "light_brown_red": (0.76, 0.46, 0.34), # Light brown skin tone with reddish hue
            "medium_brown_red": (0.66, 0.38, 0.28),# Medium brown skin tone with reddish hue
            "dark_brown_red": (0.54, 0.30, 0.22),  # Dark brown skin tone with reddish hue
            "ebony_red": (0.40, 0.20, 0.16),       # Ebony tone with reddish hue
            "mahogany_red": (0.46, 0.22, 0.20),    # Mahogany skin tone with reddish hue
            "deep_brown_red": (0.32, 0.14, 0.10),   # Deep brown/black tone with reddish hue

            # Light skin tones
            "ivory": (1.0, 0.94, 0.86),           # Light ivory
            "porcelain": (1.0, 0.96, 0.91),       # Very light porcelain
            "fair_pink": (1.0, 0.87, 0.82),       # Fair skin with pink undertone
            "fair_peach": (1.0, 0.89, 0.80),      # Fair skin with peach undertone
            "alabaster": (0.98, 0.92, 0.87),      # Pale alabaster tone
            "peach_cream": (1.0, 0.92, 0.84),     # Light with warm peach undertones
            "ivory_pink": (0.96, 0.88, 0.85),     # Ivory with a hint of pink
            "light_beige": (0.94, 0.88, 0.78),    # Soft beige for fair skin
            "rosy_fair": (1.0, 0.88, 0.85),       # Light skin with rosy undertone
            "neutral_fair": (0.95, 0.89, 0.84),   # Neutral fair tone
            "porcelain_peach": (1.0, 0.90, 0.85), # Porcelain with slight peach

            # Slightly darker fair tones
            "cream": (0.98, 0.91, 0.80),          # Cream with warm undertones
            "warm_ivory": (0.98, 0.89, 0.82),     # Ivory with warm undertones
            "cool_fair": (0.90, 0.85, 0.80),      # Fair with cool undertones
            "pale_beige": (0.96, 0.88, 0.77),     # Light beige tone
            "porcelain_fair": (0.96, 0.91, 0.86), # Very fair, porcelain-like
            "almond": (0.94, 0.88, 0.78),         # Light with an almond hint
            "pale_rose": (0.98, 0.89, 0.85),      # Light with soft pink hue
            "sand": (0.94, 0.86, 0.76),           # Light sandy tone

            # Bone color
            "white_bone": (0.95, 0.91, 0.82),      # White bone color for natural visualizing
            "bone": (0.73, 0.64, 0.50),            # Classic bone color
            "ivory_bone": (0.83, 0.71, 0.52),      # Ivory bone tone
            "off_white_bone": (0.92, 0.85, 0.68),  # Slightly off-white bone color
            "light_bone": (0.87, 0.76, 0.57),      # Light bone shade
            "dark_bone": (0.60, 0.50, 0.36),       # Darker bone color
            "beige_bone": (0.80, 0.70, 0.55),      # Beige with bone undertone
            "yellow_bone": (0.91, 0.78, 0.46),     # Yellowish bone tone
            "grey_bone": (0.75, 0.70, 0.62),       # Greyish bone color
            "chalk_bone": (0.96, 0.94, 0.80),      # Chalky white bone tone
            "aged_bone": (0.67, 0.57, 0.43),       # Aged, weathered bone
            "sandstone_bone": (0.85, 0.73, 0.55),  # Sandstone-inspired bone color
            "cream_bone": (0.92, 0.84, 0.68),      # Creamy bone tone

            # Muscle colors
            "muscle_red": (0.8, 0.0, 0.0),      # Deep red tone for muscle tissue
            "muscle_pink": (1.0, 0.75, 0.79),   # Lighter pink for muscle fibers
            "deep_red": (0.6, 0.0, 0.0),        # Darker red for deep muscle layers
            "bright_red": (1.0, 0.2, 0.2),      # Bright red tone for active muscle
            "dark_salmon": (0.91, 0.59, 0.48),  # A warm tone for muscle representation
            "flesh_tone": (0.94, 0.76, 0.65),   # Flesh-like tone with reddish hues
            "peach": (1.0, 0.85, 0.73),         # Soft peach for connective muscle tissue

            # Basic colors
            "black": (0.0, 0.0, 0.0),
            "white": (1.0, 1.0, 1.0),
            "red": (1.0, 0.0, 0.0),
            "green": (0.0, 1.0, 0.0),
            "blue": (0.0, 0.0, 1.0),
            "yellow": (1.0, 1.0, 0.0),
            "cyan": (0.0, 1.0, 1.0),
            "magenta": (1.0, 0.0, 1.0),
        };

        self.visionSeparation = 0.010;
        self.thread = None;
        self.isInitialized = False;
        self.leftCameraName = "";
        self.rightCameraName = "";
        self.centerCameraName = "";
        self.upperCameraName = "";
        self.lowerCameraName = "";
        self.DICOMMinValue = 0;
        self.DICOMMaxValue = 4000;
    
    #**************************************************** Supporting function
    def rainbowColor(self, iValue):
        """
        Maps an integer from 0 to 255 to a rainbow color (red to violet).
    
        Parameters:
            value (int): An integer from 0 to 255 representing the color position in the rainbow.
    
        Returns:
            (real, real, real): A tuple representing the RGB color.
        """
        # Ensure value is in the range 0-255
        iValue = max(0, min(255, iValue))
    
        # Map the value to a hue range from 0 (red) to 270 (violet) in HSV
        hue = (iValue / 255) * 270 / 360  # Normalize hue to a value between 0 and 0.75
    
        # Convert HSV to RGB (with full saturation and brightness)
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    
        # Convert RGB values to the 0-255 range
        return (r, g, b);
    def getScreenSize(self, resolutionName):
        """
        Returns the screen resolution (width, height) for a given resolution name.

        Parameters:
            resolution_name (str): A string representing the resolution name.
                                   Expected values are standard names like 'hd', 'fullhd', 'qhd', '4k', etc.
    
        Returns:
            tuple: A tuple (width, height) representing the screen resolution in pixels.
        """
        resolutions = {
            'sd': (640, 480),          # Standard Definition (SD)
            'hd': (1280, 720),         # High Definition (HD)
            'hd+': (1600, 900),        # HD+
            'fullhd': (1920, 1080),    # Full High Definition (Full HD)
            '2k': (2048, 1080),        # 2K Resolution
            'qhd': (2560, 1440),       # Quad High Definition (QHD) or 2K
            'wqhd': (3440, 1440),      # Wide Quad High Definition (WQHD) - Ultra-wide monitors
            'uhd': (3840, 2160),       # Ultra High Definition (UHD) or 4K
            '4k': (3840, 2160),        # 4K UHD Resolution
            '5k': (5120, 2880),        # 5K Resolution
            '8k': (7680, 4320),        # 8K UHD Resolution
            '8k ultra wide': (10240, 4320), # 8K Ultra-wide monitors
            'wxga': (1366, 768),       # Wide Extended Graphics Array (WXGA)
            'wsxga': (1600, 1024),     # Wide Super XGA (WSXGA)
            'wuxga': (1920, 1200),     # Wide Ultra XGA (WUXGA)
            'whuxga': (7680, 4800),    # Wide Hexadecatuple Ultra XGA (WHUXGA)
            'cga': (320, 200),         # Color Graphics Adapter (CGA)
            'vga': (640, 480),         # Video Graphics Array (VGA)
            'svga': (800, 600),        # Super VGA (SVGA)
            'xga': (1024, 768),        # Extended Graphics Array (XGA)
            'sxga': (1280, 1024),      # Super XGA (SXGA)
            'uxga': (1600, 1200),      # Ultra XGA (UXGA)
            'qvga': (320, 240),        # Quarter VGA (QVGA)
            'hvga': (480, 320),        # Half VGA (HVGA)
            'wvga': (800, 480),        # Wide VGA (WVGA)
        }
    
        # Convert to lowercase to handle case-insensitive input
        return resolutions.get(resolutionName.lower(), "Unknown resolution")
    def generateSphere(self, radius=1.0, center=(0.0, 0.0, 0.0), subDiv=3, sphereType='icosphere'):
        """
        Generates a 3D sphere mesh using trimesh, with customizable radius and center.
    
        Parameters:
            radius (float): Radius of the sphere. Default is 1.0.
            center (tuple): Coordinates of the sphere center as (x, y, z). Default is (0.0, 0.0, 0.0).
            subdivisions (int): Number of subdivisions for mesh refinement. Higher values create smoother spheres. Default is 3.
            sphere_type (str): Type of sphere mesh to generate ('icosphere' or 'uv_sphere').
                               'icosphere' creates a sphere with triangular faces,
                               'uv_sphere' creates a sphere with quad faces.
    
        Returns:
            trimesh.Trimesh: A sphere mesh object centered at the specified coordinates.
        """
        # Generate the base sphere at the origin
        if sphereType == 'icosphere':
            sphere = trimesh.creation.icosphere(subdivisions=subDiv, radius=radius)
        elif sphereType == 'uv_sphere':
            sphere = trimesh.creation.uv_sphere(radius=radius)
        else:
            raise ValueError("sphere_type must be either 'icosphere' or 'uv_sphere'")
    
        # Move the sphere to the desired center
        sphere.apply_translation(np.array(center))
    
        return sphere
    def copyActors(self, sourceRenderer, targetRenderer):
        actors = sourceRenderer.GetActors()
        actors.InitTraversal()
    
        for i in range(actors.GetNumberOfItems()):
            actor = actors.GetNextItem()
            newActor = vtk.vtkActor()
            newActor.ShallowCopy(actor)  # Use ShallowCopy to duplicate properties
            targetRenderer.AddActor(newActor)
    def addCameraRepresentation(self, cameraName, color=(1, 0, 0)):
        # Create a box source
        boxSource = vtk.vtkCubeSource();
        boxSource.SetXLength(0.05);  # Width
        boxSource.SetYLength(0.027);  # Height
        boxSource.SetZLength(0.027);  # Depth

        # Create a mapper and actor
        boxMapper = vtk.vtkPolyDataMapper();
        boxMapper.SetInputConnection(boxSource.GetOutputPort());

        boxActor = vtk.vtkActor();
        boxActor.SetMapper(boxMapper);
        boxActor.GetProperty().SetColor(color);

        boxActor.GetProperty().SetAmbient(0.5)   # Increase ambient light response
        boxActor.GetProperty().SetDiffuse(0.8)   # Make it more responsive to light
        boxActor.GetProperty().SetSpecular(0.5)  # Add a slight shininess effect
        boxActor.GetProperty().SetSpecularPower(20)  # Sharper specular highlight

        # Store the actor for updates
        self.subCameras[cameraName]["BoxActor"] = boxActor;

        # Add the actor to the corresponding camera's renderer
        self.mainRenderer.AddActor(boxActor);
    def updateCameraBox(self, cameraName):
        if cameraName not in self.subCameras:
            return;

        # Get camera properties
        camera = self.subCameras[cameraName]["Renderer"].GetActiveCamera();
        position = np.array(camera.GetPosition());
        focalPoint = np.array(camera.GetFocalPoint());
        viewUp = np.array(camera.GetViewUp());

        # Compute camera orientation vectors
        forwardVec = focalPoint - position
        forwardVec /= np.linalg.norm(forwardVec);  # Normalize
        upVec = viewUp / np.linalg.norm(viewUp);

        # Compute right vector using cross product
        rightVec = np.cross(forwardVec, upVec);
        rightVec /= np.linalg.norm(rightVec);

        # Recompute up vector to ensure orthogonality
        upVec = np.cross(rightVec, forwardVec);

        # Construct transformation matrix
        transformMatrix = vtk.vtkMatrix4x4();
        for i in range(3):
            transformMatrix.SetElement(i, 0, rightVec[i]);   # Right vector
            transformMatrix.SetElement(i, 1, upVec[i]);      # Up vector
            transformMatrix.SetElement(i, 2, forwardVec[i]); # Forward vector
            transformMatrix.SetElement(i, 3, position[i]);   # Position

        # Apply transformation
        transform = vtk.vtkTransform();
        transform.SetMatrix(transformMatrix);

        # Update actor
        self.subCameras[cameraName]["BoxActor"].SetUserTransform(transform);
    def opencvToVTKProjection(self, K, near=0.1, far=1000.0):
        """
        Converts OpenCV camera intrinsic matrix K into a VTK-compatible projection matrix.
        K is a 3x3 matrix: 
            [ fx  0  cx ]
            [  0 fy  cy ]
            [  0  0   1 ]
    
        near, far: Clipping planes
        """
        fx, fy = K[0, 0], K[1, 1]  # Focal lengths
        cx, cy = K[0, 2], K[1, 2]  # Principal points
        width, height = 2 * cx, 2 * cy  # Extract image size

        P = vtk.vtkMatrix4x4()
    
        # Convert OpenCV K to VTK projection matrix
        P.SetElement(0, 0,  2 * fx / width)  # Scale X
        P.SetElement(1, 1,  2 * fy / height) # Scale Y
        P.SetElement(0, 2,  1 - (2 * cx / width))  # Shift X
        P.SetElement(1, 2,  (2 * cy / height) - 1) # Shift Y
        P.SetElement(2, 2, -(far + near) / (far - near))  # Perspective depth
        P.SetElement(2, 3, -2 * far * near / (far - near))  # Depth scaling
        P.SetElement(3, 2, -1)  # Perspective division
        P.SetElement(3, 3,  0)   # Perspective mode

        return P    
   
    #**************************************************** Coordinate system functions
    ## For single renderer
    def addCoordinateSystem(self, coordName, axesLength=0.05, coneRadius=0.2, cylinderRadius=0.05):
        # Checking initializing
        if (not self.isInitialized): 
            raise ValueError("The visual interace should be initialized first.");

        # Define axes actor
        axesActor = vtk.vtkAxesActor()
        axesActor.SetTotalLength(axesLength, axesLength, axesLength)  # Set length of axes
        axesActor.SetConeRadius(coneRadius); # Set the cone radius
        axesActor.SetCylinderRadius(cylinderRadius); # Set the cylinder radius
        axesActor.GetXAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        axesActor.GetYAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        axesActor.GetZAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()

        # Adding actor into the library
        self.CoordinateSystems[coordName] = {
            "Actor": axesActor
        }

        # Add actor into the renderer
        self.mainRenderer.AddActor(axesActor);

    ## For specific renderers
    def addCoordinateSystemToRenderer(self, renderID, coordName, axesLength=0.05, coneRadius=0.2, cylinderRadius=0.05):
        # Checking initializing
        if (not self.isInitialized): 
            raise ValueError("The visual interace should be initialized first.");

        # Define axes actor
        axesActor = vtk.vtkAxesActor()
        axesActor.SetTotalLength(axesLength, axesLength, axesLength)  # Set length of axes
        axesActor.SetConeRadius(coneRadius); # Set the cone radius
        axesActor.SetCylinderRadius(cylinderRadius); # Set the cylinder radius
        axesActor.GetXAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        axesActor.GetYAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        axesActor.GetZAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()

        # Adding actor into the library
        self.MultiRenderCoordinateSystems[renderID][coordName] = {
            "Actor": axesActor
        }

        # Add actor into the renderer
        self.renderers[renderID].AddActor(axesActor);

    #**************************************************** Surface mesh interfacing functions
    ## For single renderer
    def addMesh(self, meshName, mesh, colorName="white"):
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");
        
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("The inMesh must be a trimesh.Trimesh object.");

        # Conver the trimesh to vtk data
        vertices = mesh.vertices;
        faces = mesh.faces;

        points = vtk.vtkPoints();
        for vertex in vertices:
            points.InsertNextPoint(vertex[0], vertex[1], vertex[2]);

        polyData = vtk.vtkPolyData();
        polyData.SetPoints(points);

        cells = vtk.vtkCellArray();
        for face in faces:
            cells.InsertNextCell(3, face);

        polyData.SetPolys(cells);

        mapper = vtk.vtkPolyDataMapper();
        mapper.SetInputData(polyData);

        actor = vtk.vtkActor();
        actor.SetMapper(mapper);

        actor.GetProperty().SetColor(self.colorDict.get(colorName.lower(), (1.0, 1.0, 1.0)));

        self.SurfaceMeshes[meshName] = {
            "Actor": actor,
            "PolyData": polyData,
            "Points": points
        }
        self.mainRenderer.AddActor(actor);
    def addMeshWithRGB(self, meshName, mesh, color=(1.0, 1.0, 1.0)):
        """
        Adds a mesh to the visual interface with an RGB color.
    
        Parameters:
            meshName (str): Unique name of the mesh.
            mesh (trimesh.Trimesh): The mesh object to be added.
            color (tuple): RGB values (r, g, b) ranging from 0 to 1.
        """
        if not self.isInitialized:
            raise ValueError("The visual interface should be initialized first.")

        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("The mesh must be a trimesh.Trimesh object.")

        # Convert the trimesh to VTK data
        vertices = mesh.vertices
        faces = mesh.faces

        points = vtk.vtkPoints()
        for vertex in vertices:
            points.InsertNextPoint(vertex[0], vertex[1], vertex[2])

        polyData = vtk.vtkPolyData()
        polyData.SetPoints(points)

        cells = vtk.vtkCellArray()
        for face in faces:
            cells.InsertNextCell(3, face)

        polyData.SetPolys(cells)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polyData)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Set the color from the (r, g, b) input
        actor.GetProperty().SetColor(color)

        self.SurfaceMeshes[meshName] = {
            "Actor": actor,
            "PolyData": polyData,
            "Points": points
        }
        self.mainRenderer.AddActor(actor)
    def removeMesh(self, meshName):
        # Checking initializing
        if (not self.isInitialized): 
            raise ValueError("The visual interace should be initialized first.");

        # Checking mesh existing
        if meshName not in self.SurfaceMeshes:
            raise ValueError(f"Mesh with name '{meshName}' does not exist.")

        # Retrieve the mesh actor and remove it from the renderer
        actor = self.SurfaceMeshes[meshName]["Actor"]
        self.mainRenderer.RemoveActor(actor)

        # Remove the mesh from internal storage
        del self.SurfaceMeshes[meshName]
    def updateMeshVertices(self, meshName, newVertices):
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");
        
        if meshName in self.SurfaceMeshes:
            points = self.SurfaceMeshes[meshName]["Points"];
            points.Reset();
            for vertex in newVertices:
                points.InsertNextPoint(vertex[0], vertex[1], vertex[2]);
            points.Modified();
        else:
            print(f"Mesh '{meshName}' not found.");
    def updateMesh(self, meshName, newMesh):
        if not self.isInitialized:
            raise ValueError("The visual interface should be initialized first.")

        if meshName not in self.SurfaceMeshes:
            raise ValueError(f"Mesh '{meshName}' not found.")

        # Extract the vtkPoints and vtkPolyData for the mesh
        meshData = self.SurfaceMeshes[meshName]
        points = meshData["Points"]
        polyData = meshData["PolyData"]

        # Update the vertices
        vertices = newMesh.vertices
        points.Reset()
        for vertex in vertices:
            points.InsertNextPoint(vertex[0], vertex[1], vertex[2])
        points.Modified()

        # Update the faces
        faces = newMesh.faces
        cells = vtk.vtkCellArray()
        for face in faces:
            cells.InsertNextCell(3, face)
        polyData.SetPolys(cells)
        polyData.Modified();
    def setMeshColorName(self, meshName, colorName):
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        if meshName in self.SurfaceMeshes:
            color = self.colorDict.get(colorName.lower());
            if color:
                actor = self.SurfaceMeshes[meshName]["Actor"];
                actor.GetProperty().SetColor(color);
            else:
                print(f"Color '{colorName}' not found. Available colors are: {list(self.color_dict.keys())}");
        else:
            print(f"Mesh '{meshName}' not found.");
    def setMeshColorRGB(self, meshName, red, green, blue):
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        if meshName in self.SurfaceMeshes: 
            if 0 <= red <= 1 and 0 <= green <= 1 and 0 <= blue <= 1: 
                actor = self.meshes[meshName]["Actor"];
                actor.GetProperty().SetColor(red, green, blue);
            else: 
                print("RGB values must be between 0 and 1.");
        else: 
            print(f"Mesh '{meshName}' not found.");
    def setMeshRainbowColor(self, meshName, iRainBowValue):
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        if meshName in self.SurfaceMeshes:
            color = self.rainbowColor(iRainBowValue);
            actor = self.SurfaceMeshes[meshName]["Actor"];
            actor.GetProperty().SetColor(color[0], color[1], color[2]);
        else:
            print(f"Mesh '{meshName}' not found.");
    def setMeshOpacity(self, meshName, alpha):
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        if meshName in self.SurfaceMeshes:
            if 0 <= alpha <= 1:
                actor = self.SurfaceMeshes[meshName]["Actor"];
                actor.GetProperty().SetOpacity(alpha);
            else:
                print("Alpha value must be between 0 and 1.");
        else:
            print(f"Mesh '{meshName}' not found.");

    ## For Sepcific renderers
    def addSurfaceMeshToRenderer(self, renderID, meshName, mesh, colorName="white"):
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");
        
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("The inMesh must be a trimesh.Trimesh object.");

        # Conver the trimesh to vtk data
        vertices = mesh.vertices;
        faces = mesh.faces;

        points = vtk.vtkPoints();
        for vertex in vertices:
            points.InsertNextPoint(vertex[0], vertex[1], vertex[2]);

        polyData = vtk.vtkPolyData();
        polyData.SetPoints(points);

        cells = vtk.vtkCellArray();
        for face in faces:
            cells.InsertNextCell(3, face);

        polyData.SetPolys(cells);

        mapper = vtk.vtkPolyDataMapper();
        mapper.SetInputData(polyData);

        actor = vtk.vtkActor();
        actor.SetMapper(mapper);

        actor.GetProperty().SetColor(self.colorDict.get(colorName.lower(), (1.0, 1.0, 1.0)));

        self.MultiRenderSurfaceMeshes[renderID][meshName] = {
            "Actor": actor,
            "PolyData": polyData,
            "Points": points
        }
        self.renderers[renderID].AddActor(actor);
    def removeSurfaceMeshFromRenderer(self, renderID, meshName):
        """
        Removes a surface mesh with the specified name from a given renderer and internal storage.

        Parameters:
        renderID (str): The ID of the renderer from which to remove the mesh.
        meshName (str): The name of the mesh to remove.

        Raises:
        ValueError: If the renderID does not exist or if the mesh name does not exist for the given renderID.
        """
        # Check if the renderer ID exists
        if renderID not in self.MultiRenderSurfaceMeshes:
            raise ValueError(f"Renderer with ID '{renderID}' does not exist.")
    
        # Check if the mesh exists in the specified renderer
        if meshName not in self.MultiRenderSurfaceMeshes[renderID]:
            raise ValueError(f"Mesh with name '{meshName}' does not exist in renderer '{renderID}'.")
    
        # Retrieve the mesh actor and remove it from the specified renderer
        actor = self.MultiRenderSurfaceMeshes[renderID][meshName]["Actor"]
        self.renderers[renderID].RemoveActor(actor)
    
        # Remove the mesh from internal storage
        del self.MultiRenderSurfaceMeshes[renderID][meshName]
    def updateMeshVerticesInRenderer(self, renderID, meshName, newVertices):
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");
        
        if meshName in self.MultiRenderSurfaceMeshes[renderID]:
            points = self.MultiRenderSurfaceMeshes[renderID][meshName]["Points"];
            points.Reset();
            for vertex in newVertices:
                points.InsertNextPoint(vertex[0], vertex[1], vertex[2]);
            points.Modified();
        else:
            print(f"Mesh '{meshName}' not found.");
    def updateMeshInRenderer(self, renderID, meshName, newMesh):
        """
        Updates the vertices and faces of a mesh in a specific renderer using a trimesh.Trimesh object.

        Parameters:
        renderID (str): The ID of the renderer containing the mesh.
        meshName (str): The name of the mesh to update.
        newMesh (trimesh.Trimesh): The new mesh data containing updated vertices and faces.

        Raises:
        ValueError: If the visual interface is not initialized, the renderID is invalid, or the mesh is not found.
        """
        if not self.isInitialized:
            raise ValueError("The visual interface should be initialized first.")

        if renderID not in self.MultiRenderSurfaceMeshes:
            raise ValueError(f"Renderer with ID '{renderID}' does not exist.")
    
        if meshName not in self.MultiRenderSurfaceMeshes[renderID]:
            raise ValueError(f"Mesh '{meshName}' not found in renderer '{renderID}'.")

        # Access the existing points and polyData for the mesh
        meshData = self.MultiRenderSurfaceMeshes[renderID][meshName]
        points = meshData["Points"]
        polyData = meshData["PolyData"]

        # Update the vertices
        vertices = newMesh.vertices
        points.Reset()
        for vertex in vertices:
            points.InsertNextPoint(vertex[0], vertex[1], vertex[2])
        points.Modified()

        # Update the faces
        faces = newMesh.faces
        cells = vtk.vtkCellArray()
        for face in faces:
            cells.InsertNextCell(3, face)
        polyData.SetPolys(cells)
        polyData.Modified()
    def setMeshColorNameInRenderer(self, renderID, meshName, colorName):
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        if meshName in self.MultiRenderSurfaceMeshes[renderID]:
            color = self.colorDict.get(colorName.lower());
            if color:
                actor = self.MultiRenderSurfaceMeshes[renderID][meshName]["Actor"];
                actor.GetProperty().SetColor(color);
            else:
                print(f"Color '{colorName}' not found. Available colors are: {list(self.color_dict.keys())}");
        else:
            print(f"Mesh '{meshName}' not found.");
    def setMeshColorRGBInRenderer(self, renderID, meshName, red, green, blue):
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        if meshName in self.MultiRenderSurfaceMeshes[renderID]: 
            if 0 <= red <= 1 and 0 <= green <= 1 and 0 <= blue <= 1: 
                actor = self.meshes[meshName]["Actor"];
                actor.GetProperty().SetColor(red, green, blue);
            else: 
                print("RGB values must be between 0 and 1.");
        else: 
            print(f"Mesh '{meshName}' not found.");
    def setMeshRainbowColorInRenderer(self, renderID, meshName, iRainBowValue):
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        if meshName in self.MultiRenderSurfaceMeshes[renderID]:
            color = self.rainbowColor(iRainBowValue);
            actor = self.MultiRenderSurfaceMeshes[renderID][meshName]["Actor"];
            actor.GetProperty().SetColor(color[0], color[1], color[2]);
        else:
            print(f"Mesh '{meshName}' not found.");
    def setMeshOpacityInRenderer(self, renderID, meshName, alpha):
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        if meshName in self.MultiRenderSurfaceMeshes[renderID]:
            if 0 <= alpha <= 1:
                actor = self.MultiRenderSurfaceMeshes[renderID][meshName]["Actor"];
                actor.GetProperty().SetOpacity(alpha);
            else:
                print("Alpha value must be between 0 and 1.");
        else:
            print(f"Mesh '{meshName}' not found.");

    #**************************************************** Sphere interfacing functions
    ## For single renderer
    def addColorNameSphereMesh(self, sphereName, sphereCenter, sphereRadius, colorName="red", subDiv = 1, sphereType='icosphere'):
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");
        
        # Generate the sphere mesh
        sphereMesh = self.generateSphere(sphereRadius, sphereCenter, subDiv, sphereType);

        # Conver the trimesh to vtk data
        vertices = sphereMesh.vertices;
        faces = sphereMesh.faces;
        points = vtk.vtkPoints();
        for vertex in vertices:
            points.InsertNextPoint(vertex[0], vertex[1], vertex[2]);
        polyData = vtk.vtkPolyData();
        polyData.SetPoints(points);
        cells = vtk.vtkCellArray();
        for face in faces:
            cells.InsertNextCell(3, face);
        polyData.SetPolys(cells);
        mapper = vtk.vtkPolyDataMapper();
        mapper.SetInputData(polyData);
        actor = vtk.vtkActor();
        actor.SetMapper(mapper);

        actor.GetProperty().SetColor(self.colorDict.get(colorName.lower(), (1.0, 1.0, 1.0)));

        self.SphereMeshes[sphereName] = {
            "Actor": actor,
            "PolyData": polyData,
            "Points": points
        }
        self.mainRenderer.AddActor(actor);
    def addColorSphereMesh(self, sphereName, sphereCenter, sphereRadius, color=(1.0, 1.0, 1.0), subDiv=1, sphereType='icosphere'):
        """
        Adds a sphere mesh to the scene using RGB color values in the range (0,1).

        Parameters:
            sphereName (str): Unique name of the sphere.
            sphereCenter (tuple): Coordinates of the sphere center (x, y, z).
            sphereRadius (float): Radius of the sphere.
            color (tuple): RGB color values in the format (R, G, B), where each value is between 0 and 1.
            subDiv (int): Number of subdivisions for mesh refinement.
            sphereType (str): Type of sphere ('icosphere' or 'uv_sphere').
        """
        if not self.isInitialized:
            raise ValueError("The visual interface should be initialized first.")

        # Generate sphere mesh
        sphereMesh = self.generateSphere(sphereRadius, sphereCenter, subDiv, sphereType)

        # Convert the trimesh to VTK data
        vertices = sphereMesh.vertices
        faces = sphereMesh.faces
        points = vtk.vtkPoints()
        for vertex in vertices:
            points.InsertNextPoint(vertex[0], vertex[1], vertex[2])
        polyData = vtk.vtkPolyData()
        polyData.SetPoints(points)
        cells = vtk.vtkCellArray()
        for face in faces:
            cells.InsertNextCell(3, face)
        polyData.SetPolys(cells)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polyData)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Set color directly from input (expected format: (R, G, B) in 0-1 range)
        actor.GetProperty().SetColor(color)

        # Store and add the sphere to the renderer
        self.SphereMeshes[sphereName] = {"Actor": actor, "PolyData": polyData, "Points": points}
        self.mainRenderer.AddActor(actor);
    def updateSphereCenter(self, sphereName, newCenter):
        """
        Quickly updates the center of a sphere in SphereMeshes using a transformation.

        Parameters:
            sphereName (str): The name of the sphere to update.
            newCenter (tuple): The new center coordinates (x, y, z).
        """
        if not self.isInitialized:
            raise ValueError("The visual interface should be initialized first.")

        if sphereName in self.SphereMeshes:
            actor = self.SphereMeshes[sphereName]["Actor"]

            # Compute translation vector
            oldCenter = np.mean([actor.GetMapper().GetInput().GetPoint(i) for i in range(actor.GetMapper().GetInput().GetNumberOfPoints())], axis=0)
            translation = np.array(newCenter) - oldCenter

            # Apply transformation directly to the actor
            transform = vtk.vtkTransform()
            transform.Translate(translation)
            actor.SetUserTransform(transform)
        else:
            print(f"Sphere '{sphereName}' not found.");
    def setSphereColorName(self, sphereName, colorName):
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");
        
        if sphereName in self.SphereMeshes:
            color = self.colorDict.get(colorName.lower());
            if color:
                actor = self.SphereMeshes[sphereName]["Actor"];
                actor.GetProperty().SetColor(color);
            else:
                print(f"Color '{colorName}' not found. Available colors are: {list(self.color_dict.keys())}");
        else:
            print(f"Mesh '{sphereName}' not found.");
    def setSphereRainbowColor(self, sphereName, rainBowValue):
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");
        
        if sphereName in self.SphereMeshes:
            color = self.rainbowColor(rainBowValue);
            actor = self.SphereMeshes[sphereName]["Actor"];
            actor.GetProperty().SetColor(color[0], color[1], color[2]);
        else:
            print(f"Mesh '{sphereName}' not found.");

    ## For Sepcific renderers
    def addSphereMeshToRenderer(self, renderID, sphereName, sphereCenter, sphereRadius, colorName="red", subDiv = 1, sphereType='icosphere'):
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");
        
        # Generate the sphere mesh
        sphereMesh = self.generateSphere(sphereRadius, sphereCenter, subDiv, sphereType);

        # Conver the trimesh to vtk data
        vertices = sphereMesh.vertices;
        faces = sphereMesh.faces;
        points = vtk.vtkPoints();
        for vertex in vertices:
            points.InsertNextPoint(vertex[0], vertex[1], vertex[2]);
        polyData = vtk.vtkPolyData();
        polyData.SetPoints(points);
        cells = vtk.vtkCellArray();
        for face in faces:
            cells.InsertNextCell(3, face);
        polyData.SetPolys(cells);
        mapper = vtk.vtkPolyDataMapper();
        mapper.SetInputData(polyData);
        actor = vtk.vtkActor();
        actor.SetMapper(mapper);

        actor.GetProperty().SetColor(self.colorDict.get(colorName.lower(), (1.0, 1.0, 1.0)));

        self.MultiRenderSphereMeshes[renderID][sphereName] = {
            "Actor": actor,
            "PolyData": polyData,
            "Points": points
        }
        self.renderers[renderID].AddActor(actor);
    def setSphereColorNameInRenderer(self, renderID, sphereName, colorName):
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");
        
        if sphereName in self.MultiRenderSphereMeshes[renderID]:
            color = self.colorDict.get(colorName.lower());
            if color:
                actor = self.MultiRenderSphereMeshes[renderID][sphereName]["Actor"];
                actor.GetProperty().SetColor(color);
            else:
                print(f"Color '{colorName}' not found. Available colors are: {list(self.color_dict.keys())}");
        else:
            print(f"Mesh '{sphereName}' not found.");
    def setSphereRainbowColorInRenderer(self, renderID, sphereName, rainBowValue):
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");
        
        if sphereName in self.MultiRenderSphereMeshes[renderID]:
            color = self.rainbowColor(rainBowValue);
            actor = self.MultiRenderSphereMeshes[renderID][sphereName]["Actor"];
            actor.GetProperty().SetColor(color[0], color[1], color[2]);
        else:
            print(f"Mesh '{sphereName}' not found.");

    #**************************************************** Point set interfacing functions
    ## For single renderer
    def addPointSet(self, setName, pointSet, inColorName="red", size=5.0):
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        # Create vtkPoints object and add each point to it
        points = vtk.vtkPoints()
        for coord in pointSet:
            points.InsertNextPoint(coord)

        # Use vtkPolyData to store points and a vertex glyph filter to render them as simple points
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)

        # Vertex glyph filter
        glyphFilter = vtk.vtkVertexGlyphFilter()
        glyphFilter.SetInputData(polydata)
        glyphFilter.Update()

        # Mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyphFilter.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Set color and point size
        actor.GetProperty().SetColor(self.colorDict.get(inColorName.lower(), (1.0, 1.0, 1.0)));
        actor.GetProperty().SetPointSize(size);

        # Store the actor in dictionary and add to renderer
        self.PointSets[setName] = actor
        self.mainRenderer.AddActor(actor)

    ## For specific renderers
    def addPointSetToRenderer(self, renderID, setName, pointSet, inColorName="red", size=5.0):
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        # Create vtkPoints object and add each point to it
        points = vtk.vtkPoints()
        for coord in pointSet:
            points.InsertNextPoint(coord)

        # Use vtkPolyData to store points and a vertex glyph filter to render them as simple points
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)

        # Vertex glyph filter
        glyphFilter = vtk.vtkVertexGlyphFilter()
        glyphFilter.SetInputData(polydata)
        glyphFilter.Update()

        # Mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyphFilter.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Set color and point size
        actor.GetProperty().SetColor(self.colorDict.get(inColorName.lower(), (1.0, 1.0, 1.0)));
        actor.GetProperty().SetPointSize(size);

        # Store the actor in dictionary and add to renderer
        self.MultiRenderPointSets[renderID][setName] = actor
        self.renderers[renderID].AddActor(actor)

    #**************************************************** Line rendering functions
    ## For single renderer
    def addSingleLine(self, lineName, startPoint, endPoint, colorName="red", lineWidth=2.0):
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        # Create a line source
        lineSource = vtk.vtkLineSource()
        lineSource.SetPoint1(startPoint)
        lineSource.SetPoint2(endPoint)

        # Mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(lineSource.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Set color and line width
        color = self.colorDict.get(colorName.lower(), (1.0, 1.0, 1.0))
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetLineWidth(lineWidth)

        # Add the line to the renderer and store reference
        self.mainRenderer.AddActor(actor)
        self.PointSets[lineName] = {"Actor": actor, "Source": lineSource};
    def updateSingleLinePoints(self, lineName, newStartPoint, newEndPoint):
        """
        Update both the starting and ending points of a line.

        :param lineName: The unique name of the line to update.
        :param newStartPoint: A tuple (x, y, z) indicating the new starting point of the line.
        :param newEndPoint: A tuple (x, y, z) indicating the new ending point of the line.
        """
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interface should be initialized first.")

        # Check if the line exists
        if lineName in self.PointSets:
            lineSource = self.PointSets[lineName]["Source"]
            lineSource.SetPoint1(newStartPoint)
            lineSource.SetPoint2(newEndPoint)
        else:
            print(f"Line '{lineName}' not found.")
    def addMultipleLines(self, linesDict, colorName="red", lineWidth=2.0):
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        for lineName, (startPoint, endPoint) in linesDict.items():
            self.addSingleLine(lineName, startPoint, endPoint, colorName, lineWidth)
    def updateMultipleLinesPoints(self, linesDict, colorName="red", lineWidth=2.0):
        """
        Add or update multiple lines. If a line already exists, update its start and end points.
        Otherwise, add the line with the specified parameters.

        :param linesDict: A dictionary where the keys are line names, and the values are tuples of (startPoint, endPoint).
        :param colorName: The color of the lines (default is "red").
        :param lineWidth: The width of the lines (default is 2.0).
        """
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interface should be initialized first.")

        for lineName, (startPoint, endPoint) in linesDict.items():
            if lineName in self.PointSets:
                # Update existing line points
                lineSource = self.PointSets[lineName]["Source"]
                lineSource.SetPoint1(startPoint)
                lineSource.SetPoint2(endPoint)
            else:
                # Add a new line
                self.addSingleLine(lineName, startPoint, endPoint, colorName, lineWidth)
    def setLineColor(self, lineName, colorName):
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        if lineName in self.PointSets:
            color = self.colorDict.get(colorName.lower())
            if color:
                actor = self.PointSets[lineName]["Actor"]
                actor.GetProperty().SetColor(color)
            else:
                print(f"Color '{colorName}' not found. Available colors are: {list(self.colorDict.keys())}")
        else:
            print(f"Line '{lineName}' not found.")
    def setLineOpacity(self, lineName, opacity):
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        if lineName in self.PointSets:
            if 0 <= opacity <= 1:
                actor = self.PointSets[lineName]["Actor"]
                actor.GetProperty().SetOpacity(opacity)
            else:
                print("Opacity must be between 0 and 1.")
        else:
            print(f"Line '{lineName}' not found.")
    def setLineWidth(self, lineName, lineWidth):
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        if lineName in self.PointSets:
            actor = self.PointSets[lineName]["Actor"]
            actor.GetProperty().SetLineWidth(lineWidth)
        else:
            print(f"Line '{lineName}' not found.")
    def addVectorLine(self, vectorName, startPoint, directionVector, length=1.0, colorName="blue", lineWidth=2.0):
        """
        Visualize a single normal vector (or directional vector) from a specified point.

        :param vectorName: A unique name for the vector in the visualization.
        :param startPoint: A tuple (x, y, z) indicating the starting point of the vector.
        :param directionVector: A tuple (dx, dy, dz) indicating the direction of the vector.
        :param length: The length of the vector to visualize (default is 1.0).
        :param colorName: The color of the vector line (default is "blue").
        :param lineWidth: The width of the vector line (default is 2.0).
        """
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interface should be initialized first.")

        # Normalize the direction vector and scale it to the desired length
        norm = vtk.vtkMath.Norm(directionVector)
        if norm == 0:
            raise ValueError("Direction vector must not be zero.")
        normalizedVector = [c / norm for c in directionVector]
        endPoint = [
            startPoint[i] + length * normalizedVector[i] for i in range(3)
        ]

        # Add the vector as a line using the existing single-line method
        self.addSingleLine(vectorName, startPoint, endPoint, colorName, lineWidth);
    def updateVectorLinePoint(self, vectorName, startPoint, directionVector, length=1.0, colorName="blue", lineWidth=2.0):
        """
        Add or update a single normal vector (or directional vector) from a specified point.

        :param vectorName: A unique name for the vector in the visualization.
        :param startPoint: A tuple (x, y, z) indicating the starting point of the vector.
        :param directionVector: A tuple (dx, dy, dz) indicating the direction of the vector.
        :param length: The length of the vector to visualize (default is 1.0).
        :param colorName: The color of the vector line (default is "blue").
        :param lineWidth: The width of the vector line (default is 2.0).
        """
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interface should be initialized first.")

        # Normalize the direction vector and scale it to the desired length
        norm = vtk.vtkMath.Norm(directionVector)
        if norm == 0:
            raise ValueError("Direction vector must not be zero.")
        normalizedVector = [c / norm for c in directionVector]
        endPoint = [startPoint[i] + length * normalizedVector[i] for i in range(3)]

        if vectorName in self.PointSets:
            # If vector already exists, update the start and end points
            lineSource = self.PointSets[vectorName]["Source"]
            lineSource.SetPoint1(startPoint)
            lineSource.SetPoint2(endPoint)
        else:
            # Otherwise, add a new vector line
            self.addSingleLine(vectorName, startPoint, endPoint, colorName, lineWidth)

    ## For specific renderers
    def addSingleLineToRenderer(self, renderID, lineName, startPoint, endPoint, colorName="red", lineWidth=2.0):
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        # Create a line source
        lineSource = vtk.vtkLineSource();
        lineSource.SetPoint1(startPoint);
        lineSource.SetPoint2(endPoint);

        # Mapper and actor
        mapper = vtk.vtkPolyDataMapper();
        mapper.SetInputConnection(lineSource.GetOutputPort());

        actor = vtk.vtkActor();
        actor.SetMapper(mapper);

        # Set color and line width
        color = self.colorDict.get(colorName.lower(), (1.0, 1.0, 1.0));
        actor.GetProperty().SetColor(color);
        actor.GetProperty().SetLineWidth(lineWidth);

        # Add the line to the renderer and store reference
        self.renderers[renderID].AddActor(actor);
        self.MultiRenderPointSets[renderID][lineName] = {"Actor": actor, "Source": lineSource};
    def addMultipleLinesToRenderer(self, renderID, linesDict, colorName="red", lineWidth=2.0):
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        for lineName, (startPoint, endPoint) in linesDict.items():
            self.addSingleLineToRenderer(lineName, renderID, startPoint, endPoint, colorName, lineWidth)
    def setLineColor(self, renderID, lineName, colorName):
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        if lineName in self.MultiRenderPointSets[renderID]:
            color = self.colorDict.get(colorName.lower())
            if color:
                actor = self.MultiRenderPointSets[renderID][lineName]["Actor"]
                actor.GetProperty().SetColor(color)
            else:
                print(f"Color '{colorName}' not found. Available colors are: {list(self.colorDict.keys())}")
        else:
            print(f"Line '{lineName}' not found.")
    def setLineOpacity(self, renderID, lineName, opacity):
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        if lineName in self.MultiRenderPointSets[renderID]:
            if 0 <= opacity <= 1:
                actor = self.MultiRenderPointSets[renderID][lineName]["Actor"]
                actor.GetProperty().SetOpacity(opacity)
            else:
                print("Opacity must be between 0 and 1.")
        else:
            print(f"Line '{lineName}' not found.")
    def setLineWidth(self, renderID, lineName, lineWidth):
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        if lineName in self.MultiRenderPointSets[renderID]:
            actor = self.MultiRenderPointSets[renderID][lineName]["Actor"]
            actor.GetProperty().SetLineWidth(lineWidth)
        else:
            print(f"Line '{lineName}' not found.")

    #**************************************************** For DICOM images function
    ## For single renderer
    def _processDICOMImageArray(self, numpyArray):
        # Convert to VTK format
        vtkArray = numpy_to_vtk(num_array=numpyArray.ravel(), deep=True, array_type=vtk.VTK_INT)
        return vtkArray
    def addDICOMImage(self, imageName, numpyArray):
        """
        Loads a DICOM image from a NumPy array and displays it.
        Parameters:
            imageName (str): Unique name for the DICOM image in the visualization.
            numpyArray (np.ndarray): Image data stored in a NumPy array.
        """
        if not self.isInitialized:
            raise ValueError("The visual interface should be initialized first.")

        # Convert NumPy array to VTK image 
        vtkArray = self._processDICOMImageArray(numpyArray);   
        dims = numpyArray.shape;
        imageData = vtk.vtkImageData();        
        imageData.SetDimensions(dims[1], dims[0], 1);
        imageData.GetPointData().SetScalars(vtkArray);

        # Create a mapper and actor
        mapper = vtk.vtkImageMapper();
        minValue = numpyArray.min();
        maxValue = numpyArray.max();
        mapper.SetInputData(imageData);
        mapper.SetColorWindow(maxValue - minValue);
        mapper.SetColorLevel((minValue - maxValue) // 2);
        actor = vtk.vtkActor2D();
        actor.SetMapper(mapper);

        # Store the image actor
        self.DICOMImages[imageName] = {
            "Actor": actor,
            "Mapper": mapper,
            "ImageData": imageData
        }
        self.mainRenderer.AddActor(actor)
    def updateDICOMImage(self, imageName, numpyArray):
        """
        Updates an existing DICOM image in the visualization.
        Parameters:
            imageName (str): Name of the existing DICOM image.
            numpyArray (np.ndarray): New image data stored in a NumPy array.
        """
        if not self.isInitialized:
            raise ValueError("The visual interface should be initialized first.")

        if imageName in self.DICOMImages:
            # Convert NumPy array to VTK image
            vtkArray = vtkArray = self._processDICOMImageArray(numpyArray);
            imageData = vtk.vtkImageData()
            dims = numpyArray.shape
            imageData.SetDimensions(dims[1], dims[0], 1)  # Adjust dimensions for VTK
            imageData.GetPointData().SetScalars(vtkArray)

            # Update the image data in the mapper
            self.DICOMImages[imageName]["Mapper"].SetInputData(imageData)
            self.DICOMImages[imageName]["ImageData"] = imageData

            # Render to apply the update
            self.render()
        else:
            print(f"DICOM image '{imageName}' not found.")
    def centerDICOMImage(self, imageName):
        """
        Moves the DICOM image to the center of the render window.

        Parameters:
            imageName (str): The unique name of the DICOM image to center.
        """
        if imageName not in self.DICOMImages:
            raise ValueError(f"No image found with name '{imageName}'.")

        # Get image actor and image dimensions
        actor = self.DICOMImages[imageName]["Actor"]
        imageData = self.DICOMImages[imageName]["ImageData"]
        dims = imageData.GetDimensions()  # (width, height, depth)
        imageWidth, imageHeight = dims[0], dims[1]

        # Ensure render window size is up-to-date
        self.renderWindow.Render()
        windowWidth, windowHeight = self.renderWindow.GetSize()

        # Compute position to center the image
        xPos = (windowWidth - imageWidth) // 2
        yPos = (windowHeight - imageHeight) // 2

        # Set display position
        actor.SetDisplayPosition(xPos, yPos)
    def setDICOMImageVisualParams(self, imageName, window, level):
        """
        Sets the color window and level for the specified DICOM image.
    
        Parameters:
            imageName (str): Unique name of the DICOM image in the visualization.
            window (float): The window value, controlling contrast.
            level (float): The level value, controlling brightness.
        """
        if imageName in self.DICOMImages:
            mapper = self.DICOMImages[imageName]["Mapper"]
            mapper.SetColorWindow(window)
            mapper.SetColorLevel(level)
        else:
            print(f"DICOM image '{imageName}' not found.")
    def getDICOMImageActor(self, imageName):
        """
        Retrieves the DICOM image actor from the visual interface.
    
        Parameters:
            imageName (str): The name of the DICOM image.
    
        Returns:
            vtk.vtkActor2D: The actor representing the DICOM image, or None if not found.
        """
        if imageName in self.DICOMImages:
            return self.DICOMImages[imageName]["Actor"]
        else:
            print(f"Error: DICOM image '{imageName}' not found.")
            return None
    def getDICOMImageSize(self, imageName):
        """
        Retrieves the dimensions (width, height) of a DICOM image.

        Parameters:
            imageName (str): The name of the DICOM image.

        Returns:
            tuple: (width, height) in pixels.
        """
        if imageName in self.DICOMImages:
            imageData = self.DICOMImages[imageName]["ImageData"]
            dims = imageData.GetDimensions()  # (width, height, depth)
            return dims[0], dims[1]
        else:
            print(f"Error: DICOM image '{imageName}' not found.")
            return None

    #**************************************************** For DICOM Volume function
    def _processDICOMVolumeArray(self, numpyArray):
        """Convert NumPy array to VTK format for volume rendering"""
        vtkArray = numpy_to_vtk(num_array=numpyArray.ravel(), deep=True, array_type=vtk.VTK_INT)
        return vtkArray
    def addDICOMVolume(self, volumeName, numpyArray, pixelSpacing):
        """
        Loads a 3D volume from a NumPy array and displays it using VTK volume rendering.
        
        Parameters:
            volumeName (str): Unique name for the volume.
            numpyArray (np.ndarray): 3D volume data stored in a NumPy array.
        """
        self.DICOMinValue = numpyArray.min(); self.DICOMMaxValue = numpyArray.max();
        vtkArray = self._processDICOMVolumeArray(numpyArray);
        dims = numpyArray.shape;

        volumeData = vtk.vtkImageData();
        volumeData.SetDimensions(dims[2], dims[1], dims[0]);  # Ensure correct ordering
        volumeData.GetPointData().SetScalars(vtkArray);
        volumeData.SetSpacing(pixelSpacing[0], pixelSpacing[1], pixelSpacing[2]);

        # Create volume mapper
        volumeMapper = vtk.vtkSmartVolumeMapper();
        volumeMapper.SetInputData(volumeData);

        # Define volume properties (color and opacity)
        volumeProperty = vtk.vtkVolumeProperty();

        # Color Transfer Function
        colorFunc = vtk.vtkColorTransferFunction()
        colorFunc.AddRGBPoint(self.DICOMinValue, 0.0, 0.0, 0.0)  # Black for low intensity
        colorFunc.AddRGBPoint((self.DICOMMaxValue - self.DICOMinValue)//2, 0.5, 0.5, 0.5)  # Mid-intensity gray
        colorFunc.AddRGBPoint(self.DICOMMaxValue, 1.0, 1.0, 1.0)  # White for high intensity
        volumeProperty.SetColor(colorFunc)

        # Opacity Transfer Function
        opacityFunc = vtk.vtkPiecewiseFunction()
        opacityFunc.AddPoint(self.DICOMinValue, 0.0)  # Fully transparent for very low values
        opacityFunc.AddPoint(0, 0.3)  # Low opacity for subtle details
        opacityFunc.AddPoint((self.DICOMMaxValue - self.DICOMinValue)//2, 0.7)  # Mid-opacity for soft tissue visibility
        opacityFunc.AddPoint(self.DICOMMaxValue, 1.0)  # Fully opaque for high-intensity structures
        volumeProperty.SetScalarOpacity(opacityFunc)

        volumeProperty.ShadeOn();
        volumeProperty.SetAmbient(0.4);
        volumeProperty.SetDiffuse(0.7);
        volumeProperty.SetSpecular(0.3);
        volumeProperty.SetSpecularPower(10);
        volumeProperty.SetInterpolationTypeToLinear();

        # Create volume actor
        volume = vtk.vtkVolume();
        volume.SetMapper(volumeMapper);
        volume.SetProperty(volumeProperty);

        self.DICOMVolumes[volumeName] = {
            "Actor": volume,
            "Mapper": volumeMapper,
            "VolumeData": volumeData
        }
        self.mainRenderer.AddVolume(volume)
    def updateDICOMVolume(self, volumeName, numpyArray):
        """
        Updates an existing volume with new data.
        
        Parameters:
            volumeName (str): Name of the existing volume.
            numpyArray (np.ndarray): New volume data.
        """
        if volumeName in self.DICOMVolumes:
            vtkArray = self._processDICOMVolumeArray(numpyArray)
            volumeData = vtk.vtkImageData()
            dims = numpyArray.shape
            volumeData.SetDimensions(dims[2], dims[1], dims[0])
            volumeData.GetPointData().SetScalars(vtkArray)

            self.DICOMVolumes[volumeName]["Mapper"].SetInputData(volumeData)
            self.DICOMVolumes[volumeName]["VolumeData"] = volumeData
        else:
            print(f"Volume '{volumeName}' not found.")
    def setDICOMVolumeVisualParams(self, volumeName, lowerThreshold, upperThreshold):
        """
        Sets up intensity thresholding for a DICOM volume, ensuring correct voxel range (0-255).
    
        Parameters:
            volumeName (str): Unique name of the DICOM volume.
            lowerThreshold (float): The minimum intensity value to be visualized.
            upperThreshold (float): The maximum intensity value to be visualized.
        """

        if volumeName not in self.DICOMVolumes:
            print(f"Error: DICOM volume '{volumeName}' not found.")
            return

        volume = self.DICOMVolumes[volumeName]["Actor"]
        volumeProperty = volume.GetProperty()

        # Ensure threshold values stay within valid voxel range [0, 255]
        lowerThreshold = max(self.DICOMMinValue, lowerThreshold)
        upperThreshold = min(self.DICOMMaxValue, upperThreshold)

        # **Opacity Transfer Function: Hide values outside thresholds**
        opacityFunc = vtk.vtkPiecewiseFunction();
        opacityFunc.AddPoint(self.DICOMMinValue, 0.0);  # Fully transparent for very low intensities
        opacityFunc.AddPoint(lowerThreshold, 0.0);  # Hide values below lower threshold
        opacityFunc.AddPoint((lowerThreshold + upperThreshold) / 2, 1.0);  # Mid-opacity transition
        opacityFunc.AddPoint(upperThreshold, 1.0);  # Fully opaque above upper threshold
        opacityFunc.AddPoint(self.DICOMMaxValue, 0.0);  # Hide values above upper threshold
        volumeProperty.SetScalarOpacity(opacityFunc);

        # **Preserve Grayscale Mapping via Color Transfer Function**
        colorFunc = vtk.vtkColorTransferFunction();
        colorFunc.AddRGBPoint(lowerThreshold, 0.0, 0.0, 0.0);  # Black for low intensity
        colorFunc.AddRGBPoint((lowerThreshold + upperThreshold) / 2, 0.5, 0.5, 0.5);  # Gray for mid-range intensity
        colorFunc.AddRGBPoint(upperThreshold, 1.0, 1.0, 1.0);  # White for high intensity
        volumeProperty.SetColor(colorFunc);

    #**************************************************** For Chessboard interfacing function
    def addChessboard(self, chessBoardName, position=(0, 0.3, 1.0), orientation=(90, 1, 0, 0), boardSize=(8, 8), squareSize=1.0, boardThickness=0.1, colors=('white', 'black')):
        if not self.mainRenderer:
            print("Error: mainRenderer is not initialized.")
            return

        offsetX = (boardSize[0] * squareSize) / 2.0
        offsetY = (boardSize[1] * squareSize) / 2.0

        points = vtk.vtkPoints()
        polys = vtk.vtkCellArray()
        colorsData = vtk.vtkUnsignedCharArray()
        colorsData.SetNumberOfComponents(3)  # RGB

        pointIndex = 0

        for i in range(boardSize[0] + 1):
            for j in range(boardSize[1] + 1):
                # Calculate corner points of the square
                p1 = (i * squareSize - offsetX, j * squareSize - offsetY, 0)
                p2 = ((i + 1) * squareSize - offsetX, j * squareSize - offsetY, 0)
                p3 = ((i + 1) * squareSize - offsetX, (j + 1) * squareSize - offsetY, 0)
                p4 = (i * squareSize - offsetX, (j + 1) * squareSize - offsetY, 0)

                # Add points to vtkPoints
                points.InsertNextPoint(p1)
                points.InsertNextPoint(p2)
                points.InsertNextPoint(p3)
                points.InsertNextPoint(p4)

                # Create polygon (square)
                polygon = vtk.vtkPolygon()
                polygon.GetPointIds().SetNumberOfIds(4)
                polygon.GetPointIds().SetId(0, pointIndex)
                polygon.GetPointIds().SetId(1, pointIndex + 1)
                polygon.GetPointIds().SetId(2, pointIndex + 2)
                polygon.GetPointIds().SetId(3, pointIndex + 3)

                polys.InsertNextCell(polygon)

                # Add color
                color = self.colorDict[colors[(i + j) % 2]]
                colorsData.InsertNextTuple3(*[int(c * 255) for c in color]) #convert from float 0-1 to int 0-255

                pointIndex += 4

        # Create PolyData
        polyData = vtk.vtkPolyData()
        polyData.SetPoints(points)
        polyData.SetPolys(polys)
        polyData.GetCellData().SetScalars(colorsData)

        # Create Mapper and Actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polyData)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Apply transform
        transform = vtk.vtkTransform()
        transform.Translate(position)
        transform.RotateWXYZ(orientation[0], orientation[1], orientation[2], orientation[3])
        actor.SetUserTransform(transform)

        # Add to renderer
        self.mainRenderer.AddActor(actor)

        self.ChessBoards[chessBoardName] = {
            "Actor": actor
        }
    
    #**************************************************** Camera interfacing functions
    ## For single renderer
    #### For single camera
    def resetMainCamera(self):
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        # Reset camera
        self.mainRenderer.ResetCamera();
    def setMainCameraPositionFocalPointViewUp(self, position, focalPoint, viewUp):
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        # Set camera focal point and view up        
        camera = self.mainRenderer.GetActiveCamera();
        camera.SetPosition(position);
        camera.SetFocalPoint(focalPoint);
        camera.SetViewUp(viewUp);
    def setMainCameraPosition(self, position):
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        # Set Camera position
        camera = self.mainRenderer.GetActiveCamera();
        camera.SetPosition(position);
    def setMainCameraFocalPoint(self, focalPoint):
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        # Set camera focal point
        camera = self.mainRenderer.GetActiveCamera();
        camera.SetFocalPoint(focalPoint);
    def setMainCameraViewUp(self, viewUp):
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        # Set camera view up
        camera = self.mainRenderer.GetActiveCamera();
        camera.SetViewUp(viewUp);
    def getMainCameraPosition(self):
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        # Set camera position
        camera = self.mainRenderer.GetActiveCamera();
        position = camera.GetPosition();
        return position;
    def getMainCameraFocalPoint(self):
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        # Set camera focal point
        camera = self.mainRenderer.GetActiveCamera();
        focalPoint = camera.GetFocalPoint();
        return focalPoint;
    def getMainCameraViewUp(self):
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        # Set camera view up
        camera = self.mainRenderer.GetActiveCamera();
        viewUp = camera.GetViewUp();
        return viewUp;

    #### For multiple cameras
    def addNewCamera(self, cameraName, cameraViewPort, cameraPosition, cameraViewUp, cameraFocalPoint, cameraBackgroundColor="white", cameraBoxColor="red",
                     intrinsicMatrix=None):
        # Check initialization
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        # Create renderer for the second camera
        renderer = vtk.vtkRenderer();
        xmin, ymin, xmax, ymax = cameraViewPort[0], cameraViewPort[1], cameraViewPort[2], cameraViewPort[3];
        renderer.SetViewport(xmin, ymin, xmax, ymax);
        self.renderWindow.AddRenderer(renderer);

        # Create new camera
        camera = vtk.vtkCamera();
        camera.SetPosition(cameraPosition);
        camera.SetViewUp(cameraViewUp);
        camera.SetFocalPoint(cameraFocalPoint);

        # Setting camera parameters
        if intrinsicMatrix is not None:
            vtkProjectionMatrix = self.opencvToVTKProjection(intrinsicMatrix);
            camera.SetExplicitProjectionTransformMatrix(vtkProjectionMatrix)
            camera.SetUseExplicitProjectionTransformMatrix(True)

        # Add camera to the renderer
        renderer.SetActiveCamera(camera);

        # Copy all actors from the main renderer to the current renderer
        self.copyActors(self.mainRenderer, renderer);

        # Set the background color
        renderer.SetBackground(self.colorDict.get(cameraBackgroundColor.lower(), (1.0, 1.0, 1.0)));

        # Add buffers for the new camera
        self.subCameras[cameraName] = {
            "Renderer": renderer
        }

        # Add the camera representation
        color = self.colorDict.get(cameraBoxColor.lower());
        if not color: color = np.array([1, 0, 0]);
        self.addCameraRepresentation(cameraName, color);
        self.updateCameraBox(cameraName);

        # Attach observer to track camera movement
        camera.AddObserver("ModifiedEvent", lambda obj, event: self.updateCameraBox(cameraName));
    def setLeftCameraName(self, leftCameraName):
        self.leftCameraName = leftCameraName;
    def setRightCameraName(self, rightCameraName):
        self.rightCameraName = rightCameraName;
    def setCenterCameraName(self, centerCameraName):
        self.centerCameraName = centerCameraName;
    def setUpperCameraName(self, upperCameraName):
        self.upperCameraName = upperCameraName;
    def setLowerCameraName(self, lowerCameraName):
        self.lowerCameraName = lowerCameraName;
    def linkCenterLeftCameras(self, centerCameraName, leftCameraName, visionSeparation):
        # Checking existing
        if (not self.isInitialized): return;

        # Check if the name exist in the camera
        if (centerCameraName not in self.subCameras): return;
        if (leftCameraName not in self.subCameras): return;

        # Getting buffers
        centerCamera = self.subCameras[centerCameraName]["Renderer"].GetActiveCamera();
        leftCamera = self.subCameras[leftCameraName]["Renderer"].GetActiveCamera();

        # Define callback function
        def updateLeftCamera(obj, event):
            # Getting buffers
            centerCamera = self.subCameras[centerCameraName]["Renderer"].GetActiveCamera();
            leftCamera = self.subCameras[leftCameraName]["Renderer"].GetActiveCamera();

            # Get left camera transformation details
            centerPos = np.array(centerCamera.GetPosition());
            centerFocal = np.array(centerCamera.GetFocalPoint());
            centerViewUp = np.array(centerCamera.GetViewUp());

            # Compute left camera's right vector (perpendicular to view direction & view up)
            centerViewDirection = centerFocal - centerPos;
            centerViewDirection /= np.linalg.norm(centerViewDirection);  # Normalize

            leftVector = np.cross(centerViewDirection, centerViewUp);  # Compute rightward vector
            leftVector /= np.linalg.norm(leftVector);  # Normalize
            
            # Compute new right camera position by applying stereo offset in the rightVector direction
            rightPos = centerPos - visionSeparation * leftVector;
            rightFocal = centerFocal - visionSeparation * leftVector;

            # Update right camera with the new computed transformation
            leftCamera.SetPosition(rightPos);
            leftCamera.SetFocalPoint(rightFocal);
            leftCamera.SetViewUp(centerViewUp);

            # Trigger a render update
            self.renderWindow.Render();

        # Attach observer to the left camera
        centerCamera.AddObserver("ModifiedEvent", updateLeftCamera);
    def linkCenterRightCameras(self, centerCameraName, rightCameraName, visionSeparation):
        # Checking existing
        if (not self.isInitialized): return;

        # Check if the name exist in the camera
        if (centerCameraName not in self.subCameras): return;
        if (rightCameraName not in self.subCameras): return;

        # Getting buffers
        centerCamera = self.subCameras[centerCameraName]["Renderer"].GetActiveCamera();
        rightCamera = self.subCameras[rightCameraName]["Renderer"].GetActiveCamera();

        # Define callback function
        def updateRightCamera(obj, event):
            # Getting buffers
            centerCamera = self.subCameras[centerCameraName]["Renderer"].GetActiveCamera();
            rightCamera = self.subCameras[rightCameraName]["Renderer"].GetActiveCamera();

            # Get left camera transformation details
            centerPos = np.array(centerCamera.GetPosition());
            centerFocal = np.array(centerCamera.GetFocalPoint());
            centerViewUp = np.array(centerCamera.GetViewUp());

            # Compute left camera's right vector (perpendicular to view direction & view up)
            centerViewDirection = centerFocal - centerPos;
            centerViewDirection /= np.linalg.norm(centerViewDirection);  # Normalize

            leftVector = np.cross(centerViewDirection, centerViewUp);  # Compute rightward vector
            leftVector /= np.linalg.norm(leftVector);  # Normalize
            rightVector = -leftVector;

            # Compute new right camera position by applying stereo offset in the rightVector direction
            rightPos = centerPos - visionSeparation * rightVector;
            rightFocal = centerFocal - visionSeparation * rightVector;

            # Update right camera with the new computed transformation
            rightCamera.SetPosition(rightPos);
            rightCamera.SetFocalPoint(rightFocal);
            rightCamera.SetViewUp(centerViewUp);

            # Trigger a render update
            self.renderWindow.Render();

        # Attach observer to the left camera
        centerCamera.AddObserver("ModifiedEvent", updateRightCamera);
    def linkCenterUpperCameras(self, centerCameraName, upperCameraName, visionSeparation):
        # Checking existing
        if (not self.isInitialized): return;

        # Check if the name exist in the camera
        if (centerCameraName not in self.subCameras): return;
        if (upperCameraName not in self.subCameras): return;

        # Getting buffers
        centerCamera = self.subCameras[centerCameraName]["Renderer"].GetActiveCamera();
        upperCamera = self.subCameras[upperCameraName]["Renderer"].GetActiveCamera();

        # Define call back function
        def updateUpperCamera(obj, event):
            # Getting buffers
            centerCamera = self.subCameras[centerCameraName]["Renderer"].GetActiveCamera();
            upperCamera = self.subCameras[upperCameraName]["Renderer"].GetActiveCamera();

            # Get left camera transformation details
            centerPos = np.array(centerCamera.GetPosition());
            centerFocal = np.array(centerCamera.GetFocalPoint());
            centerViewUp = np.array(centerCamera.GetViewUp());

            # Compute left camera's right vector (perpendicular to view direction & view up)
            centerViewDirection = centerFocal - centerPos;
            centerViewDirection /= np.linalg.norm(centerViewDirection);  # Normalize

            upperVector = centerViewUp;

            # Compute new right camera position by applying stereo offset in the rightVector direction
            upperPos = centerPos + visionSeparation * upperVector;
            upperFocal = centerFocal + visionSeparation * upperVector;

            # Update right camera with the new computed transformation
            upperCamera.SetPosition(upperPos);
            upperCamera.SetFocalPoint(upperFocal);
            upperCamera.SetViewUp(centerViewUp);

            # Trigger a render update
            self.renderWindow.Render();

        # Attach observer to the left camera
        centerCamera.AddObserver("ModifiedEvent", updateUpperCamera);
    def linkCenterLowerCameras(self, centerCameraName, lowerCameraName, visionSeparation):
        # Checking existing
        if (not self.isInitialized): return;

        # Check if the name exist in the camera
        if (centerCameraName not in self.subCameras): return;
        if (lowerCameraName not in self.subCameras): return;

        # Getting buffers
        centerCamera = self.subCameras[centerCameraName]["Renderer"].GetActiveCamera();
        lowerCamera = self.subCameras[lowerCameraName]["Renderer"].GetActiveCamera();

        # Creating callback function
        def updateLowerCamera(obj, event):
            # Getting buffers
            centerCamera = self.subCameras[centerCameraName]["Renderer"].GetActiveCamera();
            upperCamera = self.subCameras[lowerCameraName]["Renderer"].GetActiveCamera();

            # Get left camera transformation details
            centerPos = np.array(centerCamera.GetPosition());
            centerFocal = np.array(centerCamera.GetFocalPoint());
            centerViewUp = np.array(centerCamera.GetViewUp());

            # Compute left camera's right vector (perpendicular to view direction & view up)
            centerViewDirection = centerFocal - centerPos;
            centerViewDirection /= np.linalg.norm(centerViewDirection);  # Normalize

            lowerVector = -centerViewUp;

            # Compute new right camera position by applying stereo offset in the rightVector direction
            lowerPos = centerPos + visionSeparation * lowerVector;
            lowerFocal = centerFocal + visionSeparation * lowerVector;

            # Update right camera with the new computed transformation
            lowerCamera.SetPosition(lowerPos);
            lowerCamera.SetFocalPoint(lowerFocal);
            lowerCamera.SetViewUp(centerViewUp);

            # Trigger a render update
            self.renderWindow.Render();

        # Attach observer to the left camera
        centerCamera.AddObserver("ModifiedEvent", updateLowerCamera);
    def setCameraPosition(self, cameraName, cameraPosition):
        # Check initialization
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");
        
        # Checking existing
        if cameraName in self.subCameras:
            cameraRenderer = self.subCameras[cameraName]["Renderer"];
            camera = cameraRenderer.GetActiveCamera();
            camera.SetPosition(cameraPosition);
        else:
            print(f"Mesh '{cameraName}' not found.");
    def setCameraFocalPoint(self, cameraName, cameraFocalPoint):
        # Check initialization
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");
        
        # Checking existing
        if cameraName in self.subCameras:
            cameraRenderer = self.subCameras[cameraName]["Renderer"];
            camera = cameraRenderer.GetActiveCamera();
            camera.SetFocalPoint(cameraFocalPoint);
        else:
            print(f"Mesh '{cameraName}' not found.");
    def setCameraViewUp(self, cameraName, cameraViewUp):
        # Check initialization
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");
        
        # Checking existing
        if cameraName in self.subCameras:
            cameraRenderer = self.subCameras[cameraName]["Renderer"];
            camera = cameraRenderer.GetActiveCamera();
            camera.SetViewUp(cameraViewUp);
        else:
            print(f"Mesh '{cameraName}' not found.");
    def setBackgroundColor(self, cameraName, backgroundColorName):
        # Check initialization
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");
        
        # Checking existing
        if cameraName in self.subCameras:
            cameraRenderer = self.subCameras[cameraName]["Renderer"];
            color = self.colorDict.get(backgroundColorName.lower())
            if color:
                cameraRenderer.SetBackground(color);            
        else:
            print(f"Mesh '{cameraName}' not found.");
    def getCameraPosition(self, cameraName):
        # Check initialization
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");
        
        # Checking existing
        if cameraName in self.subCameras:
            cameraRenderer = self.subCameras[cameraName]["Renderer"];
            camera = cameraRenderer.GetActiveCamera();
            return camera.GetPosition();
        else:
            print(f"Mesh '{cameraName}' not found.");
            return None;
    def getCameraViewUp(self, cameraName):
        # Check initialization
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");
        
        # Checking existing
        if cameraName in self.subCameras:
            cameraRenderer = self.subCameras[cameraName]["Renderer"];
            camera = cameraRenderer.GetActiveCamera();
            return camera.GetViewUp();
        else:
            print(f"Mesh '{cameraName}' not found.");
            return None;
    def getCameraFocalPoint(self, cameraName):
        # Check initialization
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");
        
        # Checking existing
        if cameraName in self.subCameras:
            cameraRenderer = self.subCameras[cameraName]["Renderer"];
            camera = cameraRenderer.GetActiveCamera();
            return camera.GetFocalPoint();
        else:
            print(f"Mesh '{cameraName}' not found.");
            return None;
    def captureCameraScreen(self, cameraName, targetWidth=1920, targetHeight=1080):
        # Check initialization
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");
        
        # Checking existing
        if cameraName in self.subCameras:
            # Getting the renderer of the target camera
            cameraRenderer = self.subCameras[cameraName]["Renderer"];

            # Turn off drawing of the other renders
            allRenderers = self.renderWindow.GetRenderers();
            hiddenRenderers = [];
            for r in allRenderers:
                if r != cameraRenderer:
                    r.SetDraw(False);
                    hiddenRenderers.append(r);
            
            # Set the size of the windows to the target size and save old window size
            oldWindowSize = self.renderWindow.GetSize();
            self.renderWindow.SetOffScreenRendering(1);
            self.renderWindow.SetSize(targetWidth, targetHeight);
            oldViewPortSize = cameraRenderer.GetViewport();
            cameraRenderer.SetViewport(0, 0, 1, 1);
            self.renderWindow.Render();

            windowToImageFilter = vtk.vtkWindowToImageFilter();
            windowToImageFilter.SetInput(self.renderWindow);
            windowToImageFilter.Update();
            vtkImage = windowToImageFilter.GetOutput();

            # Convert vtkImageData to a NumPy array
            vtkArray = vtk_to_numpy(vtkImage.GetPointData().GetScalars());
            npImage = vtkArray.reshape(targetHeight, targetWidth, -1);

            # Convert RGB to BGR for OpenCV compatibility
            npImage = cv2.cvtColor(npImage, cv2.COLOR_RGB2BGR);
            npImage = cv2.flip(npImage, 0);

            # Revert to the old viewport
            cameraRenderer.SetViewport(*oldViewPortSize);
            self.renderWindow.SetSize(*oldWindowSize);
            self.renderWindow.SetOffScreenRendering(0);
            for r in hiddenRenderers: r.SetDraw(True);
            
            # Return the image
            return npImage;
        else:
            print(f"'{cameraName}' not found.");
            return None;
    def updateAllCameraActors(self):
        # Iterate through all cameras and update their renderers with actors from the main renderer
        for cameraName in self.subCameras:
            targetRenderer = self.subCameras[cameraName]["Renderer"]
            targetRenderer.RemoveAllViewProps()  # Clear existing actors
            self.copyActors(self.mainRenderer, targetRenderer)

    ## For specific renderers
    def resetCameraOfRenderer(self, renderID):
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        # Reset camera
        self.renderers[renderID].ResetCamera();
    def setCameraPositionFocalPointViewUpOfRenderer(self, renderID, position, focalPoint, viewUp):
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        # Set camera focal point and view up        
        camera = self.mainRenderer.GetActiveCamera();
        camera.SetPosition(position);
        camera.SetFocalPoint(focalPoint);
        camera.SetViewUp(viewUp);
    def setCameraPositionOfRenderer(self, renderID, position):
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        # Set Camera position
        camera = self.renderers[renderID].GetActiveCamera();
        camera.SetPosition(position);
    def setCameraFocalPointOfRenderer(self, renderID, focalPoint):
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        # Set camera focal point
        camera = self.renderers[renderID].GetActiveCamera();
        camera.SetFocalPoint(focalPoint);
    def setCameraViewUpOfRenderer(self, renderID, viewUp):
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        # Set camera view up
        camera = self.renderers[renderID].GetActiveCamera();
        camera.SetViewUp(viewUp);
    def getCameraPositionOfRenderer(self, renderID):
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        # Set camera position
        camera = self.renderers[renderID].GetActiveCamera();
        position = camera.GetPosition();
        return position;
    def getCameraFocalPointOfRenderer(self, renderID):
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        # Set camera focal point
        camera = self.renderers[renderID].GetActiveCamera();
        focalPoint = camera.GetFocalPoint();
        return focalPoint;
    def getCameraViewUpOfRenderer(self, renderID):
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        # Set camera view up
        camera = self.renderers[renderID].GetActiveCamera();
        viewUp = camera.GetViewUp();
        return viewUp;

    #**************************************************** General interfacing functions
    # Initializing function
    def initializeAutoSingleRenderer(self):
        # Checking initialized
        if self.isInitialized: return;

        # Initialize rendering objects
        self.mainRenderer = vtk.vtkRenderer();        
        self.renderWindow = vtk.vtkRenderWindow();
        self.renderWindow.AddRenderer(self.mainRenderer);
        self.renderWindowInteractor = vtk.vtkRenderWindowInteractor();
        self.renderWindowInteractor.SetRenderWindow(self.renderWindow);
        self.isInitialized = True;

        # Set ambient light of the main renderer
        self.mainRenderer.SetAmbient(0.5, 0.5, 0.5);
    def initializeSingleRenderer(self, vtkWidget):
        # Checking initialized
        if self.isInitialized: return;
        self.vtkWidget = vtkWidget;

        # Initialize rendering objects
        self.mainRenderer = vtk.vtkRenderer();        
        self.renderWindow = self.vtkWidget.GetRenderWindow();
        self.renderWindow.AddRenderer(self.mainRenderer);
        self.renderWindowInteractor = self.vtkWidget;
        self.renderWindowInteractor.Initialize();

        # Setting up flag
        self.isInitialized = True;

        # Set ambient light of the main renderer
        self.mainRenderer.SetAmbient(0.5, 0.5, 0.5);
    def initializeMultipleRenderersInTheSameLayer(self, numOfViewPorts, windowWidth, windowHeight, viewPortWidthRatio=1.0, viewportHeightRatio=1.0, isVertical=False):
        # Check if already initialized
        if self.isInitialized:
            return

        # Clear existing lists
        self.renderers.clear()

        # Create a single render window and interactor
        self.renderWindow = vtk.vtkRenderWindow()
        self.renderWindow.SetSize(windowWidth, windowHeight)
        self.renderWindowInteractor = vtk.vtkRenderWindowInteractor()

        # Calculate viewport sizes in normalized coordinates
        viewportWidth = viewPortWidthRatio / numOfViewPorts if not isVertical else viewPortWidthRatio
        viewportHeight = viewportHeightRatio / numOfViewPorts if isVertical else viewportHeightRatio

        # Add renderers with calculated viewports
        for i in range(numOfViewPorts):
            renderer = vtk.vtkRenderer()

            if isVertical:
                # Stack vertically
                viewport = [
                    0.5 - viewportWidth / 2,  # Center horizontally
                    i * viewportHeight,       # Adjust vertically
                    0.5 + viewportWidth / 2,  # Center horizontally
                    (i + 1) * viewportHeight
                ]
            else:
                # Stack horizontally
                viewport = [
                    i * viewportWidth,       # Adjust horizontally
                    0.5 - viewportHeight / 2, # Center vertically
                    (i + 1) * viewportWidth, # Adjust horizontally
                    0.5 + viewportHeight / 2  # Center vertically
                ]

            renderer.SetViewport(*viewport)
            self.renderWindow.AddRenderer(renderer)
            self.renderers.append(renderer)
            self.MultiRenderCoordinateSystems[i] = {};
            self.MultiRenderSurfaceMeshes[i] = {};
            self.MultiRenderSphereMeshes[i] = {};
            self.MultiRenderPointSets[i] = {};

        # Configure interactor
        self.renderWindowInteractor.SetRenderWindow(self.renderWindow);
        interactorStyle = vtk.vtkInteractorStyleTrackballCamera();
        self.renderWindowInteractor.SetInteractorStyle(interactorStyle);
        
        # Set flag
        self.isInitialized = True

    # Setting up functions
    ## For single window
    def setBackgroundColorRGB(self, red, green, blue):
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");
        
        # Set background color rgb
        self.mainRenderer.SetBackground(red, green, blue);
    def setBackgroundColorByName(self, colorName):
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        # Setback ground color name
        color = self.colorDict.get(colorName.lower());
        if color:
            self.mainRenderer.SetBackground(color);
        else:
            color = self.colorDict.get("white");
            self.mainRenderer.SetBackground(color);
    def setWindowSize(self, width, height):
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        # Set window size
        self.renderWindow.SetSize(width, height);
    def setWindowSizeName(self, screenSizeName):
        
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");
        screen_size = self.getScreenSize(screenSizeName);
        
        # Set window size
        if screen_size is None:
            raise ValueError(f"Unknown screen size: {screenSizeName}");
        
        width, height = screen_size;
        self.renderWindow.SetSize(width, height)
    def setTrackballCameraWindowInteractor(self):
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");
        
        # Try to initialize
        try:
            self.renderWindowInteractor.Initialize();
            interactorStyle = vtk.vtkInteractorStyleTrackballCamera();
            self.renderWindowInteractor.SetInteractorStyle(interactorStyle);
        except Exception as e:
            print(f"An error occur during rendering: {e}");
    def removeAllRenderedObjects(self):
        """
        Removes all rendering objects from the main render window.
        """
        self.mainRenderer.RemoveAllViewProps()  # Clears all objects from the renderer

    ## For specific renderers
    def setRendererBackgroundColor(self, renderID, red, green, blue):
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");
        
        # Getting renderer and windows
        renderer = self.renderers[renderID];

        # Set background color rgb
        renderer.SetBackground(red, green, blue);
    def setRendererBackgroundColorName(self, renderID, colorName):
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        # Get renderer and render windows
        renderer = self.renderers[renderID];

        # Setback ground color name
        color = self.colorDict.get(colorName.lower());
        if color:
            renderer.SetBackground(color);
        else:
            color = self.colorDict.get("white");
            renderer.SetBackground(color);
        
    ## For All renderers
    def setAllRenderersBackgroundColorName(self, red, green, blue):
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        # Interation for all
        numOfRenderers = len(self.renderers);
        for i in range(numOfRenderers):
            # Getting renderer and windows
            renderer = self.renderers[i];

            # Set background color rgb
            renderer.SetBackground(red, green, blue);
    def setAllRenderersBackgroundColorName(self, colorName):
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        # Interation for all
        numOfRenderers = len(self.renderers);
        for i in range(numOfRenderers):
            # Get renderer and render windows
            renderer = self.renderers[i];

            # Setback ground color name
            color = self.colorDict.get(colorName.lower());
            if color:
                renderer.SetBackground(color);
            else:
                color = self.colorDict.get("white");
                renderer.SetBackground(color);
   
    # Rendering functions
    def startWindowInteractor(self):
        """
        Start the render window interactor and allow quitting on a specific key press.
        """
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interface should be initialized first.")
        
        # Start window interactor loop
        self.renderWindowInteractor.Start()
    def render(self):
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        # Render
        self.renderWindow.Render();
    def processEvent(self):
        # Checking initialized
        if not self.isInitialized:
            raise ValueError("The visual interace should be initialized first.");

        # Update rendering        
        self.renderWindowInteractor.ProcessEvents();

    # For tracking keyboard even
    def setKeyBoardCallBack(self, keyCallback=None):
        if not self.renderWindowInteractor:
            self.initializeInteractor();

        def defaultKeyCallback(key):
            print(f"Key pressed: {key}");

        # Use the provided callback or fallback to the default one
        keyHandler = keyCallback if keyCallback else defaultKeyCallback

        def onKeyPress(obj, event):
            interactor = obj
            key = interactor.GetKeySym()
            keyHandler(key)

        # Add observer for keypress event
        self.renderWindowInteractor.AddObserver("KeyPressEvent", onKeyPress);

        # Start the interactor if not already started
        if not self.isInitialized:
            self.renderWindowInteractor.Start();

    # For screen capturing
    def captureWindowScreen(self):
        if self.renderWindow is None:
            raise ValueError("Render window not initialized. Please set up the rendering environment first.")

        # Get the captured image as vtkImageData
        windowToImageFilter = vtk.vtkWindowToImageFilter();
        windowToImageFilter.SetInput(self.renderWindow);
        windowToImageFilter.Update();
        vtkImage = windowToImageFilter.GetOutput();

        # Convert vtkImageData to a NumPy array
        width, height = self.renderWindow.GetSize();
        vtkArray = vtk_to_numpy(vtkImage.GetPointData().GetScalars());
        npImage = vtkArray.reshape(height, width, -1);

        # Convert RGB to BGR for OpenCV compatibility
        npImage = cv2.cvtColor(npImage, cv2.COLOR_RGB2BGR);
        npImage = cv2.flip(npImage, 0);

        return npImage;
