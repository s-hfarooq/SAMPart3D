import os
import sys
import torch
import numpy as np
import trimesh
import open3d as o3d
from PIL import Image
from cog import BasePredictor, Input, Path
from typing import Any, List
import warnings
from addict import Dict
import pickle
from sklearn.preprocessing import QuantileTransformer
from sklearn.cluster import KMeans

# Suppress warnings
warnings.filterwarnings("ignore")

# Check for optional dependencies
try:
    import tinycudann as tcnn
    TCNN_AVAILABLE = True
    print("✓ tiny-cuda-nn available")
except ImportError:
    TCNN_AVAILABLE = False
    print("⚠ tiny-cuda-nn not available, some features may not work")

# Check for Flash Attention
FLASH_ATTN_AVAILABLE = False
try:
    import flash_attn
    FLASH_ATTN_AVAILABLE = True
    print("✓ Flash Attention available")
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("⚠ Flash Attention not available, using standard attention")

# Try to compile pointops if not already installed
def compile_pointops():
    try:
        import pointops
        print("✓ pointops already installed")
        return True
    except ImportError:
        print("Compiling pointops...")
        try:
            import subprocess
            pointops_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs", "pointops")
            if os.path.exists(pointops_dir):
                subprocess.check_call([sys.executable, "setup.py", "install"], cwd=pointops_dir)
                print("✓ pointops compiled successfully")
                return True
            else:
                print(f"⚠ pointops directory not found at {pointops_dir}")
                return False
        except Exception as e:
            print(f"⚠ Failed to compile pointops: {e}")
            return False

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

compile_pointops()

try:
    from pointcept.models.SAMPart3D import SAMPart3D
    from pointcept.models.builder import build_model
    from pointcept.models.utils.structure import Point
except ImportError as e:
    print(f"Warning: Could not import pointcept modules: {e}")
    # We'll handle this in setup()

try:
    from pointcept.datasets.sampart3d_util import *
except ImportError:
    print("Warning: Could not import sampart3d_util, using fallback functions")


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Flash Attention availability was already checked at module level
        
        # Initialize model
        try:
            # Create model config
            model_config = {
                "backbone": {
                    "type": "PTv3-obj",  # This is the registered name in the models registry
                    "in_channels": 9,  # xyz + rgb + normal
                    "order": ["z", "z-trans", "hilbert", "hilbert-trans"],
                    "stride": (),
                    "enc_depths": [3, 3, 3, 6, 16],
                    "enc_channels": [32, 64, 128, 256, 384],
                    "enc_num_head": [2, 4, 8, 16, 24],
                    "enc_patch_size": [1024, 1024, 1024, 1024, 1024],
                    "mlp_ratio": 4,
                    "qkv_bias": True,
                    "qk_scale": None,
                    "attn_drop": 0.0,
                    "proj_drop": 0.0,
                    "drop_path": 0.0,
                    "shuffle_orders": False,
                    "pre_norm": True,
                    "enable_rpe": False,
                    "enable_flash": FLASH_ATTN_AVAILABLE,  # Use Flash Attention if available
                    "upcast_attention": False,
                    "upcast_softmax": False,
                    "cls_mode": False
                },
                "backbone_dim": 384,
                "output_dim": 384,
                "pcd_feat_dim": 9,  # xyz + rgb + normal
                "use_hierarchy_losses": True,
                "max_grouping_scale": 2,
                "freeze_backbone": True
            }
            
            # Create model
            self.model = SAMPart3D(**model_config)
            self.model.to(self.device)
            self.model.eval()
            
            # Setup quantile transformer for feature normalization
            self.setup_quantile_transformer()
            
            # Import necessary modules for Point class
            try:
                import spconv.pytorch as spconv
                from pointcept.models.utils.serialization import encode, decode
                from pointcept.models.utils import offset2batch, batch2offset
                import torch_scatter
                print("✓ All required modules for Point class loaded")
            except ImportError as e:
                print(f"Warning: Could not import some modules for Point class: {e}")
            
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def load_pretrained_weights(self):
        """Load pretrained weights from HuggingFace or local path"""
        try:
            # Try to load from a local checkpoint if available
            checkpoint_path = "ckpt/ptv3-object.pth"
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                if 'model' in checkpoint:
                    self.model.load_state_dict(checkpoint['model'], strict=False)
                else:
                    self.model.load_state_dict(checkpoint, strict=False)
                print(f"Loaded pretrained weights from {checkpoint_path}")
            else:
                print("No pretrained weights found, using random initialization")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
    
    def setup_quantile_transformer(self):
        """Setup quantile transformer for scale normalization"""
        # Create a simple quantile transformer for scale values
        scales = np.array([[0.0], [0.5], [1.0], [1.5], [2.0]])
        self.quantile_transformer = QuantileTransformer(output_distribution='uniform')
        self.quantile_transformer.fit(scales)
        
        # Set the quantile transformer in the model
        self.model.quantile_transformer = lambda x: torch.tensor(
            self.quantile_transformer.transform(x.cpu().numpy()), 
            device=x.device, 
            dtype=x.dtype
        )

    def preprocess_mesh(self, mesh_path, sample_num=15000):
        """Preprocess mesh to point cloud with features"""
        print(f"Preprocessing mesh: {mesh_path}")
        try:
            # Load mesh - handle both Scene and Mesh objects
            loaded = trimesh.load(mesh_path)
            
            # If it's a scene, extract the first mesh
            if isinstance(loaded, trimesh.Scene):
                print("Loaded a Scene object, extracting meshes...")
                # Get all meshes from the scene
                meshes = [m for m in loaded.geometry.values() if isinstance(m, trimesh.Trimesh)]
                if not meshes:
                    raise ValueError("No meshes found in the scene")
                
                # Combine all meshes into one
                mesh = meshes[0].copy()
                for m in meshes[1:]:
                    mesh = trimesh.util.concatenate([mesh, m])
                    
                print(f"Extracted and combined {len(meshes)} meshes")
            else:
                mesh = loaded
            
            # Ensure we have a valid mesh
            if not isinstance(mesh, trimesh.Trimesh):
                raise ValueError(f"Expected a Trimesh object, got {type(mesh)}")
                
            # Sample points
            points, face_indices = trimesh.sample.sample_surface(mesh, sample_num)
            
            # Get normals
            normals = mesh.face_normals[face_indices]
            
            # Get colors if available, otherwise use default
            if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                # Interpolate vertex colors to sampled points
                colors = []
                for i, face_idx in enumerate(face_indices):
                    face = mesh.faces[face_idx]
                    # Get barycentric coordinates
                    barycentric = trimesh.triangles.points_to_barycentric(
                        triangles=mesh.triangles[face_idx].reshape(1, 3, 3),
                        points=points[i].reshape(1, 3)
                    )
                    # Interpolate colors
                    color = np.zeros(3)
                    for j in range(3):
                        color += barycentric[0, j] * mesh.visual.vertex_colors[face[j], :3]
                    colors.append(color)
                colors = np.array(colors) / 255.0  # Normalize to [0, 1]
            else:
                # Use default color (gray)
                colors = np.ones((sample_num, 3)) * 0.5
            
            # Normalize coordinates to [-1, 1]
            points = points - mesh.bounding_box.centroid
            points = points / (mesh.bounding_box.extents.max() / 2)
            
            # Create point cloud data with proper batch information
            batch = torch.zeros(sample_num, dtype=torch.int64)  # All points belong to batch 0
            offset = torch.tensor([0, sample_num], dtype=torch.int64)  # Offset for batch 0
            
            point_data = {
                'coord': torch.tensor(points, dtype=torch.float32),
                'normal': torch.tensor(normals, dtype=torch.float32),
                'color': torch.tensor(colors, dtype=torch.float32),
                'batch': batch,
                'offset': offset,
                'grid_size': torch.tensor(0.02, dtype=torch.float32)  # Default grid size for voxelization
            }
            
            # Combine features (xyz + rgb + normal)
            point_data['feat'] = torch.cat([
                point_data['coord'],
                point_data['color'],
                point_data['normal']
            ], dim=1)
            
            print(f"Preprocessed mesh with {sample_num} points")
            return point_data
            
        except Exception as e:
            print(f"Error preprocessing mesh: {e}")
            raise

    def predict(
        self,
        mesh_file: Path = Input(description="3D mesh file to segment"),
        num_points: int = Input(description="Number of points to sample from mesh", default=15000),
        num_clusters: int = Input(description="Number of segments to generate", default=10),
        scale: float = Input(description="Scale parameter for segmentation granularity (0.1-1.0)", default=0.5),
    ) -> Path:
        """Run a single prediction on the model"""
        print("Running prediction...")
        
        # Preprocess mesh to point cloud
        point_data = self.preprocess_mesh(str(mesh_file), sample_num=num_points)
        
        # Run inference
        with torch.no_grad():
            # Move data to device
            for key, value in point_data.items():
                if isinstance(value, torch.Tensor):
                    point_data[key] = value.to(self.device)
            
            # Create a data dictionary with all required fields for the model
            # The SAMPart3D model expects an input_dict with 'obj' and 'scale' keys
            point_dict = {
                'coord': point_data['coord'],
                'feat': point_data['feat'],
                'batch': point_data['batch'],
                'offset': point_data['offset'],
                'grid_size': point_data['grid_size']
            }
            
            # Format the input as expected by SAMPart3D
            # Pass scale as a scalar value, not a tensor
            input_dict = {
                'obj': point_dict,
                'scale': scale  # Pass the scalar value directly
            }
            
            # Debug information
            print(f"Input dict keys: {input_dict.keys()}")
            print(f"Object keys: {input_dict['obj'].keys()}")
            print(f"Batch shape: {input_dict['obj']['batch'].shape}")
            print(f"Offset shape: {input_dict['obj']['offset'].shape}")
            
            # Forward pass through the model - the model will handle Point creation and serialization
            features = self.model(input_dict)
            
            # Convert features to numpy for clustering
            features_np = features.cpu().numpy()
            
            # Perform clustering on features
            kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(features_np)
            labels = kmeans.labels_
            
            # Save results
            output_file = Path(f"/tmp/segmentation_result_{os.path.basename(str(mesh_file))}.ply")
            self.save_segmentation(point_data['coord'].cpu().numpy(), labels, str(output_file))
            
            print(f"Segmentation completed with {num_clusters} segments")
            return output_file
    
    def save_segmentation(self, points, labels, output_path):
        """Save segmented point cloud to PLY file"""
        try:
            # Create colored point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # Generate random colors for segments
            num_segments = np.max(labels) + 1
            colors = np.random.random((num_segments, 3))
            point_colors = colors[labels]
            
            pcd.colors = o3d.utility.Vector3dVector(point_colors)
            
            # Save to file
            o3d.io.write_point_cloud(output_path, pcd)
            print(f"Segmentation saved to {output_path}")
        except Exception as e:
            print(f"Error saving segmentation: {e}")
            raise
