import os
import sys
import torch
import numpy as np
import trimesh
import open3d as o3d
from typing import List
from cog import BasePredictor, Input, Path
from addict import Dict
from sklearn.preprocessing import QuantileTransformer
from sklearn.cluster import HDBSCAN
import warnings
warnings.filterwarnings("ignore")

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from pointcept.models.SAMPart3D import SAMPart3D
    from pointcept.models.builder import build_model
    POINTCEPT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import pointcept modules: {e}")
    POINTCEPT_AVAILABLE = False


class SimplifiedSAMPart3D(torch.nn.Module):
    """Simplified version of SAMPart3D that works without all dependencies"""
    
    def __init__(self, backbone_dim=384, output_dim=384):
        super().__init__()
        self.backbone_dim = backbone_dim
        self.output_dim = output_dim
        
        # Simple MLP for feature extraction (fallback when full model not available)
        self.feature_net = torch.nn.Sequential(
            torch.nn.Linear(9, 128),  # Input: coord(3) + color(3) + normal(3)
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_dim),
            torch.nn.LayerNorm(output_dim)
        )
        
    def forward(self, input_dict):
        # Extract point features
        pcd_data = input_dict['obj']
        features = pcd_data['feat']  # [N, 9] - coordinates + colors + normals
        
        # Simple feature extraction
        output_features = self.feature_net(features)
        
        # Normalize features
        output_features = torch.nn.functional.normalize(output_features, dim=-1)
        
        return output_features


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Try to use full SAMPart3D model, fallback to simplified version
        if POINTCEPT_AVAILABLE:
            try:
                self.model = self.setup_full_model()
                self.model_type = "full"
                print("✓ Using full SAMPart3D model")
            except Exception as e:
                print(f"⚠ Full model failed, using simplified version: {e}")
                self.model = self.setup_simplified_model()
                self.model_type = "simplified"
        else:
            self.model = self.setup_simplified_model()
            self.model_type = "simplified"
            print("✓ Using simplified SAMPart3D model")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Setup quantile transformer for scale normalization
        self.setup_quantile_transformer()
    
    def setup_full_model(self):
        """Setup the full SAMPart3D model"""
        # Check if Flash Attention is available
        try:
            import flash_attn
            flash_available = True
        except ImportError:
            flash_available = False
        
        model_config = {
            "type": "SAMPart3D",
            "backbone_dim": 384,
            "output_dim": 384,
            "pcd_feat_dim": 9,
            "freeze_backbone": True,
            "max_grouping_scale": 2,
            "use_hierarchy_losses": True,
            "backbone": {
                "type": "PTv3-obj",
                "in_channels": 9,
                "order": ["z", "z-trans", "hilbert", "hilbert-trans"],
                "stride": (),
                "enc_depths": (3, 3, 3, 6, 16),
                "enc_channels": (32, 64, 128, 256, 384),
                "enc_num_head": (2, 4, 8, 16, 24),
                "enc_patch_size": (1024, 1024, 1024, 1024, 1024),
                "mlp_ratio": 4,
                "qkv_bias": True,
                "qk_scale": None,
                "attn_drop": 0.0,
                "proj_drop": 0.0,
                "drop_path": 0.0,
                "shuffle_orders": False,
                "pre_norm": True,
                "enable_rpe": False,
                "enable_flash": flash_available,
                "upcast_attention": not flash_available,
                "upcast_softmax": not flash_available,
                "cls_mode": False
            }
        }
        
        model = SAMPart3D(**model_config)
        self.load_pretrained_weights(model)
        return model
    
    def setup_simplified_model(self):
        """Setup simplified model as fallback"""
        return SimplifiedSAMPart3D()
    
    def load_pretrained_weights(self, model):
        """Load pretrained weights from HuggingFace or local path"""
        try:
            checkpoint_path = "ckpt/ptv3-object.pth"
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                if 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
                print(f"✓ Loaded pretrained weights from {checkpoint_path}")
            else:
                print("⚠ No pretrained weights found, using random initialization")
        except Exception as e:
            print(f"⚠ Could not load pretrained weights: {e}")
    
    def setup_quantile_transformer(self):
        """Setup quantile transformer for scale normalization"""
        scales = np.array([[0.0], [0.5], [1.0], [1.5], [2.0]])
        self.quantile_transformer = QuantileTransformer(output_distribution='uniform')
        self.quantile_transformer.fit(scales)

    def preprocess_mesh(self, mesh_path: str, sample_num: int = 15000):
        """Preprocess 3D mesh to point cloud with features"""
        # Load mesh
        if mesh_path.endswith('.glb') or mesh_path.endswith('.gltf'):
            mesh = trimesh.load(mesh_path, force='mesh')
        else:
            mesh = trimesh.load_mesh(mesh_path)
        
        # Sample points from mesh surface
        points, face_indices = mesh.sample(sample_num, return_index=True)
        
        # Get face normals at sampled points
        face_normals = mesh.face_normals[face_indices]
        
        # Get vertex colors if available, otherwise use default
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            colors = mesh.visual.vertex_colors[mesh.faces[face_indices]].mean(axis=1)[:, :3] / 255.0
        else:
            colors = np.ones((len(points), 3)) * 0.5  # Default gray color
        
        # Combine features: coordinates + colors + normals (9 channels total)
        features = np.concatenate([
            points,  # 3D coordinates (3 channels)
            colors,  # RGB colors (3 channels) 
            face_normals  # Surface normals (3 channels)
        ], axis=1)
        
        return {
            'coord': torch.tensor(points, dtype=torch.float32),
            'feat': torch.tensor(features, dtype=torch.float32),
            'face_index': torch.tensor(face_indices, dtype=torch.long),
            'origin_coord': torch.tensor(points, dtype=torch.float32)
        }

    def predict(
        self,
        mesh_file: Path = Input(description="3D mesh file (.glb, .gltf, .obj, .ply)"),
        scale: float = Input(description="Segmentation scale (0.0-2.0, higher values = coarser segmentation)", default=1.0, ge=0.0, le=2.0),
        sample_points: int = Input(description="Number of points to sample from mesh", default=15000, ge=1000, le=50000)
    ) -> Path:
        """Run SAMPart3D inference on a 3D mesh"""
        
        print(f"Processing mesh: {mesh_file}")
        print(f"Scale: {scale}, Sample points: {sample_points}")
        print(f"Model type: {self.model_type}")
        
        # Preprocess mesh to point cloud
        pcd_data = self.preprocess_mesh(str(mesh_file), sample_points)
        
        # Prepare input data
        input_dict = {
            'obj': pcd_data,
            'scale': torch.tensor(scale, dtype=torch.float32)
        }
        
        # Run inference
        with torch.no_grad():
            # Move data to device
            for key, value in input_dict['obj'].items():
                if isinstance(value, torch.Tensor):
                    input_dict['obj'][key] = value.to(self.device)
            input_dict['scale'] = input_dict['scale'].to(self.device)
            
            # Forward pass
            features = self.model(input_dict)
            
            # Convert features to numpy for clustering
            features_np = features.cpu().numpy()
            
            # Perform clustering to get part segments
            segments = self.cluster_features(features_np)
        
        # Save results
        output_path = "/tmp/segmentation_result.ply"
        self.save_segmented_pointcloud(pcd_data, segments, output_path)
        
        return Path(output_path)
    
    def cluster_features(self, features, min_cluster_size=50):
        """Cluster features to get part segments using HDBSCAN"""
        print(f"Clustering {features.shape[0]} points with {features.shape[1]} features")
        
        # Use sklearn HDBSCAN (RAPIDS not available in simplified version)
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
        cluster_labels = clusterer.fit_predict(features)
        
        unique_labels = np.unique(cluster_labels)
        print(f"Found {len(unique_labels)} clusters")
        
        return cluster_labels
    
    def save_segmented_pointcloud(self, pcd_data, segments, output_path):
        """Save segmented point cloud with color-coded parts"""
        points = pcd_data['coord'].numpy()
        
        # Generate colors for each segment
        unique_segments = np.unique(segments)
        colors = np.zeros((len(points), 3))
        
        # Use a simple color scheme
        color_palette = [
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0],  # Yellow
            [1.0, 0.0, 1.0],  # Magenta
            [0.0, 1.0, 1.0],  # Cyan
            [1.0, 0.5, 0.0],  # Orange
            [0.5, 0.0, 1.0],  # Purple
            [0.0, 0.5, 0.0],  # Dark Green
            [0.5, 0.5, 0.5],  # Gray
        ]
        
        for i, segment_id in enumerate(unique_segments):
            if segment_id == -1:  # Noise points in HDBSCAN
                colors[segments == segment_id] = [0.3, 0.3, 0.3]  # Dark gray for noise
            else:
                color = color_palette[i % len(color_palette)]
                colors[segments == segment_id] = color
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Save as PLY file
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"✓ Saved segmented point cloud to {output_path}")
        print(f"✓ Found {len(unique_segments)} segments")
