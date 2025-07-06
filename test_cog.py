#!/usr/bin/env python3
"""
Simple test script to validate the Cog implementation for SAMPart3D
"""

import os
import sys
import tempfile
import numpy as np
import trimesh

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_mesh():
    """Create a simple test mesh for testing"""
    # Create a simple cube mesh
    mesh = trimesh.creation.box(extents=[2, 2, 2])
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.obj', delete=False)
    mesh.export(temp_file.name)
    return temp_file.name

def test_cog_predictor():
    """Test the Cog predictor"""
    try:
        from predict import Predictor
        
        # Initialize predictor
        print("Initializing SAMPart3D predictor...")
        predictor = Predictor()
        predictor.setup()
        print("✓ Predictor setup successful")
        
        # Create test mesh
        print("Creating test mesh...")
        test_mesh_path = create_test_mesh()
        print(f"✓ Test mesh created: {test_mesh_path}")
        
        # Test preprocessing
        print("Testing mesh preprocessing...")
        pcd_data = predictor.preprocess_mesh(test_mesh_path, sample_num=1000)
        print(f"✓ Preprocessing successful, got {pcd_data['coord'].shape[0]} points")
        
        # Test feature clustering (without full model inference)
        print("Testing feature clustering...")
        dummy_features = np.random.randn(1000, 384)  # Dummy features
        segments = predictor.cluster_features(dummy_features, min_cluster_size=10)
        print(f"✓ Clustering successful, found {len(np.unique(segments))} segments")
        
        # Clean up
        os.unlink(test_mesh_path)
        print("✓ All tests passed!")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing SAMPart3D Cog implementation...")
    success = test_cog_predictor()
    sys.exit(0 if success else 1)
