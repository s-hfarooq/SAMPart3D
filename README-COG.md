# SAMPart3D Cog Implementation for Replicate

This directory contains the Cog implementation for deploying SAMPart3D on Replicate.com.

## Files Added for Cog Deployment

- `cog.yaml` - Cog configuration file with environment setup and dependencies
- `predict.py` - Main prediction interface implementing the Cog BasePredictor
- `requirements-cog.txt` - Python dependencies specific to Cog deployment
- `test_cog.py` - Local test script to validate the implementation
- `README-COG.md` - This documentation file

## Prerequisites

1. **Cog Installation**: Install Cog on your local machine
   ```bash
   sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
   sudo chmod +x /usr/local/bin/cog
   ```

2. **Docker**: Ensure Docker is installed and running

3. **GPU Support**: NVIDIA GPU with CUDA 12.1 support for optimal performance

## Local Testing

1. **Build the Cog container**:
   ```bash
   cog build
   ```

2. **Test with a sample mesh**:
   ```bash
   cog predict -i mesh_file=@path/to/your/mesh.glb -i scale=1.0 -i sample_points=15000
   ```

3. **Run the test script**:
   ```bash
   python test_cog.py
   ```

## Model Inputs

- **mesh_file**: 3D mesh file (.glb, .gltf, .obj, .ply)
- **scale**: Segmentation scale (0.0-2.0, higher values = coarser segmentation)
- **sample_points**: Number of points to sample from mesh (1000-50000)

## Model Output

- Segmented point cloud saved as a PLY file with color-coded parts

## Deployment to Replicate

1. **Push to Replicate**:
   ```bash
   cog push r8.im/your-username/sampart3d
   ```

2. **Set up model on Replicate.com**:
   - Go to replicate.com and create a new model
   - Connect your pushed image
   - Configure model settings

## Key Features

- **GPU Acceleration**: Uses CUDA 12.1 for fast inference
- **Flexible Input**: Supports multiple 3D mesh formats
- **Scalable Segmentation**: Adjustable segmentation granularity
- **Robust Dependencies**: Handles optional RAPIDS dependencies with sklearn fallback
- **Error Handling**: Comprehensive error handling and logging

## Dependencies

### Core Dependencies
- PyTorch 2.1.0 with CUDA 12.1
- Point Transformer V3 (PTv3) backbone
- Tiny-CUDA-NN for MLP networks
- SpConv for sparse convolutions
- Flash Attention for efficient attention

### Optional Dependencies
- RAPIDS (cuDF/cuML) for GPU-accelerated clustering
- Falls back to scikit-learn if RAPIDS unavailable

## Architecture

The SAMPart3D model consists of:
1. **PTv3 Backbone**: Point Transformer V3 for feature extraction
2. **Instance Network**: MLP for instance-level features
3. **Position Network**: MLP for positional embeddings
4. **Clustering**: HDBSCAN for part segmentation

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `sample_points` parameter
2. **RAPIDS Installation Failed**: The model will automatically fall back to sklearn
3. **Flash Attention Issues**: Ensure CUDA 12.1 compatibility

### Performance Tips

- Use GPU instances for best performance
- Adjust `scale` parameter for desired segmentation granularity
- Balance `sample_points` between quality and speed

## License

This implementation follows the same license as the original SAMPart3D repository.
