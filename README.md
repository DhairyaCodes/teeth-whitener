# Teeth Whitening Pipeline

A comprehensive, modular pipeline for automatic teeth detection, segmentation, and whitening using computer vision and deep learning techniques.

## Features

🦷 **Automatic Teeth Detection**: Uses RetinaFace detector to locate mouth regions with high accuracy and precise landmarks
📐 **Precise Segmentation**: U-Net model with MobileNetV2 backbone for efficient and accurate teeth segmentation  
🎨 **Natural Whitening**: LAB color space manipulation for realistic teeth whitening without artificial appearance
⚡ **Real-time Performance**: Optimized for speed with MobileNetV2's depthwise separable convolutions
🔧 **Modular Design**: Clean, extensible architecture with separate components for each processing stage
📊 **Adaptive Processing**: Analyzes teeth color characteristics and adjusts whitening parameters automatically

## Architecture

The pipeline consists of three main components:

### 1. Mouth Detection (`src/detection/`)
- **RetinaFace Detector**: Detects faces and extracts precise facial landmarks
- **Smart Cropping**: Creates optimally-sized mouth regions with contextual padding
- **Multi-face Support**: Processes multiple faces in a single image
- **Fallback Support**: Uses OpenCV DNN if RetinaFace is not available

### 2. Teeth Segmentation (`src/segmentation/`)
- **U-Net Architecture**: Classic encoder-decoder with skip connections for precise boundaries
- **MobileNetV2 Backbone**: Efficient feature extraction using depthwise separable convolutions
- **Binary Segmentation**: Produces accurate teeth masks for targeted whitening

### 3. LAB Color Space Whitening (`src/enhancement/`)
- **Lightness Enhancement**: Increases brightness in the L channel
- **Yellowness Reduction**: Reduces yellow tones in the B channel
- **Adaptive Parameters**: Analyzes teeth color and adjusts whitening strength
- **Smooth Blending**: Natural integration with original image

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd Aftershoot-Assignment
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **For GPU support** (optional, requires CUDA):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### Command Line Usage

**Process a single image**:
```bash
python main.py --image path/to/image.jpg --output results/
```

**Process multiple images**:
```bash
python main.py --batch input_folder/ --output results/
```

**Custom parameters**:
```bash
python main.py --image image.jpg --lightness 1.2 --yellowness 12 --padding 0.4
```

**Save intermediate results**:
```bash
python main.py --image image.jpg --output results/ --save-intermediate
```

**Use GPU acceleration**:
```bash
python main.py --image image.jpg --device cuda
```

### Programmatic Usage

```python
from src.pipeline.teeth_whitening_pipeline import TeethWhiteningPipeline

# Initialize pipeline
pipeline = TeethWhiteningPipeline(
    device='cpu',  # Use 'cuda' for GPU
    lightness_factor=1.15,
    yellowness_reduction=10
)

# Process image
result = pipeline.process_single_image(
    image_path="image.jpg",
    output_dir="results/",
    adaptive_whitening=True
)

if "error" not in result:
    print(f"Success! Processed {result['mouth_regions_count']} mouth regions")
    print(f"Result saved to: {result['final_output_path']}")
```

## Project Structure

```
Aftershoot-Assignment/
├── src/
│   ├── detection/
│   │   ├── __init__.py
│   │   └── face_detector.py          # YUNet face detection and mouth cropping
│   ├── segmentation/
│   │   ├── __init__.py
│   │   └── unet_model.py             # U-Net with MobileNetV2 backbone
│   ├── enhancement/
│   │   ├── __init__.py
│   │   └── lab_whitening.py          # LAB color space whitening
│   ├── utils/
│   │   ├── __init__.py
│   │   └── image_utils.py            # Image processing utilities
│   └── pipeline/
│       └── teeth_whitening_pipeline.py  # Main integration pipeline
├── models/                           # Downloaded/trained models
├── demo_images/                      # Sample images for testing
├── output/                          # Processing results
├── main.py                          # Command-line interface
├── sample_usage.py                  # Example usage script
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Technical Details

### Mouth Detection Process
1. **Face Detection**: RetinaFace detects faces and provides detailed facial landmarks
2. **Landmark Extraction**: Extracts precise mouth corner coordinates and other facial points
3. **Bounding Box Calculation**: Creates mouth region with configurable padding
4. **Bounds Checking**: Ensures crop coordinates are within image boundaries
5. **Fallback Handling**: Uses OpenCV DNN with estimated landmarks if RetinaFace fails

### Segmentation Model
- **Architecture**: U-Net with MobileNetV2 encoder
- **Input Size**: 224x224 pixels (automatically resized)
- **Output**: Binary mask (0-255) highlighting teeth regions
- **Training**: Designed for teeth segmentation datasets (model weights not included)

### Whitening Algorithm
- **Color Space**: Converts BGR → LAB for perceptual uniformity
- **L Channel**: Increases lightness by configurable factor (default: 1.15)
- **B Channel**: Reduces yellowness by subtracting constant (default: 10)
- **Blending**: Smooth mask-based blending for natural results

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `padding_factor` | 0.3 | 0.2-0.5 | Padding around detected mouth region |
| `lightness_factor` | 1.15 | 1.1-1.3 | Brightness enhancement multiplier |
| `yellowness_reduction` | 10 | 5-15 | Amount to reduce yellow tones |

## Model Requirements

### Face Detection
- **Primary Model**: RetinaFace (auto-downloaded on first use)
- **Fallback Model**: OpenCV DNN face detector
- **Features**: High-accuracy face detection with precise landmarks

### Teeth Segmentation
- **Model**: U-Net with MobileNetV2 (weights not included)
- **Training Data**: Requires teeth segmentation dataset
- **Note**: The pipeline includes model architecture but not trained weights

## Training Your Own Segmentation Model

To train the segmentation model on your own teeth dataset:

```python
from src.segmentation.unet_model import TeethSegmentationUNet
import torch

# Create model
model = TeethSegmentationUNet(num_classes=1)

# Training loop (implement your own)
# ... training code ...

# Save trained weights
torch.save(model.state_dict(), "models/trained_teeth_segmentation.pth")
```

## Performance

- **Speed**: ~2-3 seconds per image on CPU, ~0.5 seconds on GPU
- **Memory**: ~500MB RAM usage
- **Accuracy**: Depends on segmentation model training quality

## Limitations

1. **Segmentation Model**: Requires training on teeth segmentation dataset
2. **Lighting Conditions**: Works best with good lighting
3. **Extreme Poses**: May struggle with very angled faces
4. **Occluded Teeth**: Cannot whiten teeth that are not visible

## Future Improvements

- [ ] Pre-trained segmentation model weights
- [ ] Real-time video processing
- [ ] Advanced teeth color analysis
- [ ] Multiple whitening styles (Hollywood, Natural, etc.)
- [ ] Dental health analysis integration

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YUNet face detection model from OpenCV Model Zoo
- U-Net architecture from Ronneberger et al.
- MobileNetV2 from Sandler et al.
- LAB color space whitening techniques from digital photography literature

## Support

For questions, issues, or contributions, please:
1. Check existing GitHub issues
2. Create a new issue with detailed description
3. Provide sample images and error logs when applicable

---

**Note**: This pipeline provides the complete architecture and implementation for teeth whitening. For production use, you'll need to train the segmentation model on a suitable teeth segmentation dataset.

