# AI Teeth Whitening Pipeline

A comprehensive, modular pipeline that automatically detects teeth in photographs and applies a natural, realistic whitening effect using deep learning and advanced image processing.

## Features

* 🦷 **Automatic Mouth Detection**: Employs the RetinaFace detector to accurately locate faces and mouth regions in any image.
* 📐 **Precise Teeth Segmentation**: Leverages a trained U-Net model with a MobileNetV2 backbone to generate precise pixel-level masks of the teeth.
* 🎨 **Adaptive Whitening**: Intelligently analyzes the color of the teeth in the LAB color space and applies a proportional whitening effect, preventing unnatural glowing on already-white teeth and effectively brightening yellowed areas.
* ⚡ **Efficient Performance**: Optimized for speed, making it suitable for interactive applications.

## Architecture

The pipeline is engineered in three distinct, modular stages:

1.  **Mouth Detection (`teeth_whitener/detection/`)**: The `MouthDetector` class takes an input image, uses RetinaFace to find facial landmarks, and intelligently crops the mouth region with appropriate padding for the next stage.
2.  **Teeth Segmentation (`teeth_whitener/segmentation/`)**: The `predictor` module loads the pre-trained U-Net model (`best_teeth_segmentation_model.pth`) and processes the cropped mouth image, outputting a clean binary mask of the teeth.
3.  **Adaptive Whitening (`teeth_whitener/enhancement/`)**: The `LABTeethWhitener` class takes the original mouth crop and the generated mask. It converts the image to the LAB color space to independently adjust lightness (gamma correction) and reduce yellowness proportionally, ensuring a natural look.

## Setup and Installation

Follow these steps to get the project running on your local machine.

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/DhairyaCodes/teeth-whitener
    cd teeth-whitener
    ```

2.  **Create a Conda Environment** (Recommended):
    This project requires **Python 3.11**.
    ```bash
    conda create -n teeth_whitener python=3.11 -y
    conda activate teeth_whitener
    ```

3.  **Install Dependencies**:
    Install all required packages using the provided `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

You can run the pipeline either through the command-line interface or the interactive Streamlit web application.

### 1. Run the Command-Line Pipeline (`pipeline.py`)

This script is ideal for processing one or more images directly.

* **To process a single image**:
    ```bash
    python pipeline.py --images "path/to/your/image.jpg"
    ```
* **To process multiple images** (up to 5 at a time):
    ```bash
    python pipeline.py --images "image1.jpg" "image2.jpg"
    ```

The script will display a plot showing the original image, the generated mask, and the final whitened result for each image provided.

### 2. Launch the Streamlit Web App (`app.py`)

This provides a user-friendly interface to upload and process one photo at a time.

* **To launch the app**:
    ```bash
    streamlit run streamlit_app.py
    ```
* Your web browser will open with the application running. Simply upload an image to see the results.

## Project Structure

```
teeth-whitener/
├── teeth_whitener/
│   ├── __init__.py
│   ├── detection/
│   │   └── face_detector.py
│   ├── enhancement/
│   │   └── lab_whitening.py
│   └── segmentation/
│       └── predictor.py
├── models-weights/
│   └── best_teeth_segmentation_model.pth
├── streamlit_app.py
├── pipeline.py
├── requirements.txt
└── README.md
