# 🚀 MRI Brain Age Prediction with Ensemble Deep Learning 🚀

This repository hosts a brain age prediction project that leverages T1-weighted MRI scans to estimate brain age using an ensemble of three deep learning models: EfficientNetV2, Swin Transformer, and TinyViT. The ensemble combines predictions from axial, coronal, and sagittal views for enhanced accuracy, implemented in Python with PyTorch.

Explore the full workflow in our Colab Notebook: [brain_age_prediction.ipynb](https://colab.research.google.com/drive/1SOyfXVAInAG3c_is2r8Oj0X0nH1nSpCv?usp=sharing) 📓.

## Table of Contents

*   [📖 Overview](#-overview)
*   [✨ Features](#-features)
*   [⚙️ Requirements](#️-requirements)
*   [🔧 Setup](#-setup)
*   [📊 Dataset](#-dataset)
*   [▶️ Usage](#️-usage)
*   [📁 File Structure](#-file-structure)
*   [📈 Results](#-results)
*   [🤝 Contributing](#-contributing)
*   [📜 License](#-license)
*   [🙏 Acknowledgments](#-acknowledgments)

## 📖 Overview

The project predicts brain age from T1-weighted MRI scans by extracting 2D slices from three standard anatomical views and training specialized deep learning models on each view:

| View     | Model            | Slice Extraction Logic             | Example Input Shape (Slice) |
| :------- | :--------------- | :--------------------------------- | :-------------------------- |
| Axial    | EfficientNetV2   | All `[:, :, i]` slices           | `[1, 256, 256]`             |
| Coronal  | Swin Transformer | All `[:, i, :]` slices           | `[1, 256, 150]`             |
| Sagittal | TinyViT          | All `[i, :, :]` slices           | `[1, 256, 150]`             |

The ensemble method then averages the age predictions derived from each of these view-specific models to produce a final, potentially more robust, age estimate. The workflow includes separate training scripts for each model/view combination and an ensemble prediction script that evaluates the final performance using Mean Absolute Error (MAE).

## ✨ Features

*   **Multi-View Processing:** Extracts and processes all axial, coronal, and sagittal slices from 3D MRI volumes (assuming input shape like 256x256x150).
*   **Deep Learning Models:** Utilizes fine-tuned versions of EfficientNetV2, Swin Transformer, and TinyViT for the age regression task.
*   **Ensemble Prediction:** Combines outputs from the three view-specific models (simple averaging) for potentially improved accuracy and robustness.
*   **Visualization:** Generates PNG images showing representative slices for each view during data loading and potentially during evaluation.
*   **Evaluation:** Computes the Mean Absolute Error (MAE) and per-sample deviations, saving results to a CSV file.
*   **Robustness:** Includes basic error handling for missing files or metadata.

## ⚙️ Requirements

### Software
*   **Python:** 3.8+
*   **Libraries:** See `requirements.txt` or install manually:
    ```bash
    pip install torch torchvision timm nibabel pandas numpy matplotlib scikit-learn tqdm
    ```

### Hardware
*   **GPU:** Strongly recommended for training the models due to computational intensity. A GPU with >= 8GB VRAM is advisable (e.g., NVIDIA T4, V100, A100).
*   **CPU:** Sufficient for running the final ensemble *inference* if needed, but will be significantly slower.
*   **Storage:** Sufficient space for the dataset, Python environment, and saved model weights (potentially several GB). Google Drive is used in the Colab example.

### Git
*   **Git LFS:** Required for cloning the repository if the `.ipynb` notebook file is large, as indicated.

## 🔧 Setup

### Checklist
*   [ ] Install Git and Git LFS
*   [ ] Clone the repository
*   [ ] Install Python dependencies
*   [ ] Prepare dataset path
*   [ ] Prepare or download model weights path

### Steps

1.  **Install Git LFS** (if needed for large files like the notebook):
    *   Download and install Git LFS from [https://git-lfs.github.com/](https://git-lfs.github.com/).
    *   Enable it for your user account (run once):
        ```bash
        git lfs install
        ```
    *   *Note:* The main notebook (`brain_age_prediction.ipynb`, potentially >100 MB) might be stored using Git LFS. Cloning without LFS installed might result in pointer files instead of the actual notebook.

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/<your-username>/<your-repo-name>.git
    cd <your-repo-name>
    ```
    *(Replace `<your-username>/<your-repo-name>` with the actual repository URL)*

3.  **Install Dependencies:**
    *   (Recommended) Create a virtual environment:
        ```bash
        python -m venv venv
        source venv/bin/activate # Linux/macOS
        # venv\Scripts\activate # Windows
        ```
    *   Install from `requirements.txt` (if provided):
        ```bash
        pip install -r requirements.txt
        ```
    *   Or install manually:
        ```bash
        pip install torch torchvision timm nibabel pandas numpy matplotlib scikit-learn tqdm
        ```

4.  **Mount Google Drive (if using Google Colab):**
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

5.  **Prepare Dataset:**
    *   Place your T1-weighted MRI scans (standardized, registered) in a directory accessible by the scripts. The default path used is `/content/drive/MyDrive/T1_Dataset_Standardized_Registered`.
    *   Ensure a `participants.tsv` file is present in the *same directory* as the MRI scans (or adjust the path in the scripts). This file must contain `participant_id` and `age` columns.
    *   Verify that your `.nii.gz` files are named in a format where the participant ID can be extracted correctly (e.g., `<scan_id>_T1w_std_reg.nii.gz` where `<scan_id>` matches an ID in `participants.tsv`). The script assumes an input shape like (H, W, D) where H, W, D might be around 256x256x150 or similar.

6.  **Prepare Model Weights Paths:**
    *   The scripts expect trained model weights to be saved in specific subdirectories within `/content/drive/MyDrive/MRI_Age_Prediction_Ensemble/`.
    *   If training from scratch, these directories and `.pth` files will be created by the training scripts.
    *   If running only inference, ensure the pre-trained `.pth` files exist at these locations:
        *   `/content/drive/MyDrive/MRI_Age_Prediction_Ensemble/EfficientNetV2_Axial/EfficientNetV2_Axial_best.pth`
        *   `/content/drive/MyDrive/MRI_Age_Prediction_Ensemble/Swin_Coronal/Swin_Coronal_best.pth`
        *   `/content/drive/MyDrive/MRI_Age_Prediction_Ensemble/TinyViT_Sagittal/TinyViT_Sagittal_best.pth`

## 📊 Dataset

The project assumes a dataset consisting of:

1.  **T1-weighted MRI scans:** In NIfTI format (`.nii.gz`), preferably standardized and registered to a common template. The code was developed with shapes around 256x256x150.
2.  **Metadata File:** A tab-separated values file (`participants.tsv`) located in the same directory as the MRI scans, containing at least two columns:
    *   `participant_id`: Matches the identifier derived from the `.nii.gz` filenames.
    *   `age`: The chronological age of the participant at the time of the scan (as a float).

**Example Directory Structure (on Google Drive):**
