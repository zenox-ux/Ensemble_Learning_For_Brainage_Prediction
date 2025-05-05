MRI Brain Age Prediction with Ensemble Deep Learning
This repository contains code for predicting brain age from T1-weighted MRI scans using an ensemble of three deep learning models: EfficientNetV2 (axial view), Swin Transformer (coronal view), and TinyViT (sagittal view). The ensemble combines predictions from these models to estimate brain age, leveraging multi-view MRI data for improved accuracy. The project is implemented in Python using PyTorch and processes .nii.gz files from a standardized dataset.
Table of Contents

Overview
Features
Requirements
Setup
Dataset
Usage
File Structure
Results
Contributing
License
Acknowledgments

Overview
The project aims to predict brain age from T1-weighted MRI scans by extracting specific 2D slices (axial, coronal, sagittal) and training specialized deep learning models for each view. The models are:

EfficientNetV2 (tf_efficientnetv2_s): Processes axial slices (volume[:, :, 75], 256x256).
Swin Transformer (swin_tiny_patch4_window7_224): Processes coronal slices (volume[:, 128, :], 256x150).
TinyViT (tiny_vit_5m_224.dist_in22k): Processes sagittal slices (volume[128, :, :], 256x150).

The ensemble script combines predictions using a mean ensemble technique, achieving robust age predictions. The project includes training scripts for each model and an ensemble prediction script that evaluates deviations from actual ages using Mean Absolute Error (MAE).
Features

Multi-View Processing: Extracts axial, coronal, and sagittal slices from 3D MRI volumes (256x256x150).
Deep Learning Models: Fine-tuned EfficientNetV2, Swin Transformer, and TinyViT for regression tasks.
Ensemble Prediction: Averages predictions from three models for improved accuracy.
Visualization: Generates PNG visualizations of MRI slices for each processed scan.
Evaluation: Computes MAE and per-sample deviations, saving results as CSV.
Error Handling: Skips invalid files or metadata, ensuring robust processing.

Requirements

Python: 3.8+
Libraries:pip install torch torchvision timm nibabel pandas numpy matplotlib scikit-learn tqdm


Hardware: GPU recommended for training (e.g., NVIDIA CUDA-enabled GPU); CPU sufficient for inference.
Storage: Google Drive for dataset and model weights (or local equivalent).

Setup

Clone the Repository:
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>


Install Dependencies:
pip install -r requirements.txt

Or manually install:
pip install torch torchvision timm nibabel pandas numpy matplotlib scikit-learn tqdm


Mount Google Drive (if using Colab):
from google.colab import drive
drive.mount('/content/drive')


Prepare Dataset:

Place the dataset in /content/drive/MyDrive/T1_Dataset_Standardized.
Ensure participants.tsv contains participant_id and age columns.
Verify .nii.gz files are named <scan_id>_T1w.nii.gz with shape 256x256x150.


Download Model Weights (if not training):

Place pre-trained weights in:
/content/drive/MyDrive/MRI_Age_Prediction_Ensemble/EfficientNetV2_Axial/EfficientNetV2_Axial_best.pth
/content/drive/MyDrive/MRI_Age_Prediction_Ensemble/Swin_Coronal/Swin_Coronal_best.pth
/content/drive/MyDrive/MRI_Age_Prediction_Ensemble/TinyViT_Sagittal/TinyViT_Sagittal_best.pth





Dataset
The dataset consists of T1-weighted MRI scans in .nii.gz format, standardized to shape 256x256x150. Metadata is provided in participants.tsv, mapping participant_id to age. Example structure:
/content/drive/MyDrive/T1_Dataset_Standardized/
├── sub-001_T1w.nii.gz
├── sub-002_T1w.nii.gz
├── ...
└── participants.tsv

participants.tsv format:
participant_id,age
sub-001,45.2
sub-002,67.8
...

Usage
Training Models

Train Individual Models:Run the training scripts for each view:
python efficientnetv2_axial_train.py
python swin_coronal_train.py
python tinyvit_sagittal_train.py


Outputs model weights to /content/drive/MyDrive/MRI_Age_Prediction_Ensemble/<Model_View>/.
Saves visualizations and logs during training.


Configuration:Each training script has a config dictionary for hyperparameters (e.g., learning_rate, batch_size, num_epochs). Modify as needed.


Running Ensemble Prediction

Run Ensemble Script:
python ensemble_prediction.py


Processes .nii.gz files, extracts slices, and predicts ages.
Saves results to /content/drive/MyDrive/MRI_Age_Prediction_Ensemble/Ensemble_Results/.


Outputs:

CSV: ensemble_results.csv with columns:participant_id,actual_age,axial_pred,coronal_pred,sagittal_pred,ensemble_pred,deviation
sub-001,45.2,46.1,44.8,45.5,45.47,0.27
...


Visualizations: PNGs (slices_<scan_id>.png) showing axial, coronal, and sagittal slices.
Console: Reports MAE, summary statistics, and best/worst predictions.


Example Command in Colab:
!python ensemble_prediction.py



Notes

Test Set: The ensemble script processes all .nii.gz files. To use a specific test set, modify file_list in ensemble_prediction.py.
Ensemble Weighting: Currently uses equal weights (mean). For weighted ensemble, adjust ensemble_pred in ensemble_predict (e.g., 0.4*axial_pred + 0.3*coronal_pred + 0.3*sagittal_pred).
Memory: For large datasets, limit file_list or use a GPU to avoid RAM issues.

File Structure
<your-repo-name>/
├── efficientnetv2_axial_train.py    # Training script for EfficientNetV2 (axial)
├── swin_coronal_train.py            # Training script for Swin Transformer (coronal)
├── tinyvit_sagittal_train.py        # Training script for TinyViT (sagittal)
├── ensemble_prediction.py           # Ensemble prediction script
├── requirements.txt                 # Python dependencies
└── README.md                        # This file

External Storage (Google Drive):
/content/drive/MyDrive/
├── T1_Dataset_Standardized/         # MRI dataset
│   ├── sub-001_T1w.nii.gz
│   ├── participants.tsv
│   └── ...
└── MRI_Age_Prediction_Ensemble/     # Model weights and results
    ├── EfficientNetV2_Axial/
    │   ├── EfficientNetV2_Axial_best.pth
    │   └── <training visualizations>
    ├── Swin_Coronal/
    │   ├── Swin_Coronal_best.pth
    │   └── <training visualizations>
    ├── TinyViT_Sagittal/
    │   ├── TinyViT_Sagittal_best.pth
    │   └── <training visualizations>
    └── Ensemble_Results/
        ├── ensemble_results.csv
        ├── slices_sub-001.png
        └── ...

Results

Performance: The ensemble typically achieves lower MAE than individual models due to complementary view information.
Metrics: The ensemble script reports:
Mean Absolute Error (MAE) across all samples.
Per-sample deviations (|ensemble_pred - actual_age|).
Summary statistics and top 5 best/worst predictions.


Visualizations: Each .nii.gz file produces a PNG with axial (256x256), coronal (256x150), and sagittal (256x150) slices.

Example output:
Ensemble MAE: 3.45 years
Summary Statistics:
       actual_age  ensemble_pred  deviation
count   100.0000      100.0000   100.0000
mean     55.1234       54.9876     3.4567
...
Top 5 Best Predictions:
  participant_id  actual_age  ensemble_pred  deviation
  sub-001         45.20       45.47         0.27
  ...

Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a feature branch (git checkout -b feature/<your-feature>).
Commit changes (git commit -m "Add <your-feature>").
Push to the branch (git push origin feature/<your-feature>).
Open a Pull Request.

Please ensure code follows PEP 8 style guidelines and includes tests where applicable.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Dataset: [Provide dataset source, e.g., "OpenNeuro" or specific study, if applicable].
Libraries: Thanks to PyTorch, timm, nibabel, and other open-source contributors.
Inspiration: Research on brain age prediction using deep learning for MRI analysis.

For issues or questions, please open an issue on GitHub or contact [].
