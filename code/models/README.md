# ResNet Classification Model Training

This project uses a ResNet model to classify images, with early stopping and class imbalance handling. The model is trained and validated on a dataset of images.

## Prerequisites
- Python 3.x
- CUDA (optional, for GPU acceleration)

## Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Nazgulitos/PMLDL-MLOps.git
   cd code/models

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download the model file**

Follow the instructions in the `code/deployment/api/README.md` file to download the model file.

4. **Download the dataset**

Follow the instructions in the `code/datasets/README.md` file to download the dataset.

5. Train the model:
   ```bash
   python main.py
   ```
6. The model will be saved in the `code/deployment/api` directory as `model.pth`.
