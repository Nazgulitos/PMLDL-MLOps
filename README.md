# PMLDL-MLOps
Assignment 1 for "Practical Machine Learning and Deep Learning" Innopolis University course.

## Blond or Non-blond Classification Model Web App using Streamlit and FastAPI

This project uses a ResNet model to classify images, trained and validated on a dataset of Celeb images.

## Example of working app:<br>
<div style="display: flex; justify-content: space-between;">
   <img src="screens/screen_1.png" alt="Example 1" width="400"/>
   <img src="screens/screen_2.png" alt="Example 2" width="400"/>
</div>

## Prerequisites
- Python 3.x
- CUDA (optional, for GPU acceleration)

## Project Structure
```
├── code   
│   ├── deployment
│   │   ├── api             # FastAPI code, Dockerfile
│   │   └── app             # Streamlit app code, Dockerfile
│   ├── docker-compose.yml  # Docker Compose file for both API and web app
│   └── models              # Script for model training
├── data                    # Dataset directory (need to download)
├── models                  # Saved model file (need to download)
├── screens                 # Screenshots of working app
├── README.md               # Instructions to run app
└── .gitignore
```

## Setup Instructions

1. Clone the repository and navigate to the code/deployment directory:
   ```bash
   git clone https://github.com/Nazgulitos/PMLDL-MLOps.git
   cd code/deployment
   ```

2. Download the model file or train your own model:

    2.1. Download the `model.pth` file from the following link:
    [Download model.pth](https://drive.google.com/file/d/1K9t5hI8dZC_foQluOKY1BVETCxP1Yb8c/view?usp=sharing)

    2.2. Place the downloaded `model.pth` file in the [models] directory.

3. Run the Docker Compose file:
   ```bash
   docker-compose up --build
   ```

4. Open the app in your browser:
   ```bash
   http://localhost:8501
   ```

5. Make a prediction by uploading an image file.
