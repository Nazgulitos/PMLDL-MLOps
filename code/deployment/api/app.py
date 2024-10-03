from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import uvicorn
import torchvision.models as models

class ResNetClassificationModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetClassificationModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    def forward(self, x):
        x = self.resnet(x)
        return x

# Define the FastAPI app
app = FastAPI()

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNetClassificationModel(num_classes=2)
MODEL_PATH = "../../../models/model.pt"
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(device)

# Data transformations
transform = transforms.Compose([
    transforms.Resize((170, 170)),
    transforms.ToTensor(),
])

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the image file
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Run the image through the model
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        prediction = predicted.item()

    # Map the prediction to class names (assumed 0: Non-Blond, 1: Blond)
    class_names = ["Brunette", "Blond"]
    predicted_class = class_names[prediction]
    return {"prediction": predicted_class}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
