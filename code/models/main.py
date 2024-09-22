import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResNetClassificationModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetClassificationModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    def forward(self, x):
        x = self.resnet(x)
        return x
    
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.stop_training = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_training = True
    
def train(model, optimizer, loss_fn, train_loader, val_loader, device, ckpt_path="best.pt", early_stopping=None, epochs=100):
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate training accuracy
            _, predicted_train = torch.max(outputs, 1)
            correct_train += (predicted_train == labels).sum().item()
            total_train += labels.size(0)

        train_acc = correct_train / total_train
        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(train_acc)

        # Validation loop
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.long()  # Ensure labels are in long type (integer)

                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

                val_loss += loss.item()
                _, predicted_val = torch.max(outputs, 1)
                correct_val += (predicted_val == labels).sum().item()
                total_val += labels.size(0)

        val_acc = correct_val / total_val
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_acc)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {running_loss / len(train_loader):.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss / len(val_loader):.4f}, Val Acc: {val_acc:.4f}')

        # Early Stopping
        if early_stopping:
            early_stopping(val_loss / len(val_loader))
            if early_stopping.stop_training:
                print(f'Early stopping at epoch {epoch+1}')
                break

        # Save the model if the validation loss has improved
        if val_loss / len(val_loader) < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss / len(val_loader):.4f}. Saving checkpoint...")
            best_val_loss = val_loss / len(val_loader)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss / len(val_loader),
            }, ckpt_path)

def load_img(fname):
    img = read_image(fname)
    x = img / 255.
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(170, 170)),
    ])
    return transform(x)

def label_preprocess(img_path):
    # Image attributes
    train_features = pd.read_csv(f"{img_path}/train.csv")
    # Load and transform images
    images = torch.stack([load_img(f"{img_path}/img_align_celeba/train/{item['image_id']}") for _, item in  train_features.iterrows()])
    # Select label(s) from train_features
    labels = train_features["Blond_Hair"]
    labels.replace(-1, 0, inplace=True)
    labels = torch.from_numpy(labels.to_numpy()).float()
    processed_dataset = TensorDataset(images, labels)
    proportion = 0.8

    train_dataset, val_dataset = torch.utils.data.random_split(
        processed_dataset,
    [(int(len(images) * proportion)), len(images) - int(len(images) * proportion)],
    )

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader, labels, 




def main():
    model = ResNetClassificationModel()
    ckpt = torch.load("models/best.pt", map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    img_path = "code/datasets/archive"

    train_loader, val_loader, labels = label_preprocess(img_path)

    early_stopping = EarlyStopping(patience=5, delta=0.01)

    label_counts = pd.Series(labels.numpy()).value_counts()
    imbalance_ratio = label_counts.min() / label_counts.max()

    class_weights = torch.tensor([1.0, 1/imbalance_ratio]).to(device)  # Add class weights based on the class imbalance
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001) 

    # Train the model
    train(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        early_stopping=early_stopping
    )
    torch.save(model.state_dict(), 'models/model.pth')


if __name__ == '__main__':
    main()