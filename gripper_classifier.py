import os
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class GripperClassifier(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)  # Binary classification
        self.sigmoid = nn.Sigmoid()  # To get probabilities

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)

class DinoV2Dataset(Dataset):
    def __init__(self, feats_paths, labels_paths):
        self.feats_paths = feats_paths
        self.labels_paths = labels_paths
    
    def __len__(self):
        return len(self.feats_paths)
    
    def __getitem__(self, idx):
        features = torch.load(self.feats_paths[idx], map_location="cpu").to(torch.float32)
        labels = torch.load(self.labels_paths[idx], map_location="cpu").to(torch.float32)
        features = features.reshape(-1, 768)
        labels = labels.view(-1)
        labels = (labels > 0.5).float()  # Ensure binary labels
        return features, labels

if __name__ == "__main__":
    feature_dim = 768
    training_data = 'data/features'
    training_samples = [f for f in os.listdir(training_data) if f.endswith("_mask.pth")]

    dinov2_feats_path = [os.path.join(training_data, f.replace("_mask.pth", ".pth")) for f in training_samples]
    labels_path = [os.path.join(training_data, f) for f in training_samples]
    
    dataset = DinoV2Dataset(dinov2_feats_path, labels_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)

    classifier = GripperClassifier(in_dim=feature_dim).to("cuda")
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    num_epochs = 100
    classifier.train()
    for epoch in tqdm(range(num_epochs)):
        for features, labels in dataloader:
            features = features.squeeze().to("cuda")
            labels = labels.squeeze().to("cuda")
            optimizer.zero_grad()
            outputs = classifier(features.to(torch.float32)).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    torch.save(classifier.state_dict(), "gripper_classifier.pth")
