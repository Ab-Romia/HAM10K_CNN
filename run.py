from model import MultiTaskModel
from dataset import get_dataloaders
from train import train_model
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_loader_class, val_loader_class, train_loader_seg, val_loader_seg = get_dataloaders()

# Initialize model
model = MultiTaskModel(num_classes=7).to(device)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
criterion_class = nn.CrossEntropyLoss()
criterion_seg = nn.BCELoss()

# Train
model, history = train_model(model, train_loader_class, val_loader_class, train_loader_seg, val_loader_seg,
                             criterion_class, criterion_seg, optimizer, scheduler, epochs=10)

# Save model
torch.save(model.state_dict(), 'outputs/saved_models/multitask_model.pth')
