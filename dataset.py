import os
from sklearn.model_selection import train_test_split
import numpy as np
import glob as gb
import pandas as pd
import shutil
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch.backends.cudnn as cudnn
import torch

# Loading labels
dataset = "/kaggle/input/ham1000-segmentation-and-classification/"
label_path = dataset + "GroundTruth.csv"
mask_dir = dataset+"masks/"
images_dir = dataset + "images/"
img_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".jpg")])
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])

#%%
image_names = [os.path.splitext(f)[0] for f in img_files]    # ('ISIC_0024306', '.jpg')
mask_names = [os.path.splitext(f)[0].replace('_segmentation', '') for f in mask_files]    # ('ISIC_0024306_segmentation', 'png')

missing_masks = [f for f in image_names if f not in mask_names]

if len(missing_masks) == 0:
    print('No missing masks found.')
else:
    print(f"There are {len(missing_masks)} missing masks found:")
    print(missing_masks)
#%%
len(img_files), len(mask_files)
#%%

#%%
def img_mask_paths(img_dir, mask_dir):
    img_path = sorted(gb.glob(os.path.join(img_dir, '*.jpg')))
    mask_path = sorted(gb.glob(os.path.join(mask_dir, '*.png')))
    return np.array(img_path), np.array(mask_path)

imgs_path, masks_path = img_mask_paths(images_dir, mask_dir)
#%%
# Load the dataset
dataset = "/kaggle/input/ham1000-segmentation-and-classification/"
label_path = os.path.join(dataset, "GroundTruth.csv")
labels_df = pd.read_csv(label_path)
#%%
# Create a new column 'class' that contains the class label
labels_df['class'] = labels_df[['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']].idxmax(axis=1)

#%%
# Split dataset
train_df, temp_df = train_test_split(labels_df, test_size=0.2, stratify=labels_df['class'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['class'])
train_dir = 'train'
val_dir = 'val'
test_dir = 'test'

# Create the directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
train_df.to_csv("train/metadata.csv", index=False)
val_df.to_csv("val/metadata.csv", index=False)
test_df.to_csv("test/metadata.csv", index=False)
print("Train class distribution:\n", train_df['class'].value_counts())
print("Val class distribution:\n", val_df['class'].value_counts())
print("Test class distribution:\n", test_df['class'].value_counts())
#%%
# Function to save splits
def save_split(df, split_name):
    os.makedirs(split_name, exist_ok=True)
    for _, row in df.iterrows():
        shutil.copy(f'HAM10000_dataset/images/{row["image"]}.jpg', f'{split_name}/{row["image"]}.jpg')
        shutil.copy(f'HAM10000_dataset/masks/{row["image"]}_segmentation.png', f'{split_name}/{row["image"]}_mask.png')
# Save the splits
save_split(train_df, 'train')
save_split(val_df, 'val')
save_split(test_df, 'test')

#%%
print(f'Train: {len(train_df)} samples')
print(f'Validation: {len(val_df)} samples')
print(f'Test: {len(test_df)} samples')
#%%
# Resize images and masks
resize_transform = transforms.Resize((128, 128))

def resize_and_save(split_name):
    img_dir = os.path.join(split_name, 'images')
    mask_dir = os.path.join(split_name, 'masks')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for img_file in os.listdir(split_name):
        if img_file.endswith('.jpg'):
            img_path = os.path.join(split_name, img_file)
            mask_path = os.path.join(split_name, img_file.replace('.jpg', '_mask.png'))

            img = Image.open(img_path)
            mask = Image.open(mask_path)
            img_resized = resize_transform(img)
            mask_resized = resize_transform(mask)

            img_resized.save(os.path.join(img_dir, img_file))
            mask_resized.save(os.path.join(mask_dir, img_file.replace('.jpg', '_mask.png')))

resize_and_save('train')
resize_and_save('val')
resize_and_save('test')
#%%
from torch.utils.data import Dataset

# Convert images to tensors and normalize pixel values
class HAM10k(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, mask_t = None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_t = mask_t
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.mask_dir, self.img_files[idx].replace('.jpg', '_mask.png'))

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.mask_t(mask)

        return image, mask
#%%
# Data Augmentation
data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
mask_transform = transforms.ToTensor()
#%%
# Create PyTorch Dataset and DataLoader
train_dataset = HAM10k('train/images', 'train/masks', transform=data_transforms, mask_t=mask_transform)
val_dataset = HAM10k('val/images', 'val/masks', transform=data_transforms,  mask_t=mask_transform)
test_dataset = HAM10k('test/images', 'test/masks', transform=data_transforms,  mask_t=mask_transform)

#%%

train_loader_seg = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader_seg = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader_seg = DataLoader(test_dataset, batch_size=32, shuffle=False)
#%%
cudnn.benchmark = True

from torch.utils.data import Dataset

class HAM10kClassification(Dataset):
    def __init__(self, img_dir, metadata_csv, transform=None):
        self.img_dir = img_dir
        self.df = pd.read_csv(metadata_csv)
        self.transform = transform

        # Print columns to verify structure

        # Create a label map from the one-hot encoded columns
        self.label_map = {col: idx for idx, col in enumerate(['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'])}

        # Add a new column 'label_idx' to the DataFrame
        self.df['label_idx'] = self.df[['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']].idxmax(axis=1).map(self.label_map)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image'] + '.jpg')  # Assuming 'image' column contains image IDs
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row['label_idx'], dtype=torch.long)
        return image, label

train_dataset = HAM10kClassification('train', 'train/metadata.csv', transform=data_transforms)
val_dataset = HAM10kClassification('val', 'val/metadata.csv', transform=data_transforms)
test_dataset = HAM10kClassification('test', 'test/metadata.csv', transform=data_transforms)

#%%

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def get_dataloaders():
    return train_loader, val_loader, train_loader_seg, val_loader_seg