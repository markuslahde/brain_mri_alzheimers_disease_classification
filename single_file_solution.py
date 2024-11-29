import os 
import re
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torchvision
from sklearn.model_selection import train_test_split

#*#*#*#*#*#*#*#*#*# NOTES #*#*#*#*#*#*#*#*#*#
# - currently loads from validated_data.csv at root
# - saves validation results to /visualizations
# - no folds implemented
# - no model saving implemented
# - no test set evaluation implemented


'''
image_dirs = {
    '0_No_Impairment': 'BestMRIDataset/data/0_No_Impairment',
    '1_Very_Mild_Impairment': 'BestMRIDataset/data/1_Very_Mild_Impairment',
    '2_Mild_Impairment': 'BestMRIDataset/data/2_Mild_Impairment',
    '3_Moderate_Impairment': 'BestMRIDataset/data/3_Moderate_Impairment'
}

# Function to extract the index from filenames
def extract_index(filename):
    match = re.search(r'_(\d+)\.jpg$', filename)
    return int(match.group(1)) if match else None

# Collect all image files with inferred labels based on directory names
image_files = []
for label, image_dir in image_dirs.items():
    for f in os.listdir(image_dir):
        if f.endswith('.jpg'):
            image_files.append({
                'Image': extract_index(f),
                'Path': os.path.join(image_dir, f),
                'Category': label
            })

# Create a DataFrame from the image files
images_df = pd.DataFrame(image_files)

# Display the DataFrame
print(images_df.head())
print(len(images_df))

images_df.to_csv('validated_data.csv')
'''

data_df = pd.read_csv('BrainMRIAlzheimersDiseaseClassification/validated_data.csv')

# X = data_df.drop(['Category'], axis=1)
# y = data_df['Category']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

train_df, temp_df = train_test_split(
    data_df, test_size=0.4, stratify=data_df['Category'], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df['Category'], random_state=42
)

from tqdm import tqdm

# Transformation to convert images to tensors for calculation
to_tensor = transforms.ToTensor()

# Accumulate sums for mean and std calculation
mean_sum = torch.zeros(3)
std_sum = torch.zeros(3)
num_pixels = 0

from PIL import Image

# Calculate mean and std over all images
for img_path in tqdm(data_df['Path']):
    image = Image.open(img_path).convert("RGB")
    tensor_img = to_tensor(image)  # Convert image to tensor in [C, H, W] format
    
    mean_sum += tensor_img.mean(dim=(1, 2))  # Mean per channel
    std_sum += tensor_img.std(dim=(1, 2))  # Std per channel
    num_pixels += 1

# Mean and std across the dataset
mean_for_data_df = mean_sum / num_pixels
std_for_data_df = std_sum / num_pixels

print("Calculated Mean:", mean_for_data_df)
print("Calculated Std:", std_for_data_df)
print("Category value counts: ", data_df['Category'].value_counts())

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from torchvision import transforms

class MRIDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['Path']
        image = Image.open(img_path).convert("RGB")
        label = self.data.iloc[idx]['Category']  # Ensure labels are integers
        label = self.label_to_index(label)  # Map label to integer index
        
        if self.transform:
            image = self.transform(image)
        
        return {'img': image, 'target': label}
    
    def label_to_index(self, label):
        label_map = {'0_No_Impairment': 0, '1_Very_Mild_Impairment': 1, '2_Mild_Impairment': 2, '3_Moderate_Impairment': 3}
        return label_map[label]

import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, average_precision_score

class ModelResnet18(nn.Module):
    def __init__(self, backbone='resnet18', num_classes=4):
        super(ModelResnet18, self).__init__()  
        self.num_classes = num_classes  # Store num_classes as an attribute
        self.model = models.resnet18(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
    
class ModelResnet34(nn.Module):
    def __init__(self, backbone='resnet34', num_classes=4):
        super(ModelResnet34, self).__init__()  
        self.num_classes = num_classes  # Store num_classes as an attribute
        self.model = models.resnet34(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
    
class ModelResnet50(nn.Module):
    def __init__(self, backbone='resnet50', num_classes=4):
        super(ModelResnet50, self).__init__()  
        self.num_classes = num_classes  # Store num_classes as an attribute
        self.model = models.resnet50(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


def plot_metrics(metrics, tags):
    epochs = range(1, len(metrics['train_loss']) + 1)

    # Create a figure and store it in `fig`
    fig = plt.figure(figsize=(12, 8))

    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(epochs, metrics['train_loss'], label='Train Loss')
    plt.plot(epochs, metrics['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss over Epochs')
    plt.legend()

    # Plot balanced accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, metrics['balanced_accuracy'], label='Balanced Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Balanced Accuracy')
    plt.title(f'Balanced Accuracy over Epochs')
    plt.legend()

    # Plot ROC-AUC (if available)
    if metrics['roc_auc'][0] is not None:
        plt.subplot(2, 2, 3)
        plt.plot(epochs, metrics['roc_auc'], label='ROC-AUC')
        plt.xlabel('Epochs')
        plt.ylabel('ROC-AUC')
        plt.title('ROC-AUC over Epochs')
        plt.legend()

    # Plot Average Precision
    plt.subplot(2, 2, 4)
    plt.plot(epochs, metrics['average_precision'], label='Average Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Average Precision')
    plt.title(f'Average Precision over Epochs')
    plt.legend()  

    fig.suptitle(f'{tags}', fontsize=12, weight='bold')
    fig.canvas.manager.set_window_title(f'{tags}') 

    plt.tight_layout()
    save_path = os.path.join('BrainMRIAlzheimersDiseaseClassification/visualizations', f'{tags}.png')
    fig.savefig(save_path)
    plt.close(fig)

def validate_model(model, val_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    running_loss = 0.0
    
    criterion = nn.CrossEntropyLoss()

    # Set model to evaluation mode
    model.eval()

    all_targets = []
    all_preds = []
    all_soft_preds = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['img'].to(device)
            targets = batch['target'].to(device)
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            # Softmax predictions for metrics calculation
            soft_preds = F.softmax(outputs, dim=1)
            _, predicted = torch.max(soft_preds, 1)

            # Collect targets and predictions for metrics
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_soft_preds.extend(soft_preds.cpu().numpy())

    # Calculate mean loss
    avg_val_loss = running_loss / len(val_loader)

    # Calculate metrics using scikit-learn
    balanced_acc = balanced_accuracy_score(all_targets, all_preds)
    roc_auc = roc_auc_score(all_targets, all_soft_preds, multi_class='ovr') if model.num_classes > 2 else None
    avg_precision = average_precision_score(all_targets, all_soft_preds, average='macro')

    # Dictionary to hold validation metrics
    val_metrics = {
        'balanced_accuracy': balanced_acc,
        'roc_auc': roc_auc,
        'average_precision': avg_precision
    }
    print(f"Targets: {targets.cpu().numpy()}, Predictions: {predicted.cpu().numpy()}")
    return avg_val_loss, val_metrics

def train_model(model, train_loader, val_loader, optimizer, epochs, tags):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train() 
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer

    metrics = {
        'train_loss': [], 
        'val_loss': [], 
        'balanced_accuracy': [], 
        'roc_auc': [], 
        'average_precision': []
    }

    # best_balanced_accuracy = 0.0

    # 3. Iterate over the epochs
    for epoch in range(epochs):
        running_loss = 0.0
        ema_alpha=0.1 
        ema_prev=None
        for batch in train_loader:
            # Get the inputs and targets
            inputs = batch['img'].to(device)
            targets = batch['target'].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, targets)

            # Optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        metrics['train_loss'].append(avg_train_loss)
        
        # Validate the model and save validation metrics
        val_loss, val_metrics = validate_model(model, val_loader)        
        
        metrics['val_loss'].append(val_loss)
        metrics['balanced_accuracy'].append(val_metrics['balanced_accuracy'])
        metrics['roc_auc'].append(val_metrics['roc_auc'])
        metrics['average_precision'].append(val_metrics['average_precision'])

        print(f'Epoch {epoch} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, ',
                f'Balanced Acc: {val_metrics["balanced_accuracy"]:.4f}, ',
                f'ROC-AUC: {val_metrics["roc_auc"]:.4f}' if val_metrics["roc_auc"] is not None else 'ROC-AUC: N/A', 
                f'Avg Precision: {val_metrics["average_precision"]:.4f}')
        
    plot_metrics(metrics, tags)

def trainer(model, transform, val_test_transform, batch_size, learning_rate, epochs, tags, sampled):
        # Create datasets
        train_dataset = MRIDataset(train_df, transform=transform)
        val_dataset = MRIDataset(val_df, transform=val_test_transform)
        test_dataset = MRIDataset(test_df, transform=val_test_transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_instance = model().to(device)

        optimizer = optim.Adam(model_instance.parameters(), lr=learning_rate, weight_decay=1e-3)

        # Train the model with the selected DataLoader
        train_model(model_instance, train_loader, val_loader, optimizer, epochs, tags)

        test_metrics = validate_model(model_instance, test_loader)

        # Print test metrics
        print(f"Test Results - Balanced Accuracy: {test_metrics[1]['balanced_accuracy']:.4f}, "
            f"ROC-AUC: {test_metrics[1]['roc_auc']:.4f}" if test_metrics[1]['roc_auc'] is not None else 'ROC-AUC: N/A', 
            f"Average Precision: {test_metrics[1]['average_precision']:.4f}")



def main():
    transform_5 = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_for_data_df.tolist(), std=std_for_data_df.tolist()),
    transforms.RandomVerticalFlip(),  
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.RandomRotation(degrees=10),
    ])


    val_test_transform = transforms.Compose([
            transforms.Resize((128, 128)),            # Consistent resizing
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_for_data_df.tolist(), std=std_for_data_df.tolist()),
        ])

    model = ModelResnet18

    tags = 'Resnet18_transform_5_batch=32_lr=0.0001_epochs=20_optimizer=Adam_weight_decay=1e-3_sampled=True'

    trainer(model=model, transform=transform_5, val_test_transform=val_test_transform, batch_size=32, learning_rate=0.0001, epochs=20, tags=tags, sampled=True)

main()   