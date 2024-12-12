import os 
import re
import torch
import random
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from torchvision import datasets, transforms
import torchvision
from sklearn.model_selection import train_test_split

from dataset import MRIDataset
from model import ModelResnet18, ModelResnet34, ModelResnet50


def plot_metrics(metrics, fold, tags):
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
    save_path = os.path.join('/home/markuslahde/SAMK/ai_theme_iii/BestMRIDataset/visualizations', f'Fold_{fold}_{tags}.png')
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

def train_model(model, train_loader, val_loader, optimizer, epochs, fold, tags):
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
        best_balanced_accuracy = 0.0
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

        # Check if this epoch has the highest balanced accuracy
        if val_metrics['balanced_accuracy'] > best_balanced_accuracy:
            best_balanced_accuracy = val_metrics['balanced_accuracy']
            best_model_path = os.path.join("BestMRIDataset/data", f"best_model_fold_{fold}.pth")
            os.makedirs("BestMRIDataset/data", exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with Balanced Accuracy: {best_balanced_accuracy:.4f} at epoch {epoch}")
       
        
        metrics['val_loss'].append(val_loss)
        metrics['balanced_accuracy'].append(val_metrics['balanced_accuracy'])
        metrics['roc_auc'].append(val_metrics['roc_auc'])
        metrics['average_precision'].append(val_metrics['average_precision'])

        print(f'Epoch {epoch} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, ',
                f'Balanced Acc: {val_metrics["balanced_accuracy"]:.4f}, ',
                f'ROC-AUC: {val_metrics["roc_auc"]:.4f}' if val_metrics["roc_auc"] is not None else 'ROC-AUC: N/A', 
                f'Avg Precision: {val_metrics["average_precision"]:.4f}')
        
    plot_metrics(metrics, fold, tags)

def trainer(model, train_dataset, val_dataset, test_dataset, batch_size, learning_rate, epochs, fold, tags):

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_instance = model().to(device)

        optimizer = optim.Adam(model_instance.parameters(), lr=learning_rate, weight_decay=1e-3)

        # Train the model with the selected DataLoader
        train_model(model_instance, train_loader, val_loader, optimizer, epochs, fold, tags)

        test_metrics = validate_model(model_instance, test_loader)

        # Print test metrics
        print(f"Test Results - Balanced Accuracy: {test_metrics[1]['balanced_accuracy']:.4f}, "
            f"ROC-AUC: {test_metrics[1]['roc_auc']:.4f}" if test_metrics[1]['roc_auc'] is not None else 'ROC-AUC: N/A', 
            f"Average Precision: {test_metrics[1]['average_precision']:.4f}")
        

def five_fold_model_test(tags, model, test_dataset, batch_size=32):
    # Path to saved models and test dataset
    model_dir = "BestMRIDataset/data/"

    # Load test dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Store metrics for all models
    all_metrics = []

    # Iterate over saved models
    for fold in range(5):  # Assuming 5-fold models
        model_path = os.path.join(model_dir, f'best_model_fold_{fold}.pth')
        print(f"Loading model: {model_path}")
        
        # Load the model
        model_instance = model()  # Replace with your model initialization
        state_dict = torch.load(model_path, map_location=device)  # Load weights
        model_instance.load_state_dict(state_dict)  # Load weights into the model
        model_instance.to(device)  # Move the model to the appropriate device
        
        # Validate model
        _, metrics = validate_model(model_instance, test_loader)
        print(f"Metrics for fold {fold}: {metrics}")
        all_metrics.append(metrics)

    # Create a figure and store it in `fig`
    fig = plt.figure(figsize=(12, 8))

    # Calculate average metrics
    average_metrics = {
        'balanced_accuracy': sum(m['balanced_accuracy'] for m in all_metrics) / len(all_metrics),
        'roc_auc': sum(m['roc_auc'] for m in all_metrics if m['roc_auc'] is not None) / len(all_metrics),
        'average_precision': sum(m['average_precision'] for m in all_metrics) / len(all_metrics)
    }

    print("Average metrics across all models:")
    print(average_metrics)

    # Extract metric names and values
    metric_names = list(average_metrics.keys())
    metric_values = list(average_metrics.values())

    # Create the plot
    plt.figure(figsize=(8, 6))
    bar_container = plt.bar(metric_names, metric_values, color='skyblue')  # Use a bar plot for better visualization of metrics

    # Add values above bars for clarity
    for bar, value in zip(bar_container, metric_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}', 
                ha='center', va='bottom', fontsize=10, color='black')

    plt.title('Average metrics across all models', fontsize=14)
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Metrics', fontsize=12)
    plt.ylim(0, 1)  # Assuming metrics are normalized scores
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add a legend
    plt.legend(['Metric Values'], loc='upper left')

    # Add a suptitle and save the figure
    plt.suptitle(f'Average_model_performance', fontsize=12, weight='bold')
    plt.tight_layout()
    save_path = os.path.join('BestMRIDataset/visualizations', f'Average_model_performance_{tags}.png')
    plt.savefig(save_path)
    plt.close()


def main():
    for fold in range(5):

        train_df = pd.read_csv(os.path.join("BestMRIDataset/data", f'fold_{fold}_train.csv'))
        val_df = pd.read_csv(os.path.join("BestMRIDataset/data", f'fold_{fold}_val.csv'))
        # test_df = pd.read_csv(os.path.join("BestMRIDataset/data", f'test_data.csv'))

        from tqdm import tqdm

        # Transformation to convert images to tensors for calculation
        to_tensor = transforms.ToTensor()

        # Accumulate sums for mean and std calculation
        mean_sum = torch.zeros(3)
        std_sum = torch.zeros(3)
        num_pixels = 0

        from PIL import Image

        # Calculate mean and std over all images
        for img_path in tqdm(train_df['Path']):
            image = Image.open(img_path).convert("RGB")
            tensor_img = to_tensor(image)  # Convert image to tensor in [C, H, W] format
            
            mean_sum += tensor_img.mean(dim=(1, 2))  # Mean per channel
            std_sum += tensor_img.std(dim=(1, 2))  # Std per channel
            num_pixels += 1

        # Mean and std across the dataset
        mean_for_train_df = mean_sum / num_pixels
        std_for_train_df = std_sum / num_pixels

        print("Calculated Mean:", mean_for_train_df)
        print("Calculated Std:", std_for_train_df)
        print("Category value counts: ", train_df['Category'].value_counts())

        transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_for_train_df.tolist(), std=std_for_train_df.tolist()),
        transforms.RandomVerticalFlip(),  
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomRotation(degrees=10),
        ])

        val_test_transform = transforms.Compose([
                transforms.Resize((128, 128)),            # Consistent resizing
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_for_train_df.tolist(), std=std_for_train_df.tolist()),
            ])

        # Create datasets
        train_dataset = MRIDataset(train_df, transform=transform)
        val_dataset = MRIDataset(val_df, transform=val_test_transform)

        test_df = pd.read_csv(os.path.join("BestMRIDataset/data", f'test_data.csv'))
        test_dataset = MRIDataset(test_df, transform=val_test_transform)

        model = ModelResnet18

        tags = 'Resnet18_transform_5_batch=32_lr=0.0001_epochs=20_optimizer=Adam_weight_decay=1e-3_sampled=True'

        trainer(model=model, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset, batch_size=32, learning_rate=0.0001, epochs=20, fold=fold, tags=tags)
    five_fold_model_test(tags, model, test_dataset=test_dataset)


main()   
