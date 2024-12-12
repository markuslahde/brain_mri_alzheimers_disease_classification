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


data_df = pd.read_csv('BestMRIDataset/validated_data.csv')

X = data_df.drop(['Category'], axis=1)
y = data_df['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

# SAVE TEST SET TO CSV
test_df = X_test.copy()
test_df['Category'] = y_test.values
test_df.to_csv('BestMRIDataset/data/test_data.csv', index=False)



#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#

def get_five_folds(instances):
    """
    Parameters:
        instances: A list of dictionaries where each dictionary is an instance. 
            Each dictionary contains attribute:value pairs 
    Returns: 
        fold0, fold1, fold2, fold3, fold4
        Five folds whose class frequency distributions are 
        each representative of the entire original data set (i.e. Five-Fold 
        Stratified Cross Validation)
    """
    # Create five empty folds
    fold0 = []
    fold1 = []
    fold2 = []
    fold3 = []
    fold4 = []
 
    # Shuffle the data randomly
    random.shuffle(instances)
 
    # Generate a list of the unique class values and their counts
    classes = []  # Create an empty list named 'classes'
 
    # For each instance in the list of instances, append the value of the class
    # to the end of the classes list
    for instance in instances:
        classes.append(instance['Category'])
 
    # Create a list of the unique classes
    unique_classes = list(Counter(classes).keys())
 
    # For each unique class in the unique class list
    for uniqueclass in unique_classes:
 
        # Initialize the counter to 0
        counter = 0
         
        # Go through each instance of the data set and find instances that
        # are part of this unique class. Distribute them among one
        # of five folds
        for instance in instances:
 
            # If we have a match
            if uniqueclass == instance['Category']:
 
                # Allocate instance to fold0
                if counter == 0:
 
                    # Append this instance to the fold
                    fold0.append(instance)
 
                    # Increase the counter by 1
                    counter += 1
 
                # Allocate instance to fold1
                elif counter == 1:
 
                    # Append this instance to the fold
                    fold1.append(instance)
 
                    # Increase the counter by 1
                    counter += 1
 
                # Allocate instance to fold2
                elif counter == 2:
 
                    # Append this instance to the fold
                    fold2.append(instance)
 
                    # Increase the counter by 1
                    counter += 1
 
                # Allocate instance to fold3
                elif counter == 3:
 
                    # Append this instance to the fold
                    fold3.append(instance)
 
                    # Increase the counter by 1
                    counter += 1
 
                # Allocate instance to fold4
                else:
 
                    # Append this instance to the fold
                    fold4.append(instance)
 
                    # Reset the counter to 0
                    counter = 0
 
    # Shuffle the folds
    random.shuffle(fold0)
    random.shuffle(fold1)
    random.shuffle(fold2)
    random.shuffle(fold3)
    random.shuffle(fold4)
 
    # Return the folds
    return  fold0, fold1, fold2, fold3, fold4

# COMBINE X, y TRAIN TO MAKE FOLDS WITH get_five_folds
instances = []
for index, row in X_train.iterrows():
    instance = row.to_dict()  # Convert row to dictionary
    instance['Category'] = y_train[index]  # Add class label
    instances.append(instance)

# GET FOLDS AS LISTS
fold0, fold1, fold2, fold3, fold4 = get_five_folds(instances)

# Create a directory to save the folds if it doesn't exist
output_dir = "BestMRIDataset/data"
os.makedirs(output_dir, exist_ok=True)

# Define function to save folds with train and validation split
def save_train_val_fold(fold, fold_number, val_size=0.2):
    # Split fold into train and validation sets
    fold_df = pd.DataFrame(fold)
    train_df, val_df = train_test_split(fold_df, test_size=val_size, random_state=1, stratify=fold_df['Category'])
    
    # Save training and validation splits
    train_df.to_csv(os.path.join(output_dir, f'fold_{fold_number}_train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, f'fold_{fold_number}_val.csv'), index=False)


# Save each fold
save_train_val_fold(fold0, 0)
save_train_val_fold(fold1, 1)
save_train_val_fold(fold2, 2)
save_train_val_fold(fold3, 3)
save_train_val_fold(fold4, 4)

#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#

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