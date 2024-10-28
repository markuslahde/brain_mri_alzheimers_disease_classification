import cv2 # pip install opencv-python
from torch.utils.data import Dataset

def read_xray(path_file):
    xray= cv2.imread(path_file, cv2.IMREAD_GRAYSCALE)
    return xray

class brain_MRI_Dataset(Dataset):
    def  __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img = read_xray(self.dataset['Path'].iloc[idx])
        # for MRI read 3D data with for loop (?)
        target = read_xray(self.dataset['KL'].iloc[idx])
        name = self.dataset['Name'].iloc[idx]

        res = {
            'Name': name,
            'img': img,
            'target': target
        }

        return res


