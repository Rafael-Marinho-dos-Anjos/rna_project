
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import scipy.io as sio
from json import load
from torchvision.io import read_image
from torchvision.transforms import v2

from model.model import transforms


class MyDataset(Dataset):
    PREFIX = "pix3d/"
    
    def __init__(self, obj_class: str = None) -> None:
        super().__init__()

        self.data = load(open('pix3d/pix3d.json'))
        self.resize_mask = v2.Resize((224, 224), interpolation=v2.InterpolationMode.NEAREST)
        self.resize_img = v2.Resize((224, 224))

        dropout = []
        if obj_class in ['sofa', 'chair', 'desk', 'bed', 'bookcase', 'tool', 'misc', 'wardrobe', 'table']:
            for i in range(len(self.data)):
                if self.data[i]['category'] != obj_class:
                    dropout.append(i)
        
        for i in reversed(dropout):
            self.data.pop(i)
        

    def __getitem__(self, index):
        try:
            image = transforms(read_image(self.PREFIX + self.data[index]['img']))
            mask = self.resize_mask(read_image(self.PREFIX + self.data[index]['mask'])).type(torch.FloatTensor)
            data = torch.FloatTensor(sio.loadmat(self.PREFIX + self.data[index]['voxel'])['voxel'])
            print(self.data[index]['category'])
        
            return image, mask, data
        
        except:
            return torch.zeros((1,)), torch.zeros((1,)), torch.zeros((1,))
    
    def __len__(self):
        return len(self.data)


with open("data/category.txt", "r") as category:
    category = category.readline(-1)

ds = MyDataset(category)
train_set, test_set, val_set = random_split(ds, (0.8, 0.1, 0.1))

train_loader = DataLoader(train_set, shuffle=True)
test_loader = DataLoader(test_set)
val_loader = DataLoader(val_set)

if __name__ == "__main__":
    pass
