
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import scipy.io as sio
from json import load
from torchvision.io import read_image
from torchvision.transforms import v2

from model.model import transforms


class MyDataset(Dataset):
    PREFIX = "pix3d/"
    
    def __init__(self) -> None:
        super().__init__()

        self.data = load(open('pix3d/pix3d.json'))
        self.resize_mask = v2.Resize((224, 224), interpolation=v2.InterpolationMode.NEAREST)
        self.resize_img = v2.Resize((224, 224))

    def __getitem__(self, index):
        try:
            image = transforms(read_image(self.PREFIX + self.data[index]['img']))
            mask = self.resize_mask(read_image(self.PREFIX + self.data[index]['mask'])).type(torch.FloatTensor)
            data = torch.FloatTensor(sio.loadmat(self.PREFIX + self.data[index]['voxel'])['voxel'])
        
            return image, mask, data
        
        except:
            return torch.zeros((1,)), torch.zeros((1,)), torch.zeros((1,))
    
    def __len__(self):
        return len(self.data)
    
ds = MyDataset()
train_set, test_set, val_set = random_split(ds, (8069, 1000, 1000))

train_loader = DataLoader(train_set, shuffle=True)
test_loader = DataLoader(test_set)
val_loader = DataLoader(val_set)
        
if __name__ == "__main__":
    ds = MyDataset()
    print(ds[123][1].dtype)
