
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
        categories = set()
        if obj_class is not None:
            for i in range(len(self.data)):
                categories.add(self.data[i]['category'])
                if self.data[i]['category'] != obj_class:
                    dropout.append(i)
        
        print(categories)
        
        for i in reversed(dropout):
            self.data.pop(i)
        

    def __getitem__(self, index):
        try:
            image = read_image(self.PREFIX + self.data[index]['img'])
            mask = self.resize_mask(read_image(self.PREFIX + self.data[index]['mask'])).type(torch.FloatTensor)
            data = torch.FloatTensor(sio.loadmat(self.PREFIX + self.data[index]['voxel'])['voxel'])
            print(self.data[index]['category'])
        
            return image, mask, data
        
        except:
            return torch.zeros((1,)), torch.zeros((1,)), torch.zeros((1,))
    
    def __len__(self):
        return len(self.data)
    
ds = MyDataset('misc')
train_set, test_set, val_set = random_split(ds, (0.8, 0.1, 0.1))

train_loader = DataLoader(train_set, shuffle=True)
test_loader = DataLoader(test_set)
val_loader = DataLoader(val_set)

if __name__ == "__main__":
    import cv2

    print(len(train_loader))
    
    for i, j, k in train_loader:
        try:
            i = i.squeeze().permute(1, 2, 0).numpy()
            if i.shape[0] > 500:
                sh = ((500*i.shape[1])//i.shape[0], 500)
                i = cv2.resize(i, sh)
        except:
            continue
        i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        cv2.imshow("img", i)
        cv2.waitKey(0)
