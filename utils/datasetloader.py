import torch.utils.data as data
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision.transforms import Resize, Compose, ToTensor
import torch, time, os
import random
import cv2
import open3d as o3d

class DatasetLoader(data.Dataset):
    
    def __init__(self, root='./data/', seed=None, train=True, classes=None, batch_size=128):
        np.random.seed(seed)
        self.root = Path(root)

        if train:
            self.depth_input_paths = [root+'ModelNet10_gim_64/train_2cat/'+d for d in os.listdir(root+'ModelNet10_gim_64/train_2cat/')]
            # Randomly choose 50k images without replacement
            # self.rgb_paths = np.random.choice(self.rgb_paths, 4000, False)
        else:
            self.depth_input_paths = [root+'ModelNet10_gim_64/test_2cat/'+d for d in os.listdir(root+'ModelNet10_gim_64/test_2cat/')]
            # self.rgb_paths = np.random.choice(self.rgb_paths, 1000, False)
        
        self.length = len(self.depth_input_paths)
            
    def __getitem__(self, index):
        pathgim = self.depth_input_paths[index]        
        # pathpcd=pathgim.replace('ModelNet10_gim', 'ModelNet10_pcd').replace('png','pcd')
        # pcd = o3d.io.read_point_cloud(pathpcd)
        # pcd = torch.from_numpy(np.moveaxis(np.array(pcd.points).astype(np.float32),-1,0))
        # print(pathgim)
        gimgt=cv2.imread(pathgim,cv2.IMREAD_UNCHANGED).astype(np.float32)
        # depth_input_mod = np.moveaxis(depth_input,-1,0)
        # gimgt= Compose([Resize((100,100)), ToTensor()])(gimgt)
        gimgt=torch.from_numpy(np.moveaxis(gimgt,-1,0))
        # gimgt=gimgt-gimgt.min()
        # gimgt=gimgt/gimgt.max()
        gimgt=gimgt/255
        gimgt_rgb=gimgt*0
        gimgt_rgb[0]=gimgt[1]
        gimgt_rgb[1]=gimgt[0]
        gimgt_rgb[2]=gimgt[2]
        # pcd = pcd - pcd.min()
        # pcd_normalized = pcd/torch.abs(pcd).max()
        return gimgt_rgb

    def __len__(self):
        return self.length

if __name__ == '__main__':
    # Testing
    dataset = DatasetLoader()
    print(len(dataset))
    for item in dataset[0]:
        print(item.size())
