from PIL import Image 
import torch.utils.data as data 
import os 
import torchvision.transforms as transforms
import torch 

class UCSDAnomalyDataset(data.Dataset):
    def __init__(self, root_dir, seq_len = 10, time_stride = 1, transform=None) -> None:
        super().__init__()
        self.root_dir = root_dir 
        
        video_ids = [ids for ids in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, ids))]
   
        self.samples = []
        
        for d in video_ids:
            print(d)
            if d[-2:] == 'gt' : continue
            for t in range(1, time_stride + 1):
                for i in range(1, 200):
                    if i + (seq_len-1)*t >200:
                        break
                    self.samples.append((os.path.join(self.root_dir, d), range(i, i+(seq_len-1)*t+1, t)))
        
        self.pil_transform = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
        self.tensor_transform = transforms.Compose([
            transforms.Normalize(mean=(0.3750352255196134,), std=(0.20129592430286292,))
        ])
        
    
    def __getitem__(self, index):
        sample = [] 
        pref = self.samples[index][0]
        for fr in self.samples[index][1]:
            with open(os.path.join(pref, '{0:03d}.tif'.format(fr)), 'rb') as fin: 
                frame = Image.open(fin).convert('RGB')
                frame = self.pil_transform(frame) / 255.0
                frame = self.tensor_transform(frame)
                sample.append(frame) 
        
        sample = torch.stack(sample, axis=0)
        return sample 
    
    def __len__(self):
        return len(self.samples)
    
        
    
        

