
import os, random
from PIL import Image
from torch.utils.data import Dataset
import torch
class FolderEpisodeDataset(Dataset):
    def __init__(self, root, ways=5, shots=1, query_shots=15, episodes=100000, transform=None, split='train', seed=42):
        self.root=root; self.ways=ways; self.shots=shots; self.query_shots=query_shots; self.episodes=episodes; self.transform=transform
        classes=sorted([x for x in os.listdir(root) if os.path.isdir(os.path.join(root,x))])
        r=random.Random(seed); r.shuffle(classes); n=len(classes); a=int(.64*n); b=int(.8*n)
        self.classes={'train':classes[:a],'val':classes[a:b],'test':classes[b:]}[split]
        self.data={c:[os.path.join(root,c,x) for x in os.listdir(os.path.join(root,c))] for c in self.classes}
    def __len__(self): return self.episodes
    def __getitem__(self,i):
        rng=random.Random(i)
        cls=rng.sample(self.classes,self.ways); sx=[]; sy=[]; qx=[]; qy=[]
        for y,c in enumerate(cls):
            imgs=rng.sample(self.data[c],self.shots+self.query_shots)
            for p in imgs[:self.shots]: sx.append(self.load(p)); sy.append(y)
            for p in imgs[self.shots:]: qx.append(self.load(p)); qy.append(y)
        return {'train':(torch.stack(sx),torch.tensor(sy)), 'test':(torch.stack(qx),torch.tensor(qy))}
    def load(self,p):
        x=Image.open(p).convert('RGB'); return self.transform(x) if self.transform else x
