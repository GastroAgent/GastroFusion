
import os, random, argparse
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from backbone import ConvolutionalNeuralNetwork
from transform import get_transform
from utils import get_accuracy
from inner_loop_optimizer import gradient_update_parameters
from torch.func import functional_call

# import debugpy
# try:
#     print('Waiting for debugger attach')
#     debugpy.listen(("localhost",9504))
#     debugpy.wait_for_client()
# except Exception:
#     pass

class EpisodeFolder(Dataset):
    def __init__(self, root, ways, shots, query_shots, episodes, transform):
        self.root=root; self.ways=ways; self.shots=shots; self.query_shots=query_shots; self.episodes=episodes; self.transform=transform
        self.classes=sorted([x for x in os.listdir(root) if os.path.isdir(os.path.join(root,x))])
        self.data={c:[os.path.join(root,c,f) for f in os.listdir(os.path.join(root,c)) if os.path.isfile(os.path.join(root,c,f))] for c in self.classes}
    def __len__(self): return self.episodes
    def load(self,p):
        x=Image.open(p).convert('RGB'); return self.transform(x)
    def sample_cls(self, imgs):
        if len(imgs) == 0:
            return None, None
        # If images are insufficient, repeat sampling within the same class.
        q = random.choices(imgs, k=self.query_shots)
        s = random.choices(imgs, k=self.shots)
        return s, q
    def __getitem__(self,idx):
        random.seed(idx)
        sx,sy,qx,qy=[],[],[],[]
        valid=[c for c in self.classes if len(self.data[c]) > 0]
        # if len(valid) == 0:
        #     raise RuntimeError('No valid classes found in test directory')
        # if len(valid) < self.ways:
        #     print(f'[Warning] available classes={len(valid)} < episode_way={self.ways}, using all available classes')
        for y,c in enumerate(valid[:self.ways]):
            s,q=self.sample_cls(self.data[c])
            sx += [self.load(p) for p in s]; sy += [y]*len(s)
            qx += [self.load(p) for p in q]; qy += [y]*len(q)
        return torch.stack(sx),torch.tensor(sy),torch.stack(qx),torch.tensor(qy)


def episode_accuracy_from_logits(logits, labels, episode_way):
    """
    Evaluate only the active episode classes.
    The model classifier dimension is unchanged.
    """
    logits = logits[:, :episode_way]
    pred = torch.argmax(logits, dim=1)
    return (pred == labels).float().mean()

def run(args):
    tf=get_transform(False)
    ds=EpisodeFolder(args.test_dir,args.episode_way,args.shots,args.query_shots,args.episodes,tf)
    dl=DataLoader(ds,batch_size=1,shuffle=False)
    model=ConvolutionalNeuralNetwork(3,8,hidden_size=args.hidden_size).to(args.device)
    ck=torch.load(args.checkpoint,map_location=args.device)
    state=ck['model']
    model.load_state_dict(state, strict=True)
    M=ck['meta_params']
    acc=[]
    for i,(sx,sy,qx,qy) in enumerate(dl):
        params=dict(model.named_parameters())
        sx,sy,qx,qy=[z.squeeze(0).to(args.device) for z in (sx,sy,qx,qy)]
        loss=F.cross_entropy(model(sx),sy)
        params=gradient_update_parameters(model,M,loss,inner_lr=args.inner_lr,first_order=args.first_order,GAP=args.GAP,approx=args.approx)
        with torch.no_grad():
            logits = functional_call(model,params,(qx,))
            a = episode_accuracy_from_logits(logits, qy, args.episode_way).item()
        acc.append(a)

    a=sum(acc)/len(acc); ci=1.96*torch.tensor(acc).std()/len(acc)**0.5
    print('Test-Test Acc:',a,'CI:',ci.item())

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--test_dir',required=True); p.add_argument('--checkpoint',required=True)
    p.add_argument('--episode_way',type=int,default=5); p.add_argument('--shots',type=int,default=5); p.add_argument('--query_shots',type=int,default=1); p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--train_way',type=int,default=5); p.add_argument('--hidden_size',type=int,default=128); p.add_argument('--inner_lr',type=float,default=.01)
    p.add_argument('--first_order',action='store_true'); p.add_argument('--GAP',action='store_true'); p.add_argument('--approx',action='store_true'); p.add_argument('--use_cuda',action='store_true')
    a=p.parse_args()
    a.device=torch.device('cuda' if a.use_cuda and torch.cuda.is_available() else 'cpu')
    run(a)
