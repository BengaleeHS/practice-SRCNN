# Implemention - 구현

## Model Formulation

### Architecture

```python
class DnCNN(nn.Module):
    def __init__(self,channel=1,depth=17):
        super(DnCNN,self).__init__()
        L=[]
        L.append(nn.Conv2d(channel,64,3,padding=1,bias=False))
        L.append(nn.ReLU(inplace=True))
        for i in range(depth-2):
            L.append(nn.Conv2d(64,64,3,padding=1,bias=False))
            L.append(nn.BatchNorm2d(64,eps=0.0001,momentum=0.95))
            L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(64,channel,3,padding=1,bias=False))
        self.seq = nn.Sequential(*L)

    def forward(self,x):
        return (self.seq(x))
```

d-2개의 Conv-BN-ReLU unit과 시작, 끝부분에 Conv를 두는 구조이다. forward는 noise를 deterministic하게 예측한다.

### Parameter Initialization

kaiming normal initialize한다. batch normalization의 초기화는 일반적인 normal dist로 초기화한다.

## Data Preparation

### crop

```python
def crop_mp(paths,l,size, stride,grayscale):
    patches = []
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            left=0
            while left+size<im.width:
                upper = 0
                while upper+size<im.height:
                    cim = im.crop(box=(left,upper,left+size,upper+size))
                    cim = np.array(cim)
                    rot = np.random.randint(4)
                    flip = np.random.randint(2)
                    for _ in range(rot):
                        cim = np.rot90(cim)
                    if flip==1:
                        cim = np.fliplr(cim)
                    if(grayscale):
                        cim = np.array(Image.fromarray(cim).convert("L"))
                    patches.append(cim)
                    upper+=stride
                left+=stride
    l.extend(patches)

def prepare(path='',batch_size=128, batch_count=1600, size=40, stride=10, num_workers=4,grayscale=True):
    if num_workers > mp.cpu_count():
        return -1
    patches = []
    workers = []
    paths = glob.glob(path+'/*.png')
    paths.extend(glob.glob(path+'/*.jpg'))

    pool = mp.Pool(processes=num_workers)
    lt = pool.map(crop,paths)
    l = []
    for ll in lt:
        l.extend(ll)
    count = len(l)
    to_take = np.random.choice(count,batch_size*batch_count,replace = False)
    final = np.take(l, to_take, axis=0)
    return final
```

crop하되, 매우 많은 양이므로 multiprocessing한다. crop 후, rotation, flip을 랜덤하게 한 후 섞은 후 데이터 개수에 맞춰 선출한다.

### Dataset

```python
class TrainDataset(Dataset):
    def __init__(self,xs,noise_level) -> None:
        super(TrainDataset, self).__init__()

        self.isBlind = type(noise_level)==list
        self.sigma = noise_level

        self.xs = xs

    def __getitem__(self, index):
        t = transforms.ToTensor()(self.xs[index])
        if self.isBlind:
            x = t
            y = (t + torch.randn(t.shape)*np.random.randint(self.sigma[0],self.sigma[1])/255.0).clamp(0.,1.)
        else:
            x = t
            y = (t + torch.randn(t.shape)*self.sigma/255.0).clamp(0.,1.)
        return x, y
    
    def __len__(self):
        return len(self.xs)
```

noise를 추가한 뒤 clamp한다. x는 깨끗한 이미지, y는 noisy 이미지이다.

## Result





