import numpy as np 
from PIL import Image
import multiprocessing as mp
from torch.utils.data.dataset import Dataset
import torch
import glob
from torchvision import transforms

from matplotlib import pyplot as plt 

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

def crop(path,size=40, stride=10,grayscale=True):
    patches = []
    with Image.open(path) as im:
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
    return patches

def prepare(path='',batch_size=128, batch_count=1600, size=40, stride=10, num_workers=4,grayscale=True):
    if num_workers > mp.cpu_count():
        return -1
    patches = []
    workers = []
    paths = glob.glob(path+'/*.png')
    paths.extend(glob.glob(path+'/*.jpg'))
    paths= paths #[0:128]
    '''
    paths = np.array_split(paths, num_workers)
    l = mp.Manager().list()
    for i in range(num_workers):
        p = mp.Process(target=crop_mp,args=[paths[i],l,size,stride,grayscale])
        workers.append(p)
        p.start()

    for p in workers:
        p.join()
    
    #l = crop(paths, size,stride,grayscale)
    '''
    pool = mp.Pool(processes=num_workers)
    lt = pool.map(crop,paths)
    l = []
    for ll in lt:
        l.extend(ll)

    count = len(l)
    to_take = np.random.choice(count,batch_size*batch_count,replace = False)
    final = np.take(l, to_take, axis=0)
    return final
    
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
            y = t + torch.randn(t.shape)*np.random.randint(self.sigma[0],self.sigma[1])/255.0
        else:
            x = t
            y = t + torch.randn(t.shape)*self.sigma/255.0
        return x, y
    
    def __len__(self):
        return len(self.xs)

class TestDataset(Dataset):
    def __init__(self,path,noise_level) -> None:
        super(TestDataset, self).__init__()

        self.isBlind = type(noise_level)==list
        self.sigma = noise_level

        self.paths = glob.glob(path+'/*.png')
        self.paths.extend(glob.glob(path+'/*.jpg'))

    def __getitem__(self, index):
        im = np.array(Image.open(self.paths[index]), dtype=np.uint8)
        t = transforms.ToTensor()(im)
        if self.isBlind:
            x = t
            y = t + torch.randn(t.shape)*np.random.randint(self.sigma[0],self.sigma[1])/255.0
        else:
            x = t
            y = t + torch.randn(t.shape)*self.sigma/255.0
        return x, y
    
    def __len__(self):
        return len(self.paths)

if __name__=="__main__":
    mp.freeze_support()
    print(len(prepare(path='./CBSD432')))
    with Image.open('./CImageNet400/001.png') as im:
        im = np.array(im)
        y = transforms.ToTensor()(im)
        y += torch.randn(y.shape)*25/255.0
        print(y)
