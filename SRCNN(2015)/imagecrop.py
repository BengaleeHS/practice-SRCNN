from PIL import Image,ImageFilter
from glob import glob

stride = 14
sub = 33
i = 0
for path in glob('./T91/*.png'):
    with Image.open(path) as im:
        left=0
        while left+sub<im.width:
            upper = 0
            while upper+sub<im.height:
                cim = im.crop(box=(left,upper,left+sub,upper+sub))
                cim.save(f'./crop/{i}.png')
                upper+=stride
                i+=1
            left+=stride
print(i)