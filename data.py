import glob
import os
from torchvision import transforms
from torch.utils.data import Dataset
import numpy
from PIL import Image
# img = Image.open('ukiyoe2photo/trainB/2013-11-17 09_05_09.jpg')
#
# img = numpy.array(img)
# print(img.shape)
# # print('/*.*')
# root = 'ukiyoe2photo'
# mode = 'train'
# print(sorted(glob.glob(os.path.join(root,'{}A'.format(mode) + "/*.*"))))
def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

class ImageDataset(Dataset):
    def __init__(self, root, transforms_ = None, mode = 'train'):
        self.trans = transforms.Compose(transforms_)
        self.A = sorted(glob.glob(os.path.join(root,'{}A'.format(mode) + "/*.*"))) # A name list
        self.B = sorted(glob.glob(os.path.join(root,'{}B'.format(mode) + "/*.*"))) # B name list

    def __getitem__(self, idx):
        img_A = Image.open(self.A[idx%len(self.A)])
        img_B = Image.open(self.B[idx%len(self.B)]) # accept big small dataset
        if img_A.mode != "RGB":
            img_A = to_rgb(img_A)
        if img_B.mode != "RGB":
            img_B = to_rgb(img_B)
        img_A = self.trans(img_A)
        img_B = self.trans(img_B)
        return img_A, img_B

    def __len__(self):
        return max(len(self.A), len(self.B)) # fully use all data