import itertools
import time

import numpy as np
import torch
import torchvision as tv
import argparse

from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL.Image import Image
from torchvision.utils import make_grid, save_image

from data import ImageDataset

from models import G, D, weight_init

parser = argparse.ArgumentParser()
parser.add_argument('--num_epoch', type=int, default=200, help='num')
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--data_root', type=str, default='ukiyoe2photo')
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--b1', type=float, default=0.5)
parser.add_argument('--b2', type=float, default=0.999)
parser.add_argument('--decay', type=int, default=100)
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--channels', type=int, default=3)
parser.add_argument('--Res_num', type=int, default=9)
parser.add_argument('--epoch', type=int, default=0)

opt = parser.parse_args()
print(opt)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_shape = (opt.channels, opt.img_size, opt.img_size)

G_A2B = G(input_shape, opt.Res_num).to(device)
G_B2A = G(input_shape, opt.Res_num).to(device)
D_A = D(input_shape).to(device)
D_B = D(input_shape).to(device)

criterion_GAN = torch.nn.MSELoss().to(device)
criterion_cycle = torch.nn.L1Loss().to(device)
criterion_identity = torch.nn.L1Loss().to(device)

if opt.epoch != 0:
    # Continue Training
    G_A2B.load_state_dict(torch.load('A2B_model_at_epoch_{}.pth'.format(opt.epoch)))
    G_B2A.load_state_dict(torch.load('B2A_model_at_epoch_{}.pth'.format(opt.epoch)))
    D_A.load_state_dict(torch.load('D_A_model_at_epoch_{}.pth'.format(opt.epoch)))
    D_B.load_state_dict(torch.load('D_B_model_at_epoch_{}.pth'.format(opt.epoch)))

else:
    G_A2B.apply(weight_init)
    G_B2A.apply(weight_init)
    D_A.apply(weight_init)
    D_B.apply(weight_init)

optimizer_G = torch.optim.Adam(
    itertools.chain(G_A2B.parameters(), G_B2A.parameters()),
    lr=opt.lr,
    betas=(opt.b1, opt.b2)
)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

trans = [
    # tv.transforms.Resize(int(opt.img_size*1.12), Image.BICUBIC),
    tv.transforms.RandomCrop((opt.img_size, opt.img_size)),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]
# dataset = ImageDataset(opt.data_root, transforms_=trans, mode='train')
# print(dataset[1])
loader = DataLoader(ImageDataset(opt.data_root, transforms_=trans, mode='train'),
                    batch_size=opt.batch_size,
                    shuffle=True,
                    num_workers=0
                    )

vali_loader = DataLoader(ImageDataset(opt.data_root, transforms_=trans, mode='test'),
                         batch_size=5,
                         shuffle=True,
                         num_workers=0)

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor


def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(vali_loader))
    G_A2B.eval()
    G_B2A.eval()
    real_A = Variable(imgs["A"].type(Tensor))
    fake_B = G_A2B(real_A)
    real_B = Variable(imgs["B"].type(Tensor))
    fake_A = G_B2A(real_B)
    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, "images/%s/%s.png" % ('_sample_', batches_done), normalize=False)


now_time = time.time()

for epoch in range(opt.epoch, opt.num_epoch):
    for i, batch in enumerate(loader):

        real_A = Variable(batch[0])
        real_B = Variable(batch[1])

        valid = Variable(Tensor(torch.ones(real_A.size(0), *D_A.output_shape)), requires_grad=False)
        fake = Variable(Tensor(torch.zeros(real_A.size(0), *D_A.output_shape)), requires_grad=False)

        G_B2A.train()
        G_A2B.train()

        optimizer_G.zero_grad()

        # Use another model to create a Soft Target
        loss_identity_A = criterion_identity(G_B2A(real_A), real_A)
        loss_identity_B = criterion_identity(G_A2B(real_B), real_B)
        loss_identity = (loss_identity_B + loss_identity_A) / 2

        # Go Forward
        fake_B = G_A2B(real_A)
        loss_A2B = criterion_GAN(D_B(fake_B), valid)
        fake_A = G_B2A(real_B)
        loss_B2A = criterion_GAN(D_A(fake_A), valid)
        loss_G = (loss_B2A + loss_A2B) / 2

        # Go Back
        recover_A = G_B2A(fake_B)
        loss_rec_A = criterion_cycle(recover_A, real_A)
        recover_B = G_A2B(fake_A)
        loss_rec_B = criterion_cycle(recover_B, real_B)
        loss_cycle = (loss_rec_A + recover_B) / 2

        # 一荣俱荣，一损俱损
        loss_total = loss_G + 10 * loss_cycle + 5 * loss_identity  # Go back is more important
        loss_total.backward()
        optimizer_G.step()

        optimizer_D_A.zero_grad()
        #  See real
        loss_real = criterion_GAN(D_A(real_A),valid)
        # See fake
        loss_fake = criterion_GAN(D_A(fake_A),fake)
        loss_D_A = (loss_fake+loss_fake)/2
        loss_D_A.backward()
        optimizer_D_A.step()

        optimizer_D_B.zero_grad()
        # See real
        loss_real = criterion_GAN(D_B(real_B),valid)
        # See fake
        loss_fake = criterion_GAN(D_B(fake_B),fake)
        loss_D_B = (loss_real+loss_fake)/2
        loss_D_B.backward()
        optimizer_D_B.step()


        print(1)
