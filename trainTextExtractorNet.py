# Trains the text extraction network

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim

# fancy output
# from tqdm import tqdm

# ssim
from pytorch_ssim import SSIM

# local
from model import TextExtractor
from data_gen import PatchifyDB

if __name__ == '__main__':
    in_path = '../data/ourdata/X/s'
    target_path = '../data/ourdata/Y/s'
    model_path = './model/m1-{}-{}.pt'
    patch_size = 256
    patch_per_image = 10000

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # param
    batch_size = 8
    num_epoch = 100
    ssim_window_size = 23
    num_workers = 4

    # get data
    transform = transforms.Compose([transforms.Grayscale(),
                                    transforms.ToTensor()])
    db = PatchifyDB(in_path, target_path, patch_size,
                    patch_per_image, transform=transform)
    data_loader = DataLoader(db, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers)

    # get the model
    model = TextExtractor().to(device)
    ssim = SSIM(window_size=ssim_window_size, size_average=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001, eps=1e-07)
    
    # Train
    # ADD: make a tqdm based progress bar
    for epoch in range(num_epoch):
        for in_batch, target_batch in data_loader:
            # to GPU
            in_batch = in_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()

            outputs = model(in_batch)
            loss = 0.5*(1 - ssim(outputs, target_batch))
            loss.backward()
            optimizer.step()

    print('[%d/%d] - loss: %.5f' %
          (epoch + 1, num_epoch, loss.item()))

    # save model
    torch.save(model.state_dict(),
               model_path.format(epoch, loss))
