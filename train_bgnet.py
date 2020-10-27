# Trains the bg extraction network

import sys
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim

# ssim
from pytorch_ssim import SSIM

# local
from model import BGRestoreNet
from data_gen import PatchifyDB

if __name__ == '__main__':
    in_path = '../data/ourdata/X/s'
    target_path = '../data/ourdata/Y/s'
    model_path = './model/bgm1-{}-{}.pt'
    optimizer_state_path = './model/bgm1-opt-{}.pt'
    patch_size = 256
    patch_per_image = 10
    model_save_freq = 5

    # checkpointing
    checkpoint_load = False
    model_state = ''
    optimizer_state = ''

    device = torch.device("cuda:1" if torch.cuda.is_available()
                          else "cpu")

    # param
    batch_size = 8
    num_epoch = 100
    ssim_window_size = 23
    num_workers = 4

    # get data
    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.ToTensor()])
    db = PatchifyDB(in_path, target_path, patch_size,
                    patch_per_image, transform=transform)
    data_loader = DataLoader(db, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers)

    # get the model
    model = BGRestoreNet().to(device)
    ssim = SSIM(window_size=ssim_window_size, return_map=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001, eps=1e-07)

    if checkpoint_load:
        model.load_state_dict(torch.load(model_state))
        optimizer.load_state_dict(torch.load(optimizer_state))
        model.train()
        print('Restoring checkpoint: {} - {}'.format(
            model_state, optimizer_state))

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

            sys.stdout.write('[%d/%d] - loss: %.5f\r' %
                             (epoch + 1, num_epoch,
                              loss.item()))
            sys.stdout.flush()

        sys.stdout.write('\n')

        # save model
        if epoch % model_save_freq == 0:
            torch.save(model.state_dict(),
                       model_path.format(epoch, loss))
            torch.save(optimizer.state_dict(),
                       optimizer_state_path.format(epoch))
