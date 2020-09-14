# data generator

import os
# from skimage.io import imread
# from skimage import img_as_float
from PIL import Image
import numpy as np

from torch.utils.data import Dataset


class PatchifyDB(Dataset):
    '''
    A sample of the dataset is a random patch from the input data
    '''
    def __init__(self, input_im_path, target_im_path, patch_size,
                 transform=None):
        
        self.in_path = input_im_path
        self.target_path = target_im_path
        self.transform = transform

        assert isinstance(patch_size, (int, tuple))
        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)
        else:
            self.patch_size = patch_size

        self.in_files = sorted(os.listdir(self.in_path))
        self.target_files = sorted(os.listdir(self.target_path))
        self.numfile = len(self.in_files)

        assert (self.numfile == len(self.target_files))

    def __len__(self):
        'Total number of samples'
        return self.numfile

    def __getitem__(self, idx):
        '''
        Take a random patch and return it
        '''
        
        # im = img_as_float(imread(os.path.join(self.in_path,
        #                                       self.in_files[idx])))
        # target = img_as_float(
        #     imread(os.path.join(self.target_path,
        #                         self.target_files[idx])))

        im = Image.open(os.path.join(self.in_path, self.in_files[idx]))
        target = Image.open(os.path.join(self.target_path,
                                         self.target_files[idx]))

        # random crop
        # h, w = im.shape[:2]
        w, h = im.size
        new_h, new_w = self.patch_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        
        # x = im[top: top + new_h, left: left + new_w]
        # y = target[top: top + new_h, left: left + new_w]
        x = im.crop((left, top, left+new_w, top+new_h))
        y = target.crop((left, top, left+new_w, top+new_h))

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        # sample = {'in': x, 'target': y}
        # return sample
        return x, y


if __name__ == '__main__':
    'test code'
    import matplotlib.pyplot as plt
    from torchvision import transforms
    
    in_path = '../data/ourdata/X/s'
    target_path = '../data/ourdata/Y/s'
    patch_size = 256
    transform = transforms.Compose([transforms.Grayscale(),
                                    transforms.ToTensor()])
    db = PatchifyDB(in_path, target_path, patch_size, transform)

    sample = db[0]

    plt.subplot(1, 2, 1)
    plt.imshow(sample['in'])
    plt.subplot(1, 2, 2)
    plt.imshow(sample['target'])

    plt.ion()
    plt.show()

