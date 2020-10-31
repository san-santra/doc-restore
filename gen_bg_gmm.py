import os
import sys
import numpy as np
from PIL import Image

from method1 import restore_background

if __name__ == '__main__':
    in_path = '../data/Ourdata_allRawImage/ISI-letters_dataset/Data/'
    out_path = '../data/Ourdata_allRawImage/ISI-letters_dataset/bg'

    in_files = sorted(os.listdir(in_path))
    for i in range(len(in_files)):
        sys.stdout.write('Progress: [{}/{}]\r'.format(i, len(in_files)))

        in_file_name, ext = os.path.splitext(in_files[i])
        out_file = in_file_name+'_bg'+ext

        im_PIL = Image.open(os.path.join(in_path, in_files[i]))
        if im_PIL.mode != 'RGB':
            im_PIL = im_PIL.convert('RGB')

        im_uint8 = np.asarray(im_PIL)
        im = im_uint8/255.0

        bg = restore_background(im, im_PIL)

        bg_PIL = Image.fromarray(np.squeeze(bg).astype('uint8'))
        bg_PIL.save(os.path.join(out_path, out_file))

    print('')
