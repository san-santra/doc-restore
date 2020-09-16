import sys
import os
import torch
from PIL import Image
import numpy as np

# local
from model import TextExtractor
from lib import extract_patches, merge_patches


def extract_text(in_im, model, patch_size, stride, device):
    '''
    run the model with all the patches
    '''
    in_im = in_im.transpose((2, 0, 1))

    # the models expects patches so extract them and
    # that too in channel first order.
    patches = extract_patches(in_im, patch_size, stride)
    patches = torch.from_numpy(patches).to(device)

    out_patches = model(patches)

    out_im = merge_patches(out_patches.detach().cpu().numpy(), in_im.shape[1:], stride)

    return out_im


def extract_text_serial(in_im, model, patch_size, stride, device):
    '''
    restore patch by patch
    reduces the memory requirement
    '''

    im = in_im.transpose((2, 0, 1))

    patch_r, patch_c = patch_size
    stride_r, stride_c = stride
    nch, nrow, ncol = im.shape

    row_idxs = range(0, nrow-patch_r+1, stride_r)
    col_idxs = range(0, ncol-patch_c+1, stride_c)

    out_im = np.zeros((nrow, ncol, nch), dtype='float32')
    count = np.zeros((nrow, ncol), dtype='int')

    numpatch = len(row_idxs)*len(col_idxs)
    patch_idx = 0

    for i in row_idxs:
        for j in col_idxs:
            patch = im[:, i:i+patch_r, j:j+patch_c]
            patch = torch.from_numpy(patch[None, ...]).to(device)

            # I am not sure why this float() is necessary
            # check
            out_patch = model(patch.float())
            # out_patch ~> (1, nch, patch_r, patch_c)

            out_patch = out_patch.detach().cpu().numpy()

            out_im[i:i+patch_r, j:j+patch_c, :] += \
                out_patch[0].transpose((1, 2, 0))

            count[i:i+patch_r, j:j+patch_c] += 1

            sys.stdout.write('Progress: [{}/{}]\r'.format(patch_idx,
                                                          numpatch))

            patch_idx += 1

    sys.stdout.write('\n')

    nz = (count != 0)
    nz_r = np.tile(nz, (1, 1, nch))
    out_im[nz_r] = out_im[nz_r] / np.tile(count[..., None], (1, 1, nch))

    return out_im


def extract_text_mini_batch(in_im, model, patch_size, stride, device):
    '''
    '''

    pass


if __name__ == '__main__': 
    model_wt = './model/m1.pt'

    patch_size = (256, 256)
    stride = (10, 10)

    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python <code.py> input_image [output_image]")
        print("Output is saved in the same path if path"
              + " is not provided")

        sys.exit(0)

    input_im_path = sys.argv[1]
    if len(sys.argv) == 3:
        output_im_path = sys.argv[2]
    else:
        input_im_name, ext = os.path.splitext(input_im_path)
        output_im_path = input_im_name+"_out."+ext

    in_im = np.asarray(Image.open(input_im_path))/255.0
    if len(in_im.shape) == 2:
        in_im = np.expand_dims(in_im, -1)

    # inference on cpu
    device = torch.device('cpu')
    # model = TextExtractor()
    # model.load_state_dict(torch.load(model_wt, map_location=device))
    # model.eval()
    model = torch.load(model_wt)

    out_im = extract_text_serial(in_im, model, patch_size, stride,
                                 device)

    # save output
    out_im_PIL = Image.fromarray(out_im)
    out_im_PIL.save(output_im_path)
