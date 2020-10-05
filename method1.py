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

    out_im = merge_patches(out_patches.detach().cpu().numpy(), in_im.shape[1:],
                           stride)

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
    nz_r = np.tile(nz[..., None], (1, 1, nch))
    count_r = np.tile(count[..., None], (1, 1, nch))

    out_im[nz_r] = out_im[nz_r] / count_r[nz_r]

    return out_im


def extract_text_mini_batch(in_im, model, patch_size, stride,
                            batch_size, device):
    '''
    A balance between the above two
    '''

    im = in_im.transpose((2, 0, 1))

    patch_r, patch_c = patch_size
    stride_r, stride_c = stride
    nch, nrow, ncol = im.shape

    row_idxs = range(0, nrow-patch_r+1, stride_r)
    col_idxs = range(0, ncol-patch_c+1, stride_c)

    col_v, row_v = np.meshgrid(col_idxs, row_idxs)
    col_v = col_v.flatten()
    row_v = row_v.flatten()

    out_im = np.zeros((nrow, ncol, nch), dtype='float32')
    count = np.zeros((nrow, ncol), dtype='int')
    patch_batch = np.zeros((batch_size, nch) + patch_size, dtype='float32')

    numpatch = len(row_idxs)*len(col_idxs)

    num_batch = numpatch // batch_size
    # handle the non full sized batch separately

    for b in range(num_batch):
        b_i = b*batch_size
        b_j = b*batch_size
        for b_idx in range(batch_size):
            i = row_v[b_i + b_idx]
            j = col_v[b_j + b_idx]
            patch_batch[b_idx] = im[:, i:i+patch_r, j:j+patch_c]

        patch_batch_t = torch.from_numpy(patch_batch).to(device)
        with torch.no_grad():
            out_patch = model(patch_batch_t)
            # out_patch ~> (num_batch, nch, patch_r, patch_c)

        out_patch_b = out_patch.detach().cpu().numpy()
        for b_idx in range(batch_size):
            i = row_v[b_i + b_idx]
            j = col_v[b_j + b_idx]
            out_im[i:i+patch_r, j:j+patch_c, :] += \
                out_patch_b[b_idx].transpose((1, 2, 0))

            count[i:i+patch_r, j:j+patch_c] += 1

        sys.stdout.write('Progress: [{}/{}]\r'.format(b, num_batch))

    sys.stdout.write('\n')

    # handle the non full batch
    if numpatch % batch_size != 0:
        remaining_patches = numpatch % batch_size
        patch_batch = np.zeros((remaining_patches, 1) + patch_size,
                               dtype='float32')

        b_i = num_batch*batch_size
        b_j = num_batch*batch_size
        for b_idx in range(remaining_patches):
            i = row_v[b_i + b_idx]
            j = col_v[b_j + b_idx]
            patch_batch[b_idx] = im[:, i:i+patch_r, j:j+patch_c]

        patch_batch_t = torch.from_numpy(patch_batch).to(device)
        with torch.no_grad():
            out_patch = model(patch_batch_t)
            # out_patch ~> (num_batch, nch, patch_r, patch_c)

        out_patch_b = out_patch.detach().cpu().numpy()
        for b_idx in range(remaining_patches):
            i = row_v[b_i + b_idx]
            j = col_v[b_j + b_idx]
            out_im[i:i+patch_r, j:j+patch_c, :] += \
                out_patch_b[b_idx].transpose((1, 2, 0))

            count[i:i+patch_r, j:j+patch_c] += 1

    nz = (count != 0)
    nz_r = np.tile(nz[..., None], (1, 1, nch))
    count_r = np.tile(count[..., None], (1, 1, nch))

    out_im[nz_r] = out_im[nz_r] / count_r[nz_r]

    return out_im


if __name__ == '__main__':
    model_wt = './model/upto2017_model_ourdata.pt'

    patch_size = (256, 256)
    stride = (10, 10)
    inference_batch_size = 40

    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python <code.py> input_image [output_location]")
        print("Output is saved in the currnet path if path"
              + " is not provided")

        sys.exit(0)

    input_im_path = sys.argv[1]
    input_file_name, ext = os.path.splitext(os.path.basename(input_im_path))
    if len(sys.argv) == 3:
        output_im_loc = sys.argv[2]
    else:
        output_im_loc = '.'

    output_im_path = os.path.join(output_im_loc, input_file_name+"_out"+ext)

    in_im = np.asarray(Image.open(input_im_path))/255.0
    if len(in_im.shape) == 2:
        in_im = np.expand_dims(in_im, -1)

    # inference on cpu
    device = torch.device('cpu')
    model = TextExtractor()
    model.load_state_dict(torch.load(model_wt, map_location=device))
    model.eval()

    # main method
    # extract text
    in_im = np.expand_dims(in_im[:, :, 0], -1)
    # out_im = extract_text_serial(in_im, model, patch_size, stride,
    #                              device)
    out_im = extract_text_mini_batch(in_im, model, patch_size, stride,
                                     inference_batch_size, device)

    # save output
    out_im_PIL = Image.fromarray((np.squeeze(out_im)*255).astype(np.uint8))
    out_im_PIL.save(output_im_path)
