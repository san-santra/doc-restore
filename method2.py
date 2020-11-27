# method 2
import sys
import os
from PIL import Image
import numpy as np
import torch

from model import BGRestoreNet


def merge_patches(accum, count, patches_data, idxs):
    '''
    patches_data ~> (num_batch, nch, patch_r, patch_c)
    '''

    batch_size, nch, patch_r, patch_c = patches_data.shape

    for b_idx in range(batch_size):
        i = idxs[b_idx, 0]
        j = idxs[b_idx, 1]
        accum[i:i+patch_r, j:j+patch_c, :] += \
            patches_data[b_idx].transpose((1, 2, 0))
        # back to channel last

        count[i:i+patch_r, j:j+patch_c] += 1


def patch_process_mini_batch(model, in_im, patch_size, stride,
                             batch_size, device, debug=True):
    '''
    Apply model to in_im in mini batches
    in_im -> numpy array
    '''

    # pytorch requires data in channel first order
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
    patch_idxs = np.zeros((batch_size, 2), dtype='int')

    numpatch = len(row_idxs)*len(col_idxs)
    num_batch = numpatch // batch_size
    # handle the non full sized batch separately

    for b in range(num_batch):
        b_i = b*batch_size
        b_j = b*batch_size
        for b_idx in range(batch_size):
            i = row_v[b_i + b_idx]
            j = col_v[b_j + b_idx]
            patch_idxs[b_idx] = [i, j]
            patch_batch[b_idx] = im[:, i:i+patch_r, j:j+patch_c]

        # convert to torch before sending to model
        patch_batch_t = torch.from_numpy(patch_batch).to(device)
        with torch.no_grad():
            out_patch = model(patch_batch_t)
            # out_patch ~> (num_batch, nch, patch_r, patch_c)

        # collect the patches
        out_patch_b = out_patch.detach().cpu().numpy()
        merge_patches(out_im, count, out_patch_b, patch_idxs)

        if debug:
            sys.stdout.write('Progress: [{}/{}]\r'.format(b, num_batch))

    sys.stdout.write('\n')

    # handle the non full batch
    if numpatch % batch_size != 0:
        remaining_patches = numpatch % batch_size
        patch_batch = np.zeros((remaining_patches, nch) + patch_size,
                               dtype='float32')
        patch_idxs = np.zeros((remaining_patches, 2), dtype='int')

        b_i = num_batch*batch_size
        b_j = num_batch*batch_size
        for b_idx in range(remaining_patches):
            i = row_v[b_i + b_idx]
            j = col_v[b_j + b_idx]
            patch_idxs[b_idx] = [i, j]
            patch_batch[b_idx] = im[:, i:i+patch_r, j:j+patch_c]

        patch_batch_t = torch.from_numpy(patch_batch).to(device)
        with torch.no_grad():
            out_patch = model(patch_batch_t)
            # out_patch ~> (num_batch, nch, patch_r, patch_c)

        out_patch_b = out_patch.detach().cpu().numpy()
        merge_patches(out_im, count, out_patch_b, patch_idxs)

    if debug:
        print('Processing borders')

    # now handle the locations that are not covered
    # this can only occur in the right and bottom
    patch_end_r = row_v[-1]
    patch_end_c = col_v[-1]
    if patch_end_r + patch_r < nrow-1:
        # there are pixels at bottom that are not covered
        for i in range(ncol-patch_c+1):
            r = nrow-patch_r
            c = i
            patch = im[:, r:r+patch_r, c:c+patch_c]

            patch = torch.from_numpy(patch[None, ...]).to(device)
            with torch.no_grad():
                out_patch = model(patch)

            out_patch = out_patch.detach().cpu().numpy()

            out_im[r:r+patch_r, c:c+patch_c, :] += \
                out_patch[0].transpose((1, 2, 0))

            count[r:r+patch_r, c:c+patch_c] += 1

    if patch_end_c + patch_c < ncol-1:
        # there are pixels at right that are not covered
        for i in range(nrow-patch_r+1):
            r = i
            c = ncol-patch_c

            patch = im[:, r:r+patch_r, c:c+patch_c]

            patch = torch.from_numpy(patch[None, ...]).to(device)
            with torch.no_grad():
                out_patch = model(patch)

            out_patch = out_patch.detach().cpu().numpy()

            out_im[r:r+patch_r, c:c+patch_c, :] += \
                out_patch[0].transpose((1, 2, 0))

            count[r:r+patch_r, c:c+patch_c] += 1

    nz = (count != 0)
    nz_r = np.tile(nz[..., None], (1, 1, nch))
    count_r = np.tile(count[..., None], (1, 1, nch))

    out_im[nz_r] = out_im[nz_r] / count_r[nz_r]

    return out_im


if __name__ == '__main__':
    # param
    model_wt = './model/bgm1-960-0.13053849339485168.pt'

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

    # inference on cpu
    device = torch.device('cpu')
    model = BGRestoreNet()
    model.load_state_dict(torch.load(model_wt, map_location=device))
    model.eval()

    in_im_PIL = Image.open(input_im_path)

    if in_im_PIL.mode != 'RGB':
        in_im_PIL = in_im_PIL.convert('RGB')

    # test the bg net
    # for the time being just call the mini batch function
    in_im_uint8 = np.asarray(in_im_PIL)
    in_im = in_im_uint8/255.0

    out_im = patch_process_mini_batch(model, in_im, patch_size, stride,
                                      inference_batch_size, device)

    out_im_PIL = Image.fromarray(np.squeeze(out_im*255).astype('uint8'))
    out_im_PIL.save(output_im_path)
