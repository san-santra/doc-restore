# helper functions
import numpy as np


def extract_patches(im, patch_size, stride):
    '''
    im - a numpy array with data in channel first order
    '''

    patch_r, patch_c = patch_size
    stride_r, stride_c = stride
    nch, nrow, ncol = im.shape

    row_idxs = range(0, nrow-patch_r+1, stride_r)
    col_idxs = range(0, ncol-patch_c+1, stride_c)

    numpatch = len(row_idxs)*len(col_idxs)

    # expecting the data type to be float32
    patches = np.zeros((numpatch, nch, patch_r, patch_c),
                       dtype='float32')

    patch_idx = 0
    for i in row_idxs:
        for j in col_idxs:
            patch = im[:, i:i+patch_r, j:j+patch_c]

            patches[patch_idx] = patch
            patch_idx += 1

    return patches


def merge_patches(patches, image_size, stride):
    '''
    patches are in channel first order
    '''

    nrow, ncol = image_size
    numpatch, nch, patch_r, patch_c = patches.shape
    stride_r, stride_c = stride

    row_idxs = range(0, nrow-patch_r+1, stride_r)
    col_idxs = range(0, ncol-patch_c+1, stride_c)

    out_im = np.zeros((nch, nrow, ncol), dtype='float32')
    count = np.zeros((nrow, ncol), dtype='int')

    patch_idx = 0
    for i in row_idxs:
        for j in col_idxs:
            out_im[:, i:i+patch_r, j:j+patch_c] += patches[patch_idx]

            count[i:i+patch_r, j:j+patch_c] += 1
            patch_idx += 1
            
    nz = (count != 0)
    nz_r = np.tile(nz[None, ...], (nch, 1, 1))
    count_r = np.tile(count[None, ...], (nch, 1, 1))

    out_im[nz_r] = out_im[nz_r] / count_r[nz_r]

    return out_im.transpose((1, 2, 0))
