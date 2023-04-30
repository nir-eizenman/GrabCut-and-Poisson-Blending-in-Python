import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp
import argparse
import cv2


def poisson_blend(im_src, im_tgt, im_mask, center):
    # Get dimensions of target image
    tgt_height, tgt_width, tgt_channels = im_tgt.shape
    # Get dimensions of source image
    src_height, src_width, src_channels = im_src.shape
    # Get dimensions of target image
    mask_height, mask_width = im_mask.shape

    # calculate the offset of the source image ((x_min, y_min) is the coordinates of the top left corner of the
    # source image in the target image and (x_max, y_max) is the
    # coordinates of the bottom right corner of the source image in the target image)
    x_min, x_max = center[0] - mask_width // 2, math.ceil(center[0] + mask_width / 2)
    y_min, y_max = center[1] - mask_height // 2, math.ceil(center[1] + mask_height / 2)

    # create a diagonal matrix to be used for the laplacian matrix (D from wikipedia article)
    block_mat = sp.lil_matrix((mask_width, mask_width))
    block_mat.setdiag(-1, 1)
    block_mat.setdiag(-1, -1)
    block_mat.setdiag(4)

    # create a diagonal matrix to be used for the laplacian matrix (A from wikipedia article)
    diag_block_mat = sp.block_diag([block_mat] * mask_height).tolil()

    # complete the calculation for A by adding the -I matrices
    diag_block_mat.setdiag(-1, -1 * mask_width)
    diag_block_mat.setdiag(-1, mask_width)

    # change the laplacian matrix to a compressed sparse column matrix (CSC) for faster calculations
    lap_mat = diag_block_mat.tocsc()

    # use the diagonal matrix used to create the laplacian matrix to create the laplacian matrix for the mask
    for y in range(1, mask_height - 1):
        for vec_x in range(1, mask_width - 1):
            if im_mask[y, vec_x] == 0:
                # calculate the index of the pixel in the mask (its row in the laplacian matrix)
                k = vec_x + y * mask_width
                # set the values of the laplacian matrix to 0 for the corresponding pixel's row in the mask
                diag_block_mat[k, k - mask_width] = 0
                diag_block_mat[k, k + mask_width] = 0
                diag_block_mat[k, k - 1] = 0
                diag_block_mat[k, k + 1] = 0
                # set the value of the corresponding pixel's row to 1 in its diagonal in the laplacian matrix
                diag_block_mat[k, k] = 1

    # change the laplacian matrix to a compressed sparse column matrix (CSC) for faster calculations
    mat_b = diag_block_mat.tocsc()

    # create an image for the result (currently containing the target image)
    im_blend = np.copy(im_tgt)

    # flatten the mask to 1 dimension
    flatten_mask = im_mask.flatten()

    # calculate the correct value for the result one channel at a time
    for channel in range(im_src.shape[2]):
        flatten_target = im_tgt[y_min:y_max, x_min:x_max, channel].flatten()
        flatten_source = im_src[:, :, channel].flatten()

        # calculate the b vector for the equation (Ax = b)
        vec_b = lap_mat.dot(flatten_source)

        # set the values of the b vector to the values of the target image where the mask is 0
        vec_b[flatten_mask == 0] = flatten_target[flatten_mask == 0]

        # solve the equation (Ax = b) for x
        vec_x = spsolve(mat_b, vec_b)

        # if there are values outside the range [0, 255] clip them (255+ is set to 255 and 0- is set to 0),
        # convert the image to uint8 (unsigned int of 8 bits) and return it
        vec_x = np.clip(vec_x, 0, 255).astype(np.uint8)

        vec_x = vec_x.reshape(mask_height, mask_width)

        # update the result image with the calculated values where the mask is 255, else keep the target image values
        im_blend[y_min:y_max, x_min:x_max, channel] = np.where(im_mask == 255, vec_x,
                                                               im_tgt[y_min:y_max, x_min:x_max, channel])

    # Return the blended image
    return im_blend


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/llama.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./data/seg_GT/llama.bmp', help='mask file path')
    parser.add_argument('--tgt_path', type=str, default='./data/bg/table.jpg', help='mask file path')
    return parser.parse_args()


if __name__ == "__main__":
    # Load the source and target images
    args = parse()

    im_tgt = cv2.imread(args.tgt_path, cv2.IMREAD_COLOR)
    im_src = cv2.imread(args.src_path, cv2.IMREAD_COLOR)
    if args.mask_path == '':
        im_mask = np.full(im_src.shape, 255, dtype=np.uint8)
    else:
        im_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        im_mask = cv2.threshold(im_mask, 0, 255, cv2.THRESH_BINARY)[1]

    center = (int(im_tgt.shape[1] / 2), int(im_tgt.shape[0] / 2))

    im_clone = poisson_blend(im_src, im_tgt, im_mask, center)

    cv2.imshow('Cloned image', im_clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
