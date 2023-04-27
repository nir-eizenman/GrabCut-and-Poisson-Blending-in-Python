import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import argparse
import math

"""
Poisson Blending will allow you to blend source and target images. As we
have seen, this involves the computation of the Laplacian matrix and finding
the boundary of the binary mask (use the gradients for this). In the poisson
blend.py you should:
1. calculate the Laplacian operator
2. Solve the Poisson equation
3. Blend the source and target images using the Poisson result
4. Save the blended image
Your source and target images might no be of the same sizes, but you should
only support a case where the source mask is smaller from the target image size.
Pay attention to the matrices size, you should use sparse matrices in your code.
"""


def poisson_blend(im_src, im_tgt, im_mask, center):
    # Get dimensions of source image
    src_h, src_w, _ = im_src.shape
    # Get dimensions of target image
    mask_h, mask_w = im_mask.shape
    # Get dimensions of target image
    tgt_h, tgt_w, _ = im_tgt.shape

    # calculate the offset of the source image ((x_min, y_min) is the coordinates of the top left corner of the
    # source image in the target image and (x_max, y_max) is the
    # coordinates of the bottom right corner of the source image in the target image)
    y_min, y_max = center[1] - mask_h // 2, math.ceil(center[1] + mask_h / 2)
    x_min, x_max = center[0] - mask_w // 2, math.ceil(center[0] + mask_w / 2)

    matrix_d = sp.lil_matrix((mask_w, mask_w))
    matrix_d.setdiag(-1, -1)
    matrix_d.setdiag(4)
    matrix_d.setdiag(-1, 1)

    matrix_a = sp.block_diag([matrix_d] * mask_h).tolil()

    matrix_a.setdiag(-1, 1 * mask_w)
    matrix_a.setdiag(-1, -1 * mask_w)

    laplacian = matrix_a.tocsc()

    for y in range(1, mask_h - 1):
        for x in range(1, mask_w - 1):
            if im_mask[y, x] == 0:
                k = x + y * mask_w
                matrix_a[k, k] = 1
                matrix_a[k, k + 1] = 0
                matrix_a[k, k - 1] = 0
                matrix_a[k, k + mask_w] = 0
                matrix_a[k, k - mask_w] = 0

    matrix_b = matrix_a.tocsc()

    # create an image for the result (currently containing the target image)
    im_blend = np.copy(im_tgt)

    mask_flat = im_mask.flatten()

    for channel in range(im_src.shape[2]):
        source_flat = im_src[:, :, channel].flatten()
        target_flat = im_tgt[y_min:y_max, x_min:x_max, channel].flatten()

        vector_b = laplacian.dot(source_flat)

        vector_b[mask_flat == 0] = target_flat[mask_flat == 0]

        x = spsolve(matrix_b, vector_b)

        # if there are values outside the range [0, 255] clip them (255+ is set to 255 and 0- is set to 0),
        # convert the image to uint8 (unsigned int of 8 bits) and return it
        x = np.clip(x, 0, 255).astype(np.uint8)

        x = x.reshape(mask_h, mask_w)

        im_blend[y_min:y_max, x_min:x_max, channel] = np.where(im_mask == 255, x,
                                                               im_tgt[y_min:y_max, x_min:x_max, channel])

    # Return the blended image
    return im_blend


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/bush.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./data/seg_GT/bush.bmp', help='mask file path')
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
