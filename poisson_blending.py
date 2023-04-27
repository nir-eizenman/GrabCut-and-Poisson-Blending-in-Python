import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import argparse

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


#     # turn the mask to mask of 0 and 1
#     new_mask = np.zeros((tgt_h, tgt_w))
#     new_mask[temp_mask != 0] = 1
#
#     # Create the sparse matrix A for the Laplacian operator using scipy.sparse
#     # The size of the matrix should be equal to the number of pixels in the source image (src_h * src_w)
#     sparse_matrix_a = sp.lil_matrix((tgt_h * tgt_w, tgt_h * tgt_w))
#     # turn the matrix to a sparse matrix
#     sparse_matrix_a = sparse_matrix_a.tocsr()
#     # fill the matrix with the correct values
#     sparse_matrix_a.setdiag(-4)
#     sparse_matrix_a.setdiag(1, 1)
#     sparse_matrix_a.setdiag(1, -1)
#     sparse_matrix_a.setdiag(1, tgt_w)
#     sparse_matrix_a.setdiag(1, -tgt_w)
#
#     # Create the sparse matrix B for calculating the b vector using for the Poisson equation using scipy.sparse
#     # The size of the matrix should be equal to the number of pixels in the source image (src_h * src_w)
#     sparse_matrix_b = sp.lil_matrix((tgt_h * tgt_w, tgt_h * tgt_w))
#     # turn the matrix to a sparse matrix
#     sparse_matrix_b = sparse_matrix_b.tocsr()
#     # fill the matrix with the correct values
#     sparse_matrix_b.setdiag(1, 1)
#     sparse_matrix_b.setdiag(1, -1)
#     sparse_matrix_b.setdiag(1, tgt_w)
#     sparse_matrix_b.setdiag(1, -tgt_w)
#
#     # Create the b vector for the Poisson equation using scipy.sparse
#     # The size of the vector should be equal to the number of pixels in the source image (src_h * src_w)
#     b = np.zeros((tgt_h * tgt_w, tgt_c))
#     # fill the vector with the correct values
#     b[:, 0] = sparse_matrix_b.dot(im_src[:, :, 0].flatten())
#     b[:, 1] = sparse_matrix_b.dot(im_src[:, :, 1].flatten())
#     b[:, 2] = sparse_matrix_b.dot(im_src[:, :, 2].flatten())
#
#
#     # Solve the equation using scipy.sparse.linalg.spsolve
#     # The solution should be a vector of size (src_h * src_w)
#     x = spsolve(sparse_matrix_a, b)

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
    y_min, y_max = center[1] - mask_h // 2, center[1] + mask_h // 2
    x_min, x_max = center[0] - mask_w // 2, center[0] + mask_w // 2

    # # initialize matrix A
    # # TODO: check if the matrix is correct
    # matrix_a = sp.diags([4, -1, -1, 1, 1], [0, -1, 1, -mask_w, mask_w], shape=(mask_h * mask_w, mask_h * mask_w),
    #                     format="csr")

    matrix_d = sp.lil_matrix((mask_w, mask_w))
    matrix_d.setdiag(-1, -1)
    matrix_d.setdiag(4)
    matrix_d.setdiag(-1, 1)

    matrix_a = sp.block_diag([matrix_d] * mask_h).tolil()

    matrix_a.setdiag(-1, 1 * mask_w)
    matrix_a.setdiag(-1, -1 * mask_w)

    laplacian = matrix_a.tocsr()

    # upscale the mask to the size of the target image (use y_min, y_max, x_min, x_max to calculate the offset,
    # fill the pixels outside the mask with 0)
    new_mask = np.zeros((tgt_h, tgt_w))
    new_mask[y_min:y_max, x_min:x_max] = im_mask
    # upscale the new_mask to the size of (tgt_h * tgt_w, tgt_h * tgt_w), such that each pixel in the mask will be a
    # diagonal matrix of size (mask_w, mask_w) in the new matrix

    # *****
    # continue here
    # *****

    # create an image for the result (currently containing the target image)
    im_blend = np.copy(im_tgt)

    for channel in range(im_src.shape[2]):
        # mask_grad_x = cv2.Sobel(im_mask, cv2.CV_64F, 1, 0, ksize=1)
        # mask_grad_y = cv2.Sobel(im_mask, cv2.CV_64F, 0, 1, ksize=1)
        #
        # src_grad_x = cv2.Sobel(im_src[:, :, channel], cv2.CV_64F, 1, 0, ksize=1)
        # src_grad_y = cv2.Sobel(im_src[:, :, channel], cv2.CV_64F, 0, 1, ksize=1)
        #
        # mixed_grad_x = np.where(mask_grad_x != 0, src_grad_x, 0)
        # mixed_grad_y = np.where(mask_grad_y != 0, src_grad_y, 0)
        #
        # div_mixed_grad = cv2.subtract(cv2.Sobel(mixed_grad_x, cv2.CV_64F, 1, 0, ksize=1),
        #                               cv2.Sobel(mixed_grad_y, cv2.CV_64F, 0, 1, ksize=1))
        #
        # flat_div_mixed_grad = div_mixed_grad.flatten()

        poisson_solution = spsolve(laplacian, b[:, channel])

        poisson_solution_2d = poisson_solution.reshape(mask_h, mask_w)

        im_blend[y_min:y_max, x_min:x_max, channel] = np.where(im_mask == 255, poisson_solution_2d,
                                                               im_tgt[y_min:y_max, x_min:x_max, channel])

    # if there are values outside the range [0, 255] clip them (255+ is set to 255 and 0- is set to 0),
    # convert the image to uint8 (unsigned int of 8 bits) and return it
    im_blend = np.clip(im_blend, 0, 255).astype(np.uint8)

    # Return the blended image
    return im_blend


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/banana2.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./data/seg_GT/banana1.bmp', help='mask file path')
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
