import igraph as ig
import cv2
import argparse
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import numpy as np

GC_BGD = 0  # Hard bg pixel
GC_FGD = 1  # Hard fg pixel, will not be used
GC_PR_BGD = 2  # Soft bg pixel
GC_PR_FGD = 3  # Soft fg pixel

# initialize the mask (because we want it as a global variable)
mask = []


# Define the GrabCut algorithm function
def grabcut(img, rect, n_iter=5):
    global mask

    prev_energy = 0

    img = np.asarray(img, dtype=np.float64)

    # Assign initial labels to the pixels based on the bounding box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect

    h -= y
    w -= x

    # Initalize the inner square to Foreground
    mask[y:y + h, x:x + w] = GC_PR_FGD
    mask[rect[1] + rect[3] // 2, rect[0] + rect[2] // 2] = GC_FGD

    bgGMM, fgGMM = initalize_GMMs(img, mask)

    num_iters = 1000
    for i in range(num_iters):

        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)

        mincut_sets, curr_energy = calculate_mincut(img, mask, bgGMM, fgGMM)

        mask = update_mask(mincut_sets, mask)

        if check_convergence(np.abs(prev_energy - curr_energy)):
            break
        prev_energy = curr_energy

    # Return the final mask and the GMMs

    return mask, bgGMM, fgGMM


def initalize_GMMs(img, mask, n_components=5):
    # Extract the foreground and background pixels from the mask
    fg_data = img[np.logical_or(mask == GC_PR_FGD, mask == GC_FGD)].reshape(-1, 3)
    bg_data = img[np.logical_or(mask == GC_PR_BGD, mask == GC_BGD)].reshape(-1, 3)

    # Initialize with k-means clustering
    bg_kmeans = KMeans(n_clusters=n_components)
    bg_kmeans.fit(bg_data)
    fg_kmeans = KMeans(n_clusters=n_components)
    fg_kmeans.fit(fg_data)

    # Initialize GMMs with k-means cluster centers
    bgGMM = GaussianMixture(n_components=n_components, means_init=bg_kmeans.cluster_centers_)
    bgGMM.fit(bg_data)

    fgGMM = GaussianMixture(n_components=n_components, means_init=fg_kmeans.cluster_centers_)
    fgGMM.fit(fg_data)
    return bgGMM, fgGMM


# Define helper functions for the GrabCut algorithm
# this function takes the image, mask and the GMMs and returns the updated GMMs
def update_GMMs(img, mask, bgGMM, fgGMM):
    # TODO: implement GMM component assignment step
    fg_data = img[np.logical_or(mask == GC_PR_FGD, mask == GC_FGD)].reshape(-1, 3)
    bg_data = img[np.logical_or(mask == GC_PR_BGD, mask == GC_BGD)].reshape(-1, 3)

    # Initialize background GMM
    bg_n_components = bgGMM.n_components
    bg_covs = np.zeros((bg_n_components, 3, 3))
    bg_m = np.zeros((bg_n_components, 3))
    bg_w = np.zeros(bg_n_components)

    # Iterate over the GMM components
    # Calculate the mean and covariance of each component
    # Update the GMM weights, means and covariances
    for i in range(bg_n_components):
        # Get the pixels that belong to the current component
        component_mask = bgGMM.predict(bg_data) == i
        # Extract the pixels that belong to the current component
        component_data = bg_data[component_mask]

        # Calculate the mean and covariance of the current component
        # Update the GMM weights, means and covariances
        if len(component_data) > 0:
            # Calculate the weight of the current component by dividing the number of pixels in the component by the total number of pixels
            bg_w[i] = len(component_data) / len(bg_data)
            # Calculate the mean and covariance of the current component using the cv2.calcCovarMatrix function
            # Update the GMM weights, means and covariances
            covar, means = cv2.calcCovarMatrix(component_data, None,
                                               cv2.COVAR_NORMAL | cv2.COVAR_SCALE | cv2.COVAR_ROWS)
            bg_m[i] = means.flatten()
            bg_covs[i] = covar

    # Update the GMM weights, means and covariances
    bgGMM.weights_ = bg_w
    bgGMM.means_ = bg_m
    bgGMM.covariances_ = bg_covs

    # Update foreground GMM ( same as the background GMM )
    fg_n_components = fgGMM.n_components
    fg_w = np.zeros(fg_n_components)
    fg_m = np.zeros((fg_n_components, 3))
    fg_covs = np.zeros((fg_n_components, 3, 3))

    for i in range(fg_n_components):
        component_mask = fgGMM.predict(fg_data) == i
        component_data = fg_data[component_mask]

        if len(component_data) > 0:
            fg_w[i] = len(component_data) / len(fg_data)
            covar, means = cv2.calcCovarMatrix(component_data, None,
                                               cv2.COVAR_NORMAL | cv2.COVAR_SCALE | cv2.COVAR_ROWS)
            fg_m[i] = means.flatten()
            fg_covs[i] = covar

    fgGMM.weights_ = fg_w
    fgGMM.means_ = fg_m
    fgGMM.covariances_ = fg_covs

    # Check if any of the GMM weights are 0, if so remove the component
    bg_ind_to_remove = []
    fg_ind_to_remove = []

    for i in range(len(bgGMM.weights_)):
        if bgGMM.weights_[i] <= 0.0025:
            bg_ind_to_remove.append(i)

    for i in range(len(fgGMM.weights_)):
        if fgGMM.weights_[i] <= 0.0025:
            fg_ind_to_remove.append(i)

    if len(bg_ind_to_remove) > 0:
        bgGMM.n_components = bgGMM.n_components - len(bg_ind_to_remove)
        bgGMM.weights_ = np.delete(bgGMM.weights_, bg_ind_to_remove, axis=0)
        bgGMM.means_ = np.delete(bgGMM.means_, bg_ind_to_remove, axis=0)
        bgGMM.means_init = np.delete(bgGMM.means_init, bg_ind_to_remove, axis=0)
        bgGMM.covariances_ = np.delete(bgGMM.covariances_, bg_ind_to_remove, axis=0)
        bgGMM.precisions_ = np.delete(bgGMM.precisions_, bg_ind_to_remove, axis=0)
        bgGMM.precisions_cholesky_ = np.delete(bgGMM.precisions_cholesky_, bg_ind_to_remove, axis=0)

    if len(fg_ind_to_remove) > 0:
        fgGMM.n_components = fgGMM.n_components - len(fg_ind_to_remove)
        fgGMM.weights_ = np.delete(fgGMM.weights_, fg_ind_to_remove, axis=0)
        fgGMM.means_ = np.delete(fgGMM.means_, fg_ind_to_remove, axis=0)
        fgGMM.means_init = np.delete(fgGMM.means_init, fg_ind_to_remove, axis=0)
        fgGMM.covariances_ = np.delete(fgGMM.covariances_, fg_ind_to_remove, axis=0)
        fgGMM.precisions_ = np.delete(fgGMM.precisions_, fg_ind_to_remove, axis=0)
        fgGMM.precisions_cholesky_ = np.delete(fgGMM.precisions_cholesky_, fg_ind_to_remove, axis=0)

    return bgGMM, fgGMM


def calculate_mincut(img, mask, bgGMM, fgGMM):
    h, w = img.shape[:2]
    index_for_pixels = np.arange(h * w).reshape(h, w)
    mc_graph = ig.Graph()
    mc_graph.add_vertices(h * w + 2)
    source = h * w
    sink = h * w + 1

    flatten_img = img.reshape(h * w, -1)
    bg_data = -bgGMM.score_samples(flatten_img).reshape(h, w, 1)
    fg_data = -fgGMM.score_samples(flatten_img).reshape(h, w, 1)

    d_term = np.concatenate((bg_data, fg_data), axis=2)

    # calculate beta by summing the Euclidean distance between each pixel (its color vector) and his 8 neighbors
    # and then dividing the sum by the number of pixels, and then multiplying the result by 2 and then taking the inverse
    beta = 0

    for i in range(h):
        for j in range(w):
            # calculate the Euclidean distance between the current pixel and its 8 neighbors
            # and then sum the distances
            if i > 0:
                beta += np.linalg.norm(img[i, j] - img[i - 1, j]) ** 2
            if i < h - 1:
                beta += np.linalg.norm(img[i, j] - img[i + 1, j]) ** 2
            if j > 0:
                beta += np.linalg.norm(img[i, j] - img[i, j - 1]) ** 2
            if j < w - 1:
                beta += np.linalg.norm(img[i, j] - img[i, j + 1]) ** 2
            if i > 0 and j > 0:
                beta += np.linalg.norm(img[i, j] - img[i - 1, j - 1]) ** 2
            if i > 0 and j < w - 1:
                beta += np.linalg.norm(img[i, j] - img[i - 1, j + 1]) ** 2
            if i < h - 1 and j > 0:
                beta += np.linalg.norm(img[i, j] - img[i + 1, j - 1]) ** 2
            if i < h - 1 and j < w - 1:
                beta += np.linalg.norm(img[i, j] - img[i + 1, j + 1]) ** 2

    # calculate beta by dividing the sum of the distances by the number of pixels,
    # and then multiplying the result by 2 and then taking the inverse
    beta = beta / ((h * w * 8) - (h * 2 + w * 2 + 4))
    beta = 1 / (beta * 2)

    # recommended gamma value
    gamma = 50

    # Add edges to the graph
    edge_list = []
    weight_list = []

    # calculate N-links
    for i in range(h):
        for j in range(w):
            # calculate K value for each pixel by holding the maximum of all N-links that are connected to the pixel
            # (used to T-link, but calculated as the maximum of all N-links that are connected to the pixel)
            k = 0
            # set the node index of the current pixel
            curr_node = index_for_pixels[i, j]

            # Calculate edge weight for top neighbor
            if i > 0:
                second_node = index_for_pixels[i - 1, j]
                # color difference between the current pixel and its top left neighbor
                # calculated by the norm of the difference between the two pixel's color vectors
                colors_difference = (np.linalg.norm(img[i, j] - img[i - 1, j])) ** 2
                edge_w = gamma * np.exp(-beta * colors_difference)
                edge_list.append((curr_node, second_node))
                weight_list.append(edge_w)
                k = max(k, edge_w)

            # Calculate edge weight for left neighbor
            if j > 0:
                second_node = index_for_pixels[i, j - 1]
                # color difference between the current pixel and its top left neighbor
                # calculated by the norm of the difference between the two pixel's color vectors
                colors_difference = (np.linalg.norm(img[i, j] - img[i, j - 1])) ** 2
                edge_w = gamma * np.exp(-beta * colors_difference)
                edge_list.append((curr_node, second_node))
                weight_list.append(edge_w)
                k = max(k, edge_w)

            # Calculate edge weight for right neighbor
            if j < w - 1:
                second_node = index_for_pixels[i, j + 1]
                # color difference between the current pixel and its top left neighbor
                # calculated by the norm of the difference between the two pixel's color vectors
                colors_difference = (np.linalg.norm(img[i, j] - img[i, j + 1])) ** 2
                edge_w = gamma * np.exp(-beta * colors_difference)
                edge_list.append((curr_node, second_node))
                weight_list.append(edge_w)
                k = max(k, edge_w)

            # Calculate edge weight for bottom neighbor
            if i < h - 1:
                second_node = index_for_pixels[i + 1, j]
                # color difference between the current pixel and its top left neighbor
                # calculated by the norm of the difference between the two pixel's color vectors
                colors_difference = (np.linalg.norm(img[i, j] - img[i + 1, j])) ** 2
                edge_w = gamma * np.exp(-beta * colors_difference)
                edge_list.append((curr_node, second_node))
                weight_list.append(edge_w)
                k = max(k, edge_w)

            # calculate edge weight for top left neighbor
            if i > 0 and j > 0:
                second_node = index_for_pixels[i - 1, j - 1]
                # color difference between the current pixel and its top left neighbor
                # calculated by the norm of the difference between the two pixel's color vectors
                colors_difference = (np.linalg.norm(img[i, j] - img[i - 1, j - 1])) ** 2
                # dividing by sqrt(2) because distance between the pixels is sqrt(2) in case of diagonal neighbors
                edge_w = (1 / np.sqrt(2)) * gamma * np.exp(-beta * colors_difference)
                edge_list.append((curr_node, second_node))
                weight_list.append(edge_w)
                k = max(k, edge_w)

            # calculate edge weight for bottom left neighbor
            if i < h - 1 and j > 0:
                second_node = index_for_pixels[i + 1, j - 1]
                # color difference between the current pixel and its top left neighbor
                # calculated by the norm of the difference between the two pixel's color vectors
                colors_difference = (np.linalg.norm(img[i, j] - img[i + 1, j - 1])) ** 2
                # dividing by sqrt(2) because distance between the pixels is sqrt(2) in case of diagonal neighbors
                edge_w = (1 / np.sqrt(2)) * gamma * np.exp(-beta * colors_difference)
                edge_list.append((curr_node, second_node))
                weight_list.append(edge_w)
                k = max(k, edge_w)

            # calculate edge weight for top right neighbor
            if i > 0 and j < w - 1:
                second_node = index_for_pixels[i - 1, j + 1]
                # color difference between the current pixel and its top left neighbor
                # calculated by the norm of the difference between the two pixel's color vectors
                colors_difference = (np.linalg.norm(img[i, j] - img[i - 1, j + 1])) ** 2
                # dividing by sqrt(2) because distance between the pixels is sqrt(2) in case of diagonal neighbors
                edge_w = (1 / np.sqrt(2)) * gamma * np.exp(-beta * colors_difference)
                edge_list.append((curr_node, second_node))
                weight_list.append(edge_w)
                k = max(k, edge_w)

            # calculate edge weight for bottom right neighbor
            if i < h - 1 and j < w - 1:
                second_node = index_for_pixels[i + 1, j + 1]
                # color difference between the current pixel and its top left neighbor
                # calculated by the norm of the difference between the two pixel's color vectors
                colors_difference = (np.linalg.norm(img[i, j] - img[i + 1, j + 1])) ** 2
                # dividing by sqrt(2) because distance between the pixels is sqrt(2) in case of diagonal neighbors
                edge_w = (1 / np.sqrt(2)) * gamma * np.exp(-beta * colors_difference)
                edge_list.append((curr_node, second_node))
                weight_list.append(edge_w)
                k = max(k, edge_w)

            # get the value of Dback for the current pixel
            d_back = d_term[i, j, 0]
            # get the value of Dfore for the current pixel
            d_fore = d_term[i, j, 1]
            edge_list.append((source, curr_node))
            edge_list.append((curr_node, sink))

            # check if the current pixel is a background pixel or a foreground pixel using the mask
            if mask[i, j] == GC_PR_FGD or mask[i, j] == GC_PR_BGD:
                weight_list.append(d_fore)
                weight_list.append(d_back)
            if mask[i, j] == GC_FGD:
                weight_list.append(0)
                weight_list.append(k)
            if mask[i, j] == GC_BGD:
                weight_list.append(k)
                weight_list.append(0)

    mc_graph.add_edges(edge_list, {'weight': weight_list})

    mincut_result = mc_graph.st_mincut(source, sink, capacity='weight')

    mincut_set_list = [set(mincut_result.partition[0]), set(mincut_result.partition[1])]

    return mincut_set_list, mincut_result.value


def update_mask(mincut_sets, mask):
    h, w = mask.shape
    index_for_pixel = np.arange(h * w).reshape(h, w)
    mask_copy = np.copy(mask)
    for i in range(h):
        for j in range(w):
            if index_for_pixel[i, j] in mincut_sets[0] and (mask[i, j] == GC_PR_FGD or mask[i, j] == GC_PR_BGD):
                mask_copy[i, j] = GC_PR_BGD
            elif index_for_pixel[i, j] in mincut_sets[1] and (mask[i, j] == GC_PR_FGD or mask[i, j] == GC_PR_BGD):
                mask_copy[i, j] = GC_PR_FGD

    mask = np.copy(mask_copy)

    return mask


def check_convergence(energy):
    global mask
    is_conv = False
    if energy <= 1500:
        # change all soft background pixels to hard background pixels
        mask[mask == GC_PR_BGD] = GC_BGD
        is_conv = True
    return is_conv


def cal_metric(predicted_mask, gt_mask):
    # Convert masks to boolean arrays
    pred_mask_boolean = predicted_mask.astype(bool)
    gt_mask_boolean = gt_mask.astype(bool)

    # Calculate the number of correctly labeled pixels
    correct_pix = np.sum(pred_mask_boolean == gt_mask_boolean)
    # Calculate the total number of pixels
    total_pix = predicted_mask.size

    # Calculate the accuracy
    accuracy = correct_pix / total_pix

    # Calculate jaccard similarity
    union_of_masks = np.sum(pred_mask_boolean | gt_mask_boolean)
    inter_of_masks = np.sum(pred_mask_boolean & gt_mask_boolean)
    jaccard = inter_of_masks / union_of_masks

    return accuracy, jaccard


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='teddy', help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    return parser.parse_args()


if __name__ == '__main__':
    # Load an example image and define a bounding box around the object of interest
    args = parse()

    if args.input_img_path == '':
        input_path = f'data/imgs/{args.input_name}.jpg'
    else:
        input_path = args.input_img_path

    if args.use_file_rect:
        rect = tuple(map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' ')))
    else:
        rect = tuple(map(int, args.rect.split(',')))

    img = cv2.imread(input_path)

    # Run the GrabCut algorithm on the image and bounding box
    mask, bgGMM, fgGMM = grabcut(img, rect)
    mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]

    # Print metrics only if requested (valid only for course files)
    if args.eval:
        gt_mask = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
        acc, jac = cal_metric(mask, gt_mask)
        print(f'Accuracy={acc}, Jaccard={jac}')

    # Apply the final mask to the input image and display the results
    img_cut = img * (mask[:, :, np.newaxis])
    cv2.imshow('Original Image', img)
    cv2.imshow('GrabCut Mask', 255 * mask)
    cv2.imshow('GrabCut Result', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
