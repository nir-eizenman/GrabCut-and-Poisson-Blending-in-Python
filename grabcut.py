import numpy as np
import cv2
import argparse
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import igraph as ig

GC_BGD = 0  # Hard bg pixel
GC_FGD = 1  # Hard fg pixel, will not be used
GC_PR_BGD = 2  # Soft bg pixel
GC_PR_FGD = 3  # Soft fg pixel


# Define the GrabCut algorithm function
def grabcut(img, rect, n_iter=5):
    # Assign initial labels to the pixels based on the bounding box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect

    # this is a suggested fix from whatsapp****
    w -= x
    h -= y
    # ***************************************

    # Initalize the inner square to Foreground
    mask[y:y + h, x:x + w] = GC_PR_FGD
    mask[rect[1] + rect[3] // 2, rect[0] + rect[2] // 2] = GC_FGD

    bgGMM, fgGMM = initalize_GMMs(img, mask)

    num_iters = 1000
    for i in range(num_iters):
        # Update GMM
        print("Iteration: " + str(i) + " before updating GMMs")
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)
        print("Iteration: " + str(i) + " after updating GMMs")

        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)

        print("Iteration: " + str(i) + " after calculating mincut, energy is: " + str(energy))
        mask = update_mask(mincut_sets, mask)
        print("Iteration: " + str(i) + " after updating mask")

        if check_convergence(energy):
            break

    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM


"""
2.1 init GMMs(img, mask, n components=5):
In this method you should initialize and return two GMMs models one for
the background and one for the foreground, each of them should consist of
n components. The initialization should use kmeans. You could use your own
implementation of GMM or use any existing one, however you should pay attention
to the differences between off-the-shelf models and the use of it in the
grabcut algorithm.
"""


def initalize_GMMs(img, mask, n_components=5):
    # TODO: implement initalize_GMMs

    # Extract the foreground and background pixels from the mask
    bg_mask = (mask == GC_BGD) | (mask == GC_PR_BGD)  # bg_mask is True if the mask is either GC_BGD or GC_PR_BGD
    fg_mask = (mask == GC_FGD) | (mask == GC_PR_FGD)  # fg_mask is True if the mask is either GC_FGD or GC_PR_FGD

    # Reshape the image to a 2D array of pixels and 3 color values (RGB) (separate foreground and background pixels)
    # from a shape of (rows, cols, 3) to (rows*cols, 3)
    bg_data = img[bg_mask].reshape(-1, 3)
    fg_data = img[fg_mask].reshape(-1, 3)

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


"""
2.2 update GMMs(img, mask, bgdGMM, fgdGMM):
In this method, you need to update the Gaussian Mixture Models (GMMs)
for the foreground and background pixels based on the current mask. This step
involves calculating the mean, covariance matrix and weights of each GMM component
for the foreground and background pixels in the input image. You can
use the cv2.calcCovarMatrix() (or np.conv) function to calculate the covariance
matrix and mean values, respectively.
After calculating the mean and covariance matrix, you need to update the
GMMs using the formulae given in the GrabCut algorithm. You can refer to
the following links for more information on this step: cv2.calcCovarMatrix.
"""


# Define helper functions for the GrabCut algorithm
# this function takes the image, mask and the GMMs and returns the updated GMMs
def update_GMMs(img, mask, bgGMM, fgGMM):
    # TODO: implement GMM component assignment step
    bg_mask = (mask == GC_BGD) | (mask == GC_PR_BGD)  # bg_mask is True if the mask is either GC_BGD or GC_PR_BGD
    fg_mask = (mask == GC_FGD) | (mask == GC_PR_FGD)  # fg_mask is True if the mask is either GC_FGD or GC_PR_FGD

    # Get background and foreground pixels from the input image based on the mask
    bg_data = img[bg_mask].reshape(-1, 3)
    fg_data = img[fg_mask].reshape(-1, 3)
    print("foreground: ")
    print(fg_data)
    print("size of bg_data: " + str(bg_data.shape))
    print("size of fg_data: " + str(fg_data.shape))
    print("size of mask: " + str(mask.shape))
    print("number of 0s in mask: " + str(np.count_nonzero(mask == 0)))
    print("number of 1s in mask: " + str(np.count_nonzero(mask == 1)))
    print("number of 2s in mask: " + str(np.count_nonzero(mask == 2)))
    print("number of 3s in mask: " + str(np.count_nonzero(mask == 3)))

    # Update background GMM
    bg_n_components = bgGMM.n_components
    bg_weights = np.zeros(bg_n_components)
    bg_means = np.zeros((bg_n_components, 3))  # 3 is the number of channels ( RGB )
    # the shape is (n_components, n_channels, n_channels) because the covariance matrix is a square matrix
    bg_covs = np.zeros((bg_n_components, 3, 3))

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
            bg_weights[i] = len(component_data) / len(bg_data)
            # Calculate the mean and covariance of the current component using the cv2.calcCovarMatrix function
            cov, mean = cv2.calcCovarMatrix(component_data, None, cv2.COVAR_NORMAL | cv2.COVAR_SCALE | cv2.COVAR_ROWS)
            # Update the GMM weights, means and covariances
            bg_means[i] = mean.flatten()
            bg_covs[i] = cov

    # Update the GMM weights, means and covariances
    bgGMM.weights_ = bg_weights
    bgGMM.means_ = bg_means
    bgGMM.covariances_ = bg_covs

    # Update foreground GMM ( same as the background GMM )
    fg_n_components = fgGMM.n_components
    fg_weights = np.zeros(fg_n_components)
    fg_means = np.zeros((fg_n_components, 3))
    fg_covs = np.zeros((fg_n_components, 3, 3))

    for i in range(fg_n_components):
        component_mask = fgGMM.predict(fg_data) == i
        component_data = fg_data[component_mask]

        if len(component_data) > 0:
            fg_weights[i] = len(component_data) / len(fg_data)
            cov, mean = cv2.calcCovarMatrix(component_data, None, cv2.COVAR_NORMAL | cv2.COVAR_SCALE | cv2.COVAR_ROWS)
            fg_means[i] = mean.flatten()
            fg_covs[i] = cov

    fgGMM.weights_ = fg_weights
    fgGMM.means_ = fg_means
    fgGMM.covariances_ = fg_covs

    print("bgGMM weights: ")
    print(bgGMM.weights_)
    print("fgGMM weights: ")
    print(fgGMM.weights_)

    return bgGMM, fgGMM


"""
2.3 calculate mincut(img, mask, bgdGMM, fgdGMM):
In this method you should build a graph based on the existing mask and the
energy terms defined in the grabcut algorithm. Then a mincut should be used.
The method should return the vertices (i.e. pixels) in each segment and the
energy term corresponding to the cut. You are allowed to use any graph optimization
library, for example igraph.
"""


def calculate_mincut(img, mask, bgGMM, fgGMM):
    min_cut = [[], []]
    energy = 0
    h, w = img.shape[:2]
    img_indices = np.arange(h * w).reshape(h, w)
    graph = ig.Graph()
    graph.add_vertices(h * w + 2)
    source = h * w
    sink = h * w + 1

    # Calculate data term (used to T-link)
    data_term = np.zeros((h, w, 2))
    for i in range(h):
        for j in range(w):
            # Calculate the probability of the pixel belonging to the background and foreground
            # (this calculates D(i, s) and D(i, t) in the GrabCut algorithm for each pixel,
            # where (i, j) is the pixel and s and t are the source and sink nodes (background and foreground respectively))
            data_term[i, j, 0] = -bgGMM.score_samples(img[i, j].reshape(1, -1))
            data_term[i, j, 1] = -fgGMM.score_samples(img[i, j].reshape(1, -1))

    print("data_term: ")
    print(data_term)

    gamma = 50

    # beta = 1 / (2 * np.mean(np.sum((img[:-1, :] - img[1:, :]) ** 2, axis=2)))
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

    # calculate the number of pixels
    num_pixels = h * w
    # calculate beta by dividing the sum of the distances by the number of pixels, and then multiplying the result by 2 and then taking the inverse
    beta = beta / num_pixels
    beta = 2 * beta
    beta = 1 / beta

    print("beta: ")
    print(beta)

    # Add edges to the graph
    edges = []
    weights = []

    # calculate N-links
    for i in range(h):
        for j in range(w):
            # calculate K value for each pixel by holding the maximum of all N-links that are connected to the pixel
            # (used to T-link, but calulated as the maximum of all N-links that are connected to the pixel)
            k = 0
            # set the node index of the current pixel
            node1 = img_indices[i, j]

            # Calculate edge weight for top neighbor
            if i > 0:
                node2 = img_indices[i - 1, j]
                # color difference between the current pixel and its top left neighbor
                # calculated by the norm of the difference between the two pixel's color vectors
                color_diff = (np.linalg.norm(img[i, j] - img[i - 1, j])) ** 2
                edge_weight = gamma * np.exp(-beta * color_diff)
                edges.append((node1, node2))
                weights.append(edge_weight)
                k = max(k, edge_weight)

            # Calculate edge weight for left neighbor
            if j > 0:
                node2 = img_indices[i, j - 1]
                # color difference between the current pixel and its top left neighbor
                # calculated by the norm of the difference between the two pixel's color vectors
                color_diff = (np.linalg.norm(img[i, j] - img[i, j - 1])) ** 2
                edge_weight = gamma * np.exp(-beta * color_diff)
                edges.append((node1, node2))
                weights.append(edge_weight)
                k = max(k, edge_weight)

            # Calculate edge weight for bottom neighbor
            if i < h - 1:
                node2 = img_indices[i + 1, j]
                # color difference between the current pixel and its top left neighbor
                # calculated by the norm of the difference between the two pixel's color vectors
                color_diff = (np.linalg.norm(img[i, j] - img[i + 1, j])) ** 2
                edge_weight = gamma * np.exp(-beta * color_diff)
                edges.append((node1, node2))
                weights.append(edge_weight)
                k = max(k, edge_weight)

            # Calculate edge weight for right neighbor
            if j < w - 1:
                node2 = img_indices[i, j + 1]
                # color difference between the current pixel and its top left neighbor
                # calculated by the norm of the difference between the two pixel's color vectors
                color_diff = (np.linalg.norm(img[i, j] - img[i, j + 1])) ** 2
                edge_weight = gamma * np.exp(-beta * color_diff)
                edges.append((node1, node2))
                weights.append(edge_weight)
                k = max(k, edge_weight)

            # calculate edge weight for top left neighbor
            if i > 0 and j > 0:
                node2 = img_indices[i - 1, j - 1]
                # color difference between the current pixel and its top left neighbor
                # calculated by the norm of the difference between the two pixel's color vectors
                color_diff = (np.linalg.norm(img[i, j] - img[i - 1, j - 1])) ** 2
                # dividing by sqrt(2) because distance between the pixels is sqrt(2) in case of diagonal neighbors
                edge_weight = (1 / np.sqrt(2)) * gamma * np.exp(-beta * color_diff)
                edges.append((node1, node2))
                weights.append(edge_weight)
                k = max(k, edge_weight)

            # calculate edge weight for top right neighbor
            if i > 0 and j < w - 1:
                node2 = img_indices[i - 1, j + 1]
                # color difference between the current pixel and its top left neighbor
                # calculated by the norm of the difference between the two pixel's color vectors
                color_diff = (np.linalg.norm(img[i, j] - img[i - 1, j + 1])) ** 2
                # dividing by sqrt(2) because distance between the pixels is sqrt(2) in case of diagonal neighbors
                edge_weight = (1 / np.sqrt(2)) * gamma * np.exp(-beta * color_diff)
                edges.append((node1, node2))
                weights.append(edge_weight)
                k = max(k, edge_weight)

            # calculate edge weight for bottom left neighbor
            if i < h - 1 and j > 0:
                node2 = img_indices[i + 1, j - 1]
                # color difference between the current pixel and its top left neighbor
                # calculated by the norm of the difference between the two pixel's color vectors
                color_diff = (np.linalg.norm(img[i, j] - img[i + 1, j - 1])) ** 2
                # dividing by sqrt(2) because distance between the pixels is sqrt(2) in case of diagonal neighbors
                edge_weight = (1 / np.sqrt(2)) * gamma * np.exp(-beta * color_diff)
                edges.append((node1, node2))
                weights.append(edge_weight)
                k = max(k, edge_weight)

            # calculate edge weight for bottom right neighbor
            if i < h - 1 and j < w - 1:
                node2 = img_indices[i + 1, j + 1]
                # color difference between the current pixel and its top left neighbor
                # calculated by the norm of the difference between the two pixel's color vectors
                color_diff = (np.linalg.norm(img[i, j] - img[i + 1, j + 1])) ** 2
                # dividing by sqrt(2) because distance between the pixels is sqrt(2) in case of diagonal neighbors
                edge_weight = (1 / np.sqrt(2)) * gamma * np.exp(-beta * color_diff)
                edges.append((node1, node2))
                weights.append(edge_weight)
                k = max(k, edge_weight)

            # get the value of Dback for the current pixel
            Dback = data_term[i, j, 0]
            # get the value of Dfore for the current pixel
            Dfore = data_term[i, j, 1]
            edges.append((source, node1))
            edges.append((node1, sink))

            # check if the current pixel is a background pixel or a foreground pixel using the mask
            if mask[i, j] == GC_PR_FGD or mask[i, j] == GC_PR_BGD:
                weights.append(Dfore)
                weights.append(Dback)
            if mask[i, j] == GC_FGD:
                weights.append(0)
                weights.append(k)
            if mask[i, j] == GC_BGD:
                weights.append(k)
                weights.append(0)

    graph.add_edges(edges)
    graph.es["weight"] = weights
    print("number of edges: ", len(graph.es))

    mincut = graph.st_mincut(source, sink)
    mincut_sets = [set(mincut.partition[0]), set(mincut.partition[1])]
    print("mincut source set size: ", len(mincut_sets[0]))
    print("mincut sink set size: ", len(mincut_sets[1]))

    return mincut_sets, mincut.value


"""
2.4 update mask(img, mask, mincut sets):
In this method, you need to update the current mask based on the mincut and
return a new mask.
"""


def update_mask(mincut_sets, mask):
    h, w = mask.shape
    img_indices = np.arange(h * w).reshape(h, w)
    new_mask = np.copy(mask)
    for i in range(h):
        for j in range(w):
            if img_indices[i, j] in mincut_sets[0] and (mask[i, j] == GC_PR_FGD or mask[i, j] == GC_PR_BGD):
                new_mask[i, j] = GC_PR_BGD
            elif img_indices[i, j] in mincut_sets[1] and (mask[i, j] == GC_PR_FGD or mask[i, j] == GC_PR_BGD):
                new_mask[i, j] = GC_PR_FGD

    mask = np.copy(new_mask)

    return mask


"""
2.5 check convergence(energy):
In this method, you need to check whether the energy value has converged to
a stable minimum or not. You can use a threshold value to determine whether
the energy has converged or not. If the energy has converged, you can return
True, otherwise return False.
"""


def check_convergence(energy):
    # TODO: implement convergence check
    convergence = False
    if energy < 0.1:
        convergence = True
    return convergence


"""
2.6 cal metric(mask, gt mask):
In this method you will evaluate your segmentation results. Given two binary
images, the method will return a tuple of the accuracy (the number of pixels
that are correctly labeled divided by the total number of pixels in the image)
and the Jaccard similarity (the intersection over the union of your predicted
foreground region with the ground truth).
"""


def cal_metric(predicted_mask, gt_mask):
    # Convert masks to boolean arrays
    predicted_mask_bool = predicted_mask.astype(bool)
    gt_mask_bool = gt_mask.astype(bool)

    # Calculate the number of correctly labeled pixels
    correct_pixels = np.sum(predicted_mask_bool == gt_mask_bool)
    total_pixels = predicted_mask.size

    # Calculate the accuracy
    accuracy = correct_pixels / total_pixels

    # Calculate the Jaccard similarity
    intersection = np.sum(predicted_mask_bool & gt_mask_bool)
    union = np.sum(predicted_mask_bool | gt_mask_bool)
    jaccard_similarity = intersection / union

    return accuracy, jaccard_similarity


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='banana1', help='name of image from the course files')
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
