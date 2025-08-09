import cv2
import numpy as np

def compute_histogram_intersection(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute the histogram intersection similarity score between two grayscale images.

    This function calculates the similarity between the grayscale intensity 
    distributions of two images by computing the intersection of their 
    normalized 256-bin histograms.

    The histogram intersection is defined as the sum of the minimum values 
    in each corresponding bin of the two normalized histograms. The result 
    ranges from 0.0 (no overlap) to 1.0 (identical histograms).

    Parameters:
        img1 (np.ndarray): First input image as a 2D NumPy array (grayscale).
        img2 (np.ndarray): Second input image as a 2D NumPy array (grayscale).

    Returns:
        float: Histogram intersection score in the range [0.0, 1.0].

    Raises:
        ValueError: If either input is not a 2D array (i.e., not grayscale).
    """    
    if img1.ndim != 2 or img2.ndim != 2:
        raise ValueError("Both input images must be 2D grayscale arrays.")

    ### START CODE HERE ###
    # Step 1: Compute histograms for both images (256 bins, range 0-255)
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    
    # Step 2: Normalize the histograms to sum to 1
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    
    # Step 3: Compute the intersection by summing the minimums
    intersection = np.sum(np.minimum(hist1, hist2))
    ### END CODE HERE ###

    return float(intersection)

# Código para carregar e comparar as imagens específicas
# if __name__ == "__main__":
#    img1 = cv2.imread("img/head.png", cv2.IMREAD_GRAYSCALE)
#    img2 = cv2.imread("img/head_filtered.png", cv2.IMREAD_GRAYSCALE)
#    similarity = compute_histogram_intersection(img1, img2)
#    print(f"Similaridade entre head.png e head_filtered.png: {similarity:.4f}")