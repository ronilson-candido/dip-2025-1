import cv2 as cv
import numpy as np

def remove_salt_and_pepper_noise(image: np.ndarray) -> np.ndarray:
    """
    Removes salt and pepper noise from a grayscale image.

    Parameters:
        image (np.ndarray): Noisy input image (grayscale).

    Returns:
        np.ndarray: Denoised image.
    """
    # Apply median filter with 3x3 kernel size
    denoised_image = cv.medianBlur(image, 3)
    return denoised_image

if __name__ == "__main__":
    # Read the input image from the correct path
    noisy_image = cv.imread("img/head.png", cv.IMREAD_GRAYSCALE)
    
    # Check if image was loaded properly
    if noisy_image is None:
        raise FileNotFoundError("Could not load the image 'img/head.png'")
    
    # Remove noise
    denoised_image = remove_salt_and_pepper_noise(noisy_image)
    
    # Save the result
    cv.imwrite("img/head_filtered.png", denoised_image)