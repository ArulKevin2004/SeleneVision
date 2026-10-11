import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern


IMG_SIZE = (128, 128)


def preprocess_array(image, use_clahe=True, use_blur=True):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, IMG_SIZE)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)

    if use_blur:
        image = cv2.GaussianBlur(image, (3, 3), 0)

    return image


def preprocess_image(image_path, use_clahe=True, use_blur=True):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f'Could not read image: {image_path}')
    return preprocess_array(image, use_clahe, use_blur)


def extract_hog(image):
    return hog(
        image,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        feature_vector=True,
    )


def extract_lbp(image):
    radius = 2
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    n_bins = n_points + 2
    hist, _ = np.histogram(
        lbp.ravel(), bins=np.arange(0, n_bins + 1), range=(0, n_bins)
    )
    hist = hist.astype('float32')
    hist /= hist.sum() + 1e-7
    return hist


def extract_features(image_path, feature_type='hog', use_clahe=True, use_blur=True):
    image = preprocess_image(image_path, use_clahe, use_blur)

    if feature_type == 'hog':
        return extract_hog(image)
    if feature_type == 'lbp':
        return extract_lbp(image)
    if feature_type == 'hog_lbp':
        return np.concatenate([extract_hog(image), extract_lbp(image)])

    raise ValueError("feature_type must be 'hog', 'lbp', or 'hog_lbp'")
