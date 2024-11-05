import numpy as np
import cv2

def histogram_matching(image, target_hist):
    h, w = image.shape
    hist, _ = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())

    target_cdf = np.cumsum(target_hist)
    target_cdf_normalized = (target_cdf - target_cdf.min()) * 255 / (target_cdf.max() - target_cdf.min())

    mapping = np.interp(cdf_normalized, target_cdf_normalized, range(256))
    matched_image = mapping[image].astype(np.uint8)

    return matched_image

if __name__ == "__main__":
    image = cv2.imread('common/image1_grey.jpg', cv2.IMREAD_GRAYSCALE)
    target_image = cv2.imread('common/image2_grey.jpg', cv2.IMREAD_GRAYSCALE)

    target_hist, _ = np.histogram(target_image.flatten(), 256, [0, 256])
    matched_image = histogram_matching(image, target_hist)
    # 显示原始图像和直方图均衡化后的图像
    cv2.imshow('Original Image', image)
    cv2.imshow('Equalized Image', matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()