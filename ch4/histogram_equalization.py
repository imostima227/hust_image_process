import cv2
import numpy as np

def my_histeq(image):
    h, w = image.shape[:2]
    hist, _ = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())

    equalized_image = np.interp(image.flatten(), range(256), cdf_normalized).reshape(h, w).astype(np.uint8)
    
    return equalized_image

if __name__ == "__main__":
    # 读取灰度图像
    image = cv2.imread('common/image1_grey.jpg', cv2.IMREAD_GRAYSCALE)

    # 进行直方图均衡化
    equalized_image = my_histeq(image)
    # equalized_image = cv2.equalizeHist(image)

    # 显示原始图像和直方图均衡化后的图像
    cv2.imshow('Original Image', image)
    cv2.imshow('Equalized Image', equalized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
