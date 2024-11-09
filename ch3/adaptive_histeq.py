import numpy as np
import cv2
from scipy import interpolate
from scipy.ndimage import gaussian_filter
from histogram_equalization import my_histeq

def my_adaptive_histeq(image, block_size=90, overlap=15): # TODO: something wrong!
    h, w = image.shape[:2]
    equalized_image = np.zeros_like(image)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            hist, _ = np.histogram(block.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
            
            x = np.linspace(0, 255, num=256)
            interp_func = interpolate.interp1d(x, cdf_normalized, kind='linear')
            equalized_block = interp_func(block).reshape(block.shape).astype(np.uint8)
            # equalized_block = np.interp(block.flatten(), range(256), cdf_normalized).reshape(block.shape).astype(np.uint8)
            # equalized_block = my_histeq(block)

            # 边界平滑处理
            if i > 0 and j > 0:
                equalized_block[:overlap, :] = gaussian_filter(equalized_block[:overlap, :], sigma=2)
                equalized_block[:, :overlap] = gaussian_filter(equalized_block[:, :overlap], sigma=2)

            equalized_image[i:i+block_size, j:j+block_size] = equalized_block


            equalized_image[i:i+block_size, j:j+block_size] = equalized_block

    return equalized_image

if __name__ == "__main__":
    # 读取灰度图像
    image = cv2.imread('common/image1_grey.jpg', cv2.IMREAD_GRAYSCALE)

    # 进行直方图均衡化
    equalized_image = my_adaptive_histeq(image) 
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # equalized_image = clahe.apply(image)

    # 显示原始图像和直方图均衡化后的图像
    cv2.imshow('Original Image', image)
    cv2.imshow('Equalized Image', equalized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
