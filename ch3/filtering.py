# 包含线性滤波、中值滤波、最大值滤波
import sys
sys.path.append('./')
import cv2
import numpy as np
import matplotlib.pyplot as plt
from common.utils import salt_pepper_noise

def generate_gaussian_template(N, sigma = 1.0):
    x = np.arange(-N, N+1, 1)
    y = np.arange(-N, N+1, 1)
    x, y = np.meshgrid(x, y)

    H = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    H /= H.sum()  # normalize so the sum of all elements is 1

    return H

def generate_gaussian_first_derivative_template(N, sigma = 1.0):
    x = np.arange(-N, N+1, 1)
    y = np.arange(-N, N+1, 1)
    x, y = np.meshgrid(x, y)

    H_X = -(x / (2 * np.pi * sigma**4)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    H_Y = -(y / (2 * np.pi * sigma**4)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))

    # 高斯一阶导数滤波器（也称为高斯梯度滤波器）并不需要归一化，使其所有元素的和为1。这是因为这些滤波器的主要用途是计算图像的梯度，即图像的局部变化率，而不是平滑图像或保留图像的总体亮度。
    return H_X, H_Y

def my_filter(image, H):
    i_height, i_width = image.shape[:2]
    f_height, f_width = H.shape

    pad_height = f_height // 2
    pad_width = f_width // 2
    image_padded = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    
    dst_image = np.zeros_like(image)

    for i in range(i_height):
        for j in range(i_width):
            region = image_padded[i:i+f_height, j:j+f_width]
            dst_image[i,j] = np.sum(region * H)

    return dst_image


def median_filter(img, kernel_size):
    # 图像的高度和宽度
    h, w = img.shape[:2]

    # 边界扩展
    padding_size = kernel_size // 2
    img_padded = cv2.copyMakeBorder(img, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_REFLECT)

    # 创建一个空的输出图像
    output_img = np.zeros_like(img)

    # 对每一个像素进行中值滤波
    for y in range(h):
        for x in range(w):
            # 提取区域
            region = img_padded[y:y+kernel_size, x:x+kernel_size]
            # 找到中值
            median = np.median(region)
            # 将中值写入输出图像
            output_img[y, x] = median

    return output_img

def imnoise(img):
    noisy_img = salt_pepper_noise(img, 0.01, 0.01)
    return noisy_img

def dilate_filter(img, kernel_size):
    # 图像的高度和宽度
    h, w = img.shape[:2]

    # 边界扩展
    padding_size = kernel_size // 2
    img_padded = cv2.copyMakeBorder(img, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_REFLECT)

    # 创建一个空的输出图像
    output_img = np.zeros_like(img)

    # 对每一个像素进行最大值滤波
    for y in range(h):
        for x in range(w):
            # 提取区域
            region = img_padded[y:y+kernel_size, x:x+kernel_size]
            # 找到最大值
            max = np.max(region)
            # 将最大值值写入输出图像
            output_img[y, x] = max

    return output_img

def non_local_means(img, h=10, patch_size=5, search_window=21): # 非均值滤波 TODO:待验证
    offset = search_window // 2
    patch_offset = patch_size // 2
    rows, cols = img.shape
    padded_img = np.pad(img, ((offset, offset), (offset, offset)), 'symmetric')
    denoised_img = np.zeros_like(img, dtype=np.float64)

    for i in range(offset, offset + rows):
        for j in range(offset, offset + cols):
            i_s, i_e = i - patch_offset, i + patch_offset + 1
            j_s, j_e = j - patch_offset, j + patch_offset + 1
            patch = padded_img[i_s:i_e, j_s:j_e]

            r_s, r_e = i - offset, i + offset + 1
            c_s, c_e = j - offset, j + offset + 1
            region = padded_img[r_s:r_e, c_s:c_e]

            dists = np.array([np.sum((patch - region[k - patch_offset:k + patch_offset + 1, l - patch_offset:l + patch_offset + 1])**2) for k in range(patch_offset, patch_offset + 2 * offset) for l in range(patch_offset, patch_offset + 2 * offset)])
            weights = np.exp(-dists / h**2)
            weights /= np.sum(weights)

            patch_values = region[patch_offset:-patch_offset, patch_offset:-patch_offset].ravel()
            denoised_img[i - offset, j - offset] = np.dot(weights, patch_values)

    return denoised_img

def fft_filtering(image, filter):
    # 傅里叶变换
    # image = np.float32(image)
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    # 应用滤波器
    fshift_filtered = fshift * filter
    # fshift_filtered = fshift

    # 逆傅里叶变换
    f_filtered = np.fft.ifftshift(fshift_filtered)
    image_filtered = np.abs(np.fft.ifft2(f_filtered))
    image_filtered = cv2.convertScaleAbs(image_filtered) # 这个步骤非常关键，灰度图每个元素都是整数，不能是浮点数

    return image_filtered

# 生成高斯低通滤波器 TODO:有点想不明白为什么
def gaussian_lowpass_filter(shape, cutoff_freq):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2

    x = np.arange(cols) - ccol
    y = np.arange(rows) - crow
    xx, yy = np.meshgrid(x, y)
    radius = np.sqrt(xx ** 2 + yy ** 2)

    # 生成高斯低通滤波器
    filter = np.exp(-0.5 * (radius / cutoff_freq) ** 2)

    return filter

# 直方图平滑
def smooth_histogram(image):
    # 进行傅里叶变换
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    # 构建一个低通滤波器
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    D = 200  # 设定截止频率
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - D:crow + D, ccol - D:ccol + D] = 1

    # 应用滤波器
    fshift = fshift * mask

    # 反转频谱平移
    f_ishift = np.fft.ifftshift(fshift)

    # 进行逆傅里叶变换
    image_filtered = np.abs(np.fft.ifft2(f_ishift))

    # 将结果转换为 uint8 类型并归一化
    image_filtered = cv2.convertScaleAbs(image_filtered)
    cv2.normalize(image_filtered, image_filtered, 0, 255, cv2.NORM_MINMAX)

    return image_filtered


if __name__ == "__main__":
    # homework 4:
    H = generate_gaussian_template(200)
    image = cv2.imread('common/image1_grey.jpg', cv2.IMREAD_GRAYSCALE) # 目前只实现了读取灰度图
    J = cv2.filter2D(image, -1, H)
    # J = my_filter(image, H)
    # noisy_img = imnoise(image) # 添加噪声
    # J = median_filter(noisy_img, 3)
    # cv2.imshow('Original Image', image)
    # J = cv2.dilate(image, np.ones((5,5),np.uint8))
    # J = dilate_filter(image, 5)

    cv2.imshow('Origin Image', image)
    cv2.imshow('Filtered Image', J)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #############################################
    #homework 5:

    # image = cv2.imread('common/image1_grey.jpg', 0)
    # 设定高斯低通滤波器参数
    # cutoff_frequency = 100
    # filter = gaussian_lowpass_filter(image.shape, cutoff_frequency)

    # image_filtered = fft_filtering(image, filter)
    # image_filtered = cv2.GaussianBlur(image, (5, 5), 0)

    # 直方图平滑
    # image_filtered = smooth_histogram(image)

    # 目标边界平滑
    # edges = cv2.Canny(image,100,200)
    #image_filtered = smooth_histogram(edges)
    # filter = gaussian_lowpass_filter(edges.shape, cutoff_frequency)
    # image_filtered = fft_filtering(edges,filter)
    # print(edges.shape)
    # edges_filtered = fft_filtering(edges.shape, filter)

    # 显示原始图像和滤波后的图像
    # cv2.imshow('Original Image', edges)
    # cv2.imshow('Filtered Image', image_filtered)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    
   