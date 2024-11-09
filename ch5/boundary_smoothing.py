import numpy as np
import cv2
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

# 生成高斯低通滤波器
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

if __name__ == "__main__":
    image = cv2.imread('common/image1_grey.jpg', cv2.IMREAD_GRAYSCALE) # 目前只实现了读取灰度图
    # 直方图平滑
    image_filtered = smooth_histogram(image)

    # 目标边界平滑
    cutoff_frequency = 100
    edges = cv2.Canny(image,100,200)
    image_filtered = smooth_histogram(edges)
    filter = gaussian_lowpass_filter(edges.shape, cutoff_frequency)
    # image_filtered = fft_filtering(edges,filter)
    # print(edges.shape)
    edges_filtered = fft_filtering(edges, filter)

    # 显示原始图像和滤波后的图像
    cv2.imshow('Original Image', edges)
    cv2.imshow('Filtered Image', edges_filtered)
    cv2.waitKey(0)
    cv2.destroyAllWindows()