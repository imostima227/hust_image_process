import cv2
import numpy as np

# 读取灰度图像
image = cv2.imread('common/image1_grey.jpg', cv2.IMREAD_GRAYSCALE)

# 计算像素值的累积分布函数
histogram, bins = np.histogram(image.flatten(), 256, [0,256]) 
cdf = histogram.cumsum()
cdf_normalized = cdf / cdf.max()

# 计算新图像中5%和95%像素值对应的累积概率
low_percentile = 0.05
high_percentile = 0.95
low_value = np.interp(low_percentile, cdf_normalized, bins[:-1]) # bins[:-1]表示0~255
high_value = np.interp(high_percentile, cdf_normalized, bins[:-1])

# 灰度分段线性变换
new_image = np.interp(image, [low_value, high_value], [0, 255])

# 显示新图像
cv2.imshow('Original Image', image)
cv2.imshow('New Image', new_image.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
