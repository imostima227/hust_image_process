import cv2
import numpy as np

# 读取图像
image = cv2.imread('cell_image.jpg', 0)

# 自适应直方图均衡化
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(image)

# 中值滤波
median_filtered = cv2.medianBlur(clahe_image, 5)

# 边缘增强
sobel_x = cv2.Sobel(median_filtered, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(median_filtered, cv2.CV_64F, 0, 1, ksize=3)
sobel_image = np.sqrt(sobel_x**2 + sobel_y**2)

# 显示增强后的图像
cv2.imshow('Enhanced Image', sobel_image.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
