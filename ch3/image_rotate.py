import numpy as np
import cv2

from image_zoom import my_imresize
from interpolation import bilinear_interpolation

def my_imrotate(image, angle):
    # 获取图像尺寸
    h, w = image.shape[:2]
    
    # 转换角度为弧度
    theta = np.radians(angle)
    
    # 计算旋转后的图像尺寸
    new_w = int(abs(w * np.cos(theta)) + abs(h * np.sin(theta)))
    new_h = int(abs(w * np.sin(theta)) + abs(h * np.cos(theta)))

    # 计算新的旋转中心
    center = (w // 2, h // 2)
    center_n = (new_w // 2, new_h // 2)
    
    # 创建一个空白图像用于存储旋转后的图像
    rotated_image = np.zeros((new_h, new_w, 3), dtype=image.dtype)
    
    # 逆时针旋转图像
    for x in range(new_w):
        for y in range(new_h):
            # 计算原图像中对应的坐标
            orig_x = (x - center_n[0]) * np.cos(-theta) - (y - center_n[1]) * np.sin(-theta) + center[0]
            orig_y = (x - center_n[0]) * np.sin(-theta) + (y - center_n[1]) * np.cos(-theta) + center[1]
        
            # 检查原图像坐标是否在范围内
            if 0 <= orig_x < w and 0 <= orig_y < h:
                if isinstance(orig_x, int) and isinstance(orig_y, int):
                    rotated_image[y, x] = image[orig_y, orig_x]
                else:
                    rotated_image[y, x] = bilinear_interpolation(image, orig_y, orig_x)
    
    return rotated_image

if __name__ == "__main__":
    img = cv2.imread("image1.jpg",1)
    zoom_image = my_imresize(img, 0.5, 0.5)
    rotate_img = my_imrotate(zoom_image, 45)
    cv2.imshow("rotate_img", rotate_img)
    cv2.waitKey(0)