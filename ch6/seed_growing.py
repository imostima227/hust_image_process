import cv2
import numpy as np



# 种子点生长算法
def region_growing(image, seed_point, seed_gray, threshold):
    # 初始化结果图像
    result_image = np.zeros_like(image)
    queue = [seed_point]
    
    while queue:
        x, y = queue.pop(0)
        
        if result_image[y, x] == 0 and abs(image[y, x] - seed_gray) <= threshold:
            result_image[y, x] = 255
            
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < image.shape[1] and 0 <= new_y < image.shape[0]:
                    queue.append((new_x, new_y))
    
    return result_image


if __name__ == "__main__":
    # 读取灰度图像
    image = cv2.imread('common/image1_grey.jpg', cv2.IMREAD_GRAYSCALE)

    # 定义种子点的坐标
    image_shape = image.shape
    seed_point = (int(image_shape[0]/2), int(image_shape[1]/2))


    # 初始化种子点灰度值
    seed_gray = image[seed_point[1], seed_point[0]]

    # 定义阈值范围
    threshold = 100

    # 定义生长方向
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    # 运行种子点生长算法
    result_image = region_growing(image,  seed_point, seed_gray, threshold)

    # 显示结果
    cv2.imshow('Result Image', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
