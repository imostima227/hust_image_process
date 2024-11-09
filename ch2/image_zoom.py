# 设一幅大小为M×N的灰度图像I中，现要变成（放大或缩小）为 P×Q的图像J，请写出J的生成算法。
# 【参考函数：imresize】
import cv2
import numpy as np
from interpolation import bilinear_interpolation

def my_imresize(img, zoom_h, zoom_w):
	'''
	图像缩小—等间隔采样
	:param img: 原图
	:param zoom_h: 行缩放比例
	:param zoom_w: 列缩放比例
	:return:
	'''
	img_h, img_w ,channels = img.shape    # 获取图像大小
	new_h, new_w = int(img_h*zoom_h), int(img_w*zoom_w)  # 缩放后的图片大小
	zoom_img = np.zeros((new_h,new_w,channels),dtype= np.uint8)  # 创建缩放后的图片大小的矩阵


	for i in range(new_h):
		for j in range(new_w):
				i_ori = i / zoom_h
				j_ori = j / zoom_w
				zoom_img[i, j] = bilinear_interpolation(img, i_ori, j_ori)
	return zoom_img

if __name__ == "__main__":
	img = cv2.imread("image1.jpg",1)
	zoom_h = 1.5
	zoom_w = 1.5
	zoom_img = my_imresize(img, zoom_h, zoom_w)
	cv2.imshow("zoom_img", zoom_img)
	cv2.waitKey(0)