import cv2

def rgb2grey(image, output):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f'{output}.jpg', gray_image)


if __name__ == "__main__":
    image = cv2.imread('common/image2.jpg',1)
    rgb2grey(image, "image2_grey")