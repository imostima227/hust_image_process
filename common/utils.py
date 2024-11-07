import cv2
import numpy as np

def rgb2grey(image, output):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f'{output}.jpg', gray_image)

def salt_pepper_noise(image, salt_prob, pepper_prob):
    # 椒盐噪声图像
    noisy_image = np.copy(image)

    # 盐噪声
    num_salt = np.ceil(salt_prob * image.size).astype(int)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    for coord in zip(*coords):
        noisy_image[coord] = 255

    # 胡椒噪声
    num_pepper = np.ceil(pepper_prob * image.size).astype(int)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    for coord in zip(*coords):
        noisy_image[coord] = 0

    return noisy_image


if __name__ == "__main__":
    image = cv2.imread('common/image2.jpg',1)
    rgb2grey(image, "image2_grey")