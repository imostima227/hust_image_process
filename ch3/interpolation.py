import numpy as np 
 
def bilinear_interpolation(img, y, x):
    x1, y1 = int(np.floor(x)), int(np.floor(y))
    x2, y2 = min(x1 + 1, img.shape[1] - 1), min(y1 + 1, img.shape[0] - 1)

    Q11 = img[y1, x1]
    Q12 = img[y2, x1]
    Q21 = img[y1, x2]
    Q22 = img[y2, x2]

    x_weight = x - x1
    y_weight = y - y1

    top_interp = Q11 * (1 - x_weight) + Q21 * x_weight
    bottom_interp = Q12 * (1 - x_weight) + Q22 * x_weight

    final_interp = top_interp * (1 - y_weight) + bottom_interp * y_weight

    return final_interp
