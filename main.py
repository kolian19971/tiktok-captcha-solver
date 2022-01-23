import cv2 as cv
# import numpy as np
# from matplotlib import pyplot as plt

# 1. Соответствие шаблонов
img = cv.imread('captcha/10/full.jpeg')
template = cv.imread('captcha/10/figure.png')

# красим в серый и контурим
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = cv.GaussianBlur(img, (3, 3), 0)
img = cv.Canny(img, 10, 250)

# красим в серый и контурим
template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
template = cv.GaussianBlur(template, (3, 3), 0)
template = cv.Canny(template, 10, 250)

h, w = template.shape[:2]  # rows->h, cols->w

# 6 методы сопоставления
# methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
#            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

meth = 'cv.TM_CCOEFF'

img2 = img.copy()

# Истинное значение метода сопоставления
method = eval(meth)
res = cv.matchTemplate(img, template, method)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

# Если это соответствие квадратичной разности TM_SQDIFF или нормализованная квадратная разность совпадения TM_SQDIFF_NORMED, возьмите минимальное значение
if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
    top_left = min_loc
else:
    top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

# Рисуем прямоугольник
# cv.rectangle(img2, top_left, bottom_right, 255, 2)

# выводим средние значения
print('x center')
xCenter = (top_left[0] + bottom_right[0]) / 2
print(xCenter)

print('y_center')
yCenter = (top_left[1] + bottom_right[1]) / 2
print(yCenter)

# plt.subplot(121), plt.imshow(res, cmap='gray')
# plt.xticks([]), plt.yticks([])  # Скрыть ось
# plt.subplot(122), plt.imshow(img2, cmap='gray')
# plt.xticks([]), plt.yticks([])
# plt.suptitle(meth)
# plt.show()
