# -*- coding: utf-8 -*-
import cv2
import numpy as np


def create_system(d, b, a):
    c = np.sin(d + a) * np.sin(b + a) / (np.sin(d + a + b) * np.sin(a))
    t = np.sin(a + b) / np.sin(d + a + b)
    k = np.sin(a + b) / np.sin(b)

    def _y(x):
        return t * x / (c * (1 - x) + x)

    def _det_y(x):
        tmp = c * (1 - x) + x
        return t * c / (tmp * tmp)

    def _dis_y(x):
        y = _y(x)
        return np.sqrt(y * y + k * k - 2 * y * k * np.cos(d + a))

    return _y, _dis_y, _det_y


def solve_system(d, b, a, x_scale, x1, x2, dis):
    x_scale = float(x_scale)
    _y, _dis_y, _det_y = create_system(d, b, a)
    y1, y2 = _y(x1 / x_scale), _y(x2 / x_scale)
    scale = dis / abs(y1 - y2)

    def __y(x):
        return scale * _y(x / x_scale)#通过像素的x坐标，计算实际坐标系中的y的坐标

    def __dis_y(x):
        return scale * _dis_y(x / x_scale)

    def __det_y(x):
        return scale * _det_y(x / x_scale)

    return __y, __dis_y, __det_y


def find_red_ball(img):
    kernel_4 = np.ones((4, 4), np.uint8)  # 4x4的卷积核
    #frame = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯模糊
    cv2.convertScaleAbs(img, img, alpha=1.0, beta=0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 创建mask
    mask = cv2.inRange(hsv, np.array([166, 100,97]), np.array([180, 255, 255]))# 不增强 160  ,100,  50 下午
    mask2 = cv2.inRange(hsv, np.array([0, 100, 97]), np.array([14, 255, 255]))#20  ,100,  50  下午 看不到后面的棕色
    #晚上开灯,增强对比 14 100  100  滤掉棕色
    mask = cv2.bitwise_or(mask, mask2)

    # 后处理mask
    erosion = cv2.erode(mask, kernel_4, iterations=2)
    dilation = cv2.dilate(erosion, kernel_4, iterations=2)

    # cv2.imshow('red', dilation)
    # cv2.waitKey(1)

    # 寻找轮廓
    v = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(v) == 2:
        contours, hierarchy = v
    elif len(v) == 3:
        _, contours, hierarchy = v
    else:
        return None, None, None, None

    area = []  # List[int]
    # 找到最大的轮廓
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    if len(area) == 0:
        return None, None, None, None
    max_idx = int(np.argmax(np.array(area)))
    c = np.mean(contours[max_idx], axis=0)
    dis = np.empty(shape=len(contours[max_idx]))
    for i in range(len(dis)):
        dis[i] = np.linalg.norm(c[0] - contours[max_idx][i])
    # print np.std(dis)
    print('check red  ball , get area:%s,std%s'%(area[max_idx],np.std(dis)))
    if area[max_idx] < 1800 or np.std(dis) > 10:  #800 10
        #print('check red  but ball dont get area:%s,std%s'%(area[max_idx],np.std(dis)))
        return None, None, None, None
    x, y, w, h = cv2.boundingRect(contours[max_idx])
    return x, y, w, h

