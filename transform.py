import os
import numpy as np
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import cv2


def order_points(pts):
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    (tl, bl) = left_most[np.argsort(left_most[:, 1]), :]
    dist_tl_right = dist.cdist(tl[np.newaxis], right_most, "euclidean")[0]
    (br, tr) = right_most[np.argsort(dist_tl_right)[::-1], :]
    return np.array([tl, tr, br, bl], dtype="float32")

def warp_persp(img, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	dim = None
	(h, w) = image.shape[:2]
	if width is None and height is None:
		return image
	if width is None:
		r = height / float(h)
		dim = (int(w * r), height)
	else:
		r = width / float(w)
		dim = (width, int(h * r))
	resized = cv2.resize(image, dim, interpolation = inter)
	return resized

def reorient_img(basepath, filename, pts, plot=True, save=False, save_path='./'):
    img = cv2.imread(basepath+filename)
    new_im = warp_persp(img, pts)
    old_im = cv2.polylines(img, [pts], True, (255,0,0), 4)
    if plot:
        fig = plt.figure(figsize=(10, 7))
        fig.add_subplot(1, 2, 1)
        plt.imshow(old_im)
        fig.add_subplot(1, 2, 2)
        plt.imshow(new_im)
    if save:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        cv2.imwrite(save_path+filename, new_im)
