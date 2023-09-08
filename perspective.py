import transform as transform
from scipy.spatial import distance as dist
from pylsd.lsd import lsd
import itertools
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math


class Perspective(object):
    def __int__(self, MIN_AREA_RATIO=.25, MAX_ANGLE_DIFF=45):
        self.MIN_AREA_RATIO = MIN_AREA_RATIO
        self.MAX_ANGLE_DIFF = MAX_ANGLE_DIFF

    def filter_corners(self, corners, min_dist=20):
        def check_distance(selected, corner):
            return all(dist.euclidean(sample, corner) >= min_dist
                       for sample in selected)
        filtered_corners = []
        for c in corners:
            if check_distance(filtered_corners, c):
                filtered_corners.append(c)
        return filtered_corners
    
    def angle_between_vectors_degrees(self, u, v):
        return np.degrees(math.acos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))

    def get_angle(self, p1, p2, p3):
        a = np.radians(np.array(p1))
        b = np.radians(np.array(p2))
        c = np.radians(np.array(p3))

        avec = a - b
        cvec = c - b

        return self.angle_between_vectors_degrees(avec, cvec)

    def angle_range(self, quad):
        tl, tr, br, bl = quad
        ura = self.get_angle(tl[0], tr[0], br[0])
        ula = self.get_angle(bl[0], tl[0], tr[0])
        lra = self.get_angle(tr[0], br[0], bl[0])
        lla = self.get_angle(br[0], bl[0], tl[0])

        angles = [ura, ula, lra, lla]
        return np.ptp(angles)

    def get_corners(self, img):
        """
        :input:  rescaled and canny filtered image (img)  
        :output: list of at most 10 potential corners as (x,y) tuple.
        """
        lines = lsd(img)
        corners = []
        if lines is not None:
            # separate out the horizontal and vertical lines, and draw them back onto separate canvases
            lines = lines.squeeze().astype(np.int32).tolist()
            horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            for line in lines:
                x1, y1, x2, y2, _ = line
                if abs(x2 - x1) > abs(y2 - y1):
                    (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[0])
                    cv2.line(horizontal_lines_canvas, (max(x1 - 5, 0), y1), (min(x2 + 5, img.shape[1] - 1), y2), 255, 2)
                else:
                    (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[1])
                    cv2.line(vertical_lines_canvas, (x1, max(y1 - 5, 0)), (x2, min(y2 + 5, img.shape[0] - 1)), 255, 2)
            lines = []

            # find the horizontal lines (connected-components -> bounding boxes -> final lines)
            (contours, hierarchy) = cv2.findContours(horizontal_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
            horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            for contour in contours:
                contour = contour.reshape((contour.shape[0], contour.shape[2]))
                min_x = np.amin(contour[:, 0], axis=0) + 2
                max_x = np.amax(contour[:, 0], axis=0) - 2
                left_y = int(np.average(contour[contour[:, 0] == min_x][:, 1]))
                right_y = int(np.average(contour[contour[:, 0] == max_x][:, 1]))
                lines.append((min_x, left_y, max_x, right_y))
                cv2.line(horizontal_lines_canvas, (min_x, left_y), (max_x, right_y), 1, 1)
                corners.append((min_x, left_y))
                corners.append((max_x, right_y))

            # find the vertical lines (connected-components -> bounding boxes -> final lines)
            (contours, hierarchy) = cv2.findContours(vertical_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
            vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            for contour in contours:
                contour = contour.reshape((contour.shape[0], contour.shape[2]))
                min_y = np.amin(contour[:, 1], axis=0) + 2
                max_y = np.amax(contour[:, 1], axis=0) - 2
                top_x = int(np.average(contour[contour[:, 1] == min_y][:, 0]))
                bottom_x = int(np.average(contour[contour[:, 1] == max_y][:, 0]))
                lines.append((top_x, min_y, bottom_x, max_y))
                cv2.line(vertical_lines_canvas, (top_x, min_y), (bottom_x, max_y), 1, 1)
                corners.append((top_x, min_y))
                corners.append((bottom_x, max_y))

            # find the corners
            corners_y, corners_x = np.where(horizontal_lines_canvas + vertical_lines_canvas == 2)
            corners += zip(corners_x, corners_y)

        # remove corners in close proximity
        corners = self.filter_corners(corners)
        return corners

    def is_valid_contour(self, cnt, IM_WIDTH, IM_HEIGHT):
        return (len(cnt) == 4 and cv2.contourArea(cnt) > IM_WIDTH * IM_HEIGHT * 0.25 and self.angle_range(cnt) < 45)

    def get_contour(self, rescaled_image, plot_fig=False):
        """
        :input:  rescaled image
        :output: numpy array of shape (4, 2) containing vertices of 4 corners of screen
        """
        MORPH, CANNY, HOUGH = 9, 84, 25                             # hyperparameters
        IM_HEIGHT, IM_WIDTH, _ = rescaled_image.shape
        gray = cv2.cvtColor(rescaled_image, cv2.COLOR_BGR2GRAY)     # BGR to Gray
        gray = cv2.GaussianBlur(gray, (7,7), 0)                     # Gaussian Blur
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(MORPH,MORPH))
        dilated = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)   # Closing
        edged = cv2.Canny(dilated, 0, CANNY)                        # Canny Edge Filter
        approx_contours = []

        # Method 1 : 
        test_corners = self.get_corners(edged)
        if len(test_corners)>=4:
            quads = []
            for quad in itertools.combinations(test_corners, 4):
                points = np.array(quad)
                points = transform.order_points(points)
                points = np.array([[p] for p in points], dtype = "int32")
                quads.append(points)
            quads = sorted(quads, key=cv2.contourArea, reverse=True)[:5]
            quads = sorted(quads, key=self.angle_range)
            approx = quads[0]
            if plot_fig:
                cv2.drawContours(rescaled_image, [approx], -1, (20, 20, 255), 2)
                plt.scatter(*zip(*test_corners))
                plt.imshow(rescaled_image)
                plt.show()
            if self.is_valid_contour(approx, IM_WIDTH, IM_HEIGHT):
                approx_contours.append(approx)

        (contours, hierarchy) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 80, True)
            if self.is_valid_contour(approx, IM_WIDTH, IM_HEIGHT):
                approx_contours.append(approx)
                break
        
        if not approx_contours:
            TOP_RIGHT = (IM_WIDTH, 0)
            BOTTOM_RIGHT = (IM_WIDTH, IM_HEIGHT)
            BOTTOM_LEFT = (0, IM_HEIGHT)
            TOP_LEFT = (0, 0)
            screenCnt = np.array([[TOP_RIGHT], [BOTTOM_RIGHT], [BOTTOM_LEFT], [TOP_LEFT]])
            success = False
        else:
            screenCnt = max(approx_contours, key=cv2.contourArea)
            success = True
            
        return screenCnt.reshape(4, 2), success

    def shift_perspective(self, image):
        RESCALED_HEIGHT = 500.0
        assert(image is not None)
        ratio = image.shape[0] / RESCALED_HEIGHT
        orig = image.copy()
        rescaled_image = transform.resize(image, height = int(RESCALED_HEIGHT))
        screen_cnt, success = self.get_contour(rescaled_image)
        if success:
            new_persp = transform.warp_persp(orig, screen_cnt * ratio)
        else:
            new_persp = orig[10:-10][10:-10]
        return new_persp

