import cv2, os
import math
import numpy as np
import base64
import logging

from PIL  import Image
from io   import BytesIO
import base64
import math

def logit(msg):
    print("%s" % msg)

class ImageProcessor(object):
    @staticmethod
    def show_image(img, name = "image", scale = 1.0):
        if scale and scale != 1.0:
            img = cv2.resize(img, newsize, interpolation=cv2.INTER_CUBIC)

        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(name, img)
        cv2.waitKey(1)

    @staticmethod
    def save_image(folder, img, prefix = "img", suffix = ""):
        from datetime import datetime
        filename = "%s-%s-%s.jpg" % (prefix, datetime.now().strftime('%Y%m%d-%H%M%S-%f'), suffix)
        cv2.imwrite(os.path.join(folder, filename), img)

    @staticmethod
    def _color_rate(img_gray):
        total_size = (img_gray.shape[0] * img_gray.shape[1])
        total_rate = []
        for color in [0, 76, 255]:
            y, _ = np.where(img_gray == color)
            color_rate = 0
            if y is not None and y.size > 0:
                color_rate = y.size / total_size
            total_rate.append(color_rate)
        return total_rate[0], total_rate[1], total_rate[2]

    @staticmethod
    def _crop_gray(img, beg, end):
        crop_ratios = (beg, end)
        crop_slice  = slice(*(int(x * img.shape[0]) for x in crop_ratios))
        return img[crop_slice, :]

    @staticmethod
    def _flatten_rgb_to_gray(img):
        r, g, b = cv2.split(img)
        r_filter = (r == np.maximum(np.maximum(r, g), b)) & (r >= 120) & (g < 150) & (b < 150)
        g_filter = (g == np.maximum(np.maximum(r, g), b)) & (g >= 120) & (r < 150) & (b < 150)
        b_filter = (b == np.maximum(np.maximum(r, g), b)) & (b >= 120) & (r < 150) & (g < 150)
        black = ((r<50) & (g<50)& (b<50))

        b[b_filter], b[np.invert(b_filter)] = 255, 0
        r[r_filter], r[np.invert(r_filter)] = 255, 0
        g[g_filter], g[np.invert(g_filter)] = 255, 0
        b[black], g[black], r[black] = 0, 255, 255

        flattened = cv2.merge((r, g, b))
        img_gray = cv2.cvtColor(flattened, cv2.COLOR_BGR2GRAY)
        img_gray[(np.where(img_gray == 0))] = 255
        img_gray[(np.where(img_gray == 179))] = 0
        img_gray[(np.where(img_gray == 29))] = 255

        return img_gray

    @staticmethod
    def preprocess(b64_raw_img):
        image = np.asarray(Image.open(BytesIO(base64.b64decode(b64_raw_img))))
        # brg to rgb
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def find_final_line(target):
        if not (0 in target[-50:-1, :]):
            return False

        ly = np.argmin(target, axis=0)
        lx = np.arange(target.shape[1])[ly != 0]
        #if lx.size < (target.shape[1] - 1):
            #print("black line size: %d" % lx.size)
        #    return False
        if lx.size < target.shape[1] // 2:
            return False

        ly = ly[lx]
        count = 0
        for i in np.arange(lx.size):
            if target[ly[i], lx[i]] == 0:
                count += 1
        if count > target.shape[1] // 2:
            #print("final line reached!!")
            return True

        #print('black line is too small: %d' % count)
        return False

    @staticmethod
    def find_wall_angle(target, debug = False):
        b, r, w = ImageProcessor._color_rate(target)
        if b == 0:
            return None, -1, -1, -1, -1
        if r == 0 and w == 0:
            return 180, 0, target.shape[0] - 1, target.shape[1] - 1, target.shape[0] - 1

        _y, _x = np.where(target == 0)
        if min(_y) > 0: # not wall
            return None, -1, -1, -1, -1

        left_x = min(_x)
        left_y = max(_y[_x == left_x])

        right_x = max(_x)
        right_y = max(_y[_x == right_x])

        if b < r and b < w:
            # valid road is horizontal
            if not 76 in target[0,:]:
                _w = target.shape[1]
                target[target == 0] = 255
                ly = np.argmin(target, axis=0)
                lx = np.arange(target.shape[1])[ly != 0]
                if lx.size < 10:
                    return 0, left_x, left_y, right_x, right_y
                ly = ly[lx]
                m, _ = np.polyfit(lx, ly, 1)
                if (m >= 0) and (m < 0.1):
                    return 89, left_x, left_y, right_x, right_y
                if (m <= 0) and (m > -0.1):
                    return -89, left_x, left_y, right_x, right_y
                angle = math.degrees(math.atan(-1./m))
                return angle, left_x, left_y, right_x, right_y

        max_y_set = _x[_y >= max([left_y, right_y])]
        if max_y_set.size > 10: # strange shape so use max-y to be rectangle
            left_y = left_x = max(_y)

        #if debug:
        #    print("wall: (%d, %d) ~ (%d, %d)" % (left_x, left_y, right_x, right_y))
        if right_x == left_x:
            return 0., left_x, left_y, right_x, right_y
        if right_y == left_y:
            return 90.,  left_x, left_y, right_x, right_y

        m, _ = np.polyfit([left_x, right_x], [left_y, right_y], 1)
        angle = math.degrees(math.atan(-1./m))

        if debug:
            print("angle = %.2f" % angle)

        return angle, left_x, left_y, right_x, right_y

    @staticmethod
    def test_red_angle(target, debug = False):
        black, red, white = ImageProcessor._color_rate(target)
        if black == 0 and white == 0: # full red
            return 0.
        if black == 0 and red == 0: # full white
            return 0.

        _h, _w = target.shape

        l_angle = r_angle = 90
        #left-bound of red region
        _lx = np.argmin(target, axis = 1)
        _ly = np.arange(_h)[_lx != 0]
        for ly in np.split(_ly, np.where(np.diff(_ly) != 1)[0]+1):
            if ly.size < 10:
                continue
            lx = _lx[ly]
            a, b = np.polyfit(lx, ly, 1) # y = ax + b => (0, b) ~ (-b/a, 0)
            if debug:
                print("left-angle : %.2f" % math.degrees(math.atan(-1./a)))
                if a < 0:
                    cv2.line(target, (0, int(b)), (int(-b / a), 0), 0, 2)
                else:
                    cv2.line(target, (_w-1, int(a*(_w-1)+b)), (int(-b / a), 0), 0, 2)
        #right-bound of red region
        lr_target = np.fliplr(target)
        _rx = np.argmin(lr_target, axis = 1)
        _ry = np.arange(_h)[_rx != 0]
        for ry in np.split(_ry, np.where(np.diff(_ry) != 1)[0]+1):
            if ry.size < 10:
                continue
            rx = _w - 1 - _rx[ry]
            a, b = np.polyfit(rx, ry, 1)
            if debug:
                print("right-angle : %.2f" % math.degrees(math.atan(-1./a)))
                if a < 0:
                    cv2.line(target, (0, int(b)), (int(-b / a), 0), 0, 2)
                else:
                    cv2.line(target, (_w-1, int(a*(_w-1)+b)), (int(-b / a), 0), 0, 2)
