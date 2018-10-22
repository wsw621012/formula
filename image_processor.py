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
            if y is not None or len(y) > 0:
                color_rate = len(y) / total_size
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
    def find_wall_angle(target, debug = False):
        _y, _x = np.where(target == 0)
        if len(_y) < 20:
            return None, -1, -1, -1, -1
        if len(_y) == (target.shape[0] * target.shape[1]):
            return 180, 0, target.shape[0] - 1, target.shape[1] - 1, target.shape[0] - 1

        left_x = min(_x)
        left_y = max(_y[_x == left_x])

        right_x = max(_x)
        right_y = max(_y[_x == right_x])

        max_y_set = _x[_y >= max([left_y, right_y])]
        if len(max_y_set) > 20: # strange shape so use max-y to be rectangle
            left_y = left_x = max(_y)

        #if debug:
        #    print("wall: (%d, %d) ~ (%d, %d)" % (left_x, left_y, right_x, right_y))

        if right_x != left_x:
            m, _ = np.polyfit([left_x, right_x], [left_y, right_y], 1)
            if m == 0:
                angle = 180
            else:
                angle = math.degrees(math.atan(-1./m))
        else:
            #m, _ = np.polyfit([left_y, right_y], [left_x, right_x], 1)
            #angle = math.degrees(math.atan(-m))
            angle = 0

        if debug:
            print("angle = %.2f" % angle)

        return angle, left_x, left_y, right_x, right_y

    @staticmethod
    def test_red_angle(target, debug = False):
        if not (255 in target): # full red
            return 180
        if not (76 in target): # full white
            return 180

        _h, _w = target.shape

        l_angle = r_angle = 90
        #left-bound of red region
        _lx = np.argmin(target, axis = 1)
        _ly = np.arange(_h)[_lx != 0]
        for ly in np.split(_ly, np.where(np.diff(_ly) != 1)[0]+1):
            if len(ly) < 10:
                continue
            lx = _lx[ly]
            m, _ = np.polyfit(lx, ly, 1)
            print("left angle = %.2f" % math.degrees(math.atan(-1./m)))

        #right-bound of red region
        lr_target = np.fliplr(target)
        _rx = _w - np.argmin(lr_target, axis = 1) - 1
        _ry = np.arange(_h)[_rx != (_w - 1)]
        for ry in np.split(_ry, np.where(np.diff(_ry) != 1)[0]+1):
            if len(ry) < 10:
                continue
            rx = _rx[ry]
            m, _ = np.polyfit(rx, ry, 1)
            print("right angle = %.2f" % math.degrees(math.atan(-1./m)))

    @staticmethod
    def find_road_angle(target, debug = False):
        _y, _x = np.where(target == 76) # red-part
        if len(_y) == (target.shape[0] * target.shape[1]): # full-red
            return 180, 76

        if len(_y) == 0: # full-white
            return 180, 255

        width = target.shape[1]
        height = target.shape[0]

        if not (76 in target[height - 1, :]): # white in bottom
            line_x, line_y, none_x = [], [], []
            for x in range(target.shape[1]):
                y_set = _y[np.where(_x == x)]
                if len(y_set) > 0:
                    line_y.append(max(y_set))
                    line_x.append(x)
                else:
                    none_x.append(x)
            if len(none_x) == 0:
                m, _ = np.polyfit(line_x, line_y, 1)
                return math.degrees(math.atan(-1./m)), 255
            if min(none_x) > 0 and max(none_x) < (target.shape[1] - 1):
                px = (min(none_x) + max(none_x)) // 2
                m = math.atan2(px - (target.shape[1] // 2), target.shape[0])
                return math.degrees(m), 255
            #if min(none_x) == 0 or max(none_x) == (target.shape[1] - 1):
            else:
                m, _ = np.polyfit(line_x, line_y, 1)
                return math.degrees(math.atan(-1./m)), 255

        if not (255 in target[height - 1, :]): # red in bottom
            line_x, line_y, none_x = [], [], []
            for x in range(target.shape[1]):
                y_set = _y[np.where(_x == x)]
                # skip different red blocks
                if len(y_set) < target.shape[0] - min(y_set) - 20:
                    fake_min_y = min(y_set)
                    y_set = y_set[np.where(y_set > (fake_min_y + 5))]
                    if len(y_set) < target.shape[0] - min(y_set) - 20:
                        continue
                min_y = min(y_set)
                if min_y > 0:
                    line_y.append(min_y)
                    line_x.append(x)
                else:
                    none_x.append(x)
                    if debug:
                        print("x:%d, y:0" % x)
            if len(line_x) < 2:
                ImageProcessor.save_image('./frames', target, suffix = 'no_linex')
                return 180, 76

            if len(none_x) == 0:
                m, _ = np.polyfit(line_x, line_y, 1)
                if m == 0:
                    return 180, 76
                return math.degrees(math.atan(-1./m)), 76
            if min(none_x) > 0 and max(none_x) < (target.shape[1] - 1):
                px = (min(none_x) + max(none_x)) // 2
                m = math.atan2(px - (target.shape[1] // 2), target.shape[0])
                return math.degrees(m), 76
            #if min(none_x) == 0 or max(none_x) == (target.shape[1] - 1):
            else:
                m, _ = np.polyfit(line_x, line_y, 1)
                return math.degrees(math.atan(-1./m)), 76

        return None, target[height - 1, width // 2]

    @staticmethod
    def find_red_angle(im_gray, debug = False):
        middle_img   = ImageProcessor._crop_gray(im_gray, 0.6, 0.8)
        image_height = middle_img.shape[0]
        image_width  = middle_img.shape[1]
        camera_x     = image_width / 2

        # _y, _x = np.where(middle_img == 76)
        _y, _x = np.where(middle_img == 255)

        px = 0.0
        if _x is not None and len(_x) > 0:
            px = np.mean(_x)
        if np.isnan(px):
            return 0.0

        steering_radian = math.atan2(px - camera_x, 3 * image_height)

        if debug:
            #draw the steering direction
            print("red angle px = %.2f" % px)
            cv2.rectangle(im_gray, (int(px) - 10, (im_gray.shape[0] * 6) // 10), (int(px) + 10, (im_gray.shape[0] * 8) // 10), 0, 2)
            r = im_gray.shape[0]
            x = im_gray.shape[1] // 2 + int(r * math.sin(steering_radian))
            y = im_gray.shape[0]    - int(r * math.cos(steering_radian))
            cv2.line(im_gray, (im_gray.shape[1] // 2, im_gray.shape[0] - 1), (x, y), 0, 2)

        return math.degrees(steering_radian)
