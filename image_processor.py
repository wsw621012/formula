import cv2
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
    def most_rgb(im):

        #im.thumbnail((84, 84))
        total_color = 0
        calc = {}

        for count, (r, g, b) in im.getcolors(im.size[0] * im.size[1]):
            rgb = r * 65536 + g * 256 + b
            #if rgb < 16777215:
            calc[rgb] = count
            total_color += count
        return calc, total_color
        #most = max(calc.items(), key=operator.itemgetter(1))[0]
        #print("most rgb = %d(%d) in %d" % (most, calc[most], total_color))
        #return most, calc[most] / float(total_color)

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
        filename = "%s-%s%s.jpg" % (prefix, datetime.now().strftime('%Y%m%d-%H%M%S-%f'), suffix)
        cv2.imwrite(os.path.join(folder, filename), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


    @staticmethod
    def rad2deg(radius):
        return radius / np.pi * 180.0


    @staticmethod
    def deg2rad(degree):
        return degree / 180.0 * np.pi


    @staticmethod
    def bgr2rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    @staticmethod
    def _normalize_brightness(img):
        maximum = img.max()
        if maximum == 0:
            return img
        adjustment = min(255.0/img.max(), 3.0)
        normalized = np.clip(img * adjustment, 0, 255)
        normalized = np.array(normalized, dtype=np.uint8)
        return normalized


    @staticmethod
    def _flatten_rgb(img):
        r, g, b = cv2.split(img)
        r_filter = (r == np.maximum(np.maximum(r, g), b)) & (r >= 120) & (g < 150) & (b < 150)
        g_filter = (g == np.maximum(np.maximum(r, g), b)) & (g >= 120) & (r < 150) & (b < 150)
        b_filter = (b == np.maximum(np.maximum(r, g), b)) & (b >= 120) & (r < 150) & (g < 150)
        y_filter = ((r >= 128) & (g >= 128) & (b < 100))

        r[y_filter], g[y_filter] = 255, 255
        b[np.invert(y_filter)] = 0

        b[b_filter], b[np.invert(b_filter)] = 255, 0
        r[r_filter], r[np.invert(r_filter)] = 255, 0
        g[g_filter], g[np.invert(g_filter)] = 255, 0

        flattened = cv2.merge((r, g, b))
        return flattened


    @staticmethod
    def _crop_image(img):
        bottom_half_ratios = (0.55, 1.0)
        bottom_half_slice  = slice(*(int(x * img.shape[0]) for x in bottom_half_ratios))
        bottom_half        = img[bottom_half_slice, :, :]
        return bottom_half


    @staticmethod
    def preprocess(b64_raw_img):
        img = cv2.cvtColor(np.asarray(Image.open(BytesIO(base64.b64decode(b64_raw_img)))), cv2.COLOR_BGR2RGB)
        return ImageProcessor._flatten_rgb(ImageProcessor._crop_image(img))

    @staticmethod
    def draw_lines(img):
        blurred = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        edges   = cv2.Canny(blurred, 30, 50, apertureSize=3)

        lines = cv2.HoughLinesP(edges, 6, np.pi/60, 100, 5, 5)
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(img, (x1,y1), (x2,y2), (255, 255, 255), 2)

        return img

    @staticmethod
    def wall_detection (sr, sg, sb):
        black_count = 0
        yellow_count = 0
        for i in range(len(sr) / 10):
            for j in range(len(sr[i]) / 10):
                inverse_i = len(sr)/8 + i
                inverse_j = len(sr[i]) - 1 - j

                if sr[i][j] == 0 and sg[i][j] == 0 and sb[i][j] == 0:
                    black_count += 1
                if sr[inverse_i][inverse_j] == 0 and sg[inverse_i][inverse_j] == 0 and sb[inverse_i][inverse_j] == 0:
                    yellow_count += 1

        is_left_wall=False
        is_right_wall=False

        if black_count>=320:
            is_left_wall = True
        elif yellow_count>=40:
            if yellow_count>=40:
                is_right_wall=True
        return is_left_wall, is_right_wall

    @staticmethod
    def find_steering_angle_by_color(img):
        r, g, b      = cv2.split(img)
        image_height = img.shape[0]
        image_width  = img.shape[1]
        camera_x     = image_width / 2
        image_sample = slice(0, int(image_height * 0.2))
        sr, sg, sb   = r[image_sample, :], g[image_sample, :], b[image_sample, :]
        track_list   = [sr, sg, sb]
        tracks       = map(lambda x: len(x[x > 20]), [sr, sg, sb])
        tracks_seen  = list(filter(lambda y: y > 50, tracks))

        if len(tracks_seen) == 0:
            return 0.0

        maximum_color_idx = np.argmax(tracks, axis=None)
        _target = track_list[maximum_color_idx]
        _y, _x = np.where(_target == 255)

        px = 0
        if _x is not None and len(_x) > 0:
            px = np.mean(_x)
        steering_angle = math.atan2(image_height, (px - camera_x))
        return (np.pi/2 - steering_angle) * 2.0
