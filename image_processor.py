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
        filename = "%s-%s-%s.jpg" % (prefix, datetime.now().strftime('%Y%m%d-%H%M%S-%f'), suffix)
        cv2.imwrite(os.path.join(folder, filename), img)


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
    def flatten_rgb(img):
        b, g, r = cv2.split(img)
        r_filter = (r == np.maximum(np.maximum(r, g), b)) & (r >= 120) & (g < 150) & (b < 150)
        g_filter = (g == np.maximum(np.maximum(r, g), b)) & (g >= 120) & (r < 150) & (b < 150)
        b_filter = (b == np.maximum(np.maximum(r, g), b)) & (b >= 120) & (r < 150) & (g < 150)
        y_filter = ((r >= 128) & (g >= 128) & (b < 100))

        r[y_filter], g[y_filter] = 255, 255
        b[np.invert(y_filter)] = 0

        b[b_filter], b[np.invert(b_filter)] = 255, 0
        r[r_filter], r[np.invert(r_filter)] = 255, 0
        g[g_filter], g[np.invert(g_filter)] = 255, 0

        flattened = cv2.merge((b, g, r))
        return flattened

    @staticmethod
    def preprocess(b64_raw_img):
        image = np.asarray(Image.open(BytesIO(base64.b64decode(b64_raw_img))))
        # brg to rgb
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
    def _centre_channel(b, g, r):
        x, y = (b.shape[1] - 1 )// 2, b.shape[0] - 1

        r_filter = (r == np.maximum(np.maximum(r, g), b)) & (r >= 120) & (g < 150) & (b < 150)
        g_filter = (g == np.maximum(np.maximum(r, g), b)) & (g >= 120) & (r < 150) & (b < 150)
        b_filter = (b == np.maximum(np.maximum(r, g), b)) & (b >= 120) & (r < 150) & (g < 150)
        y_filter = ((r >= 128) & (g >= 128) & (b < 100))

        r[y_filter], g[y_filter], b[np.invert(y_filter)]  = 255, 255, 0
        b[b_filter], b[np.invert(b_filter)] = 255, 0
        r[r_filter], r[np.invert(r_filter)] = 255, 0
        g[g_filter], g[np.invert(g_filter)] = 255, 0

        if r[y, x] == 0 and g[y, x] == 0 and b[y, x] == 0:
            return 'black', None

        if r[y, x] == 255 and g[y, x] == 255:
            if b[y, x] == 0:
                return 'yellow', None
            else:
                return 'white', None

        if g[y, x] == 255:
            return "green", g
        elif r[y, x] == 255:
            return "red", r
        else: #g[79, 159] == 255:
            return "blue", b

    @staticmethod
    def _find_road_angle(target):
        left_x, left_y = [], []
        right_x, right_y = [], []
        l_angle, r_angle = None, None
        for _y in range(target.shape[0]):
            y = target.shape[0] - 1 -_y
            color_beg = False
            for x in range(target.shape[1]):
                if target[y, x] == 255 and color_beg == False:
                    if len(left_x) == 0 or abs(x - left_x[-1]) < 5:
                        color_beg = True
                        if x > 0:
                            left_y.append(y)
                            left_x.append(x)

                if target[y, x] == 0 and color_beg == True:
                    if len(right_x) == 0 or abs(x - right_x[-1]) < 5:
                        if x < 320:
                            right_y.append(y)
                            right_x.append(x)
                            break

        if len(left_x) > 3:
            if max(left_x) - min(left_x) > 10:
                m, _ = np.polyfit(left_x, left_y, 1)
                l_angle = math.degrees(math.atan(-1./m))
            else:
                m, _ = np.polyfit(left_y, left_x, 1)
                l_angle = math.degrees(math.atan(-m))

        if len(right_x) > 3:
            if max(right_x) - min(right_x) > 10:
                m, _ = np.polyfit(right_x, right_y, 1)
                r_angle = math.degrees(math.atan(-1./m))
            else:
                m, _ = np.polyfit(right_y, right_x, 1)
                r_angle = math.degrees(math.atan(-m))

        return l_angle, r_angle

    @staticmethod
    def _escape_from_wall(color, b, g, r):
        height = b.shape[0]
        color_seq = []
        x = 0
        for y in range(height):
            # black
            if r[y, x] == 0 and g[y, x] == 0 and b[y, x] == 0:
                if len(color_seq) == 0 or color_seq[-1] != 'black':
                    color_seq.append('black')
                continue
            if r[y, x] == 255 and g[y, x] == 255:
                # yellow
                if b[y, x] == 0:
                    if len(color_seq) == 0 or color_seq[-1] != 'yellow':
                        color_seq.append('yellow')
                else:
                    if len(color_seq) == 0 or color_seq[-1] != 'white':
                        color_seq.append('white')
                continue
            # red
            if r[y, x] == 255 and (len(color_seq) == 0 or color_seq[-1] != 'red'):
                color_seq.append('red')
            elif g[y, x] == 255 and (len(color_seq) == 0 or color_seq[-1] != 'green'):
                color_seq.append('green')
            elif b[y, x] == 255 and (len(color_seq) == 0 or color_seq[-1] != 'blue'):
                color_seq.append('blue')

        if color == 'blue' and len(color_seq) > 2:
            if color_seq[-2] == 'red' and color_seq[-3] == 'black':
                return 90
            if color_seq[-2] == 'green' and color_seq[-3] == 'red':
                return 90
            if color_seq[-2] == 'red' and color_seq[-3] == 'green':
                return -90
            if color_seq[-2] == 'green' and color_seq[-3] == 'black':
                return -90

        if color == 'red' and len(color_seq) > 1:
            if color_seq[-2] == 'black':
                return 90
            if color_seq[-2] == 'green':
                return -90
        if color == 'red' and len(color_seq) > 3:
            if color_seq[-4] == 'black':
                return 90
            if color_seq[-4] == 'green':
                return -90

        if color == 'green' and len(color_seq) > 1:
            if color_seq[-2] == 'black':
                return -90
            if color_seq[-2] == 'red':
                return 90
        if color == 'green' and len(color_seq) > 3:
            if color_seq[-4] == 'black':
                return -90
            if color_seq[-4] == 'red':
                return 90

        return 180

    @staticmethod
    def find_median_angle(img):
        crop_img = original_img[140:240, 0:320] # become 100 * 320
        b, g, r = cv2.split(crop_img)
        color, target = ImageProcessor._centre_channel(b, g, r)
        if target is None:
            return color, 180, None
        l_angle, r_angle = ImageProcessor._find_road_angle(target)

        if l_angle is None:
            if r_angle is None or r_angle > 88. or r_angle < -88.: # parallel forward to wall
                return color, ImageProcessor._escape_from_wall(color, b, g, r), target
            else:
                return color, r_angle, target

        if r_angle is None:
            if l_angle > 88. or l_angle < -88.: # parallel forward to wall
                return color, ImageProcessor._escape_from_wall(color, b, g, r), target
            return color, l_angle, target

        return color, (r_angle + l_angle )/2, target

    @staticmethod
    def find_color_percentage(img):
        crop_img = img[160:240, 0:320] # become 80 * 320
        b, g, r = cv2.split(crop_img)
        r_filter = (r == np.maximum(np.maximum(r, g), b)) & (r >= 120) & (g < 150) & (b < 150)
        g_filter = (g == np.maximum(np.maximum(r, g), b)) & (g >= 120) & (r < 150) & (b < 150)
        b_filter = (b == np.maximum(np.maximum(r, g), b)) & (b >= 120) & (r < 150) & (g < 150)
        y_filter = ((r >= 128) & (g >= 128) & (b < 100))

        r[y_filter], g[y_filter], b[np.invert(y_filter)]  = 255, 255, 0

        color = 0
        total = 80 * 320
        zeros = np.zeros(crop_img.shape[:2], dtype = "uint8")

        b[b_filter], b[np.invert(b_filter)] = 255, 0
        if b[79, 159] == 255:
            for blue in np.nditer(b):
                if blue == 255:
                    color += 1
            return "blue", int(100 * color // total), cv2.merge([b, zeros, zeros])

        r[r_filter], r[np.invert(r_filter)] = 255, 0
        if r[79, 159] == 255:
            for red in np.nditer(r):
                if red == 255:
                    color += 1
            return "red", int(100 * color // total), cv2.merge([zeros, zeros, r])

        g[g_filter], g[np.invert(g_filter)] = 255, 0
        if g[79, 159] == 255:
            for green in np.nditer(g):
                if green == 255:
                    color += 1
            return "green", int(100 * color // total), cv2.merge([zeros, g, zeros])

        return "black", 0, None


    @staticmethod
    def find_steering_angle_by_color(img):
        r, g, b      = cv2.split(img)

        image_height = img.shape[0]
        image_width  = img.shape[1]
        camera_x     = image_width // 2
        image_sample = slice(int(image_height * 0.2), int(image_height))
        sr, sg, sb   = r[image_sample, :], g[image_sample, :], b[image_sample, :]
        is_left_wall, is_right_wall = ImageProcessor.wall_detection(sr, sg, sb)

        if is_left_wall:
            return 2*np.pi
        if is_right_wall:
            return -2*np.pi

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
        steering_radian = math.atan2(image_height, (px - camera_x))
        return (np.pi/2 - steering_radian) * 2.0
