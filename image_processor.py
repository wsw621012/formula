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
    def wall_detection (sr, sg, sb):
        black_count = 0
        yellow_count = 0
        for i in range(len(sr) // 10):
            for j in range(len(sr[i]) // 10):
                inverse_i = len(sr) // 8 + i
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
                is_right_wall = True

        return is_left_wall, is_right_wall

    @staticmethod
    def find_radian_by_color(idx, img):
        r, g, b      = cv2.split(img)
        color_list = [r, g, b]

        Ymax = img.shape[0] - 1
        Xmax  = img.shape[1] - 1
        camera_x     = Xmax // 2

        delta_x = Xmax // 20
        image_sample = slice(int((Ymax + 1)*0.2), int(Ymax + 1))
        sr, sg, sb   = r[image_sample, :], g[image_sample, :], b[image_sample, :]

        is_left_wall, is_right_wall = ImageProcessor.wall_detection(sr, sg, sb)
        if is_left_wall:
            print("hist wall")
            return 2*np.pi
        if is_right_wall:
            print("hit wall")
            return -2*np.pi

        for _target in color_list:
            if _target[Ymax, camera_x] == 255:
                break
            print("cental-target is %d:" %_target[Ymax, camera_x])

        _x = camera_x
        _y = Ymax
        while _y > 0:
            _y -= 1
            if (_x < delta_x) or (_x >= Xmax - delta_x):
                break

            if _target[_y, _x] != 255:
                if (_x >= delta_x) and _target[_y, _x - delta_x] == 255:
                    _x -= delta_x
                    continue
                if (_x <= Xmax - delta_x) and _target[_y, _x + delta_x] == 255:
                    _x += delta_x
                    continue

                if _target[_y, Xmax] == 255:
                    _x = Xmax
                    break

                if _target[_y, 0] == 255:
                    _x = 0
                    break

                break

        # bottom_left_x
        # p1 +------+ p3
        #    |      |
        # p2+-------+ p4
        x1 = x2 = 0
        x3 = x4 = Xmax
        y2 = y4 = Ymax
        y1 = y3 = _y
        for x2 in range(camera_x - 1, -1, -1):
            if _target[Ymax, x2] != 255:
                break
        if x2 == 0:
            for y2 in range(Ymax, -1, -1):
                if _target[y2, 0] != 255:
                    break
        if y2 == 0:
            y2 = Ymax
        for x1 in range(_x - 1, -1, -1):
            if _target[_y, x1] != 255:
                break
        for x4 in range(camera_x, Xmax+1, 1):
            if _target[Ymax, x4] != 255:
                break
        if x4 == Xmax:
            for y4 in range(Ymax, -1, -1):
                if _target[y4, x4] != 255:
                    break
        if y4 == 0:
            y4 = Ymax
        for x3 in range(_x, Xmax+1, 1):
            if _target[_y, x3] != 255:
                break

        #'''debug mode
        print("p1(%d, %d)~P2(%d,%d)" %(x1,y1,x2,y2))
        print("p3(%d, %d)~P4(%d,%d)" %(x3,y3,x4,y4))

        if (x1 == 0) and (x2 == 0) and (x3 > x4):
            steering_radian = math.atan2(y4 - y3, x3 - x4)
            return np.pi/2 - steering_radian
        elif (x3 == Xmax) and (x4 == Xmax) and (x2 > x1):
            steering_radian = math.atan2(y2 - y1, x1 - x2)
            return np.pi/2 - steering_radian

        #if y1 == 0 and y3 == 0:
        #    steering_radian = math.atan2(Ymax, ((x1 + x3) // 2) - camera_x)
        #    return np.pi/2 - steering_radian

        deno = float((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))
        if deno == 0: # parallel lines
            return 0.0

        Ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / deno

        Cx = x1 + Ua*(x2-x1)
        Cy = y1 + Ua*(y2-y1)

        steering_radian = np.pi/2 - math.atan2(Ymax - Cy, Cx - camera_x)


        print("C(%d, %d)" %(Cx,Cy))

        new_img = img.copy();
        cv2.line(new_img, (x2, y2), (x1, y1), (0, 0, 0))
        cv2.line(new_img, (x4, y4), (x3, y3), (0, 0, 0))
        cv2.line(new_img, (int(Cx), int(Cy)), (camera_x, Ymax), (255, 255, 255), 2)


        filename = "./frames/%02d_%.4f.jpg" % (idx, steering_radian)
        cv2.imwrite(filename, new_img)

        return steering_radian

    @staticmethod
    def find_color_and_proportion(img):
        crop_img = img[200:240, 0:320] # become 40 * 320
        r, g, b = cv2.split(crop_img)
        r_filter = (r == np.maximum(np.maximum(r, g), b)) & (r >= 120) & (g < 150) & (b < 150)
        g_filter = (g == np.maximum(np.maximum(r, g), b)) & (g >= 120) & (r < 150) & (b < 150)
        b_filter = (b == np.maximum(np.maximum(r, g), b)) & (b >= 120) & (r < 150) & (g < 150)
        y_filter = ((r >= 128) & (g >= 128) & (b < 100))

        r[y_filter], g[y_filter], b[np.invert(y_filter)]  = 255, 255, 0

        color = 0
        total = 40 * 320
        zeros = np.zeros(crop_img.shape[:2], dtype = "uint8")

        b[b_filter], b[np.invert(b_filter)] = 255, 0
        if b[39, 159] == 255:
            for blue in np.nditer(b):
                if blue == 255:
                    color += 1
            return "blue", int(100 * color // total), cv2.merge([b, zeros, zeros])

        r[r_filter], r[np.invert(r_filter)] = 255, 0
        if r[39, 159] == 255:
            for red in np.nditer(r):
                if red == 255:
                    color += 1
            return "red", int(100 * color // total), cv2.merge([zeros, zeros, r])

        g[g_filter], g[np.invert(g_filter)] = 255, 0
        if g[39, 159] == 255:
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
