import cv2
import numpy as np
import math
import sys
from image_processor import ImageProcessor

def flatten_rgb(img):
    r, g, b = cv2.split(img)
    r_filter = (r == np.maximum(np.maximum(r, g), b)) & (r >= 120) & (g < 150) & (b < 150)
    g_filter = (g == np.maximum(np.maximum(r, g), b)) & (g >= 120) & (r < 150) & (b < 150)
    b_filter = (b == np.maximum(np.maximum(r, g), b)) & (b >= 120) & (r < 150) & (g < 150)
    black = ((r<50) & (g<50)& (b<50))

    b[b_filter], b[np.invert(b_filter)] = 255, 0
    r[r_filter], r[np.invert(r_filter)] = 255, 0
    g[g_filter], g[np.invert(g_filter)] = 255, 0
    b[black], g[black], r[black] = 0, 255, 255

    return cv2.merge((r, g, b))

def find_road_angle(target):
    left_x, left_y = [], []
    right_x, right_y = [], []
    l_angle, r_angle = None, None
    for _y in range(100):
        y = target.shape[0] -_y -1
        color_beg = False
        for x in range(target.shape[1]):
            if target[y, x] == 255 and color_beg == False:
                if len(left_x) == 0 or abs(x - left_x[-1]) < 5:
                    color_beg = True
                    if x > 0:
                        left_y.append(y)
                        left_x.append(x)

            if target[y, x] == 0 and color_beg == True:
                if x < 320:
                    right_y.append(y)
                    right_x.append(x)
                break

    if len(left_x) > 3:
        print("left: from(%d, %d) to (%d, %d)" %(left_y[0], left_x[0], left_y[-1], left_x[-1]))
        if max(left_x) - min(left_x) > 10:
            m, _ = np.polyfit(left_x, left_y, 1)
            l_angle = math.degrees(math.atan(-1./m))
        else:
            m, _ = np.polyfit(left_y, left_x, 1)
            l_angle = math.degrees(math.atan(-m))

    if len(right_x) > 3:
        print("right: from(%d, %d) to (%d, %d)" %(right_y[0], right_x[0], right_y[-1], right_x[-1]))
        if max(right_x) - min(right_x) > 10:
            m, _ = np.polyfit(right_x, right_y, 1)
            r_angle = math.degrees(math.atan(-1./m))
        else:
            m, _ = np.polyfit(right_y, right_x, 1)
            r_angle = math.degrees(math.atan(-m))

    return l_angle, r_angle


def ti6_process(img):
    bottom_half_ratios = (0.6, 0.8)
    bottom_half_slice  = slice(*(int(x * img.shape[0]) for x in bottom_half_ratios))
    flatten = flatten_rgb(img)
    crop_img        = flatten[bottom_half_slice, :, :]

    image_height = crop_img.shape[0]
    image_width  = crop_img.shape[1]
    camera_x     = image_width / 2
    img_g        = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    tracks       = map(lambda x: len(x[x > 20]), [img_g])
    tracks_seen  = list(filter(lambda y: y > 2000, tracks))

    if len(list(tracks_seen)) == 0:
        print("no tracks_seen: angle = 0.0")
    else:
        _target = img_g
        _y, _x = np.where(_target == 76)

        px = np.mean(_x)
        if np.isnan(px):
            print("not a number: angle = 0.0")
        else:
            steering_angle = math.atan2(image_height*3, (px - camera_x))
            print("steering_angle = %.2f" % math.degrees(steering_angle))
            r = 240
            px = img.shape[1]
            py = img.shape[0]
            x = px // 2 + int(r * math.cos(steering_angle))
            y = py  - int(r * math.sin(steering_angle))
            cv2.line(img, (px // 2, py - 1), (x, y), (255, 0, 255), 2)

        cv2.imshow("ti6-gray", img_g)

file = './Log/IMG/%s.jpg' % sys.argv[1]
image = cv2.imread(file)

im_gray = ImageProcessor._flatten_rgb_to_gray(image)

foot = ImageProcessor._crop_gray(im_gray, 0.8, 1.0)
cv2.imshow("foot", foot)

b, r, w = ImageProcessor._color_rate(foot)
print("b:%.0f, r:%.0f, w:%.0f" %(100*b, 100*r, 100*w))

if b > 0 and r == 0:
    print("black wall: angle = 180")
else:
    target = ImageProcessor._crop_gray(im_gray, 0.6, 1.0)
    wall_angle = ImageProcessor.find_wall_angle(target)

    if wall_angle == 180:
        if r == 1:
            print("wall_angle = 180, & red = 100%% in foot, angle = 90 or -90")
        else:
            print("wall_angle = 180, & red < 100%% in foot, angle = 180")
    elif wall_angle is None:
        if r > 0:
            angle = ImageProcessor.find_red_angle(im_gray, debug = True)
            print("wall_angle is none & red > 0 => find_red_angle = %.2f" % angle)
        else:
            print("wall_angle is none & red == 0 => angle = 180")
    else:
        if r > 0:
            angle = ImageProcessor.find_red_angle(im_gray, debug = True)
            #radius = target.shape[0]
            #x = target.shape[1] // 2 + int(radius * math.sin(math.radians(angle)))
            #y = target.shape[0]    - int(radius * math.cos(math.radians(angle)))
            #cv2.line(target, (160, target.shape[0] - 1), (x, y), 0, 2)
            print("wall_angle = %.2f & red > 0 => find_red_angle = %.2f" % (wall_angle, angle))
        else:
            radius = target.shape[0]
            x = target.shape[1] // 2 + int(radius * math.sin(math.radians(wall_angle)))
            y = target.shape[0]    - int(radius * math.cos(math.radians(wall_angle)))
            cv2.line(target, (160, target.shape[0] - 1), (x, y), 0, 2)
            print("wall_angle = %.2f, set it to last_wall_angle" % wall_angle)

    cv2.imshow("target", target)

dir = ImageProcessor.find_arrow_dir(im_gray, im_gray)
cv2.imshow("gray", im_gray)

if dir is not None:
    if dir == 'B':
        if angle == 180:
            print('angle == 180 & dir == B => left')
        else:
            print('angle != 180 & dir == B => reverse')
    elif dir == 'F':
        if angle == 180:
            print('angle == 180 & dir == F => right')
        else:
            print('angle != 180 & dir == F => forward')
else:
    print("dir is none")

cv2.waitKey(0)
cv2.destroyAllWindows()

#b, g, r = cv2.split(crop_img)


'''
color, target = centre_channel(b, g, r)
if target is None:
    print("%s: 180" % color)
else:
    l_angle, r_angle = find_road_angle(target)

    if l_angle is None:
        if r_angle is None or r_angle > 88. or r_angle < -88.: # parallel forward to wall
            print("%s: %.2f" %(color, escape_from_wall(color, b, g, r)))
        else:
            print("%s: %.2f" %(color, r_angle))
    elif r_angle is None:
        if l_angle > 88. or l_angle < -88.: # parallel forward to wall
            print("%s: %.2f" %(color, escape_from_wall(color, b, g, r)))
        else:
            print("%s: %.2f" %(color, l_angle))
    else:
        print("%s: %.2f" %(color, (r_angle + l_angle )/2))


#cv2.line(flatten_img, (x1, y1), (x2, y2), (0,0,0), 2)

#cv2.imshow("flatten", flatten_img)
#cv2.waitKey(0)
#sky_img = cv2.merge((B == 255), (G == 255), (R == 255))

for h in range(3):
    img_g = G[25*(h+1):100, 0:320]
    edge_g = cv2.Canny(img_g, 50, 150, apertureSize = 3)
    lines = cv2.HoughLines(edge_g, 4, np.pi / 180, 95 - 25*(h+1), None, 0, 0)
    zeros = np.zeros(img_g.shape[:2], dtype = "uint8")
    g_img = cv2.merge([zeros, img_g, zeros])
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            print("G[%d]:rho = %.2f, theta = %.2f" % (h, rho, math.degrees(theta)))
            x1 = int(rho*math.cos(theta) - 1000*math.sin(theta))
            y1 = int(rho*math.sin(theta) + 1000*math.cos(theta))
            x2 = int(rho*math.cos(theta) + 1000*math.sin(theta))
            y2 = int(rho*math.sin(theta) - 1000*math.cos(theta))
            cv2.line(g_img, (x1, y1), (x2, y2), (255,255,255), 1)
    cv2.imshow('g_%d' % h, g_img)

edge_g = cv2.Canny(G, 50, 150, apertureSize = 3)
cv2.imshow('green', edge_g)

lines = cv2.HoughLines(edge_g, 10, np.pi / 180, 80, None, 0, 0)

if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        print("G:rho = %.2f, theta = %.2f" % (rho, theta * 180 / np.pi))
        x1 = int(rho*math.cos(theta) - 1000*math.sin(theta))
        y1 = int(rho*math.sin(theta) + 1000*math.cos(theta))
        x2 = int(rho*math.cos(theta) + 1000*math.sin(theta))
        y2 = int(rho*math.sin(theta) - 1000*math.cos(theta))
        cv2.line(flatten_img, (x1, y1), (x2, y2), (255,255,255), 1)
cv2.waitKey(0)

# ===

edge_r = cv2.Canny(R, 50, 150, apertureSize = 3)
cv2.imshow('red', edge_g)

lines = cv2.HoughLines(edge_r, 4, np.pi / 180, 80, None, 0, 0)

if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        print("R:rho = %.2f, theta = %.2f" % (rho, theta * 180 / np.pi))
        x1 = int(rho*math.cos(theta) - 1000*math.sin(theta))
        y1 = int(rho*math.sin(theta) + 1000*math.cos(theta))
        x2 = int(rho*math.cos(theta) + 1000*math.sin(theta))
        y2 = int(rho*math.sin(theta) - 1000*math.cos(theta))
        cv2.line(flatten_img, (x1, y1), (x2, y2), (255,255,255), 1)


cv2.imshow("flattened", flatten_img)
zeros = np.zeros(flatten_img.shape[:2], dtype = "uint8")

g_img = cv2.merge([zeros, G, zeros])
b_img = cv2.merge([B, zeros, zeros])
r_img = cv2.merge([zeros, zeros, R])

cv2.imshow("Sky", sky_img)
cv2.imshow("Blue", b_img)
cv2.imshow("Green", g_img)
cv2.imshow("Red", r_img)

cv2.waitKey(0)
'''
