import cv2
import numpy as np
import math
import sys
from image_processor import ImageProcessor

file = './Log/IMG/%s.jpg' % sys.argv[1]
image = cv2.imread(file)

im_gray = ImageProcessor._flatten_rgb_to_gray(image)

target = ImageProcessor._crop_gray(im_gray, 0.6, 1.0)
b, r, w = ImageProcessor._color_rate(target)
print("b:%.0f, r:%.0f, w:%.0f" %(100*b, 100*r, 100*w))

#wall_angle, lx, ly, rx, ry = ImageProcessor.find_wall_angle(target, debug = True)
wall_angle, lx, ly, rx, ry = ImageProcessor.find_wall_angle(target, debug = True)
if wall_angle == 180:
    if r == 1:
        print("wall_angle = 180, & red = 100%% in foot, angle = 90 or -90")
    else:
        print("wall_angle = 180, & red < 100%% in foot, angle = 180")
elif wall_angle is None:
    ImageProcessor.test_red_angle(target, debug = True)
    angle, color = ImageProcessor.find_road_angle(target, debug = True)
    if angle is None:
        angle = ImageProcessor.find_red_angle(im_gray, debug = True)
    if color is None:
        print("find red angle is %.2f" % angle)
    else:
        print("find road angle is %.2f, color is %.2f" % (angle, color))
    radius = target.shape[1]
    x = target.shape[1] // 2 + int(radius * math.sin(math.radians(angle)))
    y = target.shape[0]    - int(radius * math.cos(math.radians(angle)))
    cv2.line(target, (160, target.shape[0] - 1), (x, y), 0, 2)
else:
    if lx > 0 or rx < (target.shape[1] - 1):
        px = lx // 2
        m = math.atan2(px - (target.shape[1] // 2), target.shape[0] - ly)
        if lx < (target.shape[1] - 1 - rx):
            px = (target.shape[1] + rx) // 2
            m = math.atan2(px - (target.shape[1] // 2), target.shape[0] - ry)
        wall_angle =  math.degrees(m)


    radius = target.shape[1]
    x = target.shape[1] // 2 + int(radius * math.sin(math.radians(wall_angle)))
    y = target.shape[0]    - int(radius * math.cos(math.radians(wall_angle)))
    cv2.line(target, (160, target.shape[0] - 1), (x, y), 0, 2)

    #if r > 0:
    #    angle = ImageProcessor.find_red_angle(im_gray, debug = True)
    #    print("wall_angle = %.2f & red > 0 => find_red_angle = %.2f" % (wall_angle, angle))
    #else:
    #    print("wall_angle = %.2f, set it to last_wall_angle" % wall_angle)

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
