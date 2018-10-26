import cv2
import numpy as np
import math
import sys
from image_processor import ImageProcessor
from os import listdir
from os.path import isfile, join


mypath = './Log/IMG'
for f in listdir(mypath):

    if ('.jpg' not in str(f)) or (not isfile(join(mypath, f))):
        continue

    #file = './Log/IMG/%s.jpg' % sys.argv[1]
    print(join(mypath, f))
    image = cv2.imread(join(mypath, f))

    im_gray = ImageProcessor._flatten_rgb_to_gray(image)
    target = ImageProcessor._crop_gray(im_gray, 0.57, 1.0)
    target = target[:, 153:187]
    ImageProcessor.test_red_angle(target, debug = True)
    '''
    radius = target.shape[1]
    x = target.shape[1] // 2 + int(radius * math.sin(math.radians(wall_angle)))
    y = target.shape[0]    - int(radius * math.cos(math.radians(wall_angle)))
    cv2.line(target, (160, target.shape[0] - 1), (x, y), 0, 2)
    '''
    cv2.imshow("gray", im_gray)
    cv2.imshow("target", target)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
