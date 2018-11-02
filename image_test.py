import cv2
import numpy as np
import math
import sys
from image_processor import ImageProcessor
from os import listdir
from os.path import isfile, join


#mypath = './Log/IMG'
#for f in listdir(mypath):

#    if ('.jpg' not in str(f)) or (not isfile(join(mypath, f))):
#        continue

f = './Log/IMG/{0}.jpg'.format(sys.argv[1])
image = cv2.imread(f)
#    print(join(mypath, f))
#    image = cv2.imread(join(mypath, f))

im_gray = ImageProcessor._flatten_rgb_to_gray(image)
target = ImageProcessor._crop_gray(im_gray, 0.57, 1.0)
b, r, w = ImageProcessor._color_rate(target)
if b < r and b < w:
    # valid road is horizontal
    _ly = target[0,:]

    if not 76 in _ly:
        target[target == 0] = 255
        ly = np.argmin(target, axis=0)
        if sum(ly != 0) == 0:
            print("full white")
        else:
            lx = np.arange(target.shape[1])[ly != 0]
            ly = ly[lx]
            if len(ly) > 1:
                m, _ = np.polyfit(lx, ly, 1)
                angle = math.degrees(math.atan(-1./m))
            else:
                angle = 0

            print("red-white line angle:%.2f" % angle)

            radius = 1000
            x = target.shape[1] // 2 + int(radius * math.sin(math.radians(angle)))
            y = target.shape[0]    - int(radius * math.cos(math.radians(angle)))
            cv2.line(target, (160, target.shape[0] - 1), (x, y), 0, 2)
    else:
        print("not my case.")

cv2.imshow("gray", im_gray)
cv2.imshow("target", target)
cv2.waitKey(0)
cv2.destroyAllWindows()
