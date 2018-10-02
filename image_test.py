import cv2
import numpy as np
import math

def flatten_rgb(img):
    b, g, r = cv2.split(img)
    r_filter = (r == np.maximum(np.maximum(r, g), b)) & (r >= 120) & (g < 150) & (b < 150)
    g_filter = (g == np.maximum(np.maximum(r, g), b)) & (g >= 120) & (r < 150) & (b < 150)
    b_filter = (b == np.maximum(np.maximum(r, g), b)) & (b >= 120) & (r < 150) & (g < 150)
    y_filter = ((r >= 128) & (g >= 128) & (b < 100))

    sky_filter = ((r >= 150) & (g >= 150) & (b >= 150))

    r[y_filter], g[y_filter] = 255, 255
    b[np.invert(y_filter)] = 0

    b[b_filter], b[np.invert(b_filter)] = 255, 0
    r[r_filter], r[np.invert(r_filter)] = 255, 0
    g[g_filter], g[np.invert(g_filter)] = 255, 0

    r[sky_filter], g[sky_filter], b[sky_filter] = 255, 255, 255
    flattened = cv2.merge((b, g, r))
    return flattened

def draw_line(inputImage):
    inputImageGray = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(inputImageGray,150,200,apertureSize = 3)
    minLineLength = 30
    maxLineGap = 5
    lines = cv2.HoughLinesP(edges,cv2.HOUGH_PROBABILISTIC, np.pi/180, 30, minLineLength,maxLineGap)
    for x in range(0, len(lines)):
        for x1,y1,x2,y2 in lines[x]:
            #cv2.line(inputImage,(x1,y1),(x2,y2),(0,128,0),2, cv2.LINE_AA)
            pts = np.array([[x1, y1 ], [x2 , y2]], np.int32)
            cv2.polylines(inputImage, [pts], True, (255,255,255))

    cv2.imshow("Trolley_Problem_Result", inputImage)
    cv2.imshow('edge', edges)

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

def escape_from_wall(color, b, g, r):
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

def centre_channel(b, g, r):
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

#eagle_2018_09_28_16_22_14_445-> 1st: 140:240, p1 = 4, p2 = 80
#eagle_2018_09_28_16_22_17_238
#eagle_2018_09_28_16_55_37_684
#eagle_2018_09_28_16_55_38_817
img = cv2.imread("./Log/IMG/eagle_2018_10_02_22_20_42_903.jpg")
crop_img = img[140:240, 0:320] # become 100 * 320
b, g, r = cv2.split(crop_img)
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
'''
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
