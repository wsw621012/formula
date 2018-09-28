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

#eagle_2018_09_28_16_22_14_445-> 1st: 140:240, p1 = 4, p2 = 80
#eagle_2018_09_28_16_22_17_238
img = cv2.imread("./Log/IMG/eagle_2018_09_28_16_55_37_684.jpg")
crop_img = img[140:240, 0:320]
flatten_img = flatten_rgb(crop_img)

(B, G, R) = cv2.split(flatten_img)
sky_filter = ((B == 255) & (G == 255) & (R == 255))
B[sky_filter] = 0
R[sky_filter] = 0
G[sky_filter] = 0

zeros = np.zeros(flatten_img.shape[:2], dtype = "uint8")

#sky_img = cv2.merge((B == 255), (G == 255), (R == 255))
edge_g = cv2.Canny(G, 50, 150, apertureSize = 3)
cv2.imshow('green', edge_g)

lines = cv2.HoughLines(edge_g, 4, np.pi / 180, 80, None, 0, 0)

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

#g_img = cv2.merge([zeros, G, zeros])
#b_img = cv2.merge([B, zeros, zeros])
#r_img = cv2.merge([zeros, zeros, R])

#cv2.imshow("Sky", sky_img)
#cv2.imshow("Blue", b_img)
#cv2.imshow("Green", g_img)
#cv2.imshow("Red", r_img)

cv2.waitKey(0)
