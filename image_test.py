import cv2
import numpy as np

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


img = cv2.imread("./Log/IMG/eagle_2018_09_26_09_59_54_050.jpg")
cv2.imshow("original", img)
crop_img = img[200:240, 0:320]
flatten_img = flatten_rgb(crop_img)
#cv2.imshow("flattened", flatten_img)

(B, G, R) = cv2.split(flatten_img)
sky_filter = ((B == 255) & (G == 255) & (R == 255))
B[sky_filter] = 0
R[sky_filter] = 0
G[sky_filter] = 0

zeros = np.zeros(flatten_img.shape[:2], dtype = "uint8")

#sky_img = cv2.merge((B == 255), (G == 255), (R == 255))

g_img = cv2.merge([zeros, G, zeros])
b_img = cv2.merge([B, zeros, zeros])
r_img = cv2.merge([zeros, zeros, R])

#cv2.imshow("Sky", sky_img)
blue = 0
black = 0
total = 0
for b in np.nditer(B):
    if b == 255:
        blue += 1
    elif b == 0:
        black += 1
    total += 1
print("blue: %d, black: %d, total: %d" % (blue, black, total))


#bfname = 'BLUE_%d_%.2f' % (len(bp), len(bp) / (40*320))
cv2.imshow("Blue", b_img)
cv2.imshow("Green", g_img)
cv2.imshow("Red", r_img)

cv2.waitKey(0)
