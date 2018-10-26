import cv2
import numpy as np
import numpy_indexed as npi
from PIL import Image
from io import BytesIO
import base64

class CountDetection(object):
    TRAFFIC_SIGN_IDX_TURN_LEFT = 0
    TRAFFIC_SIGN_IDX_TURN_RIGHT = 1
    TRAFFIC_SIGN_IDX_UTURN_LEFT = 2
    TRAFFIC_SIGN_IDX_UTURN_RIGHT = 3
    TRAFFIC_SIGN_IDX_FORK_LEFT = 4
    TRAFFIC_SIGN_IDX_FORK_RIGHT = 5

    TRACK_ARROW_IDX_FORWARD = 6
    TRACK_ARROW_IDX_BACKWARD = 7

    class_names = ['TurnLeft', 'TurnRight', 'UTurnLeft', 'UTurnRight', 'ForkLeft', 'ForkRight' 'TrackArrowForward', 'TrackArrowBackward']

    __TRAFFIC_SIGN_OFFSET_TOP = 0
    __TRAFFIC_SIGN_OFFSET_BOTTOM = 0.45

    __TRACK_ARROW_OFFSET_TOP = 0.55
    __TRACK_ARROW_OFFSET_BOTTOM = 0.75

    __ORIGINAL_INPUT_WIDTH = 320
    __ORIGINAL_INPUT_HEIGHT = 240

    __TRAFFIC_SIGN_AREA_MIN = 32.0
    __TRAFFIC_SIGN_AREA_MAX = 720.0

    __TRACK_ARROW_AREA_MIN = 32.0
    __TRACK_ARROW_AREA_MAX = 1080.0

    __CV2_INT_MAX = 2147483647
    __CV2_INT_MIN = -2147483648

    def __init__(self, debug=False):
        self._debug = debug
        self._y_offset_traffic_sign = int(CountDetection.__TRAFFIC_SIGN_OFFSET_TOP * CountDetection.__ORIGINAL_INPUT_HEIGHT)
        self._y_offset_track_arrow = int(CountDetection.__TRACK_ARROW_OFFSET_TOP * CountDetection.__ORIGINAL_INPUT_HEIGHT)

    def get_dots_in_region(self, r, y_start, y_end, x_start, x_end):
        return np.sum(r[y_start:y_end+1, x_start:x_end+1] == 255)

    def is_arrow_rotated(self, arrow_img):
        def __consecutive(data, stepsize=1):
            return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

        arrow_img_rot90 = np.rot90(arrow_img)

        _y, _x = np.where(arrow_img_rot90 == 0)  # get coordinates of black points (the chevron part)
        y_x = np.array(list(zip(_y, _x)))    # zip back to y,x 2D array
        y_splitx = npi.group_by(y_x[:, 0]).split(y_x[:, 1])  # get group of x for the specific y
        for row in range(0, arrow_img_rot90.shape[0]):
            if len(__consecutive(y_splitx[row])) > 1:
                return True

        return False

    def flatten_r_channel_enhanced(self, img):
        b, g, r = cv2.split(img)
        b_filter = (b == np.maximum(np.maximum(b, g), r)) & (b >= 120) & (g < 150) & (r < 150)
        g_filter = (g == np.maximum(np.maximum(b, g), r)) & (g >= 120) & (b < 150) & (r < 150)
        r_filter = (r == np.maximum(np.maximum(b, g), r)) & (r >= 120) & (b < 150) & (g < 150)
        y_filter = ((b >= 128) & (g >= 128) & (r < 100))

        b[y_filter], g[y_filter] = 255, 255
        r[np.invert(y_filter)] = 0

        b[b_filter], b[np.invert(b_filter)] = 255, 0
        g[g_filter], g[np.invert(g_filter)] = 255, 0
        r[r_filter], r[np.invert(r_filter)] = 255, 0

        return cv2.merge((b, g, r))

    def classify_traffic_sign_from_features(self, img_ratio, center_dots, center_bot_dots, left_dots, right_dots,left_col_dots, right_col_dots):
        if center_dots >=10:
            if (left_col_dots > right_col_dots and right_dots > left_dots):
                return CountDetection.TRAFFIC_SIGN_IDX_TURN_RIGHT
            elif (left_col_dots < right_col_dots and right_dots < left_dots):
                return CountDetection.TRAFFIC_SIGN_IDX_TURN_LEFT
            else:
                return -1

        if center_dots <= 1:
            if center_bot_dots == 0:
                return CountDetection.TRAFFIC_SIGN_IDX_UTURN_RIGHT if left_col_dots > right_col_dots else CountDetection.TRAFFIC_SIGN_IDX_UTURN_LEFT

        if left_dots >= 77:
            return CountDetection.TRAFFIC_SIGN_IDX_FORK_RIGHT   # fork right
        if right_dots >= 77:
            return CountDetection.TRAFFIC_SIGN_IDX_FORK_LEFT   # fork left

        return -1   # un-classified


    def classify_traffic_sign_from_image(self, sign_img):
        ''' return: sign_class '''
        # filter out those distorted img
        img_ratio = sign_img.shape[1] / sign_img.shape[0]
        if img_ratio < 1.4 or img_ratio > 3:
            return -1

        img = cv2.resize(sign_img, (18, 12), interpolation=cv2.INTER_NEAREST)

        b, g, r = cv2.split(img)

        center_dots     = self.get_dots_in_region(r, 5, 8, 8, 11)
        center_bot_dots = self.get_dots_in_region(r, 8, 11, 8, 11)
        left_dots       = self.get_dots_in_region(r, 0, 11, 0, 7)
        right_dots      = self.get_dots_in_region(r, 0, 11, 10, 17)
        left_col_dots   = self.get_dots_in_region(r, 0, 11, 0, 1)
        right_col_dots  = self.get_dots_in_region(r, 0, 11, 16, 17)

        img_class = self.classify_traffic_sign_from_features(img_ratio, center_dots, center_bot_dots, left_dots, right_dots, left_col_dots, right_col_dots)

        if self._debug == True:
            cv2.namedWindow("TrafficSign", cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow("TrafficSign", 320, 0)

            # cv2.rectangle(src_img, (left, top), (right, bottom), (0, 255, 0), 3)
            cv2.rectangle(img, (7, 5), (10, 8), (0, 255, 0), 1)     # Green middle
            cv2.rectangle(img, (7, 8), (10, 11), (0, 255, 0), 1)    # Green bottom

            cv2.rectangle(img, (0, 0), (7, 11), (0, 255, 255), 1)   # Yellow Left
            cv2.rectangle(img, (10, 0), (17, 11), (0, 255, 255), 1) # Yellow Right

            cv2.rectangle(img, (0, 0), (1, 11), (255, 255, 0), 1)   # Cyan Left
            cv2.rectangle(img, (16, 0), (17, 11), (255, 255, 0), 1) # Cyan Right

            cv2.imshow("TrafficSign", img)
            cv2.waitKey(1)

        return img_class

    def _flatten_red_and_find_sign(self, img):
        ''' return:
                sign_img: an augemented (for debug) image showing sign(s). None if no sign detected
                sign_class: int
                sign_loc: array(x) where x indicating x-axis location of signs
        '''
        _flattened_r_channel_enhanced_img = self.flatten_r_channel_enhanced(img)

        # count red points
        r = _flattened_r_channel_enhanced_img[:, :, 2]
        _y, _x = np.where(r == 255)  # filter out only those red points
        red_points = len(_y)

        # nothing significant enough in red but noise, no sign
        out_boxes, out_scores, out_classes = list(), list(), list()
        if red_points < 50:
           return out_boxes, out_scores, out_classes

        # find connected componsents and classify
        connectivity = 8
        _, _, cc_stats, _ = cv2.connectedComponentsWithStats(r, connectivity, cv2.CV_32S)
        for idx, cc_stat in enumerate(cc_stats):
            if idx == 0:  # ignore background
                continue
            if cc_stat[cv2.CC_STAT_WIDTH] < 8 or cc_stat[cv2.CC_STAT_HEIGHT] < 6:  # don't care anything too small: noise
                continue
            if cc_stat[cv2.CC_STAT_WIDTH] < cc_stat[cv2.CC_STAT_HEIGHT]: # signs are wider than it's height
                continue
            # same the bounding box for training data
            sign_img = _flattened_r_channel_enhanced_img[cc_stat[cv2.CC_STAT_TOP]:cc_stat[cv2.CC_STAT_TOP]+cc_stat[cv2.CC_STAT_HEIGHT], cc_stat[cv2.CC_STAT_LEFT]:cc_stat[cv2.CC_STAT_LEFT]+cc_stat[cv2.CC_STAT_WIDTH], :]
            sign_class_tmp = self.classify_traffic_sign_from_image(sign_img)

            # something detected, there is a sign
            if sign_class_tmp in [CountDetection.TRAFFIC_SIGN_IDX_TURN_LEFT, \
                                  CountDetection.TRAFFIC_SIGN_IDX_TURN_RIGHT, \
                                  CountDetection.TRAFFIC_SIGN_IDX_FORK_LEFT, \
                                  CountDetection.TRAFFIC_SIGN_IDX_FORK_RIGHT, \
                                  CountDetection.TRAFFIC_SIGN_IDX_UTURN_LEFT, \
                                  CountDetection.TRAFFIC_SIGN_IDX_UTURN_RIGHT]:
                out_box = [cc_stat[cv2.CC_STAT_TOP] + self._y_offset_traffic_sign, \
                           cc_stat[cv2.CC_STAT_LEFT], \
                           cc_stat[cv2.CC_STAT_TOP] + cc_stat[cv2.CC_STAT_HEIGHT] + self._y_offset_traffic_sign, \
                           cc_stat[cv2.CC_STAT_LEFT] + cc_stat[cv2.CC_STAT_WIDTH]]
                out_boxes.append(out_box)

                score = max(cc_stat[cv2.CC_STAT_HEIGHT] * cc_stat[cv2.CC_STAT_WIDTH] - CountDetection.__TRAFFIC_SIGN_AREA_MIN, 0.0) / CountDetection.__TRAFFIC_SIGN_AREA_MAX
                score = min(score, 1)
                out_scores.append(score)

                out_classes.append(sign_class_tmp)
                #if self._debug:
                #print("({0}, {1}, {2}, {3}) sign detected: {4}".format(out_box[0], out_box[1], out_box[2], out_box[3], CountDetection.class_names[sign_class_tmp]))
        return out_boxes, out_scores, out_classes

    def _flatten_red_and_find_track_arrow(self, img):
        _flattened_r_channel_enhanced_img = self.flatten_r_channel_enhanced(img)

        # revert red and black for connected component
        r = _flattened_r_channel_enhanced_img[:, :, 2]
        r_inverted = r.copy()
        r_filter = (r_inverted == 255)
        r_inverted[r_filter], r_inverted[np.invert(r_filter)] = 0, 255

        # find connected componsents and classify
        connectivity = 8
        _, _, cc_stats, _ = cv2.connectedComponentsWithStats(r_inverted, connectivity, cv2.CV_32S)

        out_boxes, out_scores, out_classes = list(), list(), list()
        is_reverse, confidence = None, None
        center_upp_dots, center_bot_dots = None, None
        for idx, cc_stat in enumerate(cc_stats):
            if cc_stat[cv2.CC_STAT_LEFT] == CountDetection.__CV2_INT_MAX or cc_stat[cv2.CC_STAT_TOP] == CountDetection.__CV2_INT_MAX:
                continue
            # any top/left/right/bot on border is not our target
            #print("left, top, width, height = %d %d %d %d" % (cc_stat[cv2.CC_STAT_LEFT], cc_stat[cv2.CC_STAT_TOP], cc_stat[cv2.CC_STAT_WIDTH], cc_stat[cv2.CC_STAT_HEIGHT]))
            if cc_stat[cv2.CC_STAT_LEFT]==0 or \
               cc_stat[cv2.CC_STAT_TOP]==0 or \
               cc_stat[cv2.CC_STAT_LEFT]+cc_stat[cv2.CC_STAT_WIDTH]==r_inverted.shape[1] or \
               cc_stat[cv2.CC_STAT_TOP]+cc_stat[cv2.CC_STAT_HEIGHT]==r.shape[0]:
                continue

            # too small: noise
            if cc_stat[cv2.CC_STAT_WIDTH] < 30 or cc_stat[cv2.CC_STAT_HEIGHT] < 3:
                continue

            # make a guess
            center_upp_dots = self.get_dots_in_region(r, cc_stat[cv2.CC_STAT_TOP]+int(cc_stat[cv2.CC_STAT_HEIGHT]*0.), cc_stat[cv2.CC_STAT_TOP]+int(cc_stat[cv2.CC_STAT_HEIGHT]*.5), cc_stat[cv2.CC_STAT_LEFT]+int(cc_stat[cv2.CC_STAT_WIDTH]*.45), cc_stat[cv2.CC_STAT_LEFT]+int(cc_stat[cv2.CC_STAT_WIDTH]*.55))
            center_bot_dots = self.get_dots_in_region(r, cc_stat[cv2.CC_STAT_TOP]+int(cc_stat[cv2.CC_STAT_HEIGHT]*.5), cc_stat[cv2.CC_STAT_TOP]+int(cc_stat[cv2.CC_STAT_HEIGHT]*1.), cc_stat[cv2.CC_STAT_LEFT]+int(cc_stat[cv2.CC_STAT_WIDTH]*.45), cc_stat[cv2.CC_STAT_LEFT]+int(cc_stat[cv2.CC_STAT_WIDTH]*.55))

            # we have a larger target to classify, ignore small ones
            if confidence is not None and center_upp_dots+center_bot_dots < confidence:
                continue
            is_reverse = center_upp_dots > center_bot_dots and not self.is_arrow_rotated(r[cc_stat[cv2.CC_STAT_TOP]:cc_stat[cv2.CC_STAT_TOP]+cc_stat[cv2.CC_STAT_HEIGHT], cc_stat[cv2.CC_STAT_LEFT]:cc_stat[cv2.CC_STAT_LEFT]+cc_stat[cv2.CC_STAT_WIDTH]])
            confidence = min(100, center_upp_dots + center_bot_dots)

            if confidence > 50:
                out_box = [cc_stat[cv2.CC_STAT_TOP] + self._y_offset_track_arrow, \
                            cc_stat[cv2.CC_STAT_LEFT], \
                            cc_stat[cv2.CC_STAT_TOP] + cc_stat[cv2.CC_STAT_HEIGHT] + self._y_offset_track_arrow, \
                            cc_stat[cv2.CC_STAT_LEFT] + cc_stat[cv2.CC_STAT_WIDTH]]
                out_boxes.append(out_box)
                out_scores.append(min(confidence / 100.0, 1.0))
                if is_reverse:
                    out_classes.append(CountDetection.TRACK_ARROW_IDX_BACKWARD)
                else:
                    out_classes.append(CountDetection.TRACK_ARROW_IDX_FORWARD)

        return out_boxes, out_scores, out_classes

    def _crop_image(self, img, begin, end):
        bottom_half_ratios = (begin, end)
        bottom_half_slice  = slice(*(int(x * img.shape[0]) for x in bottom_half_ratios))
        bottom_half        = img[bottom_half_slice, :, :]
        return bottom_half

    def classify_sign_from_image(self, img):
        img_top = self._crop_image(img, 0, 0.45)
        #cv2.imshow("TrafficSign", img_top)
        out_boxes_ts, out_scores_ts, out_classes_ts = self._flatten_red_and_find_sign(img_top)

        img_mid = self._crop_image(img, 0.55, 0.75)
        out_boxes_ta, out_scores_ta, out_classes_ta = self._flatten_red_and_find_track_arrow(img_mid)

        return out_boxes_ts+out_boxes_ta, out_scores_ts+out_scores_ta, out_classes_ts+out_classes_ta
