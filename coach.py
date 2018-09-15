from image_processor import ImageProcessor
from action import Action
from time import time

import math

class PID:
    def __init__(self, Kp, Ki, Kd, max_integral, min_interval = 0.001, set_point = 0.0, last_time = None):
        self._Kp           = Kp
        self._Ki           = Ki
        self._Kd           = Kd
        self._min_interval = min_interval
        self._max_integral = max_integral

        self._set_point    = set_point
        self._last_time    = last_time if last_time is not None else time()
        self._p_value      = 0.0
        self._i_value      = 0.0
        self._d_value      = 0.0
        self._d_time       = 0.0
        self._d_error      = 0.0
        self._last_error   = 0.0
        self._output       = 0.0


    def update(self, cur_value, cur_time = None):
        if cur_time is None:
            cur_time = time()

        error   = self._set_point - cur_value
        d_time  = cur_time - self._last_time
        d_error = error - self._last_error

        if d_time >= self._min_interval:
            self._p_value   = error
            self._i_value   = min(max(error * d_time, -self._max_integral), self._max_integral)
            self._d_value   = d_error / d_time if d_time > 0 else 0.0
            self._output    = self._p_value * self._Kp + self._i_value * self._Ki + self._d_value * self._Kd

            self._d_time     = d_time
            self._d_error    = d_error
            self._last_time  = cur_time
            self._last_error = error

        return self._output

    def reset(self, last_time = None, set_point = 0.0):
        self._set_point    = set_point
        self._last_time    = last_time if last_time is not None else time()
        self._p_value      = 0.0
        self._i_value      = 0.0
        self._d_value      = 0.0
        self._d_time       = 0.0
        self._d_error      = 0.0
        self._last_error   = 0.0
        self._output       = 0.0

    def assign_set_point(self, set_point):
        self._set_point = set_point

    def get_set_point(self):
        return self._set_point

    def get_p_value(self):
        return self._p_value

    def get_i_value(self):
        return self._i_value

    def get_d_value(self):
        return self._d_value

    def get_delta_time(self):
        return self._d_time

    def get_delta_error(self):
        return self._d_error

    def get_last_error(self):
        return self._last_error

    def get_last_time(self):
        return self._last_time

    def get_output(self):
        return self._output

class Coach(object):

    #BLACK_WALL = 0
    #YELLOW_WALL = 11250176
    #MAX_HISTORY_COUNT = 50
    #MIN_HISTORY_COUNT = 10

    MAX_SPEED = 2.0
    def __init__(self):
        self._steering_pid = PID(Kp=0.3, Ki=0.01, Kd=0.1, max_integral=10)
        self.count = 0
        self.expected_angle_history = [0, 0]
        self.min_delta_angle = self.max_delta_angle_by_speed(self.MAX_SPEED)

    def max_delta_angle_by_speed(self, speed):
        max_delta_angle = 90 * math.exp(-2*speed)
        if max_delta_angle > 45:
            max_delta_angle = 45
        return max_delta_angle

    def alter_action(self, cv2_image, speed, steering_angle, throttle, brakes):
        #print("-- angle:%.2f, throttle:%.2f, speed:%.2f" % (steering_angle, throttle, speed))
        expected_angle = ImageProcessor.find_steering_angle_by_color(cv2_image)
        expected_angle = ImageProcessor.rad2deg(self._steering_pid.update(-expected_angle))
        self.expected_angle_history.append(expected_angle)
        self.expected_angle_history = self.expected_angle_history[-2:]

        max_delta_angle = self.max_delta_angle_by_speed(speed)

        print("angle: %.2f, expected: %.2f, max_delta_angle:%.2f" % (steering_angle, expected_angle, max_delta_angle))

        if speed == 0 and throttle == 0 and brakes == 0:
            return Action.Accelerate

        if expected_angle == 0:
            return Action.Accelerate

        #if abs(expected_angle) < self.min_delta_angle:
        #    return Action.Accelerate

        if expected_angle > 0 and steering_angle >= 0:
            if (expected_angle > steering_angle) and (max_delta_angle <= expected_angle):
                return Action.TurnRight
            else:
                return Action.Accelerate

        if expected_angle > 0 and steering_angle <= 0:
            if max_delta_angle <= expected_angle:
                return Action.TurnRight
            else:
                return Action.NoAction

        if expected_angle < 0 and steering_angle <= 0:
            if (expected_angle < steering_angle) and (max_delta_angle <= -expected_angle):
                return Action.TurnLeft
            else:
                return Action.Accelerate

        if expected_angle < 0 and steering_angle >= 0:
            if max_delta_angle <= -expected_angle:
                return Action.TurnLeft
            else:
                return Action.NoAction

        return Action.NoAction
