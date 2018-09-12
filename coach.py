from image_processor import ImageProcessor
from action import Action
from time import time
import numpy as np
import operator
import math
import cv2
import os
from PIL  import Image


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

    def __init__(self):
        self._steering_pid = PID(Kp=0.3, Ki=0.01, Kd=0.1, max_integral=10)
        self.count = 0
        
        self.delta_angle_history = [0,0]
        self.angle_history = [0,0]

    def alter_action(self, cv2_image, speed, steering_angle, throttle, brakes):
        #print("-- angle:%.2f, throttle:%.2f, speed:%.2f" % (steering_angle, throttle, speed))
        expected_angle = ImageProcessor.find_steering_angle_by_color(cv2_image)
        expected_angle = ImageProcessor.rad2deg(self._steering_pid.update(-expected_angle))
        self.angle_history.append(expected_angle)
        self.angle_history = self.angle_history[-2:]

        print("angle now:%.2f, should be:%.2f - speed:%.2f" % (steering_angle, expected_angle, speed))
        delta_angle = expected_angle - steering_angle
        self.delta_angle_history.append(delta_angle)
        self.delta_angle_history = self.delta_angle_history[-2:]


        if speed == 0 and throttle == 0 and brakes == 0:
            return Action.Accelerate
        elif abs(delta_angle) < 3:
            return Action.Accelerate
        elif delta_angle > 45:
            return Action.BrakeAndTurnRight
        elif delta_angle < -45:
            return Action.BrakeAndTurnLeft
        elif delta_angle > 30:
            return Action.TurnRight
        elif delta_angle < -30:
            return Action.TurnLeft
        elif delta_angle > 15:
            if (self.angle_history[0] < 0) and (self.angle_history[1] < 0) and (self.angle_history[0] < self.angle_history[1]):
                return Action.NoAction
            elif (self.angle_history[0] > 0) and (self.angle_history[1] > 0) and (self.angle_history[0] > self.angle_history[1]):
                return Action.NoAction
            else:
                return Action.AccelerateAndTurnRight
        elif delta_angle < -15:
            if (self.angle_history[0] < 0) and (self.angle_history[1] < 0) and (self.angle_history[0] < self.angle_history[1]):
                return Action.NoAction
            elif (self.angle_history[0] > 0) and (self.angle_history[1] > 0) and (self.angle_history[0] > self.angle_history[1]):
                return Action.NoAction
            else:
                return Action.AccelerateAndTurnLeft
        elif delta_angle > 0:
            if delta_angle < self.delta_angle_history[0]:
                return Action.NoAction
            else:
                return Action.Accelerate
        elif delta_angle < 0:
            if delta_angle > self.delta_angle_history[0]:
                return Action.NoAction
            else:
                return Action.Accelerate
        else:
            return Action.NoAction
        '''
        elif abs(delta_angle) > float(self.angle_step[idx]):
            if delta_angle < 0:
                if idx == 4:
                    return Action.BrakeAndTurnLeft
                elif idx == 0:
                    return Action.AccelerateAndTurnLeft
                else:
                    return Action.TurnLeft
            else:
                if idx == 4:
                    return Action.BrakeAndTurnRight
                elif idx == 0:
                    return Action.AccelerateAndTurnRight
                else:
                    return Action.TurnRight '''
