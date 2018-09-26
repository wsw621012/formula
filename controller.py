

class Controller():
    MAX_SPEED = 2.0
    MAX_ANGLE = 40

    @staticmethod
    def speed_up(throttle):
        return min(Controller.MAX_SPEED, throttle + 0.2)

    @staticmethod
    def speed_down(throttle):
        return max(0, throttle - 0.2)


    @staticmethod
    def forward(steering_angle, time, last_time):
        delta_angle = (float(time) - float(last_time)) * 100
        if steering_angle > 0:
            return max(0, steering_angle - delta_angle)
        else:
            return min(0, steering_angle + delta_angle)

    @staticmethod
    def turn_left(steering_angle, time, last_time):
        delta_angle = (float(time) - float(last_time)) * 100
        return max(-Controller.MAX_ANGLE, steering_angle - delta_angle)

    @staticmethod
    def turn_right(steering_angle, time, last_time):
        delta_angle = (float(time) - float(last_time)) * 100
        return min(Controller.MAX_ANGLE, steering_angle + delta_angle)
