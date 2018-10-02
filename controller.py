

class Controller():
    MAX_ANGLE = 40

    @staticmethod
    def forward(steering_angle, time, last_time):
        if steering_angle > 0:
            return max(0, steering_angle - 10)
        else:
            return min(0, steering_angle + 10)

    @staticmethod
    def turn_left(steering_angle, time, last_time):
        return max(-Controller.MAX_ANGLE, steering_angle - 10)

    @staticmethod
    def turn_right(steering_angle, time, last_time):
        return min(Controller.MAX_ANGLE, steering_angle + 10)
