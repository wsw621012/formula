
class Car(object):
    MAX_STEERING_ANGLE = 40.0

    def __init__(self, control_function):
        self._control_function = control_function
        self.stage = 0
        self.throttle_range = [0, -1]
        #self.throttle_range = [0, 0, 0, 0, 0]
        self.start_time = 0

    def on_dashboard(self, dashboard):
        angle    = float(dashboard["steering_angle"])
        throttle = float(dashboard["throttle"])
        brakes   = float(dashboard["brakes"])
        speed    = float(dashboard["speed"])
        time     = float(dashboard["time"])

        print("time: %.4f, speed: %.4f, angle: %.2f, throttle: %.4f, brakes: %.2f" % (time, speed, angle, throttle, brakes))

        if self.start_time == 0 and self.stage < len(self.throttle_range):
            if speed < 2:
                self.control(0, 1)
            else:
                self.start_time = time
                print("form 0 to 2, spend:%.2f" % time)
                self.control(0, self.throttle_range[self.stage])
            return

        print("time: %.4f, speed: %.4f, angle: %.2f, throttle: %.4f, brakes: %.2f" % (time, speed, angle, throttle, brakes))
        if speed > 0.05:
            if self.stage < len(self.throttle_range):
                self.control(0, self.throttle_range[self.stage])
                return

        print("from 0 to 2, throttle: %.2f, elapse: %.2f" % (self.throttle_range[self.stage], time - self.start_time))
        self.stage += 1
        self.start_time = 0
        send_restart()
        #    self.control(0.0, -0.005)
        #    return

        #if speed > 2.0:
        #    self.pending = 1

        #self.control(0, 1)

    def control(self, steering_angle, throttle):
        #convert the values with proper units
        steering_angle = min(max(steering_angle, -Car.MAX_STEERING_ANGLE), Car.MAX_STEERING_ANGLE)
        self._control_function(steering_angle, throttle)


if __name__ == "__main__":
    import shutil
    from datetime import datetime

    import socketio
    import eventlet
    import eventlet.wsgi
    from flask import Flask

    sio = socketio.Server()
    def send_control(steering_angle, throttle):
        sio.emit(
            "steer",
            data={
                'steering_angle': str(steering_angle),
                'throttle': str(throttle)
            },
            skip_sid=True)

    def send_restart():
        sio.emit(
            "restart",
            data={},
            skip_sid=True)

    car = Car(control_function = send_control)

    @sio.on('telemetry')
    def telemetry(sid, dashboard):
        if dashboard:
            car.on_dashboard(dashboard)
        else:
            sio.emit('manual', data={}, skip_sid=True)

    @sio.on('connect')
    def connect(sid, environ):
        car.control(0, 0)

    app = socketio.Middleware(sio, Flask(__name__))
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

    #producer.join()
