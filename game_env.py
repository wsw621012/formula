from action import Action, Reverse
from multiprocessing import Queue
import numpy as np

class gameEnv(object):
    MAX_STEERING_ANGLE = 40

    def __init__(self, sio):
        self.actions = len(Action)
        self.dashboard = Queue()
        self.command = Queue()
        self.sio = sio
        self.is_finished = False
        #self.steering_angle_list = [2.86, 10.36, 37.24, 45.0]
        self.steering_angle_list = np.asarray([7.47, 15.07, 22.95, 31.33, 40.54])
        #self.lap = 1
        #self.elapse = 0.

        @sio.on('telemetry')
        def telemetry(sid, msg):
            if msg:
                self._process_msg(msg)
            else:
                self.sio.emit('manual', data={}, skip_sid=True)

        @sio.on('connect')
        def connect(sid, environ):
            print("connected...")
            self._send_control(0, 0)

    def get_mix_steering_angle(self):
        return self.steering_angle_list[0]

    def _process_msg(self, msg):
        '''
        if msg['status'] != '0' or int(msg['lap']) > 1 or float(msg['time']) > 300:
            self.is_finished = True

        if int(msg['lap']) > self.lap:
            print("lap:%d, spend %.2f sec" % (int(msg['lap'])-1, (float(msg['time'])-self.elapse)))
            self.lap += 1
            self.elapse = float(msg['time'])
        '''
        self.dashboard.put(msg)

        cmd = self.command.get()
        if cmd is None:
            self.is_finished = False
            self._send_restart()
        else:
            self._send_control(float(cmd['steering_angle']), float(cmd['throttle']))

    def _send_control(self, steering_angle, throttle):
        self.sio.emit(
            "steer",
            data={
                'steering_angle': str(steering_angle),
                'throttle': str(throttle)
            },
            skip_sid=True)

    def reset(self):
        self.command.put(None)

    def _send_restart(self):
        self.is_finished = False
        #self.lap = 1
        #self.elapse = 0.
        self.sio.emit(
            "restart",
            data={},
            skip_sid=True)

    def get_state(self):
        return self.dashboard.get()

    def step(self, state, action):
        steering_angle = float(state['steering_angle'])
        new_angle = 0 # forward
        a = self.steering_angle_list
        if action == Action.TurnRight:
            new_angle = min(40, min(a[a > steering_angle]))
        elif action == Reverse.TurnRight:
            new_angle = 40
        elif action == Action.TurnLeft:
            new_angle = -min(40, min(a[a > abs(min(0, steering_angle))]))
        elif action == Reverse.TurnLeft:
            new_angle = -40

        if action >= 0:
            self._send_cmd(new_angle, 1.0) # dan said always keep highest spped
        else:
            self._send_cmd(new_angle, -1.0)

        msg = self.dashboard.get()
        msg['steering_angle'] = new_angle
        return msg, self.is_finished


    def _send_cmd(self, steering_angle, throttle):
        cmd = {
                'steering_angle': str(steering_angle),
                'throttle': str(throttle)
        }
        self.command.put(cmd)
