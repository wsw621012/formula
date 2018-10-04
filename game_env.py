from action import Action, Reverse
from multiprocessing import Queue


class gameEnv(object):
    MAX_STEERING_ANGLE = 40

    def __init__(self, sio):
        self.actions = len(Action)
        self.dashboard = Queue()
        self.command = Queue()
        self.sio = sio
        self.is_finished = False
        self.last_time = 0.0

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

    def _process_msg(self, msg):
        msg['last_time'] = self.last_time
        self.last_time = msg['time']
        if msg['status'] != '0' or int(msg['lap']) > 1 or float(msg['time']) > 300:
            print("lap = %d, spend %.2f sec" % (int(msg['lap']) -1, float(msg['time'])))
            self.is_finished = True
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
        self.sio.emit(
            "restart",
            data={},
            skip_sid=True)

    def get_state(self):
        return self.dashboard.get()

    #s1,r,d = env.step(a)
    def step(self, state, action):
        new_angle = steering_angle = float(state['steering_angle'])

        if action == Action.TurnRight or action == Reverse.TurnRight:
            new_angle = min(gameEnv.MAX_STEERING_ANGLE, steering_angle + 10)
        elif action == Action.TurnLeft or action == Reverse.TurnLeft:
            new_angle = max(-gameEnv.MAX_STEERING_ANGLE, steering_angle - 10)
        else: # Action.Forward or Reverse.Forward:
            if steering_angle > 0:
                new_angle = max(0, steering_angle - 10)
            else:
                new_angle = min(0, steering_angle + 10)

        if action >= 0:
            self._send_cmd(new_angle, 1.) # dan said always keep highest spped
        else:
            self._send_cmd(new_angle, -1.)

        msg = self.dashboard.get()
        return msg, self.is_finished

    def _send_cmd(self, steering_angle, throttle):
        cmd = {
                'steering_angle': str(steering_angle),
                'throttle': str(throttle)
        }
        self.command.put(cmd)
