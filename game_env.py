from action import Action
from multiprocessing import Queue
from controller import Controller

class gameEnv(object):
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
        if msg['status'] != '0' or int(msg['lap']) > 1 or float(msg['time']) > 180:
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
        new_throttle = throttle = float(state['throttle'])
        #new_brakes = brakes = float(state['brakes'])

        time = float(state['time'])
        last_time = float(state['last_time'])

        if action == Action.TurnLeft:
            new_angle = Controller.turn_left(steering_angle, time, last_time)
        elif action == Action.TurnRight:
            new_angle = Controller.turn_right(steering_angle, time, last_time)
        else: # forward
            new_angle = Controller.forward(steering_angle, time, last_time)

        self._send_cmd(new_angle, 1.0) # dan said always keep highest spped

        msg = self.dashboard.get()
        return msg, self.is_finished

    def _send_cmd(self, steering_angle, throttle):
        cmd = {
                'steering_angle': str(steering_angle),
                'throttle': str(throttle)
        }
        self.command.put(cmd)
