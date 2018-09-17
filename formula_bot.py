import os

from actor_critic_network import AC_Network
from formula_a3c import Worker

import numpy as np

import threading
import math

import tensorflow as tf
import socketio
import eventlet
import eventlet.wsgi
from flask import Flask
from action import Action
from multiprocessing import Queue

import logging

def logit(msg):
    print("%s" % msg)

class FormulaGame(object):
    MAX_SPEED = 2.0
    #STEP_ANGLE = 2

    SPEED_UP = 1.0 # 1 m/sec^2
    SPEED_DOWN = 0.0 # -0.5 m/sec^2
    BRAKE = -1.0 # -2 m/sec^2

    def __init__(self, sio):
        self.queue = Queue(1)
        self.command = Queue(1)
        self.sio = sio
        self.is_finished = False
        self.min_delta_angle = self.max_delta_angle_by_speed(self.MAX_SPEED)

        @sio.on('telemetry')
        def telemetry(sid, dashboard):
            if dashboard:
                if dashboard['status'] != '0' or int(dashboard['lap']) > 5 or float(dashboard["time"]) > 180:
                    print("lap = %d, spend %.2f sec" % (int(dashboard['lap']) -1, float(dashboard["time"])))
                    self.is_finished = True

                self.queue.put(dashboard)

                cmd = self.command.get()
                if cmd is None:
                    self.is_finished = False
                    self.new_episode()
                    #self.send_control(0, 0)
                else:
                    self.send_control(float(cmd['steering_angle']), float(cmd['throttle']))
                #self.command.task_done()
            else:
                sio.emit('manual', data={}, skip_sid=True)

        @sio.on('connect')
        def connect(sid, environ):
            self.send_control(0, 0)
            print("on connect")

    def max_delta_angle_by_speed(self, speed):
        max_delta_angle = 90 * math.exp(-1.5*speed)
        if max_delta_angle > 45:
            max_delta_angle = 45
        return max_delta_angle

    def exec_action(self, action, speed, steering_angle, throttle, brakes):
        max_delta_angle = self.max_delta_angle_by_speed(speed)

        print("action = %s, max_delta_angle = %.2f, speed = %.2f, angle = %.2f, throttle = %.2f" % (action.name, max_delta_angle, speed, steering_angle, throttle))
        print("===")
        if action == 1:
            #Accelerate
            if abs(steering_angle) < self.min_delta_angle:
                return 0.0, 1
            if steering_angle >= max_delta_angle:
                return max_delta_angle, 1
            if steering_angle <= -max_delta_angle:
                return -max_delta_angle, 1
            return 0.0, 1
        elif action == 2:
            #TurnLeft
            return -max_delta_angle, 0.05
        elif action == 3:
            #TurnRight
            return max_delta_angle, 0.05
        else:
            #NoAction
            #return steering_angle, throttle
            return 0, 0

    def is_episode_finished(self):
        return self.is_finished

    def get_state(self):
        msg = self.queue.get()
        #self.queue.task_done()
        return msg

    def send_none(self):
        self.command.put(None)

    def send_cmd(self, steering_angle, throttle):
        cmd = {
                'steering_angle': str(steering_angle),
                'throttle': str(throttle)
        }
        self.command.put(cmd)

    def send_control(self, steering_angle, throttle):
        self.sio.emit(
            "steer",
            data={
                'steering_angle': str(steering_angle),
                'throttle': str(throttle)
            },
            skip_sid=True)

    def new_episode(self):
        self.is_finished = False
        self.sio.emit(
            "restart",
            data={},
            skip_sid=True)


gamma = .99 # discount rate for advantage estimation and reward discounting
load_model = True
model_path = './model'
s_size = 80 * 80 * 3 # Observations are greyscale frames of 84 * 84 * (1:gray, 3:RGB)
a_size = len(Action) # Action(enum)
max_episode_length = 300

if __name__ == "__main__":
    tf.reset_default_graph()

    #Create a directory to save model
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    #Create a directory to save episode playback gifs to
    if not os.path.exists('./frames'):
        os.makedirs('./frames')

    sio = socketio.Server()

    with tf.device("/cpu:0"):
        global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
        trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
        master_network = AC_Network(s_size,a_size,'global',None) # Generate global network
        #num_workers = multiprocessing.cpu_count() # Set workers to number of available CPU threads

        # Create worker classes
        worker = Worker(FormulaGame(sio),0,s_size,a_size,trainer,model_path,global_episodes)
        saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        if load_model == True:
            print ('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess,ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate threat.

        worker_work = lambda: worker.work(max_episode_length,gamma,sess,coord,saver)
        t = threading.Thread(target=(worker_work))
        t.start()

        #sleep(0.5)

        app = socketio.Middleware(sio, Flask(__name__))
        eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
        #coord.join(worker_threads)
