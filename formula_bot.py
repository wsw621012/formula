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
    THROTTLE_STEP = 1.0 / 5

    def __init__(self, sio):
        self.angle_step = [15, 12, 9, 6, 3]
        self.max_speed_range = [0.0, 0.4, 0.8, 1.2, 1.6]
        self.queue = Queue(1)
        self.command = Queue(1)
        self.sio = sio
        self.is_finished = False

        @sio.on('telemetry')
        def telemetry(sid, dashboard):
            if dashboard:
                if dashboard['status'] != '0' or float(dashboard['time']) > 90.0:
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

    def reward(self, speed, throttle, brakes):
        # crash
        if speed > 0 and throttle > 0:
            return -1.0

        x = (float(speed) / float((self.MAX_SPEED)) * 100.0) + 0.99
        base = 10
        log_speed =  max(0.0, math.log(x, base) / 2.0)
        if log_speed <= 0.0:
            return -0.04
        elif brakes > 0.0:
            return 0.0
        else:
            return log_speed

    def exec_action(self, action, speed, steering_angle, throttle, brakes):

        idx = 0;
        for i in range(5):
            idx = 4 - i
            if speed >= self.max_speed_range[idx]:
                break;
        print("action = %s" % (action.name))

        if action == 1:
            #Accelerate
            return 0.0, min(1.0, throttle + self.THROTTLE_STEP)
        elif action == 2:
            #Brake
            return 0.0, max(0.0, throttle - self.THROTTLE_STEP)
        elif action == 3:
            #TurnLeft
            angle = min(0, steering_angle) - float(self.angle_step[idx])
            return max(-45.0, angle), throttle
        elif action == 4:
            #TurnRight
            angle = max(0, steering_angle) + float(self.angle_step[idx])
            return min(45.0, angle), throttle
        elif action == 5:
            #AccelerateAndTurnLeft
            angle = min(0, steering_angle) - float(self.angle_step[idx])
            return max(-45, angle), throttle + self.THROTTLE_STEP
        elif action == 6:
            #AccelerateAndTurnRight
            angle = max(0, steering_angle) + float(self.angle_step[idx])
            return min(45, angle), throttle + self.THROTTLE_STEP
        elif action == 7:
            #BrakeAndTurnLeft
            angle = min(0, steering_angle) - float(self.angle_step[idx])
            return max(-45, angle), max(0, throttle - self.THROTTLE_STEP)
        elif action == 8:
            #BrakeAndTurnRight
            angle = max(0, steering_angle) + float(self.angle_step[idx])
            return min(45, angle), max(0, throttle - self.THROTTLE_STEP)
        else:
            #NoAction
            return steering_angle, throttle

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
        num_workers = 1
        workers = []
        # Create worker classes
        for i in range(num_workers):
            workers.append(Worker(FormulaGame(sio),i,s_size,a_size,trainer,model_path,global_episodes))
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
        worker_threads = []
        for worker in workers:
            worker_work = lambda: worker.work(max_episode_length,gamma,sess,coord,saver)
            t = threading.Thread(target=(worker_work))
            t.start()
            #sleep(0.5)
            worker_threads.append(t)

        app = socketio.Middleware(sio, Flask(__name__))
        eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
        coord.join(worker_threads)
