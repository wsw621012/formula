from __future__ import division

import numpy as np
import random, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import scipy.misc
import os, sys
import threading

from multiprocessing import Queue
from action import Action, Reverse
from game_env import gameEnv
from image_processor import ImageProcessor
from count_sign_classify import CountDetection

#env = gameEnv(partial=False,size=5)

class Qnetwork():
    def __init__(self,h_size, env):
        #The network recieves a frame from the game, flattened into an array.
        #It then resizes it and processes it through four convolutional layers.
        self.scalarInput =  tf.placeholder(shape=[None,21168],dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput,shape=[-1,84,84,3])
        self.conv1 = slim.conv2d( \
            inputs=self.imageIn,num_outputs=32,kernel_size=[8,8],stride=[4,4],padding='VALID', biases_initializer=None)
        self.conv2 = slim.conv2d( \
            inputs=self.conv1,num_outputs=64,kernel_size=[4,4],stride=[2,2],padding='VALID', biases_initializer=None)
        self.conv3 = slim.conv2d( \
            inputs=self.conv2,num_outputs=64,kernel_size=[3,3],stride=[1,1],padding='VALID', biases_initializer=None)
        self.conv4 = slim.conv2d( \
            inputs=self.conv3,num_outputs=h_size,kernel_size=[7,7],stride=[1,1],padding='VALID', biases_initializer=None)

        #We take the output from the final convolutional layer and split it into separate advantage and value streams.
        self.streamAC,self.streamVC = tf.split(self.conv4,2,3)
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([h_size//2,env.actions]))
        self.VW = tf.Variable(xavier_init([h_size//2,1]))
        self.Advantage = tf.matmul(self.streamA,self.AW)
        self.Value = tf.matmul(self.streamV,self.VW)

        #Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keepdims=True))
        self.predict = tf.argmax(self.Qout,1)

        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions,env.actions,dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

class experience_buffer():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self,size):
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,5])

def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

#def updateTarget(op_holder, sess):
#    for op in op_holder:
#        sess.run(op)


def update_job(myBuffer, op_holder, queue, sess, mainQN, targetQN):
    while True:
        batch_size = queue.get()
        trainBatch = myBuffer.sample(batch_size) #Get a random batch of experiences.
        #Below we perform the Double-DQN update to the target Q-values
        Q1 = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,3])})
        Q2 = sess.run(targetQN.Qout,feed_dict={targetQN.scalarInput:np.vstack(trainBatch[:,3])})
        end_multiplier = -(trainBatch[:,4] - 1)
        doubleQ = Q2[range(batch_size),Q1]
        targetQ = trainBatch[:,2] + (y*doubleQ * end_multiplier)
        #Update the network with our target values.
        _ = sess.run(mainQN.updateModel, \
            feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,0]),mainQN.targetQ:targetQ, mainQN.actions:trainBatch[:,1]})

        for op in op_holder: #Update the target network toward the primary network.
            sess.run(op)


batch_size = 32 #How many experiences to use for each training step.
update_freq = 2048 #How often to perform a training step.
y = .99 #Discount factor on the target Q-values
startE = 0.9 #Starting chance of random action
endE = 0.1 #Final chance of random action
annealing_steps = 800000. #How many steps of training to reduce startE to endE.
num_episodes = 100 #How many episodes of game environment to train network with.
pre_train_steps = 10000 #How many steps of random actions before training begins.
max_epLength = 10000 #The max allowed length of our episode.
load_model = True #Whether to load a saved model.
h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 #Rate to update target network toward primary network

class Worker():
    def __init__(self, env):
        self.env = env
        self.stop = False
        #self.frame_count = 0
        self.coach_actions = []
        self.last_wall_angle = None
        self.update_channel = Queue()
        self._sign_detection = CountDetection()
        self.sign_dir = None
        self.sign_count_down = 0
        self.bottom_line = []
        self.last_bottom_line = []
        self.reward_offset = 4

    def _detect_wall(self, im_gray, action):

        target = ImageProcessor._crop_gray(im_gray, 0.6, 1.0)
        self.env.is_finished = ImageProcessor.find_final_line(target)
        if self.env.is_finished:
            return None

        wall_angle, lx, ly, rx, ry = ImageProcessor.find_wall_angle(target)

        if wall_angle is None:
            self.last_wall_angle = None
            return None

        if wall_angle == 180:
            self.coach_actions.append(Reverse.Backward)
            return 180

        if 0 in target[0,:] and (ry == (target.shape[0] - 1) or ly == (target.shape[0] - 1)):
            if lx > 0 and lx > (target.shape[1] - rx):
                self.coach_actions.append(Reverse.TurnRight)
            elif (target.shape[1] - rx - 1) > lx:
                self.coach_actions.append(Reverse.TurnLeft)
            else:
                self.coach_actions.append(Reverse.Backward)
            return 180

        # degrees:74 is atan(320 / 95)
        # degrees:60 is atan(160 / 95)
        if action >= 0 and abs(wall_angle) < 90:
            self.last_wall_angle = wall_angle

        #wall_distance = target.shape[0] - ((ly + ry) // 2)
        wall_distance = target.shape[0] - max([ly, ry])
        #if (ly + ry) // 2 >= 25: # too close wall
        if wall_distance < 75:
            if action < 0:
                if self.last_wall_angle is None:
                    self.coach_actions.append(Reverse.Backward)
                elif self.last_wall_angle > 0:
                    self.coach_actions.append(Reverse.TurnLeft)
                else:
                    self.coach_actions.append(Reverse.TurnRight)
                return wall_angle

            if lx == 0 and rx == target.shape[1] - 1:
                if wall_angle > 0:
                    self.coach_actions.append(Action.TurnRight)
                    return wall_angle
                if wall_angle < 0:
                    self.coach_actions.append(Action.TurnLeft)
                    return wall_angle

        if lx > 0 and lx > (target.shape[1] - rx) and max(ly, ry) > 7: # obstacle was approach left-hand-side
            self.coach_actions.append(Action.TurnLeft)
            return wall_angle

        if (target.shape[1] - rx - 1) > lx and max(ly, ry) > 7:
            self.coach_actions.append(Action.TurnRight)
            return wall_angle

        #print("no coach action when wall_angle is %.2f" % wall_angle)
        return wall_angle

    def processState(self, state, action = 0):
        image = ImageProcessor.preprocess(state['image'])
        del state['image']
        im_gray = ImageProcessor._flatten_rgb_to_gray(image)

        self.last_bottom_line = self.bottom_line
        self.bottom_line = []
        _ly = im_gray[-60, :]
        for ly in np.split(_ly, np.where(np.diff(_ly) != 0)[0]+1):
            self.bottom_line.append((ly[0], len(ly)))

        #steering_angle = float(state['steering_angle'])
        #angle = self._detect_wall(im_gray, steering_angle, action)
        angle = self._detect_wall(im_gray, action)
        state['wall'] = 'n' if angle is None else 'y'

        '''
        pos , prop , sign= self._sign_detection.classify_sign_from_image(image)
        if len(sign) != 0:
            if sign[0] == 5: # right
                self.sign_dir = 'right'
                self.sign_count_down = 15
            elif sign[0] == 4: #left
                self.sign_dir = 'left'
                self.sign_count_down = 15
        '''
        '''
        if self.sign_count_down > 0:
            if self.sign_dir == 'right':
                if self.last_wall_angle is None and len(self.coach_actions) == 0:
                    self.coach_actions.append(Action.TurnRight)
            elif self.sign_dir == 'left':
                if self.last_wall_angle is None and len(self.coach_actions) == 0:
                    self.coach_actions.append(Action.TurnLeft)
            self.sign_count_down -= 1
            if self.sign_count_down == 0:
                print("sign action over")
                self.sign_dir = None
        '''
        #if angle == None:
        #    angle = ImageProcessor.find_color_angle(im_gray)

        # no wall
        #state['angle'] = str(angle)
        image = image[100:240, 0:320]
        return np.reshape(scipy.misc.imresize(image, [84,84]), [21168]) / 255.0

    #compare 2 image
    def mse(self, icon1, icon2):
	    err = np.sum((icon1 - icon2.astype("float")) ** 2)
	    err /= float(icon1.shape[0] * icon2.shape[1])
	    return err

    def reward(self, state, state_):

        #last_steering_angle, steering_angle = float(state['steering_angle']), float(state_['steering_angle'])
        #previous_angle, current_angle = float(state['angle']), float(state_['angle'])
        if state_['wall'] == 'y':
            return -1

        color_seq, last_color_seq = [item[0] for item in self.bottom_line], [item[0] for item in self.last_bottom_line]
        color_count, last_color_count = len(self.bottom_line), len(self.last_bottom_line)

        if (0 in color_seq) and (not self.env.is_finished):
            return -1.

        if not np.array_equal(color_seq, last_color_seq):
            if color_count > 1 and last_color_count > 1:
                if color_seq[0] == last_color_seq[0] and abs(self.last_bottom_line[0][1] - self.bottom_line[0][1]) < self.reward_offset:
                    return 1
                if color_seq[-1] == last_color_seq[-1] and abs(self.last_bottom_line[-1][1] - self.bottom_line[-1][1]) < self.reward_offset:
                    return 1
            return -1.

        if color_count > 1: # in the same relative-position
            offset = self.last_bottom_line[0][1] - self.bottom_line[0][1]
            #print('shift %d pixel' % offset)

            if abs(offset) < self.reward_offset:
                return 1.

            if color_count == 3 and color_seq[1] == 76 and self.bottom_line[1][1] < 65: # in the middle way
                return 1

        return -1

    def run(self):
        tf.reset_default_graph()
        mainQN = Qnetwork(h_size, env)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        #trainables = tf.trainable_variables()
        #Make a path for our model to be saved in.
        with tf.Session() as sess:
            sess.run(init)
            ckpt = tf.train.get_checkpoint_state(".")
            saver.restore(sess, ckpt.model_checkpoint_path)

            self.coach_actions = []
            state = self.env.get_state()
            state['steering_angle'] = 0
            s = self.processState(state)
            self.last_steering_angle = 0
            #d = False
            while True:    #Choose an action by greedily (with e chance of random action) from the Q-network
                if len(self.coach_actions) > 0:
                    a = self.coach_actions.pop(0)
                else:
                    a = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:[s]})[0]

                state, d = self.env.step(state, a)
                s = self.processState(state, a)

            self.env.reset()

    def train(self, path, debug = False):
        tf.reset_default_graph()

        mainQN = Qnetwork(h_size, env)
        targetQN = Qnetwork(h_size, env)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=5)
        trainables = tf.trainable_variables()

        targetOps = updateTargetGraph(trainables,tau)
        myBuffer = experience_buffer()
        #Set the rate of random action decrease.
        e = startE
        stepDrop = (startE - endE)/annealing_steps

        #create lists to contain total rewards and steps per episode
        rList = []
        total_steps = 0

        #Make a path for our model to be saved in.
        if not os.path.exists(path):
            os.makedirs(path)

        with tf.Session() as sess:
            sess.run(init)

            updateThread = threading.Thread(target=update_job, args=(myBuffer, targetOps, self.update_channel, sess,mainQN, targetQN,))  # <- note extra ','
            updateThread.start()

            if load_model == True:
                print('Loading Model...')
                ckpt = tf.train.get_checkpoint_state(path)
                saver.restore(sess,ckpt.model_checkpoint_path)
            for eps in range(num_episodes):
                round_step_counter = 0
                coach_count = 0
                self.coach_actions = []
                state = self.env.get_state()
                state['steering_angle'] = 0
                s = self.processState(state)
                self.env.is_finished = False
                d = False
                rAll = 0
                last_action = 0
                episodeBuffer = experience_buffer()
                j = 0
                while j < max_epLength:    #Choose an action by greedily (with e chance of random action) from the Q-network
                    j += 1
                    if len(self.coach_actions) > 0:
                        a = self.coach_actions.pop(0)
                        coach_count += 1
                        if debug:
                            print("XX %s(%d)" % ((Action(a).name if a>=0 else Reverse(a).name), round_step_counter))

                    elif np.random.rand(1) < e or total_steps < pre_train_steps:
                        a = np.random.choice(3, 1, p=[0.3, 0.4, 0.3])[0]

                        if debug:
                            print("-- %s(%d)" % (Action(a).name, round_step_counter))
                    else:
                        a = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:[s]})[0]
                        if debug:
                            print("OO %s(%d)" % (Action(a).name, round_step_counter))

                    state_, d = self.env.step(state, a)
                    s1 = self.processState(state_, a)

                    if len(self.coach_actions) > 0:
                        r = -1
                    else:
                        r = self.reward(state, state_)
                    round_step_counter += 1

                    if a >= 0:
                        total_steps += 1
                        episodeBuffer.add(np.reshape(np.array([s,a,r,s1,d]),[1,5])) #Save the experience to our episode buffer.
                        if total_steps == pre_train_steps + 1:
                            print("pre-trainning end.")

                    if total_steps > pre_train_steps:
                        if e > endE:
                            e -= stepDrop

                        if total_steps % (update_freq) == 0:
                            self.update_channel.put_nowait(batch_size)
                            #updateTarget(targetOps,sess) #Update the target network toward the primary network.

                    rAll += r
                    s = s1
                    state = state_
                    if d:
                        if round_step_counter < 10:
                            d = False
                            self.env.is_finished = False
                        else:
                            print("round:%d, step_counter = %d, coach rate = %.2f%%" % (eps, round_step_counter, (100*coach_count/round_step_counter)))
                            break
                    #if (len(rList) < 1) or (rAll/round_step_counter > max(rList)):

                    #if len(rList) >= 1:
                    #    print("add to learn - r:%.2f(%.2f)." % (rAll/round_step_counter, max(rList)))

                #if d and ((len(rList) < 4 ) or (rAll/round_step_counter > np.mean(rList[-4:]))):
                if d:
                    myBuffer.add(episodeBuffer.buffer)
                    rList.append(rAll/round_step_counter)
                else:
                    print("not finished or score lower than mean score.")
                #Periodically save the model.
                #if (eps + 1) % 4 == 0:
                model_path = '%s/model-%d.ckpt' % (path, eps)
                saver.save(sess,path+'/model-'+str(eps)+'.ckpt')
                print("Saved Model: %s" % model_path)

                if len(rList) >= 4:
                    print("(random:%.2f%%)%d, %d, %.2f" %(e*100, round_step_counter, rAll, np.mean(rList[-4:])))
                else:
                    print("(random:%.2f%%)%d, %d, %.2f" %(e*100, round_step_counter, rAll, np.mean(rList)))

                if self.stop == True:
                    break

                self.env.reset()
            saver.save(sess,path+'/model-'+str(eps)+'.ckpt')
            updateThread.join()

        #print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")

if __name__ == "__main__":
    import socketio
    import eventlet
    import eventlet.wsgi
    from flask import Flask

    sio = socketio.Server()

    env = gameEnv(sio)

    worker = Worker(env)

    path = "./dqn" #The path to save our model to.

    if len(sys.argv) > 1 and sys.argv[1] == 'run':
        print("execution mode...")
        #path = './candidate_models'
        worker_work = lambda: worker.run()
        t = threading.Thread(target=(worker_work))
        t.start()

    else:
        print("training mode...")
        worker_work = lambda: worker.train(path, debug = False)
        t = threading.Thread(target=(worker_work))
        t.start()

    app = socketio.Middleware(sio, Flask(__name__))
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

    worker.stop = True
    print("game over")
