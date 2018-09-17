import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal

from PIL  import Image
from io   import BytesIO
import base64
import math

from coach import Coach

from actor_critic_network import AC_Network

from action import Action
from random import choice

from time import time
from image_processor import ImageProcessor


def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# Processes screen image to produce cropped and resized image.
def process_frame(frame):
    #s = frame[10:-10,30:-30]
    s = scipy.misc.imresize(frame, [80,80])
    s = np.reshape(s,[np.prod(s.shape)]) / 255.0
    return s


class Worker():
    def __init__(self,game,name,s_size,a_size,trainer,model_path,global_episodes):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_"+str(self.number))

        self.coach = Coach()

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(s_size,a_size,self.name,trainer)
        self.update_local_ops = update_target_graph('global',self.name)

        self.actions = np.identity(a_size,dtype=bool).tolist()

        self.car = game

    # Discounting function used to calculate discounted returns.
    def discount(self, x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    def train(self,rollout,sess,gamma,bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:,0]
        actions = rollout[:,1]
        rewards = rollout[:,2]
        next_observations = rollout[:,3]
        values = rollout[:,5]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = self.discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = self.discount(advantages,gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v:discounted_rewards,
            self.local_AC.inputs:np.vstack(observations),
            self.local_AC.actions:actions,
            self.local_AC.advantages:advantages,
            self.local_AC.state_in[0]:self.batch_rnn_state[0],
            self.local_AC.state_in[1]:self.batch_rnn_state[1]}

        v_l,p_l,e_l,g_n,v_n, self.batch_rnn_state,_ = sess.run([self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.state_out,
            self.local_AC.apply_grads],
            feed_dict=feed_dict)

        return v_l / len(rollout),p_l / len(rollout),e_l / len(rollout), g_n,v_n

    def reward(self, is_crash ,speed, throttle, brakes):
        # crash
        if is_crash:
            return -1

        x = (float(speed) / 2.0 * 100.0) + 0.99
        base = 10
        log_speed =  max(0.0, math.log(x, base) / 2.0)
        if log_speed <= 0.0:
            return -0.04
        else:
            return log_speed

    def work(self,max_episode_length,gamma,sess,coord,saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print ("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                #episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                d = False

                payload = self.car.get_state()

                steering_angle = float(payload["steering_angle"])
                throttle = float(payload["throttle"])
                speed = float(payload["speed"])
                brakes = float(payload["brakes"])
                cv2_img = ImageProcessor.preprocess(payload["image"])

                s = process_frame(cv2_img)

                rnn_state = self.local_AC.state_init
                self.batch_rnn_state = rnn_state

                print("episode:%d" % episode_count)

                while self.car.is_episode_finished() == False:
                    #Take an action using probabilities from policy network output.
                    a_dist,v,rnn_state = sess.run([self.local_AC.policy,self.local_AC.value,self.local_AC.state_out],
                        feed_dict={self.local_AC.inputs:[s],
                        self.local_AC.state_in[0]:rnn_state[0],
                        self.local_AC.state_in[1]:rnn_state[1]})

                    aa = np.random.choice(a_dist[0],p=a_dist[0])
                    print("nn choice aa = %s" % aa)
                    aa = np.argmax(a_dist == aa)
                    print("np.argmax(a_dist == aa) = %d" % aa)
                    action = Action(aa)
                    print("nn choice action:%s" % action.name)
                    #if episode_count % 20 != 0:
                    action = self.coach.alter_action(cv2_img, speed, steering_angle, throttle, brakes)

                    steering_angle, throttle = self.car.exec_action(action, speed, steering_angle, throttle, brakes)
                    #print("angle:%.2f, throttle:%.2f, speed:%.2f" % (steering_angle, throttle, speed))
                    self.car.send_cmd(steering_angle, throttle)

                    # get response
                    payload = self.car.get_state()

                    cv2_img = ImageProcessor.preprocess(payload["image"])
                    steering_angle = float(payload["steering_angle"])
                    throttle = float(payload["throttle"])
                    speed = float(payload["speed"])
                    brakes = float(payload["brakes"])
                    is_crash = ImageProcessor.wall_detection(cv2_img)

                    r = self.reward(is_crash, speed, throttle, brakes)
                    #print("reward = %s" % r)
                    d = self.car.is_episode_finished()
                    if is_crash:
                        d = True

                    if d == False:
                        s_ = process_frame(cv2_img)
                    else:
                        s_ = s

                    episode_buffer.append([s,action,r,s_,d,v[0,0]])
                    episode_values.append(v[0,0])

                    episode_reward += r
                    s = s_
                    total_steps += 1
                    episode_step_count += 1

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer) == 30 and d != True and episode_step_count != max_episode_length - 1:
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        v1 = sess.run(self.local_AC.value,
                            feed_dict={self.local_AC.inputs:[s],
                            self.local_AC.state_in[0]:rnn_state[0],
                            self.local_AC.state_in[1]:rnn_state[1]})[0,0]

                        try:
                            v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,v1)
                            episode_buffer = []
                            sess.run(self.update_local_ops)
                            #saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                            #print ("Saved Model")
                        except:
                            print("exception occurred when train")

                    if d == True:
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the episode buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,0.0)


                # Periodically save gifs of episodes, model parameters, and summary statistics.
                #if episode_count % 5 == 0 and episode_count != 0:
                if episode_count != 0:
                    '''if self.name == 'worker_0' and episode_count % 25 == 0:
                        time_per_step = 0.05
                        images = np.array(episode_frames)
                        make_gif(images,'./frames/image'+str(episode_count)+'.gif',
                            duration=len(images)*time_per_step,true_image=True,salience=False)
                    '''
                    #if episode_count % 50 == 0 and self.name == 'worker_0':
                    if episode_count % 5 == 0 and self.name == 'worker_0':
                        saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                        print ("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.increment)

                episode_count += 1
                self.car.send_none()
                print("episode_count is: %s" % episode_count)
