import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.mujoco import mujoco_env
from gym import utils
import numpy as np
import csv
from pyquaternion import Quaternion
# from numpy import linalg as npla

class DyrosRedEnv(mujoco_env.MujocoEnv, utils.EzPickle):
  # metadata = {'render.modes': ['human'],
  #         'video.frames_per_second' : 144
  #         }
  def __init__(self):
        f_ref_qpos = open('/home/dg/Downloads/mocap_data/training_reference/q_pos.csv', 'r', encoding='utf-8')
        f_ref_qvel = open('/home/dg/Downloads/mocap_data/training_reference/q_vel.csv', 'r', encoding='utf-8')
        f_ref_com = open('/home/dg/Downloads/mocap_data/training_reference/com_pos_ref.csv', 'r', encoding='utf-8')
        f_ref_foot = open('/home/dg/Downloads/mocap_data/training_reference/foot_pos_ref.csv', 'r', encoding='utf-8')

        f_ref_pelv_pos = open('/home/dg/Downloads/mocap_data/training_reference/pelv_pos_abs.csv', 'r', encoding='utf-8')
        f_ref_pelv_quat = open('/home/dg/Downloads/mocap_data/training_reference/pelv_quat_abs.csv', 'r', encoding='utf-8')
        rdr_ref_qpos = csv.reader(f_ref_qpos)
        # rdr_ref_quat = csv.reader(f_ref_quat)
        rdr_ref_qvel = csv.reader(f_ref_qvel)
        # rdr_ref_qaxis = csv.reader(f_ref_qaxis)
        rdr_ref_com = csv.reader(f_ref_com)
      #   rdr_ref_rwrist = csv.reader(f_ref_rwrist)
      #   rdr_ref_lwrist = csv.reader(f_ref_lwrist)
        rdr_ref_foot = csv.reader(f_ref_foot)
        # rdr_ref_rfoot = csv.reader(f_ref_rfoot)
        # rdr_ref_lfoot = csv.reader(f_ref_lfoot)
        rdr_ref_pelv_pos = csv.reader(f_ref_pelv_pos)
        rdr_ref_pelv_quat = csv.reader(f_ref_pelv_quat)
        self.phase_size = 38
        def get_data(data, rows):
              data_batch = []
              for line in data:
                    line = [float(i) for i in line]
                    data_batch.append(line)
              return np.asarray(np.reshape(data_batch, (rows, -1)))
        self.ref_qpos = get_data(rdr_ref_qpos, 39)
        # self.ref_quat = get_data(rdr_ref_quat, 39)
        self.ref_qvel = get_data(rdr_ref_qvel, 38)
        # self.ref_qaxis = get_data(rdr_ref_qaxis, 39)
        self.ref_com = get_data(rdr_ref_com, 38)
      #   self.ref_rwrist = get_data(rdr_ref_rwrist, 39)
      #   self.ref_lwrist = get_data(rdr_ref_lwrist, 39)
        self.ref_foot = get_data(rdr_ref_foot, 38)
        # self.ref_rfoot = get_data(rdr_ref_rfoot, 39)
        # self.ref_lfoot = get_data(rdr_ref_lfoot, 39)
        self.ref_pelv_pos = get_data(rdr_ref_pelv_pos, 39)
        self.ref_pelv_quat = get_data(rdr_ref_pelv_quat, 39)

        self.eps_start_time = 0
        self.start_phase = np.random.random_sample()
        # self.start_phase = 0
        self.start_phase_count = int((self.phase_size*self.start_phase)%self.phase_size)
        self.time_count = self.start_phase_count
        self.time = 0
        self.hip_jnt_start = 0

        self.vel_des = np.zeros(3)
        self.vel_des[0] = 1
        mujoco_env.MujocoEnv.__init__(self, '/home/dg/tsallis_actor_critic_mujoco/custom_gym/custom_gym/envs/assets/dyros_red_robot_leg_fixed.xml', 40)
        utils.EzPickle.__init__(self)

  def step(self, action):
        data = self.sim.data

        totalmass = np.sum(np.expand_dims(self.model.body_mass, 1))
        alive_bonus = 0.001
        wp = 0.5; wr = 0.2; wv = 0.05; we = 0.15; wc = 0.1
        # wp = 0.7; wv = 0.3; we = 0.00; wc = 0.00
        
        self.phase_size = 38

        pos_before = mass_center(self.model, self.sim)

        # self.torque = self.pdcontrol(action)
        self.additional_command = 0.2*np.tanh(action)
        self.position_command = self.ref_qpos[(self.time_count+1)%self.phase_size, :] + self.additional_command
        # self.torque = self.pdcontrol(self.position_command)
        self.torque = self.pdcontrol(action)
        self.do_simulation(self.torque, self.frame_skip)

        pos_after = mass_center(self.model, self.sim)

      #   if self.eps_start_time == self.time:


        self.time = data.time              
        self.time_count = (round((self.time-self.eps_start_time)/self.dt) + self.start_phase_count)%self.phase_size 
        self.phase = [self.time_count/self.phase_size] 

        # init_qpos_state = self.init_qpos
        # init_qvel_state = self.init_qvel

        # init_qpos_state[0:3] = self.ref_pelv_pos[self.time_count, 0:3]
        # # init_qpos_state[2] += 0.073
        # init_qpos_state[3:7] = self.ref_pelv_quat[self.time_count, 0:4]
         
        # init_qpos_state[7:19] = self.ref_qpos[self.time_count ,:]
        # init_qvel_state[6:18] = self.ref_qvel[self.time_count ,:]
        # self.set_state(
        #     init_qpos_state, init_qvel_state 
        #     )

        vel_error = self.vel_des - (pos_after - pos_before) / self.dt
        vel_error[0] = np.maximum(vel_error[0], 0)
        vel_error[1] = np.maximum(vel_error[1], 0)
        vel_error[2] = np.maximum(vel_error[2], 0)
        # print( "dt :", self.dt )
        # print( "time_count:", self.time_count )
        # print( "start_phase_count:", self.start_phase_count )
        reward_target = np.exp(-2.5*np.square(vel_error*np.array([1, 0, 0])).sum())
        #lin_vel_cost = 1/(np.abs(2.5 * np.square(vel_error*np.array([1, 0, 0])).sum())+1)
      #   quad_ctrl_cost = np.exp(-0.0001 * np.square(data.qfrc_actuator).sum())

      #   quad_impact_cost = .5e-6 * (np.square(data.cfrc_ext).sum())
      #   quad_impact_cost = max(quad_impact_cost, 1)
      #   quad_impact_cost = np.exp(-1 * quad_impact_cost)
        # reward_imitation = wp*self.calc_joint_quat_reward()+wr*self.calc_root_reward()+wv*self.calc_joint_vel_reward()+we*self.calc_limb_pos_reward() +wc*self.calc_com_pos_reward()
        reward_imitation = 0
        reward  = 0.8*reward_imitation +0.2*alive_bonus#+ 0.3*reward_target #+ 0.2*alive_bonus
        qpos = self.sim.data.qpos
        qpos[3] = np.clip(qpos[3], -1, 1)
        done = bool((qpos[2] < 0.45) or (qpos[2] > 1.7))
        done = bool(np.arccos(qpos[3])*2 > np.pi/3 or (done))  # pelv orientation angle
        # done = bool((qpos[12] < -np.pi/3) or (qpos[12] > np.pi/3) or (done))  # right ankle roll
        # done = bool((qpos[18] < -np.pi/3) or (qpos[18] > np.pi/3) or (done))  # left ankle roll
        # done = bool((qpos[7] < -np.pi/3) or (qpos[7] > np.pi/3) or (done))      # right hip yaw
        # done = bool((qpos[13] < -np.pi/3) or (qpos[13] > np.pi/3) or (done))    # left hip yaw
        # done = bool((qpos[8] < -np.pi/3) or (qpos[8] > np.pi/3) or (done))      # right hip roll
        # done = bool((qpos[14] < -np.pi/3) or (qpos[14] > np.pi/3) or (done))      # left hip roll
        # print( "pelv angle:", np.arccos(qpos[3])*2*180/np.pi )

        done = False

        self.time += self.dt;
        return self._get_obs(), reward, done, dict(reward_target=reward_target,reward_imitation = reward_imitation
         , reward_pos=self.calc_joint_quat_reward(), reward_vel=self.calc_joint_vel_reward(), reward_ee=self.calc_limb_pos_reward(), 
         reward_com = self.calc_com_pos_reward(), reward_root = self.calc_root_reward(), action = self.additional_command)


  def reset(self):
        c = 0.01
        self.eps_start_time = self.sim.data.time      
        self.start_phase = np.random.random_sample()
        # self.start_phase = 0
        self.start_phase_count = int((self.phase_size*self.start_phase)%self.phase_size)
      
      
        init_qpos_state = self.init_qpos
        init_qvel_state = self.init_qvel
        # init_qpos_state[0:3] = self.ref_pelv_pos[self.start_phase_count, 0:3]
        # init_qpos_state[3:7] = self.ref_pelv_quat[self.start_phase_count, 0:4]
        # init_qpos_state[7:19] = self.ref_qpos[self.start_phase_count ,:]
        # init_qvel_state[6:18] = self.ref_qvel[self.start_phase_count ,:]
        init_qpos_state[0:12] = self.ref_qpos[self.start_phase_count ,:]
        init_qvel_state[0:12] = self.ref_qvel[self.start_phase_count ,:]
        self.set_state(
            init_qpos_state, # + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            init_qvel_state  # + self.np_random.uniform(low=-c, high=c, size=self.model.nv)  
            )
        return self._get_obs()

  def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 0.03
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20
        
  def _get_obs(self):
        data = self.sim.data
        # pelv_ori_rel_global = Quaternion(data.body_xquat[1, :])
        # com_pos_global = mass_center(self.model, self.sim)
        # rfoot_pos_rel_pelv = (pelv_ori_rel_global.inverse).rotate((data.xipos[8] - data.body_xpos[1])) # rfoot
        # lfoot_pos_rel_pelv = (pelv_ori_rel_global.inverse).rotate((data.xipos[15] - data.body_xpos[1])) # lfoot
        # com_pos_rel_pelv = (pelv_ori_rel_global.inverse).rotate((com_pos_global - data.body_xpos[1])) 

        # return np.concatenate([data.qpos.flat[2:], data.qvel.flat, self.phase])
      #   return np.concatenate([data.body_xquat])
        return np.concatenate([data.qpos.flat, data.qvel.flat])
        # return np.concatenate([rfoot_pos_rel_pelv, lfoot_pos_rel_pelv, com_pos_rel_pelv])

  def get_xquat(self, i):
        data = self.sim.data
        body_quat = Quaternion()
        body_quat = Quaternion(data.body_xquat[i+2, :])
        return body_quat        

  def pdcontrol(self, position):
        data = self.sim.data
        torque = np.zeros(12);
      #   torque = 625 * (position - data.qpos[7:19]) + 50 * (-data.qvel[6:18]);
        # torque = 900 * (position - data.qpos[7:19]) + 10 * (-data.qvel[6:18]);
        torque = 2500 * (position - data.qpos) + 1 * (-data.qvel);
        # torque[0:4] = 600 * (position[0:4] - data.qpos[0:4]) + 11 * (-data.qvel[0:4]);
        # torque[4:6] = 100 * (position[4:6] - data.qpos[4:6]) + 1 * (-data.qvel[4:6]);
        # torque[6:10] = 600 * (position[6:10] - data.qpos[6:10]) + 11 * (-data.qvel[6:10]);
        # torque[10:12] = 100 * (position[10:12] - data.qpos[10:12]) + 1 * (-data.qvel[10:12]);
      #   torque[0:4] = 600 * (position[0:4] - data.qpos[7:11]) + 11 * (-data.qvel[6:10]);
      #   torque[4:6] = 100 * (position[4:6] - data.qpos[11:13]) + 1 * (-data.qvel[10:12]);
      #   torque[6:10] = 600 * (position[6:10] - data.qpos[13:17]) + 11 * (-data.qvel[12:16]);
      #   torque[10:12] = 100 * (position[10:12] - data.qpos[17:19]) + 1 * (-data.qvel[16:18]);
        return torque

  def quat_diff_square(self, q1, q2):
        q_diff = Quaternion()
        q1_quat = Quaternion(q1)
        q2_quat = Quaternion(q2)
        q_diff = q1_quat/q2_quat
        q_diff[0] = np.clip(q_diff[0], -1, 1)
        q_diff_angle = np.arccos(q_diff[0])*2
        return np.square(q_diff_angle)

  def calc_root_reward(self):
        data = self.sim.data
        root_err = 0
        root_pos_diff_sum = 0
        root_rot_diff_sum = 0
      #   root_vel_diff_sum = 0
      #   root_ang_vel_diff_sum = 0
        
        # root_pos_diff_sum += np.square(data.qpos[2] - self.ref_pelv_pos[self.time_count, 2])
        # root_rot_diff_sum += self.quat_diff_square(data.qpos[3:7], self.ref_pelv_quat[self.time_count, 0:4])
      #   quat_diff_sum += 0.5*np.square(data.qpos[7:10] - self.ref_qpos[self.time_count, 0:3]).sum()         # R hip
      #   quat_diff_sum += 0.3*np.square(data.qpos[10] - self.ref_qpos[self.time_count, 3]).sum()             # R knee
      #   quat_diff_sum += 0.2*np.square(data.qpos[11:13] - self.ref_qpos[self.time_count, 4:6]).sum()        # R ankle
      #   quat_diff_sum += 0.5*np.square(data.qpos[13:16] - self.ref_qpos[self.time_count, 6:9]).sum()            # L hip
      #   quat_diff_sum += 0.3*np.square(data.qpos[16] - self.ref_qpos[self.time_count, 9]).sum()                 # L knee
      #   quat_diff_sum += 0.2*np.square(data.qpos[17:19] - self.ref_qpos[self.time_count, 10:12]).sum()          # L ankle
        root_err = root_pos_diff_sum + 0.1*root_rot_diff_sum #+ 0.01*root_vel_diff_sum + 0.001*root_ang_vel_diff_sum
        return np.exp(-5*root_err)


  def calc_joint_quat_reward(self):
        data = self.sim.data
        quat_diff_sum = 0
      #   quat_diff_sum += np.square(data.qpos[7:19] - self.ref_qpos[self.time_count, :]).sum()
      #   quat_diff_sum += np.square(data.qpos - self.ref_qpos[self.time_count, :]).sum()

        # quat_diff_sum += 0.5*np.square(data.qpos[7:10] - self.ref_qpos[self.time_count, 0:3]).sum()         # R hip
        # quat_diff_sum += 0.3*np.square(data.qpos[10] - self.ref_qpos[self.time_count, 3]).sum()             # R knee
        # quat_diff_sum += 0.2*np.square(data.qpos[11:13] - self.ref_qpos[self.time_count, 4:6]).sum()        # R ankle
        # quat_diff_sum += 0.5*np.square(data.qpos[13:16] - self.ref_qpos[self.time_count, 6:9]).sum()            # L hip
        # quat_diff_sum += 0.3*np.square(data.qpos[16] - self.ref_qpos[self.time_count, 9]).sum()                 # L knee
        # quat_diff_sum += 0.2*np.square(data.qpos[17:19] - self.ref_qpos[self.time_count, 10:12]).sum()          # L ankle
        return np.exp(-2*quat_diff_sum)

  def calc_joint_vel_reward(self):
        data = self.sim.data
        qvel_diff_sum = 0
      #   qvel_diff_sum += np.square(data.qvel[6:18]-self.ref_qvel[self.time_count, :] ).sum()
      #   qvel_diff_sum += np.square(data.qvel-self.ref_qvel[self.time_count, :] ).sum()
        # qvel_diff_sum += 0.5*np.square(data.qvel[6:9] - self.ref_qvel[self.time_count, 0:3]).sum()         # R hip
        # qvel_diff_sum += 0.3*np.square(data.qvel[9] - self.ref_qvel[self.time_count, 3]).sum()             # R knee
        # qvel_diff_sum += 0.2*np.square(data.qvel[10:12] - self.ref_qvel[self.time_count, 4:6]).sum()        # R ankle
        # qvel_diff_sum += 0.5*np.square(data.qvel[12:15] - self.ref_qvel[self.time_count, 6:9]).sum()            # L hip
        # qvel_diff_sum += 0.3*np.square(data.qvel[15] - self.ref_qvel[self.time_count, 9]).sum()                 # L knee
        # qvel_diff_sum += 0.2*np.square(data.qvel[16:18] - self.ref_qvel[self.time_count, 10:12]).sum()          # L ankle
        return np.exp(-0.1*qvel_diff_sum)

  def calc_limb_pos_reward(self):
        #right foot
        data = self.sim.data

        # pelv_ori_rel_global = Quaternion(data.body_xquat[1, :])
        # rfoot_pos_rel_pelv = (pelv_ori_rel_global.inverse).rotate((data.xipos[8] - data.body_xpos[1])) # rfoot
        # lfoot_pos_rel_pelv = (pelv_ori_rel_global.inverse).rotate((data.xipos[15] - data.body_xpos[1])) # lfoot
        # rwrist_pos_rel_pelv = (pelv_ori_rel_global.inverse).rotate((data.xipos[26] - data.body_xpos[1])) # rfoot
        # lwrist_pos_rel_pelv = (pelv_ori_rel_global.inverse).rotate((data.xipos[34] - data.body_xpos[1])) # lfoot
        limb_pos_diff_sum = 0
        # limb_pos_diff_sum += np.square(rfoot_pos_rel_pelv - self.ref_foot[self.time_count, 0:3]).sum()
        # limb_pos_diff_sum += np.square(lfoot_pos_rel_pelv - self.ref_foot[self.time_count, 3:6]).sum()
        # print("limb_pos_diff_sum: ", limb_pos_diff_sum)
        # limb_pos_diff_sum = 0
        # limb_pos_diff_sum += np.square(rfoot_pos_rel_pelv - self.ref_rfoot[self.time_count, :]).sum()
        # limb_pos_diff_sum += np.square(lfoot_pos_rel_pelv - self.ref_lfoot[self.time_count, :]).sum()
      #   limb_pos_diff_sum += np.square(rwrist_pos_rel_pelv - self.ref_rwrist[self.time_count, :]).sum()
      #   limb_pos_diff_sum += np.square(lwrist_pos_rel_pelv - self.ref_lwrist[self.time_count, :]).sum()

        # return 1/(np.abs(40*limb_pos_diff_sum)+1)
        return np.exp(-40*limb_pos_diff_sum)

  def calc_com_pos_reward(self):
        #right foot
        data = self.sim.data
        # com_pos_global = mass_center(self.model, self.sim)
        # pelv_ori_rel_global = Quaternion(data.body_xquat[1, :])
        # com_pos_rel_pelv = (pelv_ori_rel_global.inverse).rotate((com_pos_global - data.body_xpos[1])) 
        
        limb_com_diff_sum = 0
        # limb_com_diff_sum += np.square(com_pos_rel_pelv - self.ref_com[self.time_count, :]).sum()
        # print("limb_com_diff_sum: ", limb_com_diff_sum)
        # return 1/(np.abs(10*limb_com_diff_sum)+1)
        return np.exp(-10*limb_com_diff_sum)
    

       


def mass_center(model, sim):
      mass = np.expand_dims(model.body_mass, 1)
      xpos = sim.data.xipos
      return (np.sum(mass * xpos, 0) / np.sum(mass))

# def quaternion_multiply(quaternion0, quaternion1):
#     w0, x0, y0, z0 = quaternion0
#     w1, x1, y1, z1 = quaternion1
#     return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
#                      x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
#                      -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
#                      x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0])

