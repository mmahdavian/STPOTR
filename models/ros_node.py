###############################################################################
# Pose Transformers (POTR): Human Motion Prediction with Non-Autoregressive 
# Transformers
# 
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by 
# Angel Martinez <angel.martinez@idiap.ch>,
# 
# This file is part of 
# POTR: Human Motion Prediction with Non-Autoregressive Transformers
# 
# POTR is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
# 
# POTR is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with POTR. If not, see <http://www.gnu.org/licenses/>.
###############################################################################

"""Model function to deploy POTR models for visualization and generation."""

import rospy
import numpy as np
import os
import sys
import argparse
import json
import time
import cv2

from matplotlib import image
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

thispath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, thispath+"/../")

import training.seq2seq_model_fn as seq2seq_model_fn
import models.PoseTransformer as PoseTransformer
import models.PoseEncoderDecoder as PoseEncoderDecoder
import data.H36MDataset_v3 as H36MDataset_v3
import utils.utils as utils
#import radam.radam as radam
import training.transformer_model_fn as tr_fn
import tqdm
from pose_publisher.msg import Skeleton3D17
from pose_publisher.msg import Skeleton3DBuffer
from zed_interfaces.msg import Keypoint3D
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from std_msgs.msg import Float64MultiArray
from sympy import Point3D, Line3D


_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#_DEVICE = torch.device('cpu')


class human_prediction():
    def __init__(self,model):
        rospy.init_node('listener', anonymous=True)
        self.model = model
        self.norm_stats = {}
        self.norm_stats['mean'] = np.array([ 0.00660166, -0.00322862, -0.00074547, -0.00221925,  0.32131838,
        0.05040703, -0.01361359,  0.69785828,  0.09532178, -0.00660162,
        0.00322858,  0.00074546, -0.01506641,  0.32316435,  0.05134183,
       -0.02408792,  0.70626347,  0.09823843,  0.00577709, -0.21263408,
       -0.02852573,  0.01207891, -0.43795797, -0.05560767,  0.01407008,
       -0.49628542, -0.05891722,  0.01702867, -0.58822308, -0.07295712,
        0.00417502, -0.38859674, -0.04970666, -0.0071068 , -0.19491813,
       -0.02284074, -0.00568425, -0.13399005, -0.00880117,  0.0170771 ,
       -0.38437933, -0.04920271,  0.01767753, -0.19377204, -0.02285681,
        0.01422233, -0.16066915, -0.01378928])
        self.norm_stats['std'] = np.array([0.09752762, 0.02463142, 0.08864842, 0.18086745, 0.19258509,
       0.19703887, 0.2124409 , 0.24669765, 0.24913149, 0.09752732,
       0.02463133, 0.08864804, 0.18512949, 0.20038966, 0.20181931,
       0.21314536, 0.25163   , 0.25110496, 0.05994541, 0.05127974,
       0.07128643, 0.11012195, 0.09926957, 0.13492376, 0.13778828,
       0.12397334, 0.16564003, 0.14164487, 0.12879104, 0.17520238,
       0.14193176, 0.09618481, 0.15208769, 0.2274719 , 0.13292273,
       0.22023694, 0.24284202, 0.22032088, 0.24409384, 0.1440386 ,
       0.09937461, 0.15516004, 0.23021911, 0.14233924, 0.22319722,
       0.24973414, 0.23667015, 0.25038966])
        self.norm_stats['mean_traj'] = np.array([-0.3049573 , -0.24428056,  2.96069035])
        self.norm_stats['std_traj'] = np.array([1.53404053, 0.33958351, 3.70011473])

        self.skeleton_subscriber = rospy.Subscriber("/pose_publisher/3DSkeletonBuffer", Skeleton3DBuffer, self.Predict)
        self.my_counter = 0
  #      self.obj_pub = rospy.Publisher('/obj_position', PointCloud , queue_size=1)    
        self.publisher = rospy.Publisher('/potrtr/predictions', Skeleton3DBuffer, queue_size=1)
        self.publisher_heading = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)

    
    def skeleton_to_inputs(self, skeletonbuffer):

        skeletonbuffer = np.array(skeletonbuffer.skeleton_3d_17_flat.data).reshape(skeletonbuffer.shape)
        first_hip = skeletonbuffer[0:1,0:1,:]

        enc_input = skeletonbuffer[:,1:,:] - skeletonbuffer[:,0:1,:] 
        enc_traj = skeletonbuffer[:,0:1,:]
        dec_input = np.array([enc_input[-1,:,:]]*20)
        dec_traj = np.array([enc_traj[-1,:,:]]*20)

        enc_input = np.array(enc_input).reshape(1,5,48) #reshape (5,3,16) to (5,48)
        enc_traj = np.array(enc_traj).reshape(1,5,3)
        dec_input = np.array(dec_input).reshape(1, 20,48)
        dec_traj = np.array(dec_traj).reshape(1,20,3)
        return enc_input, enc_traj, dec_input, dec_traj, first_hip

    def normalize_data(self,data):
        """Data normalization with mean and std removing dimensions with low std."""
        tmp_data = data
        tmp_data = tmp_data - self.norm_stats['mean']
        tmp_data = np.divide(tmp_data, self.norm_stats['std'])
        data = tmp_data
        return data

    def normalize_data_traj(self,data):
        tmp_traj = data
        tmp_traj = tmp_traj - self.norm_stats['mean_traj']
        tmp_traj = np.divide(tmp_traj, self.norm_stats['std_traj'])
        traj = tmp_traj
        return traj

    def unnormalize_mine(self,data):
        """Unnormalize data and pads with zeros the minor joints.

        Args:
        norm_seq: A numpy array. Normalized sequence [batch_size, seq_length,
            n_major_joints*dof]

        Returnrs:
        Numpy array of shape [batch_size, seq_length, 99]
        """
        batch_size, seq_length, D = data.shape
        dof = D//16

        # unnormalize input sequence
        sequence = data*self.norm_stats['std'] + self.norm_stats['mean']

        # batch_size x seq_length x n_major_joints x dof (or joint dim)
        sequence = sequence.reshape((batch_size, seq_length, 16, dof))
        sequence = np.reshape(sequence, [batch_size, seq_length, -1])
        return sequence

    def unnormalize_mine_traj(self,data):
        """Unnormalize data and pads with zeros the minor joints.

        Args:
        norm_seq: A numpy array. Normalized sequence [batch_size, seq_length,
            n_major_joints*dof]

        Returnrs:
        Numpy array of shape [batch_size, seq_length, 99]
        """
        batch_size, seq_length, D = data.shape
        dof = D//1

        # unnormalize input sequence
        sequence = data*self.norm_stats['std_traj'] + self.norm_stats['mean_traj']

        # batch_size x seq_length x n_major_joints x dof (or joint dim)
        sequence = sequence.reshape((batch_size, seq_length, 1, dof))
        sequence = np.reshape(sequence, [batch_size, seq_length, -1])
        return sequence

    import numpy as np # Scientific computing library for Python
 
    def get_quaternion_from_euler(self, roll, pitch, yaw):
        """
        Convert an Euler angle to a quaternion.
        
        Input
            :param roll: The roll (rotation around x-axis) angle in radians.
            :param pitch: The pitch (rotation around y-axis) angle in radians.
            :param yaw: The yaw (rotation around z-axis) angle in radians.
        
        Output
            :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
        """
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        
        return [qx, qy, qz, qw]
  
    def predictions_to_msg(self, prediction, trajectory, seq, first_hip):
        
        traj_pred = trajectory.reshape(20,1,3) #+ first_hip
        pose_pred = prediction.reshape(20,16,3) + traj_pred
        output = np.concatenate((traj_pred, pose_pred),axis=1)

        skeletonbuffer_msg = Skeleton3DBuffer()
        array_msg =  Float64MultiArray()
        array_msg.data = output.flatten()

        skeletonbuffer_msg.skeleton_3d_17_flat = array_msg
        skeletonbuffer_msg.shape = output.shape
        skeletonbuffer_msg.seq = seq

        return skeletonbuffer_msg

    def goal_from_pose(self, pose, traj):
        pose = pose.reshape(20,16,3)
        a = pose[19,0]
        b = pose[19,3]
        c = traj[0, 19, 0:3]
        msg = PoseStamped()
        msg.header.frame_id = 'camera'
        msg.pose.position.x = c[0]
        msg.pose.position.y = c[1]
        msg.pose.position.z = c[2]

        # p1, p2, p3 = Point3D(a[0], a[1], 0), Point3D(b[0], b[1], 0), Point3D(c[0], c[1], 0)
        # L = Line3D(p1, p2)
        # P = L.perpendicular_line(p3)
        
        orientation = np.arctan2(b[2]-a[2], b[0]-a[0])
        quaternions = self.get_quaternion_from_euler(0, orientation, 0)
        msg.pose.orientation.x = quaternions[0]
        msg.pose.orientation.y = quaternions[1]
        msg.pose.orientation.z = quaternions[2]
        msg.pose.orientation.w = quaternions[3]
        return msg

    def Predict(self,skeletonbuffer):
# =============================================================================
        with torch.no_grad():
            enc_input, enc_traj, dec_input, dec_traj, first_hip = self.skeleton_to_inputs(skeletonbuffer) 
            enc_input_normalized = self.normalize_data(enc_input)
            dec_input_normalized = self.normalize_data(dec_input)

            enc_traj_normalized = self.normalize_data_traj(enc_traj)
            dec_traj_normalized = self.normalize_data_traj(dec_traj)

            enc_traj_normalized = enc_traj_normalized - enc_traj_normalized[:,0,:]
            dec_traj_normalized = dec_traj_normalized - enc_traj_normalized[:,0,:]

            # sample['decoder_inputs_traj'] = sample['decoder_inputs_traj'] - sample['encoder_inputs_traj'][0,:,0].reshape(-1,1,3)
            # sample['decoder_outputs_traj'] = sample['decoder_outputs_traj'] - sample['encoder_inputs_traj'][0,:,0].reshape(-1,1,3)
            # sample['encoder_inputs_traj'] = sample['encoder_inputs_traj'] - sample['encoder_inputs_traj'][0,:,0].reshape(-1,1,3)

            enc_inputs_normalized = torch.from_numpy(enc_input_normalized).float().to(_DEVICE)
            dec_inputs_normalized = torch.from_numpy(dec_input_normalized).float().to(_DEVICE)
                        
            enc_inputs_traj_normalized = torch.from_numpy(enc_traj_normalized).float().to(_DEVICE)
            dec_inputs_traj_normalized = torch.from_numpy(dec_traj_normalized).float().to(_DEVICE)
          
# =============================================================================
#             enc_inputs = torch.squeeze(enc_inputs)
#             dec_inputs = torch.squeeze(dec_inputs)
#   
#             enc_inputs_traj = torch.squeeze(enc_inputs_traj)
#             dec_inputs_traj = torch.squeeze(dec_inputs_traj)
# =============================================================================
            
            t1 = time.time()
            prediction = model(
                enc_inputs_normalized,
                dec_inputs_normalized,
                enc_inputs_traj_normalized,
                dec_inputs_traj_normalized,
                get_attn_weights=True
            )
            t2=time.time()
  
            classes = prediction[1]
            traj_prediction = prediction[-1]
            traj_prediction = traj_prediction[-1].cpu().numpy() #.reshape(20,1,3)
  
            prediction = prediction[0]
            prediction = prediction[-1].cpu().numpy() #.reshape(20,16,3)
            preds = self.unnormalize_mine(prediction)
            preds_traj = self.unnormalize_mine_traj(traj_prediction)
            self.publisher.publish(self.predictions_to_msg(preds,preds_traj, skeletonbuffer.seq, first_hip))    
            self.publisher_heading.publish(self.goal_from_pose(preds, preds_traj))
            maximum_estimation_time = params['target_seq_len']/params['frame_rate']
            
            self.my_counter +=1
# =============================================================================
    
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str,default="/home/autolab/workspace/3dpose/potrtr/training/model/config.json")
    parser.add_argument('--model_file', type=str,default="/home/autolab/workspace/3dpose/potrtr/training/model/best_epoch_fde_0002_best_sofar.pt")
    args = parser.parse_args()
    params = json.load(open(args.config_file))

    # train_dataset_fn, eval_dataset_fn = tr_fn.dataset_factory_total(params)
    
    pose_encoder_fn, pose_decoder_fn = \
        PoseEncoderDecoder.select_pose_encoder_decoder_fn(params)
        
    traj_encoder_fn, traj_decoder_fn = \
        PoseEncoderDecoder.select_traj_encoder_decoder_fn(params)
                  
  
    for k,v in params.items():
      print('[INFO] (POTRFn@main) {}: {}'.format(k, v))
  
    model = PoseTransformer.model_factory(
          params, 
          pose_encoder_fn, 
          pose_decoder_fn,
          traj_encoder_fn, 
          traj_decoder_fn 
      )
    
    model.load_state_dict(torch.load(args.model_file, map_location=_DEVICE))
    model.to(_DEVICE)
    model.eval()
              
    human_prediction(model)
    rospy.spin()
    
    
    
    
    
