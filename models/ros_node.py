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
import tf2_ros
import tf2_geometry_msgs

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
from visualization_msgs.msg import Marker
from sympy import Point3D, Line3D


_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#_DEVICE = torch.device('cpu')

def quaternion_to_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[3]
    q1 = Q[0]
    q2 = Q[1]
    q3 = Q[2]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

class human_prediction():
    def __init__(self,model,parents):
        rospy.init_node('listener', anonymous=True)
        self.model = model
        self.parents = parents
        self.norm_stats = {}
        self.norm_stats['mean'] = np.array([ 0.00660166, -0.00322862, -0.00074547, -0.00221925,  0.32131838,
        0.05040703, -0.01361359,  0.69785828,  0.09532178, -0.00660162,
        0.00322858,  0.00074546, -0.01506641,  0.32316435,  0.05134183,
       -0.02408792,  0.70626347,  0.09823843])
        self.norm_stats['std'] = np.array([0.09752762, 0.02463142, 0.08864842, 0.18086745, 0.19258509,
       0.19703887, 0.2124409 , 0.24669765, 0.24913149, 0.09752732,
       0.02463133, 0.08864804, 0.18512949, 0.20038966, 0.20181931,
       0.21314536, 0.25163   , 0.25110496])
        self.norm_stats['mean_traj'] = np.array([-0.3049573 , -0.24428056,  2.96069035])
        self.norm_stats['std_traj'] = np.array([1.53404053, 0.33958351, 3.70011473])
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(100.0))  # tf buffer length
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.skeleton_subscriber = rospy.Subscriber("/pose_publisher/3DSkeletonBuffer", Skeleton3DBuffer, self.Predict)
        self.my_counter = 0
  #      self.obj_pub = rospy.Publisher('/obj_position', PointCloud , queue_size=1)    
        self.publisher = rospy.Publisher('/potrtr/predictions', Skeleton3DBuffer, queue_size=1)
        self.publisher_heading = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
        self.publisher_heading_marker = rospy.Publisher('/marker/heading', Marker, queue_size=1)
        self.publisher_left = rospy.Publisher('/marker/left', Marker, queue_size=1)
        self.publisher_right = rospy.Publisher('/marker/right', Marker, queue_size=1)

    
    def skeleton_to_inputs(self, skeletonbuffer):

        skeletonbuffer = np.array(skeletonbuffer.skeleton_3d_17_flat.data).reshape(skeletonbuffer.shape)
        skeletonbuffer[:, :, 1:3] = skeletonbuffer[:, :, 2:0:-1]
        skeletonbuffer[:,:,1] = skeletonbuffer[:,:,1]
        first_hip = skeletonbuffer[0:1,0:1,:]

        enc_input = skeletonbuffer[:,1:7,:] - skeletonbuffer[:,0:1,:] 
        enc_traj = skeletonbuffer[:,0:1,:]
        dec_input = np.array([enc_input[-1,:,:]]*20)
        dec_traj = np.array([enc_traj[-1,:,:]]*20)

        enc_input = np.array(enc_input).reshape(1,5,18) #reshape (5,3,6) to (5,18)
        enc_traj = np.array(enc_traj).reshape(1,5,3)
        dec_input = np.array(dec_input).reshape(1, 20,18)
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
        dof = D//6

        # unnormalize input sequence
        sequence = data*self.norm_stats['std'] + self.norm_stats['mean']

        # batch_size x seq_length x n_major_joints x dof (or joint dim)
        sequence = sequence.reshape((batch_size, seq_length, 6, dof))
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
  
    def predictions_to_msg(self, output, seq):

        skeletonbuffer_msg = Skeleton3DBuffer()
        array_msg =  Float64MultiArray()
        array_msg.data = output.flatten()

        skeletonbuffer_msg.skeleton_3d_17_flat = array_msg
        skeletonbuffer_msg.shape = output.shape
        skeletonbuffer_msg.seq = seq

        return skeletonbuffer_msg
    
    def get_transform_camera_world(self, pose_stamped):
        transform = self.tf_buffer.lookup_transform('world',
                                       # source frame:
                                       pose_stamped.header.frame_id,
                                       # get the tf at the time the pose was valid
                                       pose_stamped.header.stamp,
                                       # wait for at most 1 second for transform, otherwise throw
                                       rospy.Duration(1.0))

        pose_transformed = tf2_geometry_msgs.do_transform_pose(pose_stamped, transform)
        return pose_transformed


    def goal_from_pose(self, poses, index=19, ahead=0):
        a = poses[index, 1, 0:3]
        b = poses[index, 4, 0:3]
        c = poses[index, 0, 0:3]
        msg = PoseStamped()
        orientation = np.arctan2(b[1]-a[1], b[0]-a[0]) + np.pi/2
        
        msg.header.frame_id = 'camera'
        msg.pose.position.x = c[0] + np.cos(orientation) * ahead
        msg.pose.position.y = c[1] + np.sin(orientation) * ahead
        msg.pose.position.z = 0
        
        quaternions = self.get_quaternion_from_euler(0, 0, orientation)
        msg.pose.orientation.x = quaternions[0]
        msg.pose.orientation.y = quaternions[1]
        msg.pose.orientation.z = quaternions[2]
        msg.pose.orientation.w = quaternions[3]
        new_msg = self.get_transform_camera_world(msg)
        return new_msg

    def Predict(self,skeletonbuffer):
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
            
#            t1 = time.time()
            prediction = model(
                enc_inputs_normalized,
                dec_inputs_normalized,
                enc_inputs_traj_normalized,
                dec_inputs_traj_normalized,
                get_attn_weights=True
            )
 #           t2=time.time()
  
            classes = prediction[1]
            traj_prediction = prediction[-1]
            traj_prediction = traj_prediction[-1].cpu().numpy() #.reshape(20,1,3)
  
            prediction = prediction[0]
            prediction = prediction[-1].cpu().numpy() #.reshape(20,6,3)
            preds = self.unnormalize_mine(prediction)
            preds_traj = self.unnormalize_mine_traj(traj_prediction)

            # # batch=0
            # fig = plt.figure(figsize=(4,4))
            # ax = fig.add_subplot(111, projection='3d')
            # ax.set_xlim3d([-1,1])
            # ax.set_ylim3d([-1,1])
            # ax.set_zlim3d([-1,1])

            # for i in range(20):
            #     pos = preds[batch][i].reshape(6,3)
            #     pos = np.concatenate((np.array([[0,0,0]]),pos),axis=0) + preds_traj[batch][i]
            #     for j, j_parent in enumerate(self.parents):    
            #         if j_parent == -1:
            #             continue                
            #         ax.plot([pos[j, 0], pos[j_parent, 0]],
            #                 [pos[j, 2], pos[j_parent, 2]],
            #                 [pos[j, 1], pos[j_parent, 1]], zdir='y', c='blue')

            # for i in range(5):
            #     pos = enc_input[batch][i].reshape(6,3)
            #     pos = np.concatenate((np.array([[0,0,0]]),pos),axis=0) + enc_traj[batch][i]
            #     for j, j_parent in enumerate(self.parents):    
            #         if j_parent == -1:
            #             continue                
            #         ax.plot([pos[j, 0], pos[j_parent, 0]],
            #                 [pos[j, 2], pos[j_parent, 2]],
            #                 [pos[j, 1], pos[j_parent, 1]], zdir='y', c='red')

            # fig.savefig('test.png')

            #------------
            traj_pred = preds_traj.reshape(20,1,3) #+ first_hip
            pose_pred = preds.reshape(20,6,3) + traj_pred
            output = np.concatenate((traj_pred, pose_pred),axis=1)
            output[:, :, 1:3] = output[:, :, 2:0:-1]
            output[:,:,1] = output[:,:,1]
            self.publisher.publish(self.predictions_to_msg(output ,skeletonbuffer.seq))    
            index_pose = 10
            self.publisher_heading.publish(self.goal_from_pose(output, index=index_pose, ahead=2))
            
            msg = Marker()
            msg.type = 0
            msg.scale.x = 1
            msg.scale.y = 0.1
            msg.scale.z = 0.1
            msg.color.r = 1
            msg.color.a = 0.5
            pose_stamped = self.goal_from_pose(output, index=index_pose)
            msg.pose = pose_stamped.pose
            # msg.pose.position.x = output[5, 0, 0]
            # msg.pose.position.y = output[5, 0, 1]
            # msg.pose.position.z = output[5, 0, 2]
            msg.header = pose_stamped.header
            self.publisher_heading_marker.publish(msg)
            maximum_estimation_time = params['target_seq_len']/params['frame_rate']

            msg.pose.position.x = output[index_pose, 1, 0]
            msg.pose.position.y = output[index_pose, 1, 1]
            msg.pose.position.z = output[index_pose, 1, 2]
            msg.scale.x = 0.2
            msg.scale.y = 0.2
            msg.scale.z = 0.2
            msg.color.g = 1
            msg.color.r = 0
            msg.type = 2
            self.publisher_left.publish(msg)

            msg.pose.position.x = output[index_pose, 4, 0]
            msg.pose.position.y = output[index_pose, 4, 1]
            msg.pose.position.z = output[index_pose, 4, 2]
            msg.type = 2
            msg.color.b = 1
            msg.color.r = 0
            self.publisher_right.publish(msg)

            self.my_counter +=1
# =============================================================================
    
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str,default="/home/autolab/workspace/3dpose/potrtr/training/model/legs_noisy/config.json")
    parser.add_argument('--model_file', type=str,default="/home/autolab/workspace/3dpose/potrtr/training/model/legs_noisy/best_epoch_fde_0173.pt")
    parser.add_argument('--data_path', type=str, default="/home/autolab/workspace/3dpose/potrtr/data/h3.6m/")

    args = parser.parse_args()
    params = json.load(open(args.config_file))
    parents, offset, rot_ind, exp_map_ind = utils.load_constants(args.data_path)

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
              
    human_prediction(model,parents)
    rospy.spin()
    
    
    
    
    
