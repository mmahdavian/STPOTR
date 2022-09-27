###############################################################################
# (STPOTR): Simultaneous Human Trajectory and Pose Prediction Using a 
# Non-Autoregressive Transformer for Robot Following Ahead
# 
# Copyright (c) 2022 MARS Lab at Simon Fraser University
# Written by 
# Mohammad Mahdavian <mmahdavi@sfu.ca>,
# 
# This file is part of 
# STPOTR: Simultaneous Human Trajectory and Pose Prediction Using a 
# Non-Autoregressive Transformer for Robot Following Ahead
# 
# STPOTR is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
# 
# STPOTR is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with STPOTR. If not, see <http://www.gnu.org/licenses/>.
###############################################################################

"""ros node for publishing predictions"""

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
import models.STPoseTransformer as STPoseTransformer
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
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
from std_msgs.msg import Header



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
    def __init__(self,model,parents, mode=1):
        rospy.init_node('listener', anonymous=True)
        self.model = model
        self.parents = parents
        self.norm_stats = {}

        self.norm_stats['mean']=np.array([ 0.00660166, -0.00322862, -0.00074547, -0.00221925,  0.32131838,
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
        self.norm_stats['mean_traj']=np.array([-0.3049573 , -0.24428056,  2.96069035])
        self.norm_stats['std_traj'] = np.array([1.53404053, 0.33958351, 3.70011473])
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(100.0))  # tf buffer length
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.mode = mode

        self.skeleton_subscriber = rospy.Subscriber("/pose_publisher/3DSkeletonBuffer", Skeleton3DBuffer, self.Predict, queue_size=1)
        self.my_counter = 0
  #      self.obj_pub = rospy.Publisher('/obj_position', PointCloud , queue_size=1)    
        self.publisher = rospy.Publisher('/potrtr/predictions', Skeleton3DBuffer, queue_size=1)
        self.publisher_heading = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
        self.publisher_heading_marker = rospy.Publisher('/marker/heading', Marker, queue_size=1)
        self.publisher_left = rospy.Publisher('/marker/left', Marker, queue_size=1)
        self.publisher_right = rospy.Publisher('/marker/right', Marker, queue_size=1)
        self.pc2_publisher = []
        for i in range(20):
            self.pc2_publisher.append(rospy.Publisher('/pose_publisher/predict_fram'+str(i), PointCloud, queue_size=1))


    
    def skeleton_to_inputs(self, skeletonbuffer):

        skeletonbuffer = np.array(skeletonbuffer.skeleton_3d_17_flat.data).reshape(skeletonbuffer.skeleton_3d_17_flat_shape)
        skeletonbuffer[:, :, 1:3] = skeletonbuffer[:, :, 2:0:-1]
        first_hip = skeletonbuffer[0:1,0:1,:]

        enc_input = skeletonbuffer[:,1:17,:] - skeletonbuffer[:,0:1,:]
        enc_traj = skeletonbuffer[:,0:1,:]
        dec_input = np.array([enc_input[-1,:,:]]*20)
        dec_traj = np.array([enc_traj[-1,:,:]]*20)

        enc_input = np.array(enc_input).reshape(1,5,48) #reshape (5,3,6) to (5,18)
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
  
    def predictions_to_msg(self, output, seq):

        skeletonbuffer_msg = Skeleton3DBuffer()
        array_msg =  Float64MultiArray()
        array_msg.data = output.flatten()

        skeletonbuffer_msg.skeleton_3d_17_flat = array_msg
        skeletonbuffer_msg.shape = output.shape
        #skeletonbuffer_msg.seq = seq

        return skeletonbuffer_msg
    
    def pose_transform(self, skeleton_buffer, Transform):
        skeleton_buffer = np.concatenate((skeleton_buffer, np.ones((skeleton_buffer.shape[0],skeleton_buffer.shape[1],1))),axis=2)
        skeleton_buffer_shape = skeleton_buffer.shape
        skeleton_buffer = np.reshape(skeleton_buffer, (skeleton_buffer_shape[0]*skeleton_buffer_shape[1],skeleton_buffer_shape[2]))
        skeleton_buffer_t = np.transpose(skeleton_buffer)
        skeleton_buffer_t = np.matmul(Transform, skeleton_buffer_t)
        skeleton_buffer = np.transpose(skeleton_buffer_t)
        skeleton_buffer = np.reshape(skeleton_buffer, (skeleton_buffer_shape[0],skeleton_buffer_shape[1],skeleton_buffer_shape[2]))
        return skeleton_buffer[:,:,:3]

    def predictions_to_pointcloud(self, predictions, transform, transform_shape):
        transform = np.array(transform).reshape(transform_shape)
        predictions_global_frame = self.pose_transform(predictions, transform)
        for i in range(20):
      #  i = 19
           self.pc2_publisher[i].publish(self.get_point_clouds(predictions[i,:,:], relative_to="camera"))
                           
    def get_point_clouds(self, skeleton_3d, relative_to="world"):
        pc = PointCloud()
        pc.header.stamp = rospy.Time.now()
        pc.header.frame_id = relative_to
        for i in range(skeleton_3d.shape[0]):
            pc.points.append(Point32(x=skeleton_3d[i][0],y=skeleton_3d[i][1],z=skeleton_3d[i][2]))
        return pc

    def get_transform_camera_world(self, pose_stamped):
        not_found = True
        while not_found:
            try:
                transform = self.tf_buffer.lookup_transform('world',
                                               # source frame:
                                               pose_stamped.header.frame_id,
                                               # get the tf at the time the pose was valid
                                               pose_stamped.header.stamp,
                                               # wait for at most 1 second for transform, otherwise throw
                                               rospy.Duration(1.0))
                not_found = False
                time.sleep(0.05)
            except Exception as e:
                print(e)
    

        pose_transformed = tf2_geometry_msgs.do_transform_pose(pose_stamped, transform)
        return pose_transformed


    def goal_from_pose(self, poses, index=19, ahead=0):
        a = poses[index, 1, 0:3]
        b = poses[index, 4, 0:3]
        c = poses[index, 0, 0:3]
        msg = PoseStamped()
        orientation = np.arctan2(b[1]-a[1], b[0]-a[0]) + np.pi/2
        
        msg.header.frame_id = 'camera'
        msg.header.stamp = rospy.Time.now()
        if self.mode == 0:
            msg.pose.position.x = c[0] + np.cos(orientation) * ahead
            msg.pose.position.y = c[1] + np.sin(orientation) * ahead
        elif self.mode == 1:
            msg.pose.position.x = c[0] + np.cos(orientation+np.pi/2) * ahead
            msg.pose.position.y = c[1] + np.sin(orientation+np.pi/2) * ahead
        elif self.mode == 2:
            msg.pose.position.x = c[0] + np.cos(orientation-np.pi/2) * ahead
            msg.pose.position.y = c[1] + np.sin(orientation-np.pi/2) * ahead
        msg.pose.position.z = 0
        
        quaternions = self.get_quaternion_from_euler(0, 0, orientation)
        msg.pose.orientation.x = quaternions[0]
        msg.pose.orientation.y = quaternions[1]
        msg.pose.orientation.z = quaternions[2]
        msg.pose.orientation.w = quaternions[3]
        new_msg = self.get_transform_camera_world(msg)
        new_msg.pose.orientation.x = 0
        new_msg.pose.orientation.y = 0
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
            
            prediction = model(
                enc_inputs_normalized,
                dec_inputs_normalized,
                enc_inputs_traj_normalized,
                dec_inputs_traj_normalized,
                get_attn_weights=True
            )
  
            classes = prediction[1]
            traj_prediction = prediction[-1]
            traj_prediction = traj_prediction[-1].cpu().numpy() #.reshape(20,1,3)
  
            prediction = prediction[0]
            prediction = prediction[-1].cpu().numpy() #.reshape(20,6,3)
            preds = self.unnormalize_mine(prediction)
            preds_traj = self.unnormalize_mine_traj(traj_prediction)

            #------------
            traj_pred = preds_traj.reshape(20,1,3) #+ first_hip
            pose_pred = preds.reshape(20,16,3) + traj_pred
            output = np.concatenate((traj_pred, pose_pred),axis=1)
            output[:, :, 1:3] = output[:, :, 2:0:-1]
           # self.publisher.publish(self.predictions_to_msg(output ,skeletonbuffer.seq))    
            self.predictions_to_pointcloud(output, skeletonbuffer.transform.data, skeletonbuffer.transform_shape)
            index_pose = 10
            self.publisher_heading.publish(self.goal_from_pose(output, index=index_pose, ahead=1.8))
            
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
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str,default="../training/model/new/config.json")
    parser.add_argument('--model_file', type=str,default="../training/model/new/best_epoch_fde_0002_best_sofar.pt")
    parser.add_argument('--data_path', type=str, default="../data/h3.6m/")

    parser.add_argument('--mode', type=int, default=0)

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
  
    model = STPoseTransformer.model_factory(
          params, 
          pose_encoder_fn, 
          pose_decoder_fn,
          traj_encoder_fn, 
          traj_decoder_fn
      )
    
    model.load_state_dict(torch.load(args.model_file, map_location=_DEVICE))
    model.to(_DEVICE)
    model.eval()
              
    human_prediction(model,parents, mode=args.mode)
    rospy.spin()
    
