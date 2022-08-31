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
from zed_interfaces.msg import Skeleton3D

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#_DEVICE = torch.device('cpu')


class human_prediction():
    def __init__(self,model):
        rospy.init_node('listener', anonymous=True)
        self.model = model
        self.skeleton_subscriber = rospy.Subscriber("/pose_publisher/3DSkeletonBuffer", Skeleton3DBuffer, self.Predict)
        self.my_counter=0
  #      self.obj_pub = rospy.Publisher('/obj_position', PointCloud , queue_size=1)    
    def skeleton_to_inputs(self, skeleton):
        enc_input = []
        enc_traj = []
        dec_input = []
        dec_traj = []
        first_hip = list(skeleton.skeleton_3d_17[0].keypoints)[0]
        for body in skeleton.skeleton_3d_17:
            hip = list(body.keypoints)[0]
            keypoints = list(body.keypoints)[1:17]
            enc_input.append(np.array([np.array(kp.kp)-np.array(hip.kp) for kp in keypoints]))
            enc_traj.append(np.array(hip.kp)-np.array(first_hip.kp))

        dec_input = np.array([[kp.kp for kp in keypoints]]*20)
        dec_traj = np.array([np.array(hip.kp)-np.array(first_hip.kp)]*20)

        enc_input = np.array(enc_input).reshape(1,5,48) #reshape (5,3,16) to (5,48)
        enc_traj = np.array(enc_traj).reshape(1,5,3)
        dec_input = np.array(dec_input).reshape(1, 20,48)
        dec_traj = np.array(dec_traj).reshape(1,20,3)
        return enc_input, enc_traj, dec_input, dec_traj

    def Predict(self,skeleton):
        print("skeleton received,",skeleton.header.stamp)
        
        enc_input, enc_traj, dec_input, decc_traj = self.skeleton_to_inputs(skeleton)    
# =============================================================================
        with torch.no_grad():

            
            # sample['decoder_inputs_traj'] = sample['decoder_inputs_traj'] - sample['encoder_inputs_traj'][0,:,0].reshape(-1,1,3)
            # sample['decoder_outputs_traj'] = sample['decoder_outputs_traj'] - sample['encoder_inputs_traj'][0,:,0].reshape(-1,1,3)
            # sample['encoder_inputs_traj'] = sample['encoder_inputs_traj'] - sample['encoder_inputs_traj'][0,:,0].reshape(-1,1,3)

            enc_inputs = torch.from_numpy(enc_input).float().to(_DEVICE)
            dec_inputs = torch.from_numpy(dec_input).float().to(_DEVICE)
                        
            enc_inputs_traj = torch.from_numpy(enc_traj).float().to(_DEVICE)
            dec_inputs_traj = torch.from_numpy(decc_traj).float().to(_DEVICE)
          
# =============================================================================
#             enc_inputs = torch.squeeze(enc_inputs)
#             dec_inputs = torch.squeeze(dec_inputs)
#   
#             enc_inputs_traj = torch.squeeze(enc_inputs_traj)
#             dec_inputs_traj = torch.squeeze(dec_inputs_traj)
# =============================================================================
            
            t1 = time.time()
            prediction = model(
                enc_inputs,
                dec_inputs,
                enc_inputs_traj,
                dec_inputs_traj,
                get_attn_weights=True
            )
            t2=time.time()
  
            classes = prediction[1]
            traj_prediction = prediction[-1]
            traj_prediction = traj_prediction[-1].cpu().numpy()
  
            prediction = prediction[0]
            prediction = prediction[-1].cpu().numpy()
  
            preds = eval_dataset_fn.dataset.unnormalize_mine(prediction)
            preds_traj = eval_dataset_fn.dataset.unnormalize_mine_traj(traj_prediction)
            
            maximum_estimation_time = params['target_seq_len']/params['frame_rate']
            
            self.my_counter +=1
# =============================================================================
    
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str,default="/home/mohammad/Mohammad_ws/future_pose_prediction/potr/training/corrected3/config/config.json")
    parser.add_argument('--model_file', type=str,default="/home/mohammad/Mohammad_ws/future_pose_prediction/potr/training/corrected3/models/best_epoch_fde_0176_best_sofar.pt")
    parser.add_argument('--data_path', type=str, default="/home/mohammad/Mohammad_ws/future_pose_prediction/potr/data/h3.6m/")
    args = parser.parse_args()
    params = json.load(open(args.config_file))

    if args.data_path is not None:
      params['data_path'] = args.data_path
    args.data_path = params['data_path']
    
    
    train_dataset_fn, eval_dataset_fn = tr_fn.dataset_factory_total(params)
    
    
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
              
    human_prediction(model,eval_dataset_fn)
    rospy.spin()
    
    
    
    
    
