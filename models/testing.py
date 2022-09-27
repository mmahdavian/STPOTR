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

"""Model function to deploy STPOTR models for visualization and generation."""

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
import models.STPoseTransformer as STPoseTransformer
import models.PoseEncoderDecoder as PoseEncoderDecoder
import data.H36MDataset_v3 as H36MDataset_v3
#import data.AMASSDataset as AMASSDataset
import utils.utils as utils
#import radam.radam as radam
import training.transformer_model_fn as tr_fn
import tqdm

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#_DEVICE = torch.device('cpu')


def plot_conf_mat(matrix):
  import matplotlib.pyplot as plt
  import matplotlib
  fig, ax = plt.subplots(figsize=(30,30))
  #im = ax.imshow(matrix, cmap='Wistia')
  im = ax.imshow(matrix, cmap='Blues')

  action_labels = ['A%02d'%i for i in range(1, 61, 1)]
  ax.set_xticks(np.arange(len(action_labels)))
  ax.set_yticks(np.arange(len(action_labels)))

  ax.set_xticklabels(action_labels, fontdict={'fontsize':15})#, rotation=90)
  ax.set_yticklabels(action_labels, fontdict={'fontsize':15})

  ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

  for i in range(len(action_labels)):
    for j in range(len(action_labels)):
      # color= "w" if round(matrix[i, j],2) < nmax else "black"
      text = ax.text(j, i, round(matrix[i, j], 2),
          ha="center", va="center", color="black", fontsize=10)

  plt.ylabel("")
  plt.xlabel("")
  # ax.set_title("Small plot")
  fig.tight_layout()
  #plt.show()
  plt.savefig('confusion_matrix.png')
  plt.close()

def crop_image(img):
  size = max(img.shape[0], img.shape[1])
  h = int(size*0.30)
  w = int(size*0.30)
  cy = img.shape[0]//2
  cx = img.shape[1]//2

  crop = img[cy-h//2:cy+h//2, cx-w//2:cx+w//2]
  return crop

def compute_ade(pred, gt):
    diff = pred - gt
    dist = np.linalg.norm(diff, axis=2).mean(axis=1).mean(axis=0)
    return dist.min()

def compute_fde(pred, gt):
    diff = pred - gt
    dist = np.linalg.norm(diff, axis=2)[:, -1].mean(axis=0)
    return dist.min() 

def compute_stats(pred, gt, mrt):
    _,f,_ = gt.shape
    diff = pred - gt
    pos_errors = []
    for i in range(f):
        err = diff[:,i]
        err = np.linalg.norm(err, axis=1).mean()
        pos_errors.append(err)
    return pos_errors

def Calc_error_h36mdataset():
      parser = argparse.ArgumentParser()
      parser.add_argument('--config_file', type=str,default="../pretrained_model/config/config.json")
      parser.add_argument('--model_file', type=str,default="../pretrained_model/models/pretrained_model.pt")
      parser.add_argument('--data_path', type=str, default="../data/h3.6m/")
      args = parser.parse_args()

      seq_shape = (15, 8, 25, 63)    
      seq_shape_traj = (15, 8, 25, 3)
      
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
        print('[INFO] (STPOTRFn@main) {}: {}'.format(k, v))
    
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
           
      total_errors = 0 
      total_ade = 0
      total_fde = 0
      total_errors_traj = 0 
      total_ade_traj = 0
      total_fde_traj = 0     

      sample = next(iter(eval_dataset_fn))

      with torch.no_grad():

          enc_inputs = sample['encoder_inputs'].to(_DEVICE)
          dec_inputs = sample['decoder_inputs'].to(_DEVICE)
          sample['decoder_inputs_traj'] = sample['decoder_inputs_traj'] - sample['encoder_inputs_traj'][0,:,0].reshape(-1,1,3)
          sample['decoder_outputs_traj'] = sample['decoder_outputs_traj'] - sample['encoder_inputs_traj'][0,:,0].reshape(-1,1,3)
          sample['encoder_inputs_traj'] = sample['encoder_inputs_traj'] - sample['encoder_inputs_traj'][0,:,0].reshape(-1,1,3)
          
          enc_inputs_traj = sample['encoder_inputs_traj'].to(_DEVICE)
          dec_inputs_traj = sample['decoder_inputs_traj'].to(_DEVICE)
        

          gts = np.squeeze(sample['decoder_outputs'].cpu().numpy())
          ins = np.squeeze(sample['encoder_inputs'].cpu().numpy())
          gts_traj = np.squeeze(sample['decoder_outputs_traj'].cpu().numpy())
          ins_traj = np.squeeze(sample['encoder_inputs_traj'].cpu().numpy())

          ins = eval_dataset_fn.dataset.unnormalize_mine(ins)
          ins_traj = eval_dataset_fn.dataset.unnormalize_mine_traj(ins_traj)
      
          gts = eval_dataset_fn.dataset.unnormalize_mine(gts)
          gts_traj = eval_dataset_fn.dataset.unnormalize_mine_traj(gts_traj)
       
          enc_inputs = torch.squeeze(enc_inputs)
          dec_inputs = torch.squeeze(dec_inputs)

          enc_inputs_traj = torch.squeeze(enc_inputs_traj)
          dec_inputs_traj = torch.squeeze(dec_inputs_traj)
        
          prediction = model(
              enc_inputs,
              dec_inputs,
              enc_inputs_traj,
              dec_inputs_traj,
              get_attn_weights=True
          )

          classes = prediction[1]
          traj_prediction = prediction[-1]
          traj_prediction = traj_prediction[-1].cpu().numpy()

          prediction = prediction[0]
          prediction = prediction[-1].cpu().numpy()

          preds = eval_dataset_fn.dataset.unnormalize_mine(prediction)
          preds_traj = eval_dataset_fn.dataset.unnormalize_mine_traj(traj_prediction)
          
          maximum_estimation_time = params['target_seq_len']/params['frame_rate']
          
          errors = compute_stats(preds,gts,maximum_estimation_time)
          ADE = compute_ade(preds,gts)
          FDE = compute_fde(preds,gts)
          total_errors += np.array(errors)      
          total_ade += ADE
          total_fde += FDE  
          
          errors_traj = compute_stats(preds_traj,gts_traj,maximum_estimation_time)
          ADE_traj = compute_ade(preds_traj,gts_traj)
          FDE_traj = compute_fde(preds_traj,gts_traj)
          total_errors_traj += np.array(errors_traj)      
          total_ade_traj += ADE_traj
          total_fde_traj += FDE_traj
          
          print(ADE,FDE,ADE_traj,FDE_traj)

              
      maximum_time = params['target_seq_len']/params['frame_rate']
      
      total_errors = total_errors
      dt = maximum_time/params['target_seq_len']
      
      print("result of evaluation on data")
    
      for i in range(params['target_seq_len']):
          print(str((i+1)*dt)[:5]+"  ", end=" ")
      print(" ")
      for i in range(params['target_seq_len']):
          print(str(total_errors[i])[:5], end=" ")
      print(" ")
      
      
      avg_ADE = total_ade 
      avg_FDE = total_fde 
      print(avg_ADE,avg_FDE)
      print()

      total_errors_traj = total_errors_traj
      print("result of evaluation for hip traj on data")
    
      for i in range(params['target_seq_len']):
          print(str((i+1)*dt)[:5]+"  ", end=" ")
      print(" ")
      for i in range(params['target_seq_len']):
          print(str(total_errors_traj[i])[:5], end=" ")
      print(" ")
      
      avg_ADE_traj = total_ade_traj 
      avg_FDE_traj = total_fde_traj
      print(avg_ADE_traj,avg_FDE_traj)
      return 0

class visual():
    def __init__(self):
        self.norm_stats={}
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
            
    
    def visualize_h36mdataset(self):
          parser = argparse.ArgumentParser()
          parser.add_argument('--config_file', type=str,default="../pretrained_model/config/config.json")
          parser.add_argument('--model_file', type=str,default="../pretrained_model/models/pretrained_model.pt")
          parser.add_argument('--data_path', type=str, default="../data/h3.6m/")
          args = parser.parse_args()
        
          parents, offset, rot_ind, exp_map_ind = utils.load_constants(args.data_path)

          seq_shape = (15, 8, 25, 63)    
          seq_shape_traj = (15, 8, 25, 3)
          
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
            print('[INFO] (STPOTRFn@main) {}: {}'.format(k, v))
          
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
                  
          _ALL_ACTIONS = [
                ("Directions",0),( "Discussion",1),( "Eating",2),( "Greeting",3), ("Phoning",4),
                ("Photo",5),("Posing",6), ("Purchases",7), ("Sitting",8), ("SittingDown",9), ("Smoking",10),
                ("Waiting",11), ("Walking",12), ("WalkDog",13), ("WalkTogether",14)]
    
          sample = next(iter(eval_dataset_fn))

          with torch.no_grad():
              enc_inputs = sample['encoder_inputs'].to(_DEVICE)
              dec_inputs = sample['decoder_inputs'].to(_DEVICE)
              sample['decoder_inputs_traj'] = sample['decoder_inputs_traj'] - sample['encoder_inputs_traj'][0,:,0].reshape(-1,1,3)
              sample['decoder_outputs_traj'] = sample['decoder_outputs_traj'] - sample['encoder_inputs_traj'][0,:,0].reshape(-1,1,3)
              sample['encoder_inputs_traj'] = sample['encoder_inputs_traj'] - sample['encoder_inputs_traj'][0,:,0].reshape(-1,1,3)
              
              enc_inputs_traj = sample['encoder_inputs_traj'].to(_DEVICE)
              dec_inputs_traj = sample['decoder_inputs_traj'].to(_DEVICE)
            
    
              gts = np.squeeze(sample['decoder_outputs'].cpu().numpy())
              ins = np.squeeze(sample['encoder_inputs'].cpu().numpy())
              gts_traj = np.squeeze(sample['decoder_outputs_traj'].cpu().numpy())
              ins_traj = np.squeeze(sample['encoder_inputs_traj'].cpu().numpy())
    
              ins = eval_dataset_fn.dataset.unnormalize_mine(ins)
              ins_traj = eval_dataset_fn.dataset.unnormalize_mine_traj(ins_traj)
          
              gts = eval_dataset_fn.dataset.unnormalize_mine(gts)
              gts_traj = eval_dataset_fn.dataset.unnormalize_mine_traj(gts_traj)
           
              enc_inputs = torch.squeeze(enc_inputs)
              dec_inputs = torch.squeeze(dec_inputs)
    
              enc_inputs_traj = torch.squeeze(enc_inputs_traj)
              dec_inputs_traj = torch.squeeze(dec_inputs_traj)
        
              prediction = model(
                  enc_inputs, 
                  dec_inputs, 
                  enc_inputs_traj, 
                  dec_inputs_traj, 
                  get_attn_weights=True
              )
              
              classes = prediction[1]
              traj_prediction = prediction[-1]
              traj_prediction = traj_prediction[-1].cpu().numpy()
    
              prediction = prediction[0]
              prediction = prediction[-1].cpu().numpy()
    
              preds = eval_dataset_fn.dataset.unnormalize_mine(prediction)
              preds_traj = eval_dataset_fn.dataset.unnormalize_mine_traj(traj_prediction)
              
              joints_left=[3, 4, 5, 10, 11, 12]
              joints_right=[0, 1, 2, 13, 14, 15]

              elev = -125
              azim = -45
              frames3 = [1059]
              
              k_num=0
              fig = plt.figure(figsize=(30,30))
    
              for kk in frames3:
                  batch=kk
                  ### prediction graph
                  ax = fig.add_subplot(1,2,1, projection='3d')
                  ax.view_init(elev=elev, azim=azim)
                  ax.set_xlim3d([-3,3])
                  ax.set_ylim3d([-3,3])
                  ax.set_zlim3d([-1,1])
                  
                  ax.set_zticks([])
                  ax.set_xticks([])
                  ax.set_yticks([])    
    
                  for i in range(5):
                      if i!=0 and i!=4:
                          continue
                      pos = ins[batch][i].reshape(16,3)
                      pos = np.concatenate((np.array([[0,0,0]]),pos),axis=0) + ins_traj[batch][i] - ins_traj[batch][0]
        
                      for j, j_parent in enumerate(parents):    
                          if j_parent == -1:
                              continue                
                          ax.plot([pos[j, 0], pos[j_parent, 0]],
                                  [pos[j, 2], pos[j_parent, 2]],
                                  [pos[j, 1], pos[j_parent, 1]], zdir='z', c='blue',alpha=(i+1)*0.15+0.25)
                  
                  in_gt_traj = ins_traj[batch] - ins_traj[batch][0]
                  ax.plot(in_gt_traj[:,0],in_gt_traj[:,2],in_gt_traj[:,1],'b--')
                  
                  for i in range(20):
                      if i!=6 and i!=13 and i!=19:
                          continue
                      pos = preds[batch][i].reshape(16,3)
                      pos = np.concatenate((np.array([[0,0,0]]),pos),axis=0) + preds_traj[batch][i] - ins_traj[batch][0]
        
                      for j, j_parent in enumerate(parents):    
                          if j_parent == -1:
                              continue                
                          ax.plot([pos[j, 0], pos[j_parent, 0]],
                                  [pos[j, 2], pos[j_parent, 2]],
                                  [pos[j, 1], pos[j_parent, 1]], zdir='z', c='darkred',alpha=(i+1)*0.04+ 0.2)
                          
                  out_pred_traj = preds_traj[batch] - ins_traj[batch][0]
                  ax.plot(out_pred_traj[:,0],out_pred_traj[:,2],out_pred_traj[:,1],'b--')                            
                  ### ground truth graph
                  ax = fig.add_subplot(1,2,2, projection='3d')
                  ax.view_init(elev=elev, azim=azim)

                  ax.set_xlim3d([-3,3])
                  ax.set_ylim3d([-3,3])  
                  ax.set_zlim3d([-1,1])
                  ax.set_zticks([])
                  ax.set_xticks([])
                  ax.set_yticks([])      
    
                  for i in range(5):
                      if i!=0 and i!=4:
                          continue
                      pos = ins[batch][i].reshape(16,3)
                      pos = np.concatenate((np.array([[0,0,0]]),pos),axis=0) + ins_traj[batch][i] - ins_traj[batch][0]
    
                      for j, j_parent in enumerate(parents):    
                          if j_parent == -1:
                              continue                
                          ax.plot([pos[j, 0], pos[j_parent, 0]],
                                  [pos[j, 2], pos[j_parent, 2]],
                                  [pos[j, 1], pos[j_parent, 1]], zdir='z', c='blue',alpha=(i+1)*0.15+0.25)
                          
                  in_gt_traj = ins_traj[batch] - ins_traj[batch][0]
                  ax.plot(in_gt_traj[:,0],in_gt_traj[:,2],in_gt_traj[:,1],'b--')
    
                  for i in range(20):
                      if i!=6 and i!=13 and i!=19:
                          continue
                      pos = gts[batch][i].reshape(16,3)
                      pos = np.concatenate((np.array([[0,0,0]]),pos),axis=0) + gts_traj[batch][i] - ins_traj[batch][0]
                      for j, j_parent in enumerate(parents):    
                          if j_parent == -1:
                              continue                
                          ax.plot([pos[j, 0], pos[j_parent, 0]],
                                  [pos[j, 2], pos[j_parent, 2]],
                                  [pos[j, 1], pos[j_parent, 1]], zdir='z', c='darkgreen',alpha=(i+1)*0.04+ 0.2)
    
                  out_gt_traj = gts_traj[batch] - ins_traj[batch][0]
                  ax.plot(out_gt_traj[:,0],out_gt_traj[:,2],out_gt_traj[:,1],'b--')
    
                  fig.savefig('./human3d_prediction_vs_gt'+str(kk)+'.png')
                  k_num+=1
                  plt.subplots_adjust(wspace=0, hspace=0)
              return 0

if __name__ == '__main__':
   
#   visual().visualize_h36mdataset()
   Calc_error_h36mdataset()









