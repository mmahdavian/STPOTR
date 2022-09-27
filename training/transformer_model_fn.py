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

"""Implments the model function for the POTR model."""

import numpy as np
import os
import sys
import argparse
import json
import time

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

from data.h36m_dataset import Human36mDataset


_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_NSEEDS = 8

class STPOTRModelFn(seq2seq_model_fn.ModelFn):
  def __init__(self,
               params,
               train_dataset_fn,
               eval_dataset_fn,
               eval_dataset_fn_total,
               pose_encoder_fn=None,
               pose_decoder_fn=None,
               traj_encoder_fn=None,
               traj_decoder_fn=None):
    super(STPOTRModelFn, self).__init__(
      params, train_dataset_fn, eval_dataset_fn,eval_dataset_fn_total, pose_encoder_fn, pose_decoder_fn, traj_encoder_fn, traj_decoder_fn)
    self._loss_fn = self.layerwise_loss_fn

  def smooth_l1(self, decoder_pred, decoder_gt):
    l1loss = nn.SmoothL1Loss(reduction='mean')
    return l1loss(decoder_pred, decoder_gt)

  def loss_l1(self, decoder_pred, decoder_gt):
    return nn.L1Loss(reduction='mean')(decoder_pred, decoder_gt)

  def loss_activity(self, logits, class_gt):                                     
    """Computes entropy loss from logits between predictions and class."""
    return nn.functional.cross_entropy(logits, class_gt, reduction='mean')

  def compute_class_loss(self, class_logits, class_gt):
    """Computes the class loss for each of the decoder layers predictions or memory."""
    class_loss = 0.0
    for l in range(len(class_logits)):
      class_loss += self.loss_activity(class_logits[l], class_gt)

    return class_loss/len(class_logits)

  def select_loss_fn(self):
    if self._params['loss_fn'] == 'mse':
      return self.loss_mse
    elif self._params['loss_fn'] == 'smoothl1':
      return self.smooth_l1
    elif self._params['loss_fn'] == 'l1':
      return self.loss_l1
    else:
      raise ValueError('Unknown loss name {}.'.format(self._params['loss_fn']))

  def layerwise_loss_fn(self, decoder_pred, decoder_gt, decoder_pred_traj, decoder_gt_traj, class_logits=None, class_gt=None):
    """Computes layerwise loss between predictions and ground truth."""
    pose_loss = 0.0
    loss_fn = self.select_loss_fn()

    ### pose loss
    for l in range(len(decoder_pred)):
      pose_loss += loss_fn(decoder_pred[l], decoder_gt)

    pose_loss = pose_loss/len(decoder_pred)
    
    ### trajectory loss
    trajectory_loss = 0
    loss_fn_traj = self.select_loss_fn()
    for l in range(len(decoder_pred_traj)):
      trajectory_loss += loss_fn_traj(decoder_pred_traj[l], decoder_gt_traj)
      
    trajectory_loss = trajectory_loss/len(decoder_pred_traj)
    
    ### velocity loss
    loss_fn_velocity = self.select_loss_fn()
    velocity_loss = 0
    b,f,j = decoder_pred[-1].shape
    for l in range(len(decoder_pred)):
        velocity_loss += loss_fn_velocity((decoder_pred[0][:,1:f,:]-decoder_pred[0][:,0:f-1,:]) , (decoder_gt[:,1:f,:]-decoder_gt[:,0:f-1,:]))
        
    velocity_loss = velocity_loss/len(decoder_pred)
    
    if class_logits is not None:
      return pose_loss, self.compute_class_loss(class_logits, class_gt), trajectory_loss , velocity_loss

    return pose_loss, None , trajectory_loss , velocity_loss

  def init_model(self, pose_encoder_fn=None, pose_decoder_fn=None, traj_encoder_fn=None, traj_decoder_fn=None):
    self._model = STPoseTransformer.model_factory(
        self._params, 
        pose_encoder_fn, 
        pose_decoder_fn,
        traj_encoder_fn, 
        traj_decoder_fn  
    )
    
  def select_optimizer(self):
    optimizer = optim.AdamW(
        self._model.parameters(), lr=self._params['learning_rate'],
        betas=(0.9, 0.999),
        weight_decay=self._params['weight_decay']
    )
    return optimizer

def dataset_factory_total(params):
  if params['dataset'] == 'h36m_v3':
    return H36MDataset_v3.dataset_factory_total(params)
  else:
    raise ValueError('Unknown dataset {}'.format(params['dataset']))


def dataset_factory(params):
  if params['dataset'] == 'h36m_v3':
    return H36MDataset_v3.dataset_factory(params)
  else:
    raise ValueError('Unknown dataset {}'.format(params['dataset']))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_prefix', type=str, default='trained_model')
  parser.add_argument('--batch_size', type=int, default=16)
  parser.add_argument('--data_path', type=str,default='../data/h3.6m/')
  parser.add_argument('--learning_rate', type=float, default=1e-4)
  parser.add_argument('--max_epochs', type=int, default=500)
  parser.add_argument('--steps_per_epoch', type=int, default=200)
  parser.add_argument('--action', type=str, default='all')
  parser.add_argument('--use_one_hot',  action='store_true')
  parser.add_argument('--init_fn', type=str, default='xavier_init')
  parser.add_argument('--include_last_obs', action='store_true')
  parser.add_argument('--model_dim', type=int, default=256)
  parser.add_argument('--model_dim_traj', type=int, default=32)
  parser.add_argument('--num_encoder_layers', type=int, default=4)
  parser.add_argument('--num_decoder_layers', type=int, default=4)
  parser.add_argument('--num_heads', type=int, default=4)
  parser.add_argument('--dim_ffn', type=int, default=2048)
  parser.add_argument('--dropout', type=float, default=0.3)
  parser.add_argument('--source_seq_len', type=int, default=5)                  
  parser.add_argument('--target_seq_len', type=int, default=20)
  parser.add_argument('--frame_rate', type=int, default=10)
  parser.add_argument('--max_gradient_norm', type=float, default=0.1)
  parser.add_argument('--lr_step_size',type=int, default=400)
  parser.add_argument('--gamma',type=float, default=0.1)
  parser.add_argument('--learning_rate_fn',type=str, default='step')
  parser.add_argument('--warmup_epochs', type=int, default=100)
  parser.add_argument('--pose_format', type=str, default='rotmat')
  parser.add_argument('--remove_low_std', action='store_true')
  parser.add_argument('--remove_global_trans', action='store_true')
  parser.add_argument('--loss_fn', type=str, default='mse')
  parser.add_argument('--pad_decoder_inputs', action='store_true')
  parser.add_argument('--pad_decoder_inputs_mean', action='store_true')
  parser.add_argument('--use_wao_amass_joints', action='store_true')
  parser.add_argument('--non_autoregressive', action='store_true')
  parser.add_argument('--pre_normalization', action='store_true')
  parser.add_argument('--use_query_embedding', action='store_true')
  parser.add_argument('--predict_activity', action='store_true')
  parser.add_argument('--use_memory', action='store_true')
  parser.add_argument('--query_selection',action='store_true')
  parser.add_argument('--activity_weight', type=float, default=1.0)
  parser.add_argument('--pose_embedding_type', type=str, default='gcn_enc')
  parser.add_argument('--traj_embedding_type', type=str, default='gcn_enc')
  parser.add_argument('--encoder_ckpt', type=str, default=None)
  parser.add_argument('--dataset', type=str, default='h36m_v3')
  parser.add_argument('--skip_rate', type=int, default=5)
  parser.add_argument('--eval_num_seeds', type=int, default=_NSEEDS)
  parser.add_argument('--copy_method', type=str, default=None)
  parser.add_argument('--finetuning_ckpt', type=str, default=None)
  parser.add_argument('--pos_enc_alpha', type=float, default=10)
  parser.add_argument('--pos_enc_beta', type=float, default=500)
  parser.add_argument('--gt_ratio', type=float, default=0)
  parser.add_argument('--weight_decay', type=float, default=0.001)
  parser.add_argument('--std', type=float, default=0.0005)
  parser.add_argument('--std_traj', type=float, default=0.0005)
  parser.add_argument('--noisy', action='store_true')
  parser.add_argument('--heading', action='store_true')

  args = parser.parse_args()
  params = vars(args)
  train_dataset_fn, eval_dataset_fn = dataset_factory(params)
  train_dataset_fn_total, eval_dataset_fn_total = dataset_factory_total(params)

  params['input_dim'] = train_dataset_fn.dataset._data_dim
  params['pose_dim'] = train_dataset_fn.dataset._pose_dim
  params['input_dim_traj'] = 3
  params['pose_dim_traj'] = 3
  
  pose_encoder_fn, pose_decoder_fn = \
      PoseEncoderDecoder.select_pose_encoder_decoder_fn(params)

  traj_encoder_fn, traj_decoder_fn = \
      PoseEncoderDecoder.select_traj_encoder_decoder_fn(params)

  for k,v in params.items():
    print('[INFO] (STPOTRFn@main) {}: {}'.format(k, v))

  utils.create_dir_tree(params['model_prefix'])

  config_path = os.path.join(params['model_prefix'], 'config', 'config.json')        
  with open(config_path, 'w') as file_:
    json.dump(params, file_, indent=4)

  model_fn = STPOTRModelFn(
      params, train_dataset_fn, 
      eval_dataset_fn, 
      eval_dataset_fn_total,
      pose_encoder_fn, 
      pose_decoder_fn, 
      traj_encoder_fn, 
      traj_decoder_fn,
  )
  model_fn.train()
  