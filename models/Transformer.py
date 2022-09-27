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


"""Implementation of the Transformer for sequence-to-sequence decoding.

Implementation of the transformer for sequence to sequence prediction as in
[1] and [2].

[1] https://arxiv.org/pdf/1706.03762.pdf
[2] https://arxiv.org/pdf/2005.12872.pdf
"""

import numpy as np
import os
import sys
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment

thispath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, thispath+"/../")

import utils.utils as utils
import utils.PositionEncodings as PositionEncodings
import models.TransformerEncoder as Encoder
import models.TransformerDecoder as Decoder

class Transformer(nn.Module):
  def __init__(self,
              num_encoder_layers=6,
              num_decoder_layers=6,
              model_dim=256,
              model_dim_traj=128,
              num_heads=8,
              dim_ffn=2048,
              dropout=0.1,
              init_fn=utils.normal_init_,
              use_query_embedding=False,
              pre_normalization=False,
              query_selection=False,
              end_attention=True,
              target_seq_len=25):
    """Implements the Transformer model for sequence-to-sequence modeling."""
    super(Transformer, self).__init__()
    self._model_dim = model_dim
    self._model_dim_traj = model_dim_traj
    self._num_heads = num_heads
    self._dim_ffn = dim_ffn
    self._dropout = dropout
    self._use_query_embedding = use_query_embedding
    self._query_selection = query_selection
    self._tgt_seq_len = target_seq_len
    self._end_attention = end_attention
    self._encoder = Encoder.TransformerEncoder(
        num_layers=num_encoder_layers,
        model_dim=model_dim,
        num_heads=num_heads,
        dim_ffn=dim_ffn,
        dropout=dropout,
        init_fn=init_fn,
        pre_normalization=pre_normalization
    )
    
    self._decoder = Decoder.TransformerDecoder(
        num_layers=num_decoder_layers,
        model_dim=model_dim,
        num_heads=num_heads,
        dim_ffn=dim_ffn,
        dropout=dropout,
        init_fn=init_fn,
        use_query_embedding=use_query_embedding,
        pre_normalization=pre_normalization
    )
    
    self._encoder_traj = Encoder.TransformerEncoder(
        num_layers=num_encoder_layers,
        model_dim=model_dim_traj,
        num_heads=num_heads,
        dim_ffn=dim_ffn,
        dropout=dropout,
        init_fn=init_fn,
        pre_normalization=pre_normalization
    )
    
    self._decoder_traj = Decoder.TransformerDecoder(
        num_layers=num_decoder_layers,
        model_dim=model_dim_traj,
        num_heads=num_heads,
        dim_ffn=dim_ffn,
        dropout=dropout,
        init_fn=init_fn,
        use_query_embedding=use_query_embedding,
        pre_normalization=pre_normalization
    )
    
    if self._query_selection:
      self._position_predictor = nn.Linear(self._model_dim, self._tgt_seq_len)
    self._self_attn_traj = nn.MultiheadAttention(model_dim_traj, num_heads, dropout)
    self._self_attn_end = nn.MultiheadAttention(model_dim, num_heads, dropout)
    self._self_attn_end_traj = nn.MultiheadAttention(model_dim_traj, num_heads, dropout)

    self._traj_pos_linear = nn.Linear(self._model_dim, self._model_dim_traj)

    self.qkv = nn.Linear(self._model_dim, 3*self._model_dim)
    self.qkv_traj_end = nn.Linear(self._model_dim_traj, 3*self._model_dim_traj)
    self.q_traj = nn.Linear(self._model_dim_traj, self._model_dim_traj)
    self.k_traj = nn.Linear(self._model_dim_traj, self._model_dim_traj)
    self.v_traj = nn.Linear(self._model_dim_traj, self._model_dim_traj)


  def process_index_selection(self, self_attn, one_to_one_selection=False):
    """Selection of query elments using position predictor from encoder memory.

    After prediction a maximum assignement problem is solved to get indices for
    each element in the query sequence.

    Args:
      self_attn: Encoder memory with shape [src_len, batch_size, model_dim]

    Returns:
      A tuple with two list of i and j matrix entries of m
    """
    batch_size = self_attn.size()[1]
    # batch_size x src_seq_len x model_dim
    in_pos = torch.transpose(self_attn, 0, 1)
    # predict the matrix of similitudes
    # batch_size x src_seq_len x tgt_seq_len 
    prob_matrix = self._position_predictor(in_pos)
    # apply softmax on the similutes to get probabilities on positions
    # batch_size x src_seq_len x tgt_seq_len

    if one_to_one_selection:
      soft_matrix = F.softmax(prob_matrix, dim=2)
      # predict assignments in a one to one fashion maximizing the sum of probs
      indices = [linear_sum_assignment(soft_matrix[i].cpu().detach(), maximize=True) 
                 for i in range(batch_size)
      ]
    else:
      # perform softmax by rows to have many targets to one input assignements
      soft_matrix = F.softmax(prob_matrix, dim=1)
      indices_rows = torch.argmax(soft_matrix, 1)
      indices = [(indices_rows[i], list(range(prob_matrix.size()[2]))) 
          for i in range(batch_size)
      ]

    return indices, soft_matrix 

  def forward(self,
              source_seq,
              target_seq,
              source_seq_traj,
              target_seq_traj,
              encoder_position_encodings=None,
              decoder_position_encodings=None,
              encoder_trajectory_encodings=None,
              decoder_trajectory_encodings=None,
              query_embedding=None,
              query_embedding_traj=None,
              mask_target_padding=None,
              mask_look_ahead=None,
              get_attn_weights=False,
              query_selection_fn=None):
      
    if self._use_query_embedding:
      bs = source_seq.size()[1]
      query_embedding = query_embedding.unsqueeze(1).repeat(1, bs, 1)
      query_embedding_traj = query_embedding_traj.unsqueeze(1).repeat(1, bs, 1)
      decoder_position_encodings = encoder_position_encodings
      decoder_trajectory_encodings = encoder_trajectory_encodings
    
    memory, enc_weights = self._encoder(source_seq, encoder_position_encodings)
    memory_traj, enc_weights_traj = self._encoder_traj(source_seq_traj, encoder_trajectory_encodings)

    tgt_plain = None
    # perform selection from input sequence
    if self._query_selection:
      indices, prob_matrix = self.process_index_selection(memory)
      tgt_plain, target_seq = query_selection_fn(indices)
   

    out_attn, out_weights = self._decoder(
        target_seq,
        memory,
        decoder_position_encodings,
        query_embedding=query_embedding,
        mask_target_padding=mask_target_padding,
        mask_look_ahead=mask_look_ahead,
        get_attn_weights=get_attn_weights
    )
   
    memory_copy = self._traj_pos_linear(memory)

    query = self.q_traj(memory_traj)
    key = self.k_traj(memory_copy)
    value = self.v_traj(memory_copy)
    
    attn_output_traj_pose, attn_weights_traj_pose = self._self_attn_traj(
        query, 
        key, 
        value, 
        need_weights=True
    )

    memory_traj = memory_traj.clone() + attn_output_traj_pose
    
    out_attn_traj, out_weights_traj = self._decoder_traj(
        target_seq_traj,
        memory_traj,
        decoder_trajectory_encodings,
        query_embedding=query_embedding_traj,
        mask_target_padding=mask_target_padding,
        mask_look_ahead=mask_look_ahead,
        get_attn_weights=get_attn_weights
    )
    
    output_attn = []
    output_attn_traj=[]
    
    for i in range(len(out_attn)):
        enc_dec_tot = torch.concat((memory,out_attn[i]),dim=0)
        qkv_end = self.qkv(enc_dec_tot)
        q_end,k_end,v_end = qkv_end.chunk(3,dim=-1)
        end_attn, end_attn_weights = self._self_attn_end(q_end,k_end,v_end,need_weights=True)
        output_attn.append(end_attn[-self._tgt_seq_len:])
        
        enc_dec_tot_traj = torch.concat((memory_traj,out_attn_traj[i]),dim=0) 
        qkv_end_traj = self.qkv_traj_end(enc_dec_tot_traj)
        q_end_traj,k_end_traj,v_end_traj = qkv_end_traj.chunk(3,dim=-1)
        end_attn_traj, end_attn_traj_weights = self._self_attn_end_traj(q_end_traj,k_end_traj,v_end_traj,need_weights=True)
        output_attn_traj.append(end_attn_traj[-self._tgt_seq_len:])
        

    out_weights_ = None
    enc_weights_ = None
    prob_matrix_ = None
    
    if get_attn_weights:
      out_weights_, enc_weights_ = out_weights, enc_weights

    if self._query_selection:
      prob_matrix_ =  prob_matrix

    return output_attn, memory, out_weights_, enc_weights_, (tgt_plain, prob_matrix_),output_attn_traj,memory_traj

#    return out_attn, memory, out_weights_, enc_weights_, (tgt_plain, prob_matrix_),out_attn_traj,memory_traj
