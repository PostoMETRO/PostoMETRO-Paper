# ----------------------------------------------------------------------------------------------
# PostoMETRO Official Code. Adopted from FastMETRO(https://github.com/postech-ami/FastMETRO)
# Copyright (c) Medical Imaging, Robotics, Analytical Computing & Learning. (MIRACLE.) All Rights Reserved 
# Licensed under the MIT license.
# ----------------------------------------------------------------------------------------------

"""
PostoMETRO model.
"""
from __future__ import absolute_import, division, print_function
import torch
import numpy as np
from torch import nn
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer
from .position_encoding import build_position_encoding
from .smpl_param_regressor import build_smpl_parameter_regressor
import src.modeling.data.config as cfg
from .modules import FCBlock, make_conv_layers, MixerLayer

class PostoMETRO_Body_Network(nn.Module):
    """PostoMETRO for 3D human pose and mesh reconstruction from a single RGB image"""
    def __init__(self, args, backbone, mesh_sampler, pct = None, num_joints=14, num_vertices=431):
        """
        Parameters:
            - args: Arguments
            - backbone: CNN Backbone used to extract image features from the given image
            - mesh_sampler: Mesh Sampler used in the coarse-to-fine mesh upsampling
            - num_joints: The number of joint tokens used in the transformer decoder
            - num_vertices: The number of vertex tokens used in the transformer decoder
        """
        super().__init__()
        self.args = args
        self.backbone = backbone
        self.mesh_sampler = mesh_sampler
        self.num_joints = num_joints
        self.num_vertices = num_vertices
        
        # the number of transformer layers
        if 'PostoMETRO-S' in args.model_name:
            num_enc_layers = 1
            num_dec_layers = 1
        elif 'PostoMETRO-M' in args.model_name:
            num_enc_layers = 2
            num_dec_layers = 2
        elif 'PostoMETRO-L' in args.model_name:
            num_enc_layers = 3
            num_dec_layers = 3
        else:
            assert False, "The model name is not valid"
    
        # configurations for the first transformer
        self.transformer_config_1 = {"model_dim": args.model_dim_1, "dropout": args.transformer_dropout, "nhead": args.transformer_nhead, 
                                     "feedforward_dim": args.feedforward_dim_1, "num_enc_layers": num_enc_layers, "num_dec_layers": num_dec_layers, 
                                     "pos_type": args.pos_type}
        # configurations for the second transformer
        self.transformer_config_2 = {"model_dim": args.model_dim_2, "dropout": args.transformer_dropout, "nhead": args.transformer_nhead,
                                     "feedforward_dim": args.feedforward_dim_2, "num_enc_layers": num_enc_layers, "num_dec_layers": num_dec_layers, 
                                     "pos_type": args.pos_type}
        # build transformers
        self.transformer_1 = build_transformer(self.transformer_config_1)
        self.transformer_2 = build_transformer(self.transformer_config_2)

        # dimensionality reduction
        self.dim_reduce_enc_cam = nn.Linear(self.transformer_config_1["model_dim"], self.transformer_config_2["model_dim"])
        self.dim_reduce_enc_img = nn.Linear(self.transformer_config_1["model_dim"], self.transformer_config_2["model_dim"])
        self.dim_reduce_dec = nn.Linear(self.transformer_config_1["model_dim"], self.transformer_config_2["model_dim"])
        
        # token embeddings
        self.cam_token_embed = nn.Embedding(1, self.transformer_config_1["model_dim"])
        self.joint_token_embed = nn.Embedding(self.num_joints, self.transformer_config_1["model_dim"])
        self.vertex_token_embed = nn.Embedding(self.num_vertices, self.transformer_config_1["model_dim"])
        # positional encodings
        self.position_encoding_1 = build_position_encoding(pos_type=self.transformer_config_1['pos_type'], hidden_dim=self.transformer_config_1['model_dim'])
        self.position_encoding_2 = build_position_encoding(pos_type=self.transformer_config_2['pos_type'], hidden_dim=self.transformer_config_2['model_dim'])
        # estimators
        self.xyz_regressor = nn.Linear(self.transformer_config_2["model_dim"], 3)
        self.cam_predictor = nn.Linear(self.transformer_config_2["model_dim"], 3)
        
        # 1x1 Convolution
        self.conv_1x1 = nn.Conv2d(args.conv_1x1_dim, self.transformer_config_1["model_dim"], kernel_size=1)

        # attention mask
        zeros_1 = torch.tensor(np.zeros((num_vertices, num_joints)).astype(bool)) 
        zeros_2 = torch.tensor(np.zeros((num_joints, (num_joints + num_vertices))).astype(bool)) 
        adjacency_indices = torch.load('./src/modeling/data/smpl_431_adjmat_indices.pt')
        adjacency_matrix_value = torch.load('./src/modeling/data/smpl_431_adjmat_values.pt')
        adjacency_matrix_size = torch.load('./src/modeling/data/smpl_431_adjmat_size.pt')
        adjacency_matrix = torch.sparse_coo_tensor(adjacency_indices, adjacency_matrix_value, size=adjacency_matrix_size).to_dense()
        temp_mask_1 = (adjacency_matrix == 0)
        temp_mask_2 = torch.cat([zeros_1, temp_mask_1], dim=1)
        self.attention_mask = torch.cat([zeros_2, temp_mask_2], dim=0)
        
        # learnable upsampling layer is used (from coarse mesh to intermediate mesh); for visually pleasing mesh result
        ### pre-computed upsampling matrix is used (from intermediate mesh to fine mesh); to reduce optimization difficulty
        self.coarse2intermediate_upsample = nn.Linear(431, 1723)

        # (optional) smpl parameter regressor; to obtain SMPL parameters
        if args.use_smpl_param_regressor:
            self.smpl_parameter_regressor = build_smpl_parameter_regressor()

        # using extra 2D pose token
        self.pct = pct if pct is not None else None

        if self.pct is not None:
            self.token_mixer = FCBlock(args.tokenizer_codebook_token_dim + 1, self.transformer_config_1["model_dim"])

            pct_param_dict = {
                'enc_hidden_dim': 512,
                'enc_hidden_inter_dim': 512,
                'token_inter_dim': 64,
                'enc_dropout': 0.0,
                'enc_num_blocks': 4,
                'num_joints': 34,
                'token_num': 34
            }

            self.start_embed = nn.Linear(512, pct_param_dict['enc_hidden_dim'])
            self.encoder = nn.ModuleList([
                MixerLayer(pct_param_dict['enc_hidden_dim'], pct_param_dict['enc_hidden_inter_dim'], 
                        pct_param_dict['num_joints'], pct_param_dict['token_inter_dim'], 
                        pct_param_dict['enc_dropout']) 
                for _ in range(pct_param_dict['enc_num_blocks'])
            ])
            self.encoder_layer_norm = nn.LayerNorm(pct_param_dict['enc_hidden_dim'])
            self.token_mlp = nn.Linear(pct_param_dict['num_joints'], pct_param_dict['token_num'])

            self.dim_reduce_enc_pct = nn.Linear(self.transformer_config_1["model_dim"], self.transformer_config_2["model_dim"])
 
    def make_2d_gaussian_heatmap(self, joint_coord_img, heatmap_shape, sigma = 2.5):
        x = torch.arange(heatmap_shape[1])
        y = torch.arange(heatmap_shape[0])
        yy, xx = torch.meshgrid(y, x)
        xx = xx[None, None, :, :].cuda().float()
        yy = yy[None, None, :, :].cuda().float()

        x = joint_coord_img[:, :, 0, None, None]
        y = joint_coord_img[:, :, 1, None, None]
        heatmap = torch.exp(
            -(((xx - x) / sigma) ** 2) / 2 - (((yy - y) / sigma) ** 2) / 2)
        return heatmap

    def forward(self, images):
        device = images.device
        batch_size = images.size(0)

        # preparation
        cam_token = self.cam_token_embed.weight.unsqueeze(1).repeat(1, batch_size, 1) # 1 X batch_size X 512 
        jv_tokens = torch.cat([self.joint_token_embed.weight, self.vertex_token_embed.weight], dim=0).unsqueeze(1).repeat(1, batch_size, 1) # (num_joints + num_vertices) X batch_size X 512
        attention_mask = self.attention_mask.to(device) # (num_joints + num_vertices) X (num_joints + num_vertices)

        pct_token = None
        if self.pct is not None:
            pct_out = self.pct(images, None, train=False)
            pct_pose = pct_out['part_token_feat'].clone()

            encode_feat = self.start_embed(pct_pose) # 2, 17, 512
            for num_layer in self.encoder:
                encode_feat = num_layer(encode_feat)
            encode_feat = self.encoder_layer_norm(encode_feat)
            encode_feat = encode_feat.transpose(2, 1)
            encode_feat = self.token_mlp(encode_feat).transpose(2, 1)
            pct_token_out = encode_feat.permute(1,0,2)

            pct_score = pct_out['encoding_scores']
            pct_score = pct_score.permute(1,0,2)
            pct_token = torch.cat([pct_token_out, pct_score], dim = -1)

            pct_token = self.token_mixer(pct_token) # [b, 34, 512]

        
        # extract image features through a CNN backbone
        _img_features = self.backbone(images) # batch_size X 2048 X 7 X 7
        _, _, h, w = _img_features.shape
        img_features = self.conv_1x1(_img_features).flatten(2).permute(2, 0, 1) # 49 X batch_size X 512 
        
        # positional encodings
        pos_enc_1 = self.position_encoding_1(batch_size, h, w, device).flatten(2).permute(2, 0, 1) # 49 X batch_size X 512 
        pos_enc_2 = self.position_encoding_2(batch_size, h, w, device).flatten(2).permute(2, 0, 1) # 49 X batch_size X 128 

        # first transformer encoder-decoder
        cam_features_1, enc_img_features_1, jv_features_1, pct_features_1 = self.transformer_1(img_features, cam_token, jv_tokens, pos_enc_1, pct_token = pct_token, attention_mask=attention_mask, decoder_attn_mask = None)
        
        # progressive dimensionality reduction
        reduced_cam_features_1 = self.dim_reduce_enc_cam(cam_features_1) # 1 X batch_size X 128 
        reduced_enc_img_features_1 = self.dim_reduce_enc_img(enc_img_features_1) # 49 X batch_size X 128 
        reduced_jv_features_1 = self.dim_reduce_dec(jv_features_1) # (num_joints + num_vertices) X batch_size X 128
        reduced_pct_features_1 = None
        if pct_features_1 is not None:
            reduced_pct_features_1 = self.dim_reduce_enc_pct(pct_features_1)

        # second transformer encoder-decoder
        cam_features_2, _, jv_features_2,_ = self.transformer_2(reduced_enc_img_features_1, reduced_cam_features_1, reduced_jv_features_1, pos_enc_2, pct_token = reduced_pct_features_1, attention_mask=attention_mask) 

        # estimators
        pred_cam = self.cam_predictor(cam_features_2).view(batch_size, 3) # batch_size X 3

        pred_3d_coordinates = self.xyz_regressor(jv_features_2.transpose(0, 1)) # batch_size X (num_joints + num_vertices) X 3
        pred_3d_joints = pred_3d_coordinates[:,:self.num_joints,:] # batch_size X num_joints X 3
        pred_3d_vertices_coarse = pred_3d_coordinates[:,self.num_joints:,:] # batch_size X num_vertices(coarse) X 3
        
        # coarse-to-intermediate mesh upsampling
        pred_3d_vertices_intermediate = self.coarse2intermediate_upsample(pred_3d_vertices_coarse.transpose(1,2)).transpose(1,2) # batch_size X num_vertices(intermediate) X 3
        # intermediate-to-fine mesh upsampling
        pred_3d_vertices_fine = self.mesh_sampler.upsample(pred_3d_vertices_intermediate, n1=1, n2=0) # batch_size X num_vertices(fine) X 3

        out = {}
        out['pred_cam'] = pred_cam
        out['pred_3d_joints'] = pred_3d_joints
        out['pred_3d_vertices_coarse'] = pred_3d_vertices_coarse
        out['pred_3d_vertices_intermediate'] = pred_3d_vertices_intermediate
        out['pred_3d_vertices_fine'] = pred_3d_vertices_fine

        return out
