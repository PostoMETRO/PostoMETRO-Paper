# --------------------------------------------------------------------------
# Borrowed from Pose Compositional Tokens(https://github.com/Gengzigang/PCT)
# --------------------------------------------------------------------------

import torch
import torch.nn as nn

from .pct_tokenizer import PCT_Tokenizer
from .pct_head import PCT_Head
from .modules import MixerLayer, FCBlock, BasicBlock
from .transformer import build_transformer
from .position_encoding import build_position_encoding

class PCT(nn.Module):
    def __init__(self,
                 args,
                 backbone,
                 stage_pct,
                 in_channels,
                 image_size,
                 num_joints,
                 pretrained=None,
                 tokenizer_pretrained=None):
        super().__init__()
        self.stage_pct = stage_pct
        assert self.stage_pct in ["tokenizer", "classifier"]
        self.guide_ratio = args.tokenizer_guide_ratio
        self.image_guide = self.guide_ratio > 0.0
        self.num_joints = num_joints

        self.backbone = backbone
        if self.image_guide:
            self.extra_backbone = backbone
        
        self.keypoint_head = PCT_Head(args,stage_pct,in_channels,image_size,num_joints)

        self.init_weights(pretrained, tokenizer_pretrained)

    def init_weights(self, pretrained, tokenizer):
        """Weight initialization for model."""
        if self.stage_pct == "classifier":
            self.backbone.init_weights(pretrained)
        if self.image_guide:
            self.extra_backbone.init_weights(pretrained)
        self.keypoint_head.init_weights()
        self.keypoint_head.tokenizer.init_weights(tokenizer)

    def forward(self,img, joints, train = True):
        if train:
            output = None if self.stage_pct == "tokenizer" else self.backbone(img)
            extra_output = self.extra_backbone(img) if self.image_guide else None

            p_logits, p_joints, g_logits, e_latent_loss = \
                self.keypoint_head(output, extra_output, joints, train=True)
            return {
                'cls_logits': p_logits,
                'pred_pose': p_joints,
                'encoding_indices': g_logits,
                'e_latent_loss': e_latent_loss
            }
        else:
            results = {}
    
            batch_size, _, img_height, img_width = img.shape
                
            output = None if self.stage_pct == "tokenizer" \
                else self.backbone(img) 
            extra_output = self.extra_backbone(img) \
                if self.image_guide and self.stage_pct == "tokenizer" else None
            
            p_joints, encoding_scores, out_part_token_feat = \
                self.keypoint_head(output, extra_output, joints, train=False)
            return {
                'pred_pose': p_joints,
                'encoding_scores': encoding_scores,
                'part_token_feat': out_part_token_feat
            }
