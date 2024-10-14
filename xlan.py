import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg
import lib.utils as utils
from models.att_basic_model import AttBasicModel
import blocks
from torch.nn.utils.weight_norm import weight_norm

'''
class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, features_dim, g_features, decoder_dim, attention_dim, dropout=0.5):
        """
        :param features_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.features_att = weight_norm(nn.Linear(features_dim, attention_dim))  # linear layer to transform encoded image
        self.g_att = weight_norm(nn.Linear(g_features, attention_dim))  # linear layer to transform encoded image
        self.decoder_att = weight_norm(nn.Linear(decoder_dim, attention_dim))  # linear layer to transform decoder's output
        self.full_att = weight_norm(nn.Linear(attention_dim, 1))  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, image_features, g_features, decoder_hidden):
        """
        Forward propagation.
        :param image_features: encoded images, a tensor of dimension (batch_size, 36, features_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.features_att(image_features)  # (batch_size, 36, attention_dim)
        att3 = self.features_att(g_features)  # (batch_size, 36, attention_dim)
        #print(att1.shape)
        #print(att3.shape)

        # First, add a new dimension to tensor2
        #att3 = att3.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 7, 1024)

        # Repeat tensor2 along the first and second dimensions to match tensor1
        #att3 = att3.expand(50, 38, -1, -1)  # Shape: (50, 38, 7, 1024)

        # Optionally, reduce the expanded tensor to a shape of (50, 38, 1024)
        # Here, let's take the mean along the 3rd dimension
        #att3 = att3.mean(dim=2)  # Shape: (50, 38, 1024)

        # Dynamically expand tensor2 to match the shape of tensor1
        # Add two new dimensions to tensor2
        att3 = att3.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 7, 1024)

        # Repeat tensor2 along the first and second dimensions to match tensor1
        att3 = att3.expand(att1.shape[0], att1.shape[1], -1, -1)  # Shape: (50, dynamic_dim, 7, 1024)

        # Reduce the expanded tensor along the 3rd dimension using mean
        att3 = att3.mean(dim=2)  # Shape: (50, dynamic_dim, 1024)

        att1 = att1*att3
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.dropout(self.relu(att1 + att2.unsqueeze(1)))).squeeze(2)  # (batch_size, 36)
        alpha = self.softmax(att)  # (batch_size, 36)
        attention_weighted_encoding = (image_features * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, features_dim)

        return attention_weighted_encoding,alpha


class Attention(nn.Module):
    def __init__(self):
        super(Attention,self).__init__()
        # Define a linear layer to embed att_feats to size (1800, 1024)
        self.linear_layer = nn.Linear(4, 1024)

    def Forward(self, **kwargs):
        gtt_feats = kwargs[cfg.PARAM.GTT_FEATS]  # 1800, 4
        embedded_feats = self.linear_layer(gtt_feats)  # Shape: (1800, 1024)
        g_att = nn.ReLU(embedded_feats)
        print(gtt_feats.shape)
        exit(0)
        return g_att
        
  
'''

class XLAN(AttBasicModel):
    def __init__(self):
        super(XLAN, self).__init__()
        self.num_layers = 2
        #self.geo_attention = Attention()
        # First LSTM layer
        rnn_input_size = cfg.MODEL.RNN_SIZE + cfg.MODEL.BILINEAR.DIM
        self.att_lstm = nn.LSTMCell(rnn_input_size, cfg.MODEL.RNN_SIZE)
        self.ctx_drop = nn.Dropout(cfg.MODEL.DROPOUT_LM)

        self.box_embed = nn.Sequential(nn.Linear(4, 1024),
                                       nn.ReLU(),
                                       nn.Dropout(0.3))

        #self.attention1 = Attention(1024, 1024,1024, 1024)  # attention network
        self.attention = blocks.create(
            cfg.MODEL.BILINEAR.DECODE_BLOCK,
            embed_dim=cfg.MODEL.BILINEAR.DIM,
            att_type=cfg.MODEL.BILINEAR.ATTTYPE,
            att_heads=cfg.MODEL.BILINEAR.HEAD,
            att_mid_dim=cfg.MODEL.BILINEAR.DECODE_ATT_MID_DIM,
            att_mid_drop=cfg.MODEL.BILINEAR.DECODE_ATT_MID_DROPOUT,
            dropout=cfg.MODEL.BILINEAR.DECODE_DROPOUT,
            layer_num=cfg.MODEL.BILINEAR.DECODE_LAYERS
        )
        self.att2ctx = nn.Sequential(
            nn.Linear(cfg.MODEL.BILINEAR.DIM + cfg.MODEL.RNN_SIZE, 2 * cfg.MODEL.RNN_SIZE),
            nn.GLU()
        )

    # state[0] -- h, state[1] -- c
    def Forward(self, **kwargs):
        wt = kwargs[cfg.PARAM.WT]
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]   #50,48,1024
        gtt_feats = kwargs[cfg.PARAM.GTT_FEATS]    #1800, 4
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        state = kwargs[cfg.PARAM.STATE]
        gv_feat = kwargs[cfg.PARAM.GLOBAL_FEAT]
        p_att_feats = kwargs[cfg.PARAM.P_ATT_FEATS]
        #print(att_feats.shape)
        #exit(0)
        #print(gtt_feats.shape)
        #exit(0)
        att_feats = torch.cat((gtt_feats, att_feats), dim=1)
        #att_feats = gtt_feats * att_feats

        if gv_feat.shape[-1] == 1:  # empty gv_feat   [50,1024]
            if att_mask is not None:
                gv_feat = torch.sum(att_feats * att_mask.unsqueeze(-1), 1) / torch.sum(att_mask.unsqueeze(-1), 1)
            else:
                gv_feat = torch.mean(att_feats, 1)
        # xt = self.word_embed(wt)
        xt = self.word_embed(wt.long())

        h_att, c_att = self.att_lstm(torch.cat([xt, gv_feat + self.ctx_drop(state[0][1])], 1),
                                     (state[0][0], state[1][0]))
        att, _ = self.attention(h_att, att_feats, gtt_feats, att_mask, p_att_feats, precompute=True)

        #attention_weighted_encoding = self.attention1(att_feats, gtt_feats, h_att)[0]

        ctx_input = torch.cat([att, h_att], 1)

        output = self.att2ctx(ctx_input)
        state = [torch.stack((h_att, output)), torch.stack((c_att, state[1][1]))]

        return output, state