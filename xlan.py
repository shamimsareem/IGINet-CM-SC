import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg
import lib.utils as utils
from models.att_basic_model import AttBasicModel
import blocks
from torch.nn.utils.weight_norm import weight_norm


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
