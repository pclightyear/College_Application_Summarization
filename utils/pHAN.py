import torch
from torch import nn
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from itertools import chain

# Utility variable
import sys
sys.path.insert(0, '../..')

import utils.torch as Tor
import utils.preprocess as PP

"""
AttentionNetwork
"""
class AttentionNetwork(nn.Module):
    """
        Adapted from: 
            https://www.kaggle.com/code/mlwhiz/attention-pytorch-and-keras/notebook
            https://www.kaggle.com/code/hsankesara/news-classification-using-han/notebook
            https://github.com/uvipen/Hierarchical-attention-networks-pytorch/tree/master/src
    """
    def __init__(
        self, feature_dim, context_dim, temperature=1, leaky_relu_negative_slope=0.1, 
        bias=True, epsilon=0.0, **kwargs
    ):
        super(AttentionNetwork, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.context_dim = context_dim
        self.temperature = temperature
        self.epsilon = epsilon
        
        ## transformation layer for input
        self.Wx = nn.Linear(feature_dim, context_dim, self.bias)
        nn.init.xavier_uniform_(self.Wx.weight)
        ## transformation layer for comment
        self.Wc = nn.Linear(feature_dim, context_dim, self.bias)
        nn.init.xavier_uniform_(self.Wc.weight)
        ## transformation layer for dynamic context vector
        self.Wd = nn.Linear(feature_dim, context_dim, self.bias)
        nn.init.xavier_uniform_(self.Wd.weight)
        ## context vector
        context_vector = torch.zeros(1, context_dim)
        nn.init.xavier_uniform_(context_vector)
        self.u = nn.Parameter(context_vector)
        
#         self.x_layer_norm = nn.LayerNorm(self.feature_dim)
#         self.c_layer_norm = nn.LayerNorm(self.feature_dim)
        
        ## Leaky ReLU
        self.leaky_relu_negative_slope = leaky_relu_negative_slope
        self.leaky_relu = nn.LeakyReLU(leaky_relu_negative_slope)
        
    def forward(self, x, c, eta=0, _u=None, mask=None):
        uit = self.Wx(x)
        uit = self.leaky_relu(uit) ## using relu instead of tanh, since we are not using rnn
        
        _c = self.Wc(c)
        _c = self.leaky_relu(_c)
        
        if _u is not None:
            ## sentence attention
            u = self.Wd(_u)
            u = self.leaky_relu(u)
            u = u.repeat(x.shape[0], 1)
        else:
            ## perspective attention
            u = eta * _c + (1 - eta) * self.u
        
        ## dot product with context vector
        ## note that different instance correspond to different context vector
        ait = torch.div(
            torch.sum(uit * torch.unsqueeze(u, 1), dim=-1),
            self.temperature ## softmax with temperature
        )

        ## subtract with the maximum value to prevent overflow
        max_ait, _ = torch.max(ait, dim=1, keepdim=True)
        ait = ait - max_ait.detach()
        ait = torch.exp(ait)
        
        if mask is not None:
            ait = ait * mask
            
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        a = ait / (torch.sum(ait, dim=1, keepdim=True) + self.epsilon)
        
        weighted_input = x * torch.unsqueeze(a, dim=-1)
        out = torch.sum(weighted_input, dim=1)
        
        return out, a

    
"""
PerspectiveHierarchicalAttentionNetwork
"""
class PerspectiveHierarchicalAttentionNetwork(nn.Module):
    """
        Adapted from: 
            https://www.kaggle.com/code/mlwhiz/attention-pytorch-and-keras/notebook
            https://www.kaggle.com/code/hsankesara/news-classification-using-han/notebook
            https://github.com/uvipen/Hierarchical-attention-networks-pytorch/tree/master/src
    """
    def __init__(
        self, 
        bert_model, bert_tokenizer, perspective_mean_embed, 
        num_perspective, num_sent_per_perspective, bert_dim, sent_dim, cxt_dim, prj_dim,
        sent_temperature=1, pers_temperature=1.5, dropout_rate=0.1, leaky_relu_negative_slope=0.1, 
        encode_batch_size=200, compression=True, projection=True, attention_empty_mask=False,
        freeze_bert=True, **kwargs
    ):
        super(PerspectiveHierarchicalAttentionNetwork, self).__init__(**kwargs)
        
        ## hyperparameters
        self.num_perspective = num_perspective
        self.num_sent_per_perspective = num_sent_per_perspective
        self.sent_dim = sent_dim
        self.cxt_dim = cxt_dim
        self.prj_dim = prj_dim
        self.sent_temperature = sent_temperature
        self.pers_temperature = pers_temperature
        self.dropout_rate = dropout_rate
        self.encode_batch_size = encode_batch_size
        self.compression = compression
        self.projection = projection
        self.attention_empty_mask = attention_empty_mask
        if self.attention_empty_mask:
            self.epsilon = 1e-10
        else:
            self.epsilon = 0.0
        
        ## bert embedding model
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
        self.bert_dim = bert_dim
        self.freeze_bert = freeze_bert
        ## freeze sbert model
        if freeze_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False
        if compression:
            self.compression_layer = nn.Linear(bert_dim, sent_dim)
            nn.init.xavier_uniform_(self.compression_layer.weight)
            
        ## attention network
        ## initialize context vector with cluster mean embedding
        self.perspective_mean_embed = perspective_mean_embed
        ## context vector for outliers
        outlier_ctx_vec = torch.zeros(1, bert_dim)
        nn.init.xavier_uniform_(outlier_ctx_vec)
        self.sent_att_ui = nn.Parameter(
            torch.cat([perspective_mean_embed, outlier_ctx_vec])
        )
        
        self.sentence_att_net = AttentionNetwork(sent_dim, cxt_dim, sent_temperature, leaky_relu_negative_slope, epsilon=self.epsilon)
        self.perspective_att_net = AttentionNetwork(sent_dim, cxt_dim, pers_temperature, leaky_relu_negative_slope, epsilon=self.epsilon)

        ## layer normalization (for preventing gradient vanishing)
        self.sent_layer_norm = nn.LayerNorm(self.sent_dim)
        self.pers_layer_norm = nn.LayerNorm(self.sent_dim)
        self.sum_layer_norm = nn.LayerNorm(self.sent_dim)
        
        ## dropout
        self.dropout = nn.Dropout(p=dropout_rate)
        
        ## single layer classifier
        self.grade_classifier = nn.Sequential(
            nn.Linear(sent_dim, 4)
        )
        for layer in self.grade_classifier:
            nn.init.xavier_uniform_(layer.weight)
        
        ## projection head
        if projection:
            self.project_head = nn.Linear(sent_dim, prj_dim)
            nn.init.xavier_uniform_(self.project_head.weight)
            self.leaky_relu_negative_slope = leaky_relu_negative_slope
            self.leaky_relu = nn.LeakyReLU(leaky_relu_negative_slope)
        
    def max_pooling(self, embed, att_mask):
        return torch.max((embed * att_mask.unsqueeze(-1)), axis=1).values
    
    def mean_pooling(self, embed, att_mask):
        return embed.sum(axis=1) / att_mask.sum(axis=-1).unsqueeze(-1)
    
    def encode(self, t):
        device = next(self.bert_model.parameters()).device
        
        if not self.bert_tokenizer:
            return self.bert_model.encode(t, convert_to_tensor=True, show_progress_bar=False)
        
        dataset = Tor.BatchSentenceDataset(t)
        dataloader = DataLoader(dataset, batch_size=self.encode_batch_size, shuffle=False)
        
        embed_batch_list = []
        
        for batch in dataloader:
            encoding = self.bert_tokenizer(batch, padding=True, return_tensors='pt', max_length=500, truncation='longest_first')
            for key in encoding:
                if isinstance(encoding[key], Tensor):
                    encoding[key] = encoding[key].to(device)

            if self.freeze_bert:
                with torch.no_grad():
                    embed_dict = self.bert_model(**encoding)
            else:
                embed_dict = self.bert_model(**encoding)

            att_mask = encoding['attention_mask']
            embed = embed_dict['last_hidden_state']
            embed = self.mean_pooling(embed, att_mask)
            embed_batch_list.append(embed)
            
        embed = torch.cat(embed_batch_list)
        
        return embed
    
    def forward(self, s, c, eta=0.95):
        device = next(self.parameters()).device
        
        ## encode the pseudo summary by sentence bert
        ss = list(chain.from_iterable(s))
        sss = list(chain.from_iterable(ss))
        s_embed = self.encode(sss)
        if self.compression:
            s_embed = self.compression_layer(s_embed)
        s_embed = s_embed.reshape(
            -1, self.num_perspective, self.num_sent_per_perspective, self.sent_dim
        )
        ## create empty sentence mask
        if self.attention_empty_mask:
            empty_sent_mask = torch.tensor([0 if PP.is_empty_sent(_s) else 1 for _s in sss]).to(device)
            empty_sent_mask = empty_sent_mask.reshape(-1, self.num_perspective, self.num_sent_per_perspective)
        else:
            empty_sent_mask = torch.ones(s_embed.shape[:-1]).to(device) ## test with no masking
        
        ## encode the comment by sentence bert
        c_embed = self.encode(c)
        if self.compression:
            c_embed = self.compression_layer(c_embed)
        
        sent_att_ui = self.sent_att_ui
        if self.compression:
            sent_att_ui = self.compression_layer(sent_att_ui)
        
#         print("s_shape", s_embed.shape)
#         print("s_embed", s_embed)
        
        ## get perspective embedding with sentence attention
        s_embed = s_embed.permute(1, 0, 2, 3) ## iterate through perspective
        empty_sent_mask = empty_sent_mask.permute(1, 0, 2)  ## iterate through perspective
        pers_embed_list = []
        sent_att_list = []
        
        s_embed = self.dropout(s_embed) ## dropout
        c_embed = self.dropout(c_embed) ## dropout
        
        for ith_s_embed, mask, sent_att_u in zip(s_embed, empty_sent_mask, sent_att_ui):
            ith_s_embed = self.sent_layer_norm(ith_s_embed) ## apply layer normalization
            out, sent_att = self.sentence_att_net(ith_s_embed, c_embed, _u=sent_att_u, mask=mask)
            pers_embed_list.append(out)
            sent_att_list.append(sent_att)
        
        sent_att = torch.stack(sent_att_list, 1)
        
#         print("sent_att", sent_att)
        
        pers_embed = torch.stack(pers_embed_list, 1)
#         print("pers_embed", pers_embed)
        pers_embed = self.pers_layer_norm(pers_embed) ## apply layer normalization
        pers_embed = self.dropout(pers_embed) ## dropout
        
        ## create empty perspective mask
        ## [DEBUG] empty perspective not necessary have 0 sum tensor
        if self.attention_empty_mask:
            empty_pers_mask = torch.tensor([0 if PP.is_empty_sent(''.join(_s)) else 1 for _s in ss]).to(device)
            empty_pers_mask = empty_pers_mask.reshape(-1, self.num_perspective)
        else:
            empty_pers_mask = torch.ones(pers_embed.shape[:-1]).to(device) ## test with no masking
        
#         print(pers_embed.shape)
#         print(pers_embed.shape)
        
        ## get pseudo summary embedding with perspective attention
        sum_embed, pers_att = self.perspective_att_net(pers_embed, c_embed, eta, mask=empty_pers_mask)
        sum_embed = self.sum_layer_norm(sum_embed) ## apply layer normalization
#         sum_embed = self.dropout(sum_embed) ## dropout
        
#         print("pers_att", pers_att)
    
#         print("sum_embed", sum_embed)
    
        ## project embed to calculate contrastive loss
        if self.projection:
            projected_embed = self.project_head(sum_embed)
            projected_embed = self.leaky_relu(projected_embed)
        else:
            projected_embed = sum_embed
        
        ## predict grades
        logits = self.grade_classifier(sum_embed)
        logits = nn.functional.softmax(logits, dim=-1)
        
        return projected_embed, logits, sent_att, pers_att