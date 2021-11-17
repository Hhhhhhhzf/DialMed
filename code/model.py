"""
@Time: 2021/11/15 15:49
@desc: 
"""
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class PTMDialogueEncoder(nn.Module):
    """
    Dialogue Encoder
    """
    def __init__(self, **kwargs):
        super(PTMDialogueEncoder, self).__init__()
        self.ptm = kwargs['model']
        self.gat = GAT(nfeat=kwargs['hidden_size'],
                       nhid=kwargs['hidden_size'],
                       out_feature=kwargs['hidden_size'],
                       dropout=0.3,
                       alpha=0.3,
                       nheads=1)
        self.position_encoder = PositionEncoder(kwargs['hidden_size'])
        self.attention = Attention(kwargs['hidden_size'])
        self.linear = nn.Linear(kwargs['hidden_size'], kwargs['output_size'])

    def forward(self, input_ids, token_type_ids, attention_mask, adj=None, cls_ids=None):
        utter_embedding = self.ptm(input_ids, attention_mask, token_type_ids)[0]
        new_utter_embed = []
        for batch_id, cls_id in enumerate(cls_ids):
            utter_embed = torch.index_select(utter_embedding[batch_id], 0, cls_id.to(utter_embedding.device))
            utter_embed = self.position_encoder(utter_embed)
            utter_embed = self.gat(utter_embed, adj[batch_id].to(input_ids.device))
            new_utter_embed.append(self.attention(utter_embed))
        dialogue_embedding = torch.stack(new_utter_embed)
        dialogue_embedding = self.linear(dialogue_embedding)
        return dialogue_embedding


class DDN(nn.Module):
    def __init__(self, **kwargs):
        super(DDN, self).__init__()
        pass

    def forward(self, input_ids, token_type_ids, attention_mask, adj, cls_ids, node_ids, kg_adjs, disease_ids):
        pass


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, out_feature, dropout, alpha, nheads):
        """
        reference from https://github.com/Diego999/pyGAT
        Dense version of GAT.
        """
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, out_feature, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        inputs = x
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x + inputs


class GraphAttentionLayer(nn.Module):
    """
    reference from https://github.com/Diego999/pyGAT
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Linear(in_features=in_features, out_features=out_features)
        self.a = nn.Linear(2 * out_features, 1)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = self.W(h)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(self.a(a_input).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        if self.dropout > 0:
            attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
