import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class EmbeddingModel_Optimized(nn.Module):
    def __init__(self, ent_size, rel_size, embed_size, device, reverse_dictionary, ent_dic):
        super().__init__()
        self.ent_size = ent_size
        self.rel_size = rel_size
        self.embed_size = embed_size
        self.device = device
        self.bilinear_weights = False

        initrange = 0.5 / embed_size
        self.out_embed_ent = nn.Embedding(ent_size, embed_size).to(device)
        self.out_embed_ent.weight.data.uniform_(-initrange, initrange)

        self.out_embed_rel = nn.Embedding(rel_size, embed_size).to(device)
        self.out_embed_rel.weight.data.uniform_(-initrange, initrange)

        self.in_embed_ent = nn.Embedding(ent_size, embed_size).to(device)
        self.in_embed_ent.weight.data.uniform_(-initrange, initrange)

        self.in_embed_rel = nn.Embedding(rel_size, embed_size).to(device)
        self.in_embed_rel.weight.data.uniform_(-initrange, initrange)

        self.in_embed_map = nn.Embedding(rel_size, embed_size).to(device)
        self.in_embed_map.weight.data.uniform_(-initrange, initrange)

        self.out_embed_map = nn.Embedding(rel_size, embed_size).to(device)
        self.out_embed_map.weight.data.uniform_(-initrange, initrange)

        # ---- 预计算反向id映射（GPU张量） ----
        reverse_id_list = []
        for wid, id_ in reverse_dictionary.items():  # wid 是 id，id_ 是对应的字符串
            if id_ == '<unk>':
                pass
            elif int(id_) >= ent_size:
                id_ = int(id_) - ent_size
                reverse_id_list.append(int(id_))
            else:
                reverse_id_list.append(int(id_))
        self.reverse_id_map = torch.tensor(reverse_id_list, dtype=torch.long, device=device)

        # ---- ent_dic 转成布尔mask，GPU上判断更快 ----
        ent_dic_mask = torch.zeros(len(reverse_dictionary), dtype=torch.bool)
        ent_dic_mask[list(ent_dic)] = True
        self.ent_dic_mask = ent_dic_mask.to(device)

    def forward(self, input_labels, pos_labels, neg_labels):
        # 所有输入直接在GPU上
        input_labels = input_labels.to(self.device)
        pos_labels = pos_labels.view(-1).to(self.device)
        neg_labels = neg_labels.to(self.device)
        # print('input_labels--', input_labels.size())
        # print('neg_labels--', neg_labels.size())
        # print('neg_labels--', neg_labels)

        # ---- 反向id映射（批量） ----
        input_ids  = self.reverse_id_map[input_labels]
        pos_ids    = self.reverse_id_map[pos_labels]
        neg_ids    = self.reverse_id_map[neg_labels]

        # ---- in_ent_dic 掩码 ----
        in_ent_input = self.ent_dic_mask[input_labels]
        in_ent_pos   = self.ent_dic_mask[pos_labels]
        in_ent_neg   = self.ent_dic_mask[neg_labels]

        # ---- 分组索引（所有分组用布尔mask批量取） ----
        # ent_dic_input_1: 输入in_ent_dic & pos in_ent_dic
        mask_ent_dic_1 = in_ent_input & in_ent_pos
        # ent_dic_input_2: 输入in_ent_dic & pos not_in_ent_dic
        mask_ent_dic_2 = in_ent_input & (~in_ent_pos)
        # not_in_ent_dic_input_1: 输入not_in_ent_dic & pos in_ent_dic
        mask_not_ent_dic_1 = (~in_ent_input) & in_ent_pos
        # not_in_ent_dic_input_2: 输入not_in_ent_dic & pos not_in_ent_dic
        mask_not_ent_dic_2 = (~in_ent_input) & (~in_ent_pos)

        # ---- 正样本embedding ----
        input_emb = []
        output_emb = []

        # pos_labels in ent_dic
        input_emb.append(self.in_embed_ent(input_ids[mask_ent_dic_1]))
        output_emb.append(self.out_embed_ent(pos_ids[mask_ent_dic_1]))

        # pos_labels not in ent_dic
        map_vec_1 = self.in_embed_map(pos_ids[mask_ent_dic_2])
        ent_map_1 = self.in_embed_ent(input_ids[mask_ent_dic_2])
        input_emb.append(self.map(ent_map_1, map_vec_1))
        output_emb.append(self.out_embed_rel(pos_ids[mask_ent_dic_2]))

        # not_in_ent_dic_input_1
        ent_map_3 = self.out_embed_ent(pos_ids[mask_not_ent_dic_1])
        map_vec_3 = self.out_embed_map(input_ids[mask_not_ent_dic_1])
        input_emb.append(self.in_embed_rel(input_ids[mask_not_ent_dic_1]))
        output_emb.append(self.map(ent_map_3, map_vec_3))

        # not_in_ent_dic_input_2
        input_emb.append(self.in_embed_rel(input_ids[mask_not_ent_dic_2]))
        output_emb.append(self.out_embed_rel(pos_ids[mask_not_ent_dic_2]))

        # ---- 负样本embedding ----

        # ---- 分组索引（所有分组用布尔mask批量取） ----
        # ent_dic_input_1: 输入in_ent_dic & pos in_ent_dic
        mask_ent_neg_dic_1 = in_ent_input.unsqueeze(1) & in_ent_neg
        # ent_dic_input_2: 输入in_ent_dic & pos not_in_ent_dic
        mask_ent_neg_dic_2 = in_ent_input.unsqueeze(1) & (~in_ent_neg)
        # not_in_ent_dic_input_1: 输入not_in_ent_dic & pos in_ent_dic
        mask_not_ent_neg_dic_1 = (~in_ent_input.unsqueeze(1)) & in_ent_neg
        # not_in_ent_dic_input_2: 输入not_in_ent_dic & pos not_in_ent_dic
        mask_not_ent_neg_dic_2 = (~in_ent_input.unsqueeze(1)) & (~in_ent_neg)

        # ---- 负样本embedding ----
        input_emb_neg = []
        negtive_emb = []
        # neg in ent_dic
        input_emb_neg.append(self.in_embed_ent(input_ids.unsqueeze(1)[mask_ent_neg_dic_1]))
        negtive_emb.append(self.out_embed_ent(neg_ids[mask_ent_neg_dic_1]))

        # neg not in ent_dic
        map_vec_2 = self.in_embed_map(neg_ids[mask_ent_neg_dic_2])
        ent_map_2 = self.in_embed_ent(input_ids.unsqueeze(1)[mask_ent_neg_dic_2])
        input_emb_neg.append(self.map(ent_map_2, map_vec_2))
        negtive_emb.append(self.out_embed_rel(neg_ids[mask_ent_neg_dic_2]))

        # not_in_ent_dic_input_1
        ent_map_4 = self.out_embed_ent(neg_ids[mask_not_ent_neg_dic_1])
        map_vec_4 = self.out_embed_map(input_ids.unsqueeze(1)[mask_not_ent_neg_dic_1])
        input_emb_neg.append(self.in_embed_rel(input_ids.unsqueeze(1)[mask_not_ent_neg_dic_1]))
        negtive_emb.append(self.map(ent_map_4, map_vec_4))

        # not_in_ent_dic_input_2
        input_emb_neg.append(self.in_embed_rel(input_ids.unsqueeze(1)[mask_not_ent_neg_dic_2]))
        negtive_emb.append(self.out_embed_rel(neg_ids[mask_not_ent_neg_dic_2]))

        # ---- 拼接 & bmm计算 ----
        input_embedding = torch.cat(input_emb, dim=0).unsqueeze(2)
        pos_embedding   = torch.cat(output_emb, dim=0).unsqueeze(1)
        input_neg_embedding = torch.cat(input_emb_neg, dim=0).unsqueeze(2)
        neg_embedding   = torch.cat(negtive_emb, dim=0).unsqueeze(1)
        # print('input_neg_embedding--', input_neg_embedding.size())
        # print('input_neg_embedding--', neg_embedding.size())
        # input_neg_embedding - - torch.Size([20, 200, 1])
        # input_neg_embedding - - torch.Size([20, 200, 1])

        log_pos = torch.bmm(pos_embedding, input_embedding).squeeze()
        log_neg = torch.bmm(neg_embedding, -input_neg_embedding).squeeze()
        # print('log_pos--', log_pos.size())  #torch.Size([32])
        # print('log_neg--', log_neg.size())   # torch.Size([20])

        log_neg = log_neg.reshape(log_pos.size(0), -1)
        log_pos = F.logsigmoid(log_pos)
        log_neg = F.logsigmoid(log_neg).sum(1)

        loss = -(log_pos + log_neg)
        return loss
    def getoutput(self, input_labels, neg_labels, ent_dic, reverse_dictionary):
        # === 预处理 ===
        device = self.device
      
        # 保证 long 类型
        input_labels = input_labels.to(device).long()
        neg_labels = neg_labels.to(device).long()

        all_labels = torch.cat([input_labels, neg_labels])  # shape: (2 * batch_size, )

        get_orig_id = self.reverse_id_map[all_labels].view(2, -1)
        input_orig = get_orig_id[0]
        neg_orig = get_orig_id[1]

        # 批量分类掩码
        input_is_ent = self.ent_dic_mask[input_labels]
        neg_is_ent = self.ent_dic_mask[neg_labels]

        # === 分四类掩码 ===
        mask_ent_ent = input_is_ent & neg_is_ent
        mask_ent_nonent = input_is_ent & (~neg_is_ent)
        mask_nonent_ent = (~input_is_ent) & neg_is_ent
        mask_nonent_nonent = (~input_is_ent) & (~neg_is_ent)

        input_emb_neg = []
        negtive_emb = []

        # 1. labels in ent_dic & neg in ent_dic
        if mask_ent_ent.any():
            input_emb_neg.append(self.in_embed_ent(input_orig[mask_ent_ent]))
            negtive_emb.append(self.out_embed_ent(neg_orig[mask_ent_ent]))

        # 2. labels in ent_dic & neg not in ent_dic
        if mask_ent_nonent.any():
            ent_map_2 = self.in_embed_ent(input_orig[mask_ent_nonent])
            map_vec_2 = self.in_embed_map(neg_orig[mask_ent_nonent])
            input_emb_neg.append(self.map(ent_map_2, map_vec_2))
            negtive_emb.append(self.out_embed_rel(neg_orig[mask_ent_nonent]))

        # 3. labels not in ent_dic & neg in ent_dic
        if mask_nonent_ent.any():
            ent_map_4 = self.out_embed_ent(neg_orig[mask_nonent_ent])
            map_vec_4 = self.out_embed_map(input_orig[mask_nonent_ent])
            negtive_emb.append(self.map(ent_map_4, map_vec_4))
            input_emb_neg.append(self.in_embed_rel(input_orig[mask_nonent_ent]))

        # 4. labels not in ent_dic & neg not in ent_dic
        if mask_nonent_nonent.any():
            negtive_emb.append(self.out_embed_rel(neg_orig[mask_nonent_nonent]))
            input_emb_neg.append(self.in_embed_rel(input_orig[mask_nonent_nonent]))

        self.outputs1 = torch.cat(input_emb_neg, dim=0)
        self.outputs2 = torch.cat(negtive_emb, dim=0)
        return self.outputs1, self.outputs2

    def map(self, ent, mapping):
        norm = F.normalize(mapping, p=2, dim=-1)
        ent_map = ent - torch.sum(ent * norm, dim=1, keepdim=True) * norm
        return ent_map
    def affinity(self, inputs1, inputs2):
        self.input_dim1 = len(inputs1)
        self.input_dim2 = len(inputs2)
        matrix = Variable(torch.randn(self.input_dim1, self.input_dim2))  ##即原来的self.vars['weights']
        if self.bilinear_weights:
            prod = torch.matmul(inputs2, matrix.transpose(0, 1))
            self.prod = prod
            result = torch.sum(inputs1 * prod, dim=1)
        else:
            result = torch.sum(inputs1 * inputs2, dim=1)
            # print('result',result.data.cpu().numpy().shape)
        return result

    def get_probs(self, outputs1, outputs2):
        probs = torch.sigmoid(torch.pow(self.affinity(outputs1, outputs2), 0.25))
        return probs

    def input_embeddings(self):
        return self.in_embed_ent.weight.data.cpu().numpy(), self.in_embed_rel.weight.data.cpu().numpy()

    def scale_loss(self, embedding):
        return torch.sum(
            torch.max(
                torch.sum(
                    embedding ** 2, dim=1, keepdim=True
                ) - torch.autograd.Variable(torch.FloatTensor([1.0]).cuda(self.device)),
                torch.autograd.Variable(torch.FloatTensor([0.0]).cuda(self.device))
            ))

    def orthogonal_loss(self, relation_embedding, w_embedding):
        dot = torch.sum(relation_embedding * w_embedding, dim=1, keepdim=False) ** 2
        norm = torch.norm(relation_embedding, p=1, dim=1) ** 2
        loss = torch.sum(
            torch.relu(dot / norm - torch.autograd.Variable(torch.FloatTensor([1e-5]).cuda(self.device) ** 2)))
        return loss








