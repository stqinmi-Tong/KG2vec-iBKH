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
        # input_emb_neg = []
        # negtive_emb = []
        #
        # # neg in ent_dic
        # mask_neg_ent = in_ent_input.unsqueeze(1) & in_ent_neg
        # mask_neg_rel = in_ent_input.unsqueeze(1) & (~in_ent_neg)
        #
        # # neg in ent_dic
        # input_emb_neg.append(self.in_embed_ent(input_ids.unsqueeze(1)[mask_neg_ent]))
        # negtive_emb.append(self.out_embed_ent(neg_ids[mask_neg_ent]))
        #
        # # neg not in ent_dic
        # ent_map_2 = self.in_embed_ent(input_ids.unsqueeze(1)[mask_neg_rel])
        # map_vec_2 = self.in_embed_map(neg_ids[mask_neg_rel])
        # input_emb_neg.append(self.map(ent_map_2, map_vec_2))
        # negtive_emb.append(self.out_embed_rel(neg_ids[mask_neg_rel]))

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
        #
        # # 保证 long 类型
        # input_labels = input_labels.to(device).long()
        # neg_labels = neg_labels.to(device).long()
        #
        # # ---- 批量获取原始 ID（利用预计算的 reverse_id_map） ----
        # all_labels = torch.cat([input_labels, neg_labels])  # shape: (2 * batch_size, )
        # get_orig_id = self.reverse_id_map[all_labels].view(2, -1)
        # input_orig = get_orig_id[0]
        # neg_orig = get_orig_id[1]



        # ent_mask = torch.zeros(
        #     max(max(reverse_dictionary.keys()), input_labels.max().item(),
        #         neg_labels.max().item()) + 1,
        #     dtype=torch.bool, device=device)


        # print('ent_mask',min(ent_dic), max(ent_dic), ent_mask.size())
        # ent_mask[list(ent_dic)] = True

        # 保证 long 类型
        input_labels = input_labels.to(device).long()
        neg_labels = neg_labels.to(device).long()

        # 批量获取原始 ID
        # get_orig_id = torch.tensor(
        #     [self.get_original_id(i.item(), reverse_dictionary, self.ent_size) for i in
        #      torch.cat([input_labels, neg_labels])],
        #     dtype=torch.long, device=device
        # ).view(2, -1)
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


    # def get_original_id(self, word_id, reverse_dict, ent_size):  ##ent_size=40943+1
    #     id_ = reverse_dict[int(word_id)]
    #     if id_ == '<unk>':
    #         pass
    #     elif int(id_) >= ent_size:
    #         id_ = int(id_) - ent_size
    #
    #     return id_






class EmbeddingModel(nn.Module):
    def __init__(self, ent_size, rel_size, embed_size, device):
        """
        初始化实体和关系的嵌入模型
        优化点：
        - 统一初始化范围的计算
        - 简化嵌入层初始化代码
        """
        super(EmbeddingModel, self).__init__()
        self.ent_size = ent_size
        self.rel_size = rel_size
        self.embed_size = embed_size
        self.device = device
        self.bilinear_weights = False

        # 统一初始化范围
        initrange = 0.5 / self.embed_size

        # 初始化所有嵌入层
        self._init_embeddings(initrange)

    def _init_embeddings(self, initrange):
        """统一初始化所有嵌入层"""
        # 输入和输出嵌入
        self.out_embed_ent = self._create_embedding(self.ent_size, initrange)
        self.out_embed_rel = self._create_embedding(self.rel_size, initrange)
        self.in_embed_ent = self._create_embedding(self.ent_size, initrange)
        self.in_embed_rel = self._create_embedding(self.rel_size, initrange)

        # 映射嵌入
        self.in_embed_map = self._create_embedding(self.rel_size, initrange)
        self.out_embed_map = self._create_embedding(self.rel_size, initrange)

    def _create_embedding(self, num_embeddings, initrange):
        """创建并初始化嵌入层的辅助方法"""
        embedding = nn.Embedding(num_embeddings, self.embed_size, sparse=False)
        embedding.weight.data.uniform_(-initrange, initrange)
        return embedding

    def forward(self, input_labels, pos_labels, neg_labels, ent_dic, reverse_dictionary):
        """
        input_labels: 中心词, [batch_size]
        pos_labels: 中心词周围 context window 出现过的单词 [batch_size * (window_size * 2)]
        neg_labelss: 中心词周围没有出现过的单词，从 negative sampling 得到 [batch_size, (window_size * 2 * K)]
        return: loss, [batch_size]
        前向传播计算损失
        优化点：
        - 减少数据转换
        - 向量化处理
        - 预分配内存
        """
        # 保持数据在GPU上，避免不必要的转换
        input_labels = input_labels.view(-1)
        pos_labels = pos_labels.view(-1)
        neg_labels = neg_labels.view(-1)

        # 预分配列表空间
        input_emb, output_emb = [], []
        input_emb_neg, negtive_emb = [], []

        # 批量处理正样本
        self._process_batch(input_labels, pos_labels, ent_dic, reverse_dictionary,
                            input_emb, output_emb)

        # 批量处理负样本
        neg_labels = neg_labels.view(-1, neg_labels.size(0) // input_labels.size(0))
        self._process_batch(input_labels.repeat_interleave(neg_labels.size(1)),
                            neg_labels.view(-1), ent_dic, reverse_dictionary,
                            input_emb_neg, negtive_emb)

        # 合并所有嵌入并计算损失
        return self._compute_loss(input_emb, output_emb, input_emb_neg, negtive_emb)

    def _process_batch(self, inputs, targets, ent_dic, reverse_dictionary,
                       input_emb_list, output_emb_list):
        """
        批量处理输入-目标对的辅助方法
        优化点：
        - 向量化处理
        - 减少条件判断
        """
        # 获取原始ID
        input_ids = self._get_original_ids(inputs, reverse_dictionary)
        target_ids = self._get_original_ids(targets, reverse_dictionary)

        # 创建掩码
        input_in_ent = torch.tensor([x.item() in ent_dic for x in inputs], device=self.device)
        target_in_ent = torch.tensor([x.item() in ent_dic for x in targets], device=self.device)

        # 处理四种情况
        self._process_case(input_ids, target_ids, input_in_ent, target_in_ent,
                           input_emb_list, output_emb_list, True, True)
        self._process_case(input_ids, target_ids, input_in_ent, target_in_ent,
                           input_emb_list, output_emb_list, True, False)
        self._process_case(input_ids, target_ids, input_in_ent, target_in_ent,
                           input_emb_list, output_emb_list, False, True)
        self._process_case(input_ids, target_ids, input_in_ent, target_in_ent,
                           input_emb_list, output_emb_list, False, False)

    def _process_case(self, input_ids, target_ids, input_mask, target_mask,
                      input_emb_list, output_emb_list, input_is_ent, target_is_ent):
        """
        处理特定情况的辅助方法
        优化点：
        - 批量处理
        - 减少重复代码
        """
        # 创建当前情况的掩码
        print(input_mask.shape,target_mask.shape)
        case_mask = (input_mask == input_is_ent) & (target_mask == target_is_ent)
        print("case_mask",case_mask.shape)
        # if not case_mask.any():
        #     return

        # 获取当前情况的ID
        print("input_ids",input_ids,input_ids.shape)
        print("case_mask", case_mask,case_mask.shape)
        case_input_ids = input_ids[case_mask]
        print("case_input_ids",case_input_ids)
        case_target_ids = target_ids[case_mask]
        print("ok")

        # 选择正确的嵌入层
        if input_is_ent:
            input_embed = self.in_embed_ent(case_input_ids)
            output_embed = self.out_embed_ent(case_target_ids)
            if not target_is_ent:  # 目标不是实体
                map_vec = self.in_embed_map(case_target_ids)
                input_embed = self.map(input_embed, map_vec)
                output_embed = self.out_embed_rel(case_target_ids)
        else:
            input_embed = self.in_embed_rel(case_input_ids)
            output_embed = self.out_embed_rel(case_target_ids)
            if target_is_ent:  # 目标是实体
                map_vec = self.out_embed_map(case_input_ids)
                target_embed = self.out_embed_ent(case_target_ids)
                output_embed = self.map(target_embed, map_vec)

        # 添加到列表
        input_emb_list.append(input_embed)
        output_emb_list.append(output_embed)

    def _compute_loss(self, input_emb, output_emb, input_emb_neg, negtive_emb):
        """
        计算损失的辅助方法
        优化点：
        - 简化矩阵运算
        """
        # 合并所有嵌入
        input_embedding = torch.cat(input_emb, dim=0).unsqueeze(2)
        pos_embedding = torch.cat(output_emb, dim=0).unsqueeze(1)
        input_neg_embedding = torch.cat(input_emb_neg, dim=0).unsqueeze(2)
        neg_embedding = torch.cat(negtive_emb, dim=0).unsqueeze(1)

        # 计算正负样本得分
        log_pos = torch.bmm(pos_embedding, input_embedding).squeeze()
        log_neg = torch.bmm(neg_embedding, -input_neg_embedding).squeeze()

        # 计算损失
        log_neg = log_neg.reshape(log_pos.size(0), -1)
        log_pos = F.logsigmoid(log_pos)
        log_neg = F.logsigmoid(log_neg).sum(1)

        return -(log_pos + log_neg)

    def getoutput(self, input_labels, neg_labels, ent_dic, reverse_dictionary):
        """
        获取输出嵌入
        优化点：
        - 复用_process_batch方法
        """
        input_emb_neg, negtive_emb = [], []
        self._process_batch(input_labels.view(-1), neg_labels.view(-1),
                            ent_dic, reverse_dictionary, input_emb_neg, negtive_emb)

        self.outputs1 = torch.cat(input_emb_neg, dim=0)
        self.outputs2 = torch.cat(negtive_emb, dim=0)
        return self.outputs1, self.outputs2

    def _get_original_ids(self, word_ids, reverse_dict):
        """
        批量获取原始ID的辅助方法
        优化点：
        - 向量化操作
        """
        ids = []
        for word_id in word_ids:
            id_str = reverse_dict[word_id.item()]
            if id_str == '<unk>':
                ids.append(self.ent_size - 1)
            elif int(id_str) >= self.ent_size - 1:
                ids.append(int(id_str) - (self.ent_size - 1))
            else:
                ids.append(int(id_str))
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def affinity(self, inputs1, inputs2):
        # 1. 移除不必要的成员变量存储（除非后续确实需要）
        input_dim1, input_dim2 = inputs1.size(0), inputs2.size(0)
        # 2. 使用更高效的矩阵运算方式
        if self.bilinear_weights:
            # 3. 直接创建权重矩阵到目标设备（避免不必要的Variable包装）
            weight_matrix = torch.randn(input_dim2, input_dim1,
                                        device=inputs1.device,  # 保持设备一致
                                        dtype=inputs1.dtype)  # 保持数据类型一致
            # 4. 更高效的矩阵乘法计算
            prod = inputs2 @ weight_matrix  # 等价于 matmul
            result = (inputs1 * prod).sum(dim=1)
        else:
            # 5. 点积运算优化
            result = (inputs1 * inputs2).sum(dim=1)
        return result

    def get_probs(self, outputs1, outputs2):
        """计算概率"""
        return torch.sigmoid(torch.pow(self.affinity(outputs1, outputs2), 0.25))

    def input_embeddings(self):
        """获取输入嵌入"""
        return self.in_embed_ent.weight.data.cpu().numpy(), self.in_embed_rel.weight.data.cpu().numpy()

    def scale_loss(self, embedding):
        """计算尺度损失"""
        norm_sq = torch.sum(embedding ** 2, dim=1, keepdim=True)
        return torch.sum(F.relu(norm_sq - 1.0))

    def orthogonal_loss(self, relation_embedding, w_embedding):
        """计算正交损失"""
        dot = torch.sum(relation_embedding * w_embedding, dim=1) ** 2
        norm = torch.norm(relation_embedding, p=2, dim=1) ** 2
        return torch.sum(F.relu(dot / norm - 1e-10))

    @staticmethod
    def map(ent, mapping):
        """将实体映射到关系的超平面"""
        norm = F.normalize(mapping, p=2, dim=-1)
        return ent - torch.sum(ent * norm, dim=1, keepdim=True) * norm


class EmbeddingModel_nonMap(nn.Module):
    def __init__(self, ent_size, rel_size, embed_size, gpu):
        """
        初始化实体和关系的嵌入模型
        优化点：
        - 统一初始化范围的计算
        """
        super(EmbeddingModel, self).__init__()
        self.ent_size = ent_size  # 实体字典大小
        self.rel_size = rel_size  # 关系字典大小
        self.embed_size = embed_size  # 嵌入维度
        self.device = gpu
        self.bilinear_weights = False

        # 统一初始化范围
        initrange = 0.5 / self.embed_size

        # 初始化实体和关系的嵌入矩阵
        self._init_embeddings(initrange)

    def _init_embeddings(self, initrange):
        """初始化所有嵌入矩阵的辅助方法"""
        # 输出嵌入
        self.out_embed_ent = nn.Embedding(self.ent_size, self.embed_size, sparse=False)
        self.out_embed_rel = nn.Embedding(self.rel_size, self.embed_size, sparse=False)

        # 输入嵌入
        self.in_embed_ent = nn.Embedding(self.ent_size, self.embed_size, sparse=False)
        self.in_embed_rel = nn.Embedding(self.rel_size, self.embed_size, sparse=False)

        # 统一初始化权重
        for embedding in [self.out_embed_ent, self.out_embed_rel,
                          self.in_embed_ent, self.in_embed_rel]:
            embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_labels, pos_labels, neg_labels, ent_dic, reverse_dictionary):
        """
        前向传播计算损失
        优化点：
        - 减少数据在CPU和GPU之间的传输
        - 使用向量化操作代替循环
        - 预分配列表空间
        """
        # 转换数据格式并保持在GPU上
        input_labels = input_labels.view(-1)
        pos_labels = pos_labels.view(-1)

        # 预分配列表空间
        input_emb, output_emb = [], []
        input_emb_neg, negtive_emb = [], []

        # 批量处理输入和正样本
        self._process_pairs(input_labels, pos_labels, ent_dic, reverse_dictionary,
                            input_emb, output_emb)

        # 批量处理输入和负样本
        self._process_pairs(input_labels.repeat_interleave(neg_labels.size(1)),
                            neg_labels.view(-1), ent_dic, reverse_dictionary,
                            input_emb_neg, negtive_emb)

        # 合并所有嵌入
        input_embedding = torch.cat(input_emb, dim=0).unsqueeze(2)
        pos_embedding = torch.cat(output_emb, dim=0).unsqueeze(1)
        input_emb_neg = torch.cat(input_emb_neg, dim=0).unsqueeze(2)
        neg_embedding = torch.cat(negtive_emb, dim=0).unsqueeze(1)

        # 计算正负样本的得分
        log_pos = torch.bmm(pos_embedding, input_embedding).squeeze()
        log_neg = torch.bmm(neg_embedding, -input_emb_neg).squeeze()

        # 重塑负样本得分并计算损失
        log_neg = log_neg.reshape(log_pos.size(0), -1)
        log_pos = F.logsigmoid(log_pos)
        log_neg = F.logsigmoid(log_neg).sum(1)

        loss = -(log_pos + log_neg)
        return loss

    def _process_pairs(self, inputs, targets, ent_dic, reverse_dictionary,
                       input_emb_list, output_emb_list):
        """
        处理输入-目标对的辅助方法
        优化点：
        - 批量处理数据
        - 减少条件判断
        """
        # 获取原始ID
        input_ids = self._get_original_ids(inputs.cpu().numpy(), reverse_dictionary)
        target_ids = self._get_original_ids(targets.cpu().numpy(), reverse_dictionary)

        # 创建掩码
        input_in_ent = torch.tensor([x in ent_dic for x in inputs], device=self.device)
        target_in_ent = torch.tensor([x in ent_dic for x in targets], device=self.device)

        # 处理四种情况
        self._process_case(input_ids, target_ids, input_in_ent, target_in_ent,
                           input_emb_list, output_emb_list, True, True)
        self._process_case(input_ids, target_ids, input_in_ent, target_in_ent,
                           input_emb_list, output_emb_list, True, False)
        self._process_case(input_ids, target_ids, input_in_ent, target_in_ent,
                           input_emb_list, output_emb_list, False, True)
        self._process_case(input_ids, target_ids, input_in_ent, target_in_ent,
                           input_emb_list, output_emb_list, False, False)

    def _process_case(self, input_ids, target_ids, input_mask, target_mask,
                      input_emb_list, output_emb_list, input_is_ent, target_is_ent):
        """
        处理特定情况的辅助方法
        """
        # 创建当前情况的掩码
        case_mask = (input_mask == input_is_ent) & (target_mask == target_is_ent)
        if not case_mask.any():
            return

        # 获取当前情况的ID
        case_input_ids = input_ids[case_mask]
        case_target_ids = target_ids[case_mask]

        # 选择正确的嵌入层
        input_embed = self.in_embed_ent if input_is_ent else self.in_embed_rel
        output_embed = self.out_embed_ent if target_is_ent else self.out_embed_rel

        # 添加到列表
        input_emb_list.append(input_embed(case_input_ids))
        output_emb_list.append(output_embed(case_target_ids))

    def getoutput(self, input_labels, neg_labels, ent_dic, reverse_dictionary):
        """
        获取输出嵌入
        优化点：
        - 使用向量化操作
        - 减少重复代码
        """
        # 转换数据格式
        input_labels = input_labels.view(-1)
        neg_labels = neg_labels.view(-1)

        # 预分配列表空间
        input_emb_neg, negtive_emb = [], []

        # 批量处理输入和负样本
        self._process_pairs(input_labels, neg_labels, ent_dic, reverse_dictionary,
                            input_emb_neg, negtive_emb)

        # 合并结果
        self.outputs1 = torch.cat(input_emb_neg, dim=0)
        self.outputs2 = torch.cat(negtive_emb, dim=0)

        return self.outputs1, self.outputs2

    def _get_original_ids(self, word_ids, reverse_dict):
        """
        批量获取原始ID的辅助方法
        优化点：
        - 向量化操作
        """
        ids = []
        for word_id in word_ids:
            id_str = reverse_dict[word_id.item()]
            if id_str == '<unk>':
                ids.append(self.ent_size - 1)
            elif int(id_str) >= self.ent_size - 1:
                ids.append(int(id_str) - (self.ent_size - 1))
            else:
                ids.append(int(id_str))
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def affinity(self, inputs1, inputs2):
        # 1. 移除不必要的成员变量存储（除非后续确实需要）
        input_dim1, input_dim2 = inputs1.size(0), inputs2.size(0)
        # 2. 使用更高效的矩阵运算方式
        if self.bilinear_weights:
            # 3. 直接创建权重矩阵到目标设备（避免不必要的Variable包装）
            weight_matrix = torch.randn(input_dim2, input_dim1,
                                        device=inputs1.device,  # 保持设备一致
                                        dtype=inputs1.dtype)  # 保持数据类型一致
            # 4. 更高效的矩阵乘法计算
            prod = inputs2 @ weight_matrix  # 等价于 matmul
            result = (inputs1 * prod).sum(dim=1)
        else:
            # 5. 点积运算优化
            result = (inputs1 * inputs2).sum(dim=1)
        return result


    def get_probs(self, outputs1, outputs2):
        """计算概率"""
        return torch.sigmoid(torch.pow(self.affinity(outputs1, outputs2), 0.25))

    def input_embeddings(self):
        """获取输入嵌入"""
        return self.in_embed_ent.weight.data.cpu().numpy(), self.in_embed_rel.weight.data.cpu().numpy()

    def scale_loss(self, embedding):
        """计算尺度损失"""
        norm_sq = torch.sum(embedding ** 2, dim=1, keepdim=True)
        return torch.sum(F.relu(norm_sq - 1.0))

    def orthogonal_loss(self, relation_embedding, w_embedding):
        """计算正交损失"""
        dot = torch.sum(relation_embedding * w_embedding, dim=1) ** 2
        norm = torch.norm(relation_embedding, p=2, dim=1) ** 2
        return torch.sum(F.relu(dot / norm - 1e-10))

    @staticmethod
    def map(ent, mapping):
        """将实体映射到关系的超平面"""
        norm = F.normalize(mapping, p=2, dim=-1)
        return ent - torch.sum(ent * norm, dim=1, keepdim=True) * norm


class EmbeddingModel_original(nn.Module):

    def __init__(self, ent_size, rel_size, embed_size, device):
        ''' 初始化输出和输出embedding
            函数用处不大
        '''
        super(EmbeddingModel_original, self).__init__()
        self.ent_size = ent_size  # 字典大小 30000
        self.rel_size = rel_size  # 字典大小 30000
        self.vocab_size = ent_size + rel_size + 1  # 字典大小 30000
        self.embed_size = embed_size  # 单词维度 一般是50，100，300维
        self.bilinear_weights = False
        initrange = 0.5 / self.embed_size
        self.out_embed_ent = nn.Embedding(self.ent_size, self.embed_size,sparse=False)  # 初始化一个矩阵，self.vocab_size * self.embed_size
        self.out_embed_ent.weight.data.uniform_(-initrange,initrange)  # 把矩阵中的权重初始化 设置在 -0.5 / self.embed_size到0.5 / self.embed_size间的随机值

        self.out_embed_rel = nn.Embedding(self.rel_size, self.embed_size,sparse=False)  # 初始化一个矩阵，self.vocab_size * self.embed_size
        self.out_embed_rel.weight.data.uniform_(-initrange,initrange)  # 把矩阵中的权重初始化 设置在 -0.5 / self.embed_size到0.5 / self.embed_size间的随机值

        self.in_embed_ent = nn.Embedding(self.ent_size, self.embed_size,sparse=False)  # 初始化一个矩阵，self.vocab_size * self.embed_size
        self.in_embed_ent.weight.data.uniform_(-initrange,initrange)  # 把矩阵中的权重初始化 设置在 -0.5 / self.embed_size到0.5 / self.embed_size间的随机值

        self.in_embed_rel = nn.Embedding(self.rel_size, self.embed_size,sparse=False)  # 初始化一个矩阵，self.vocab_size * self.embed_size
        self.in_embed_rel.weight.data.uniform_(-initrange,initrange)  # 把矩阵中的权重初始化 设置在 -0.5 / self.embed_size到0.5 / self.embed_size间的随机值

        self.in_embed_map = nn.Embedding(self.rel_size, self.embed_size,sparse=False)  # 初始化一个矩阵，self.vocab_size * self.embed_size
        self.in_embed_map.weight.data.uniform_(-initrange,initrange)  # 把矩阵中的权重初始化 设置在 -0.5 / self.embed_size到0.5 / self.embed_size间的随机值

        self.out_embed_map = nn.Embedding(self.rel_size, self.embed_size,sparse=False)  # 初始化一个矩阵，self.vocab_size * self.embed_size
        self.out_embed_map.weight.data.uniform_(-initrange,initrange)  # 把矩阵中的权重初始化 设置在 -0.5 / self.embed_size到0.5 / self.embed_size间的随机值

        self.device = device

    def forward(self, input_labels, pos_labels, neg_labels, ent_dic, reverse_dictionary):
        '''
        input_labels: 中心词, [batch_size]
        pos_labels: 中心词周围 context window 出现过的单词 [batch_size * (window_size * 2)]
        neg_labelss: 中心词周围没有出现过的单词，从 negative sampling 得到 [batch_size, (window_size * 2 * K)]
        return: loss, [batch_size]
        输入“抓紧”加速函数运行
        '''
        ent_dic_input_1 = []
        ent_dic_input_2 = []
        ent_dic_output_1 = []
        ent_dic_output_2 = []
        ent_dic_neg_input_1 = []
        ent_dic_neg_input_2 = []
        ent_dic_neg_output = []
        ent_dic_neg_1 = []
        ent_dic_neg_2 = []

        not_in_ent_dic_input_1 = []
        not_in_ent_dic_input_2 = []
        not_in_ent_dic_output_1 = []
        not_in_ent_dic_output_2 = []
        not_in_ent_dic_neg_input_1 = []
        not_in_ent_dic_neg_input_2 = []
        not_in_ent_dic_neg_1 = []
        not_in_ent_dic_neg_2 = []

        input_emb = []
        output_emb = []
        negtive_emb = []
        input_emb_neg = []

        input_labels = input_labels.cpu().numpy().tolist()
        pos_labels = pos_labels.view(-1).cpu().numpy().tolist()
        neg_labels = neg_labels.cpu().numpy().tolist()

        # start_time = time.time()
        for i, data in enumerate(input_labels):
            if data in ent_dic:
                input = self.get_original_id(data, reverse_dictionary, self.ent_size)
                output = self.get_original_id(pos_labels[i], reverse_dictionary, self.ent_size)
                if pos_labels[i] in ent_dic:
                    ent_dic_input_1.append(input)
                    ent_dic_output_1.append(output)
                else:
                    ent_dic_input_2.append(input)
                    ent_dic_output_2.append(output)
                for neg in neg_labels[i]:
                    negtive = self.get_original_id(neg, reverse_dictionary, self.ent_size)
                    if neg in ent_dic:
                        ent_dic_neg_input_1.append(input)
                        ent_dic_neg_1.append(negtive)
                    else:
                        ent_dic_neg_input_2.append(input)
                        ent_dic_neg_output.append(output)
                        ent_dic_neg_2.append(negtive)
            else:
                input = self.get_original_id(data, reverse_dictionary, self.ent_size)
                output = self.get_original_id(pos_labels[i], reverse_dictionary, self.ent_size)
                if pos_labels[i] in ent_dic:
                    not_in_ent_dic_input_1.append(input)
                    not_in_ent_dic_output_1.append(output)
                else:
                    not_in_ent_dic_input_2.append(input)
                    not_in_ent_dic_output_2.append(output)
                for neg in neg_labels[i]:
                    negtive = self.get_original_id(neg, reverse_dictionary, self.ent_size)
                    if neg in ent_dic:
                        not_in_ent_dic_neg_input_1.append(input)
                        not_in_ent_dic_neg_1.append(negtive)
                    else:
                        not_in_ent_dic_neg_input_2.append(input)
                        not_in_ent_dic_neg_2.append(negtive)


        # labels in ent_dic
        ent_dic_input_1 = Variable(torch.Tensor(np.array(ent_dic_input_1)).long()).cuda(self.device)
        ent_dic_input_2 = Variable(torch.Tensor(np.array(ent_dic_input_2)).long()).cuda(self.device)
        ent_dic_output_1 = Variable(torch.Tensor(np.array(ent_dic_output_1)).long()).cuda(self.device)
        ent_dic_output_2 = Variable(torch.Tensor(np.array(ent_dic_output_2)).long()).cuda(self.device)
        ent_dic_neg_input_1 = Variable(torch.Tensor(np.array(ent_dic_neg_input_1)).long()).cuda(self.device)
        ent_dic_neg_input_2 = Variable(torch.Tensor(np.array(ent_dic_neg_input_2)).long()).cuda(self.device)
        ent_dic_neg_output = Variable(torch.Tensor(np.array(ent_dic_neg_output)).long()).cuda(self.device)
        ent_dic_neg_1 = Variable(torch.Tensor(np.array(ent_dic_neg_1)).long()).cuda(self.device)
        ent_dic_neg_2 = Variable(torch.Tensor(np.array(ent_dic_neg_2)).long()).cuda(self.device)


        # pos_labels in ent_dic
        input_emb.append(self.in_embed_ent(ent_dic_input_1))
        output_emb.append(self.out_embed_ent(ent_dic_output_1))

        # pos_labels not in ent_dic
        map_vec_1 = self.in_embed_map(ent_dic_output_2)
        ent_map_1 = self.in_embed_ent(ent_dic_input_2)
        input_emb.append(self.map(ent_map_1, map_vec_1))
        output_emb.append(self.out_embed_rel(ent_dic_output_2))

        # neg in ent_dic
        input_emb_neg.append(self.in_embed_ent(ent_dic_neg_input_1))
        negtive_emb.append(self.out_embed_ent(ent_dic_neg_1))

        # neg not in ent_dic
        ent_map_2 = self.in_embed_ent(ent_dic_neg_input_2)
        map_vec_2 = self.in_embed_map(ent_dic_neg_2)
        input_emb_neg.append(self.map(ent_map_2, map_vec_2))
        negtive_emb.append(self.out_embed_rel(ent_dic_neg_2))

        # labels not in ent_dic
        not_in_ent_dic_input_1 = Variable(torch.Tensor(np.array(not_in_ent_dic_input_1)).long()).cuda(self.device)
        not_in_ent_dic_input_2 = Variable(torch.Tensor(np.array(not_in_ent_dic_input_2)).long()).cuda(self.device)
        not_in_ent_dic_output_1 = Variable(torch.Tensor(np.array(not_in_ent_dic_output_1)).long()).cuda(self.device)
        not_in_ent_dic_output_2 = Variable(torch.Tensor(np.array(not_in_ent_dic_output_2)).long()).cuda(self.device)
        not_in_ent_dic_neg_input_1 = Variable(torch.Tensor(np.array(not_in_ent_dic_neg_input_1)).long()).cuda(self.device)
        not_in_ent_dic_neg_input_2 = Variable(torch.Tensor(np.array(not_in_ent_dic_neg_input_2)).long()).cuda(self.device)
        not_in_ent_dic_neg_1 = Variable(torch.Tensor(np.array(not_in_ent_dic_neg_1)).long()).cuda(self.device)
        not_in_ent_dic_neg_2 = Variable(torch.Tensor(np.array(not_in_ent_dic_neg_2)).long()).cuda(self.device)

        # pos_labels in ent_dic:
        ent_map_3 = self.out_embed_ent(not_in_ent_dic_output_1)
        map_vec_3 = self.out_embed_map(not_in_ent_dic_input_1)
        input_emb.append(self.in_embed_rel(not_in_ent_dic_input_1))
        output_emb.append(self.map(ent_map_3, map_vec_3))

        # pos_labels not in ent_dic:
        input_emb.append(self.in_embed_rel(not_in_ent_dic_input_2))
        output_emb.append(self.out_embed_rel(not_in_ent_dic_output_2))

        # neg in ent_dic
        ent_map_4 = self.out_embed_ent(not_in_ent_dic_neg_1)
        map_vec_4 = self.out_embed_map(not_in_ent_dic_neg_input_1)
        negtive_emb.append(self.map(ent_map_4, map_vec_4))
        input_emb_neg.append(self.in_embed_rel(not_in_ent_dic_neg_input_1))

        # neg not in ent_dic
        input_emb_neg.append(self.in_embed_rel(not_in_ent_dic_neg_input_2))
        negtive_emb.append(self.out_embed_rel(not_in_ent_dic_neg_2))

        input_embedding = torch.cat([x for x in input_emb], dim=0).unsqueeze(2).cuda(self.device)
        pos_embedding = torch.cat([x for x in output_emb], dim=0).unsqueeze(1).cuda(self.device)
        input_neg_embedding = torch.cat([x for x in input_emb_neg], dim=0).unsqueeze(2).cuda(self.device)
        neg_embedding = torch.cat([x for x in negtive_emb], dim=0).unsqueeze(1).cuda(self.device)


        log_pos = torch.bmm(pos_embedding,input_embedding).squeeze().cuda(self.device)  # B * (2*C) 变成 B * (1) * embed_size x B * embed_size*1
        log_neg = torch.bmm(neg_embedding,-input_neg_embedding).squeeze().cuda(self.device)  # B * ((2*C)*K)  B * (2*C * K) * embed_size x B * embed_size*1
        log_neg = log_neg.reshape(log_pos.size(0), -1)
        log_pos = F.logsigmoid(log_pos)  # .sum(1)
        log_neg = F.logsigmoid(log_neg).sum(1)  # batch_size

        loss = -(log_pos + log_neg)

        return loss

    def getoutput(self, input_labels, neg_labels, ent_dic,
                       reverse_dictionary):  # input1, input2:user_list, y_list_tensor

        ent_dic_neg_input_1 = []
        ent_dic_neg_input_2 = []
        ent_dic_neg_1 = []
        ent_dic_neg_2 = []

        not_in_ent_dic_neg_input_1 = []
        not_in_ent_dic_neg_input_2 = []
        not_in_ent_dic_neg_1 = []
        not_in_ent_dic_neg_2 = []

        negtive_emb = []
        input_emb_neg = []

        batch_size = input_labels.size(0)

        input_labels = input_labels.cpu().numpy().tolist()
        neg_labels = neg_labels.cpu().numpy().tolist()

        # start_time = time.time()
        for i, data in enumerate(input_labels):
            if data in ent_dic:
                input = self.get_original_id(data, reverse_dictionary, self.ent_size)
                negtive = self.get_original_id(neg_labels[i], reverse_dictionary, self.ent_size)
                if neg_labels[i] in ent_dic:
                    ent_dic_neg_input_1.append(input)
                    ent_dic_neg_1.append(negtive)
                else:
                    ent_dic_neg_input_2.append(input)
                    ent_dic_neg_2.append(negtive)
            else:
                input = self.get_original_id(data, reverse_dictionary, self.ent_size)
                negtive = self.get_original_id(neg_labels[i], reverse_dictionary, self.ent_size)
                if neg_labels[i] in ent_dic:
                    not_in_ent_dic_neg_input_1.append(input)
                    not_in_ent_dic_neg_1.append(negtive)
                else:
                    not_in_ent_dic_neg_input_2.append(input)
                    not_in_ent_dic_neg_2.append(negtive)
        # labels in ent_dic
        ent_dic_neg_input_1 = Variable(torch.Tensor(np.array(ent_dic_neg_input_1)).long()).cuda(self.device)
        ent_dic_neg_input_2 = Variable(torch.Tensor(np.array(ent_dic_neg_input_2)).long()).cuda(self.device)
        ent_dic_neg_1 = Variable(torch.Tensor(np.array(ent_dic_neg_1)).long()).cuda(self.device)
        ent_dic_neg_2 = Variable(torch.Tensor(np.array(ent_dic_neg_2)).long()).cuda(self.device)

        # neg in ent_dic
        input_emb_neg.append(self.in_embed_ent(ent_dic_neg_input_1))
        negtive_emb.append(self.out_embed_ent(ent_dic_neg_1))

        # neg not in ent_dic
        ent_map_2 = self.in_embed_ent(ent_dic_neg_input_2)
        map_vec_2 = self.in_embed_map(ent_dic_neg_2)
        input_emb_neg.append(self.map(ent_map_2, map_vec_2))
        negtive_emb.append(self.out_embed_rel(ent_dic_neg_2))

        # labels not in ent_dic
        not_in_ent_dic_neg_input_1 = Variable(torch.Tensor(np.array(not_in_ent_dic_neg_input_1)).long()).cuda(
            self.device)
        not_in_ent_dic_neg_input_2 = Variable(torch.Tensor(np.array(not_in_ent_dic_neg_input_2)).long()).cuda(
            self.device)
        not_in_ent_dic_neg_1 = Variable(torch.Tensor(np.array(not_in_ent_dic_neg_1)).long()).cuda(self.device)
        not_in_ent_dic_neg_2 = Variable(torch.Tensor(np.array(not_in_ent_dic_neg_2)).long()).cuda(self.device)

        # neg in ent_dic
        ent_map_4 = self.out_embed_ent(not_in_ent_dic_neg_1)
        map_vec_4 = self.out_embed_map(not_in_ent_dic_neg_input_1)
        negtive_emb.append(self.map(ent_map_4, map_vec_4))
        input_emb_neg.append(self.in_embed_rel(not_in_ent_dic_neg_input_1))

        # neg not in ent_dic
        negtive_emb.append(self.out_embed_rel(not_in_ent_dic_neg_2))
        input_emb_neg.append(self.in_embed_rel(not_in_ent_dic_neg_input_2))

        self.outputs1 = torch.cat([x for x in input_emb_neg], dim=0).cuda(self.device)
        self.outputs2 = torch.cat([x for x in negtive_emb], dim=0).cuda(self.device)
        # print('outputs1', np.array(self.outputs1.data.cpu().numpy()).shape)#outputs1 (120, 100, 1)
        # print('outputs2', np.array(self.outputs2.data.cpu().numpy()).shape)#outputs2 (120, 1, 100)
        return self.outputs1, self.outputs2

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

    def get_original_id(self, word_id, reverse_dict, ent_size):  ##ent_size=40943+1
        id = reverse_dict[word_id]
        # print('id',id) #id 40943
        if id == '<unk>':
            id = ent_size - 1
        elif int(id) >= ent_size - 1:
            id = int(id) - int((ent_size - 1))
        ###int(id) >= ent_size时，返回的id=40943，于是乎self.in_embed_map(40943)就不对了，超出了rel_size的范围
        # 当改为int(id) >= ent_size-1时，返回的是id=0,这样就能有对应的embedding矩阵中的索引了。
        return id


    def map(self, ent, mapping):
        ####将实体映射到关系的超平面上
        norm = F.normalize(mapping, p=2, dim=-1)
        ent_map = ent - torch.sum(ent * norm, dim=1, keepdim=True) * norm
        return ent_map



