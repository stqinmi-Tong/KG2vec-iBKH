import torch
import numpy as np
from torch.autograd import Variable

class TestModel:
    def __init__(self, ent_size=100, embed_dim=10, device='cuda'):
        self.ent_size = ent_size
        self.device = device
        self.in_embed_ent = lambda x: x.float().unsqueeze(1).repeat(1, embed_dim).to(device)
        self.out_embed_ent = lambda x: x.float().unsqueeze(1).repeat(1, embed_dim).to(device)
        self.in_embed_rel = lambda x: x.float().unsqueeze(1).repeat(1, embed_dim).to(device)
        self.out_embed_rel = lambda x: x.float().unsqueeze(1).repeat(1, embed_dim).to(device)
        self.in_embed_map = lambda x: x.float().unsqueeze(1).repeat(1, embed_dim).to(device)
        self.out_embed_map = lambda x: x.float().unsqueeze(1).repeat(1, embed_dim).to(device)
        self.map = lambda a, b: a + b
        self.ent_size = ent_size

    def get_original_id(self, x, reverse_dictionary, ent_size):
        return reverse_dictionary.get(x, x % ent_size)

    # 原始版本 getoutput (保留完整逻辑)
    def getoutput_orig(self, input_labels, neg_labels, ent_dic,
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

    # 优化版本 getoutput_fast (保留我们优化后的版本)
    def getoutput_fast(self, input_labels, neg_labels, ent_dic, reverse_dictionary):
        # === 预处理 ===
        device = self.device
        ent_mask = torch.zeros(
            max(max(reverse_dictionary.values()), input_labels.max().item(), neg_labels.max().item()) + 1,
            dtype=torch.bool, device=device)
        ent_mask[list(ent_dic)] = True

        # 保证 long 类型
        input_labels = input_labels.to(device).long()
        neg_labels = neg_labels.to(device).long()

        # 批量获取原始 ID
        get_orig_id = torch.tensor(
            [self.get_original_id(i.item(), reverse_dictionary, self.ent_size) for i in
             torch.cat([input_labels, neg_labels])],
            dtype=torch.long, device=device
        ).view(2, -1)
        input_orig = get_orig_id[0]
        neg_orig = get_orig_id[1]

        # 批量分类掩码
        input_is_ent = ent_mask[input_labels]
        neg_is_ent = ent_mask[neg_labels]

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

# 单元测试
def test_getoutput_real():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TestModel(device=device)

    batch_size = 32
    input_labels = torch.randint(0, 50, (batch_size,), device=device)
    neg_labels = torch.randint(0, 50, (batch_size,), device=device)
    ent_dic = set(np.random.choice(50, size=20, replace=False))
    reverse_dictionary = {i: i*2 for i in range(50)}

    out_orig = model.getoutput_orig(input_labels, neg_labels, ent_dic, reverse_dictionary)
    out_fast = model.getoutput_fast(input_labels, neg_labels, ent_dic, reverse_dictionary)

    # 精确对比
    assert all(torch.equal(o1, o2) for o1, o2 in zip(out_orig, out_fast)), "输出不一致！"
    print("单元测试通过：优化版输出与原始输出完全一致。")

test_getoutput_real()





