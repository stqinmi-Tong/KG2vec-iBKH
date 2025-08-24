import torch
from collections import defaultdict
import numpy as np
import time
from scipy.stats import norm


def MCNS_fast(model, candidates, start_given, q_1_dict, q_2_dict,
              N_steps, input_tensor, ent_dic, reverse_dictionary, device):

    # === 预处理分布 ===
    distribution_np = norm.pdf(np.arange(0, 10, 1), 5, 1)
    distribution_np = distribution_np / distribution_np.sum()
    distribution = torch.tensor(distribution_np, dtype=torch.float32, device=device)

    # === 将字典转成张量 ===
    vocab_size = max(max(q_1_dict.keys()), max(q_2_dict.keys())) + 1
    ent_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    ent_mask[list(ent_dic)] = True

    q1_probs = torch.zeros(vocab_size, dtype=torch.float32, device=device)
    q2_probs = torch.zeros(vocab_size, dtype=torch.float32, device=device)
    for k, v in q_1_dict.items():
        q1_probs[k] = v
    for k, v in q_2_dict.items():
        q2_probs[k] = v

    q1_keys = torch.tensor(list(q_1_dict.keys()), dtype=torch.long, device=device)
    q1_weights = torch.tensor(list(q_1_dict.values()), dtype=torch.float32, device=device)
    q2_keys = torch.tensor(list(q_2_dict.keys()), dtype=torch.long, device=device)
    q2_weights = torch.tensor(list(q_2_dict.values()), dtype=torch.float32, device=device)

    # === 初始化 start ===
    user_list = input_tensor.clone().to(device)
    if start_given is None:
        is_ent = ent_mask[user_list]
        start = torch.empty_like(user_list)
        start[is_ent] = q1_keys[torch.randint(0, len(q1_keys), (is_ent.sum(),), device=device)]
        start[~is_ent] = q2_keys[torch.randint(0, len(q2_keys), (~is_ent).sum(), device=device)]
        cur_state = start
    else:
        cur_state = start_given.clone().to(device)

    walks = defaultdict(list)
    count = 0

    while True:
        count += 1
        sample_num = torch.rand(1, device=device)

        if sample_num < 0.5:
            # --- 批量采样 ---
            is_ent = ent_mask[user_list]
            y_list = torch.empty_like(user_list)

            idx_ent = torch.multinomial(q1_weights, is_ent.sum().item(), replacement=True)
            idx_nonent = torch.multinomial(q2_weights, (~is_ent).sum().item(), replacement=True)

            y_list[is_ent] = q1_keys[idx_ent]
            y_list[~is_ent] = q2_keys[idx_nonent]

            q_probs_list = torch.where(is_ent, q1_probs[y_list], q2_probs[y_list])
            q_probs_next_list = torch.where(ent_mask[cur_state], q1_probs[cur_state], q2_probs[cur_state])

        else:
            # --- 候选节点分布采样 ---
            y_list = torch.empty_like(user_list)
            q_probs_list = torch.empty_like(user_list, dtype=torch.float32)
            q_probs_next_list = torch.empty_like(user_list, dtype=torch.float32)

            for idx, node in enumerate(cur_state):
                cand_nodes = candidates[node.item()]
                if len(cand_nodes) == 10:
                    cand_tensor = torch.tensor(cand_nodes, device=device)
                    choice_idx = torch.multinomial(distribution, 1).item()
                    y = cand_tensor[choice_idx]
                    y_list[idx] = y
                    q_probs_list[idx] = distribution[choice_idx]

                    cand_next = candidates[y.item()]
                    if node.item() in cand_next:
                        idx_next = cand_next.index(node.item())
                        q_probs_next_list[idx] = distribution[idx_next]
                    else:
                        q_probs_next_list[idx] = q1_probs[node] if ent_mask[node] else q2_probs[node]
                else:
                    if ent_mask[node]:
                        y = q1_keys[torch.multinomial(q1_weights, 1)]
                        q_probs_list[idx] = q1_probs[y]
                        q_probs_next_list[idx] = q1_probs[node]
                    else:
                        y = q2_keys[torch.multinomial(q2_weights, 1)]
                        q_probs_list[idx] = q2_probs[y]
                        q_probs_next_list[idx] = q2_probs[node]
                    y_list[idx] = y

        # === 模型计算 ===
        user_list_out, y_list_out = model.getoutput(user_list, y_list, ent_dic, reverse_dictionary)
        _, cur_state_out = model.getoutput(user_list, cur_state, ent_dic, reverse_dictionary)

        p_probs = model.get_probs(user_list_out, y_list_out)
        p_probs_next = model.get_probs(user_list_out, cur_state_out)

        # === 接受概率计算 ===
        A_a_list = (p_probs * q_probs_next_list) / (p_probs_next * q_probs_list)

        if count > N_steps:
            for i in range(len(cur_state)):
                walks[user_list[i].item()].append(cur_state[i].item())
        else:
            u = torch.rand(len(cur_state), device=device)
            accept = u < torch.minimum(torch.ones_like(A_a_list), A_a_list)
            cur_state = torch.where(accept, y_list, cur_state)

        # === 判断结束条件 ===
        if sum(len(v) for v in walks.values()) == len(user_list):
            generate_examples = [walks[user.item()][0] for user in user_list]
            return torch.tensor(generate_examples, dtype=torch.long, device=device)



def MCNS_gpu(model, candidates, start_given, q_1_dict, q_2_dict,
                      N_steps, input, ent_dic, reverse_dictionary, device):
    """
    完全等价于原 MCNS_original，但在不改变随机性与输出的前提下，
    最大限度减少 CPU/GPU 往返，并用向量化完成接受率与状态更新。
    """

    # === 固定与原版一致的分布（仍用 numpy） ===
    distribution = norm.pdf(np.arange(0, 10, 1), 5, 1)
    distribution = distribution / np.sum(distribution)  # 归一化

    # 仅在需要 CPU 操作时才转为 numpy；保持与原版一致的输入语义（list / numpy）
    user_list = input.detach().cpu().numpy().tolist()

    # === 初始化起点（严格保持与原版相同的随机调用顺序与规则）===
    if start_given is None:
        start = []
        for i in user_list:
            if i in ent_dic:
                s = np.random.choice(list(q_1_dict.keys()), 1)[0]
            else:
                s = np.random.choice(list(q_2_dict.keys()), 1)[0]
            start.append(s)
    else:
        start = start_given.detach().cpu().numpy().tolist()

    cur_state = start
    if len(cur_state) != len(user_list):
        cur_state = cur_state[:len(user_list)]

    walks = defaultdict(list)
    count = 0

    # 预取 q_1 / q_2 的键与概率为 numpy 数组（查表更快；不改变随机性）
    q1_keys = np.fromiter(q_1_dict.keys(), dtype=np.int64)
    q1_probs = np.fromiter(q_1_dict.values(), dtype=np.float64)
    q2_keys = np.fromiter(q_2_dict.keys(), dtype=np.int64)
    q2_probs = np.fromiter(q_2_dict.values(), dtype=np.float64)

    # 主循环
    while True:
        count += 1
        sample_num = np.random.random()

        # === 生成 y_list 以及 q(y|x)、q(x|y)（严格逐样本、逐条件，保持原版随机次序）===
        y_list = []
        q_probs_list = []       # q(y|x)
        q_probs_next_list = []  # q(x|y)

        if sample_num < 0.5:
            # 路径 1：全局分布抽样
            for idx, i in enumerate(user_list):
                if i in ent_dic:
                    y = np.random.choice(q1_keys, 1, p=q1_probs)[0]
                    y_list.append(int(y))
                    q_probs_list.append(float(q_1_dict[y]))
                else:
                    y = np.random.choice(q2_keys, 1, p=q2_probs)[0]
                    y_list.append(int(y))
                    q_probs_list.append(float(q_2_dict[y]))

            for c in cur_state:
                if c in ent_dic:
                    q_probs_next_list.append(float(q_1_dict[c]))
                else:
                    q_probs_next_list.append(float(q_2_dict[c]))
        else:
            # 路径 2：候选集／回退策略
            for i in cur_state:
                cand_nodes = candidates[i]
                if len(cand_nodes) == 10:
                    # 从候选分布中采样（与原版相同的顺序与分布）
                    choice_idx = np.random.choice(10, 1, p=distribution)[0]
                    y = cand_nodes[choice_idx]
                    y_list.append(int(y))
                    q_probs_list.append(float(distribution[choice_idx]))  # q(y|x)

                    node_list_next = candidates[y]
                    if i in node_list_next:
                        idx_next = node_list_next.index(i)
                        q_probs_next_list.append(float(distribution[idx_next]))  # q(x|y)
                    else:
                        # 回退
                        if i in ent_dic:
                            q_probs_next_list.append(float(q_1_dict[i]))
                        else:
                            q_probs_next_list.append(float(q_2_dict[i]))
                else:
                    # 回退到全局分布
                    if i in ent_dic:
                        y = np.random.choice(q1_keys, 1, p=q1_probs)[0]
                        q_probs_next = q_1_dict[i]
                        q_probs_list.append(float(q_1_dict[y]))
                    else:
                        y = np.random.choice(q2_keys, 1, p=q2_probs)[0]
                        q_probs_next = q_2_dict[i]
                        q_probs_list.append(float(q_2_dict[y]))
                    y_list.append(int(y))
                    q_probs_next_list.append(float(q_probs_next))

        # === 以下用 GPU 张量向量化完成：模型前向、接受率、状态更新 ===
        # 转为 GPU LongTensor（一次性拷贝）
        user_t = torch.from_numpy(np.asarray(user_list, dtype=np.int64)).to(device)
        y_t    = torch.from_numpy(np.asarray(y_list, dtype=np.int64)).to(device)
        cur_t  = torch.from_numpy(np.asarray(cur_state, dtype=np.int64)).to(device)

        # 模型输出（保持与原版相同的两次 getoutput 调用）
        user_out, y_out   = model.getoutput(user_t, y_t, ent_dic, reverse_dictionary)
        _,        cur_out = model.getoutput(user_t, cur_t, ent_dic, reverse_dictionary)

        # 计算 p 概率（GPU）
        p_probs      = model.get_probs(user_out, y_out)         # shape: [B]
        p_probs_next = model.get_probs(user_out, cur_out)       # shape: [B]

        # 接受率 A_a（把 q 概率搬到 GPU，并向量化）
        qyx = torch.from_numpy(np.asarray(q_probs_list, dtype=np.float32)).to(device)
        qxy = torch.from_numpy(np.asarray(q_probs_next_list, dtype=np.float32)).to(device)

        A_a = (p_probs * qxy) / (p_probs_next * qyx)           # [B]
        A_a = torch.minimum(A_a, torch.ones_like(A_a))

        if count > N_steps:
            # 终止分支：和原版一致，按当前 cur_state 记入 walks
            for k in range(len(cur_state)):
                walks[user_list[k]].append(cur_state[k])
        else:
            # 接受-拒绝（GPU 上一次性完成）
            u = torch.rand(A_a.size(0), device=device)
            accept = (u < A_a)
            # 按位选择下一个状态
            next_state_t = torch.where(accept, y_t, cur_t)

            # 写回 CPU list 以继续后续 numpy / Python 控制逻辑（仅一次回拷）
            cur_state = next_state_t.detach().cpu().numpy().tolist()

        # 统计长度，保持与原版相同的逻辑
        length = sum(len(v) for v in walks.values())

        if length == len(user_list):
            generate_examples = []
            for user in user_list:
                d = walks[user]
                if len(d) == 1:
                    generate_examples.append(d[0])
                else:
                    generate_examples.append(d[0])
                    del walks[user][0]
            # 返回 GPU LongTensor（与原版一致）
            return torch.from_numpy(np.asarray(generate_examples, dtype=np.int64)).to(device)
        else:
            continue