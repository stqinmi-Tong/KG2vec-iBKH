import torch
import torch.utils.data as tud
import numpy as np
from tqdm import tqdm
import time
import datetime
class WordEmbeddingDataset_KG2vec(tud.Dataset):
    def __init__(self, data_path, CUDA, sampled_num):

        """
        input example:
        data_path = [[1,2,3]]
        dataset = WordEmbeddingDataset_KG2vec(data_path, dictionary=None, CUDA=False, sampled_num=5)
        output example:
        最终数据集内部存储：
        center_word = tensor([1,1,2,2,3,3])
        pos_words = tensor([2,3,1,3,1,2])
        """
        super(WordEmbeddingDataset_KG2vec, self).__init__()
        self.CUDA = CUDA
        self.sampled_num = sampled_num

        # 直接加载为LongTensor避免多次转换
        self.data = torch.LongTensor(data_path)

        # 预分配空间
        self.center_word = []
        self.pos_words = []

        # 批量处理所有路径
        self._process_paths()

        # 转换为Tensor一次完成
        self.center_word = torch.LongTensor(self.center_word)
        self.pos_words = torch.LongTensor(self.pos_words)

        # if CUDA:
        #     self.center_word = self.center_word.cuda()
        #     self.pos_words = self.pos_words.cuda()

    def _process_paths(self):
        """向量化处理所有路径"""
        for path in tqdm(self.data, desc="Processing paths"):
            for index, w in enumerate(path):
                # 获取上下文词
                if index == 0:
                    context = path[1:]
                elif index == len(path) - 1:
                    context = path[:index]
                else:
                    context = torch.cat([path[:index], path[index + 1:]])

                # 扩展样本
                self.center_word.extend([w.item()] * len(context))
                self.pos_words.extend(context.tolist())

    def __len__(self):
        return len(self.center_word)

    def __getitem__(self, idx):
        """直接返回Tensor避免每次转换"""
        center = self.center_word[idx]
        pos = self.pos_words[idx]
        # if self.CUDA:
        #     center = center.cuda()
        #     pos = pos.cuda()
        return center, pos




class WordEmbeddingDataset_nomal_neg(tud.Dataset):
    def __init__(self, data_path, dictionary, CUDA, sampled_num, reverse_dictionary, rel_dic, ent_dic, word_freqs):
        ''' 优化后的初始化函数
        Args:
            data_path: 文本数据路径或已加载的数据
            dictionary: 单词到索引的字典
            CUDA: 是否使用CUDA
            sampled_num: 负采样数量
            reverse_dictionary: 反向字典
            rel_dic: 关系词字典
            ent_dic: 实体词字典
            word_freqs: 词频统计
        '''
        super(WordEmbeddingDataset, self).__init__()

        # 数据初始化 - 只做一次类型转换
        self.data = torch.as_tensor(data_path, dtype=torch.long)
        self.word_to_idx = dictionary
        self.word_freqs = torch.as_tensor(word_freqs, dtype=torch.float)
        self.CUDA = CUDA
        self.sampled_num = sampled_num
        self.reverse_dictionary = reverse_dictionary
        self.rel_dic = set(rel_dic)  # 转换为集合提高查找效率
        self.ent_dic = set(ent_dic)

        # 预分配空间
        total_samples = len(self.data) * 4  # 预估样本数量
        self.center_words = []
        self.pos_words = []
        self.neg_words = []

        # 批量处理负采样
        all_neg_samples = torch.multinomial(
            self.word_freqs,
            total_samples * 300,  # 预生成足够多的负样本
            replacement=True
        )
        # 处理每个路径
        neg_idx = 0
        for path in tqdm(self.data, desc="Processing data"):
            path_len = len(path)
            for index, w in enumerate(path):
                # 获取上下文词
                if index == 0:
                    context = path[1:]
                elif index == path_len - 1:
                    context = path[:index]
                else:
                    context = torch.cat([path[:index], path[index + 1:]])

                # 存储中心词和正样本
                # self.center_words.append(context)
                # self.pos_words.append(torch.tensor([w]))
                self.center_word.extend([w.item()] * len(context))
                self.pos_words.extend(context.tolist())

                # 处理负样本
                pos_word = w.item()
                if self.reverse_dictionary[pos_word] == '<UNK>':
                    # UNK特殊处理
                    neg_sample = all_neg_samples[neg_idx:neg_idx + self.sampled_num]
                    self.neg_words.append(neg_sample)

                elif pos_word in self.ent_dic:
                    # 实体词处理
                    candidates = all_neg_samples[neg_idx:neg_idx + 300].tolist() + list(self.ent_dic)
                    samples = []
                    for sam in candidates:
                        if sam in self.ent_dic and sam not in samples:
                            samples.append(sam)
                            if len(samples) == self.sampled_num:
                                break
                    self.neg_words.append(torch.tensor(samples[:self.sampled_num]))

                else:
                    # 关系词处理
                    candidates = all_neg_samples[neg_idx:neg_idx + 300].tolist() + list(self.rel_dic)
                    samples = []
                    for sam in candidates:
                        if sam in self.rel_dic and sam not in samples:
                            samples.append(sam)
                            if len(samples) == self.sampled_num:
                                break
                    self.neg_words.append(torch.tensor(samples[:self.sampled_num]))
                neg_idx += 300

        # 转换为tensor并确保正确形状
        self.center_words = torch.stack(self.center_words)
        self.pos_words = torch.stack(self.pos_words)
        self.neg_words = torch.stack(self.neg_words)

        # 清理内存
        del all_neg_samples
        gc.collect()

    def __len__(self):
        return len(self.center_words)

    def __getitem__(self, idx):
        return self.center_words[idx], self.pos_words[idx], self.neg_words[idx]


class WordEmbeddingDatasetOptimized(tud.Dataset):
    def __init__(self, data_path, dictionary, CUDA, sampled_num, reverse_dictionary, rel_dic, ent_dic, word_freqs):
        super().__init__()
        self.word_to_idx = dictionary
        self.sampled_num = sampled_num
        self.CUDA = CUDA

        ent_set = set(ent_dic)
        rel_set = set(rel_dic)
        rev_dict = reverse_dictionary

        # 转为 tensor
        data_tensor = torch.tensor(data_path, dtype=torch.long)
        word_freqs_tensor = torch.tensor(word_freqs, dtype=torch.float)

        centers, positives, negs = [], [], []

        # # 一次性生成负样本矩阵（最大长度）
        # max_len = max(len(path) for path in data_tensor)
        # total_neg_samples = 300 * max_len

        # 遍历路径
        for path in tqdm(data_tensor, desc="Processing paths"):
            path_np = path.numpy()
            path_len = len(path_np)

            # 构建邻居矩阵（中心词对应的正样本）
            for idx, w in enumerate(path_np):
                neighbors = np.concatenate([path_np[:idx], path_np[idx+1:]]) if path_len > 1 else np.array([])
                if len(neighbors) == 0:
                    continue
                centers.append(neighbors)
                positives.append(np.array([w]))

                # 负样本
                neg_sample = torch.multinomial(word_freqs_tensor, 300 * len(neighbors), replacement=True).numpy()
                if rev_dict[w] == '<UNK>':
                    negs.append(neg_sample[:sampled_num])
                elif w in ent_set:
                    combined = np.unique(np.concatenate([neg_sample, list(ent_set)]))
                    negs.append(combined[:sampled_num])
                else:
                    combined = np.unique(np.concatenate([neg_sample, list(rel_set)]))
                    negs.append(combined[:sampled_num])

        # 转为 tensor 并统一送 GPU
        self.center_word = torch.tensor(np.array(centers), dtype=torch.long)
        self.pos_words = torch.tensor(np.array(positives), dtype=torch.long)
        self.neg_words = torch.tensor(np.array(negs), dtype=torch.long)

        if CUDA:
            self.center_word = self.center_word.cuda(non_blocking=True)
            self.pos_words = self.pos_words.cuda(non_blocking=True)
            self.neg_words = self.neg_words.cuda(non_blocking=True)

    def __len__(self):
        return len(self.center_word)

    def __getitem__(self, idx):
        return self.center_word[idx], self.pos_words[idx], self.neg_words[idx]

class WordEmbeddingDatasetOptimizedSGNS(tud.Dataset):
    def __init__(self, data_path, dictionary, CUDA, sampled_num, reverse_dictionary, word_freqs):
        super().__init__()
        self.word_to_idx = dictionary
        self.sampled_num = sampled_num
        self.CUDA = CUDA
        self.rev_dict = reverse_dictionary

        # 转为 tensor
        data_tensor = torch.tensor(data_path, dtype=torch.long)

        # SGNS 的负采样分布：unigram^0.75
        word_freqs = np.array(word_freqs) ** 0.75
        word_freqs = word_freqs / word_freqs.sum()
        self.word_freqs_tensor = torch.tensor(word_freqs, dtype=torch.float)

        centers, positives, negs = [], [], []

        # 遍历路径
        for path in tqdm(data_tensor, desc="Processing paths"):
            path_np = path.numpy()
            path_len = len(path_np)

            # 构建邻居矩阵（中心词对应的正样本 = path 里除了自己之外的所有词）
            for idx, w in enumerate(path_np):
                neighbors = np.concatenate([path_np[:idx], path_np[idx+1:]]) if path_len > 1 else np.array([])
                if len(neighbors) == 0:
                    continue

                for pos in neighbors:
                    centers.append(w)
                    positives.append(pos)

                    # SGNS 负样本
                    neg_sample = torch.multinomial(self.word_freqs_tensor,
                                                   self.sampled_num,
                                                   replacement=True).numpy()
                    negs.append(neg_sample)

        # 转为 tensor 并统一送 GPU
        self.center_word = torch.tensor(np.array(centers), dtype=torch.long)
        self.pos_words = torch.tensor(np.array(positives), dtype=torch.long)
        self.neg_words = torch.tensor(np.array(negs), dtype=torch.long)

        if CUDA:
            self.center_word = self.center_word.cuda(non_blocking=True)
            self.pos_words = self.pos_words.cuda(non_blocking=True)
            self.neg_words = self.neg_words.cuda(non_blocking=True)

    def __len__(self):
        return len(self.center_word)

    def __getitem__(self, idx):
        return self.center_word[idx], self.pos_words[idx], self.neg_words[idx]