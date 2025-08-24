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






    def __getitem__(self, idx):

        return self.center_word[idx], self.pos_words[idx], self.neg_words[idx]
