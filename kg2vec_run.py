import argparse
import collections
import os
import random
import json
import numpy as np
import torch
import datetime
import torch.multiprocessing as mp
from create_batch import build_graph, generate_paths,RandomSubsetSampler
from preprocessing import read_entity_from_id, read_relation_from_id, build_data
from WordEmbeddingDataset import WordEmbeddingDataset_KG2vec
from kg2vec import kg2vec_fast
from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler
def parse_args():
    parser = argparse.ArgumentParser(description="KG2vec Training Arguments")

    # Network arguments
    parser.add_argument("-data", "--data", default="../data/iBKH/",
                        help="data directory")
    parser.add_argument("-pre_emb", "--pretrained_emb", type=bool, default=False,
                        help="Use pretrained embeddings")
    parser.add_argument("-emb_size", "--EMBEDDING_SIZE", type=int, default=200,
                        help="Size of embeddings (if pretrained not used)")#100
    parser.add_argument("-skip_window", "--skip_window", type=int, default=2,
                        help="length of context window")
    parser.add_argument("-EPOCHS", "--epoch_num", type=int, default=20,
                        help="number of training epochs of kg2vec")

    parser.add_argument("-sampled_num", "--sampled_num", type=int, default=1,
                        help="number of negative samples")
    parser.add_argument("-walk_num", "--walk_num", type=int, default=2,
                        help="number of walk steps")#fb:10,wn:6
    parser.add_argument("-walk_num_edge", "--walk_num_edge", type=int, default=1,
                        help="number of edges through a walk")#fb:6,wn:3
    parser.add_argument("-lr", "--lr", type=float, default=0.01,
                        help="learning rate") #FB: 1e-3
    parser.add_argument("-BATCH_SIZE", "--batch_size_kg2vec", type=int, default=512,
                        help="Batch size for KG2vec")
    parser.add_argument("-path_length", "--path_length", type=int, default=3,
                       help="path length for KG2vec")

    # Training arguments
    parser.add_argument("-margin", "--margin", type=float, default=1,
                        help="Margin used in hinge loss")
    parser.add_argument("-outfolder", "--output_folder",
                        default="./checkpoints/",
                        help="Folder name to save the models")
    parser.add_argument("--test-record", type=str,
                        default="./checkpoints/test_record.txt")
    parser.add_argument("-gpu", "--gpu", default='1', help="Training use GPU")
    return parser.parse_args()

def load_data(args):

    train_data, validation_data, test_data, entity2id, relation2id = build_data(
        args.data, is_unweighted=False, directed=True)
    if os.path.exists(os.path.join(args.data, "training_graph.tsv")):
        graph = torch.load(os.path.join(args.data, "training_graph.tsv"))
    else:
        graph = build_graph(os.path.join(args.data, "training_triplets.tsv"), entity2id, relation2id)
        torch.save(graph, os.path.join(args.data, "training_graph.tsv"))
    print('graph has been constructed!')

    paths = generate_paths(graph, n=args.path_length)  # n=5 => 2跳路径
    return train_data, paths ###train_data=[[e1_id,r1_id,e2_id],...],paths=[[e1_id,r1_id,e2_id,r2_id,e3_id],...]


def save_data(args, train_data, paths):
    """统一保存，减少磁盘IO"""
    save_dict = {
        "train_data": train_data,
        "paths": paths

    }
    torch.save(save_dict, os.path.join(args.data, "train_package.pt"))

def load_saved_data(args):
    save_path = os.path.join(args.data, "train_package.pt")
    if os.path.exists(save_path):
        package = torch.load(save_path)
        # print('paths',package["paths"])
        return package["train_data"],  package["paths"]
    else:
        train_data, paths = load_data(args)
        save_data(args, train_data, paths)
    return train_data, paths

def path_rel2ent(path_data, ent_size):
    """筛选合适长度的路径，并将关系编码转变为与实体编码一致"""
    print('data processing')

    path_reltoent = []
    for path in tqdm(path_data, desc="Converting paths"):
        new_path = [j + ent_size if i % 2 else j for i, j in enumerate(path)]
        path_reltoent.append(new_path)

    return path_reltoent

def datatoword(data, dictionary, ent_size):
    """将实体和关系转变为词典的id"""
    unk_index = dictionary['<unk>']

    # 向量化处理，避免 Python 循环开销
    data = np.array(data, dtype=np.int64)
    rel_shifted = data[:, 1] + ent_size

    words = np.stack([
        np.vectorize(lambda x: dictionary.get(str(x), unk_index))(data[:, 0]),
        np.vectorize(lambda x: dictionary.get(str(x), unk_index))(rel_shifted),
        np.vectorize(lambda x: dictionary.get(str(x), unk_index))(data[:, 2]),
    ], axis=1)

    return words.tolist()
def word_freq(vocab):
    """优化后的词频计算函数
    Args:
        vocab: 词汇表字典，键为单词，值为词频计数

    Returns:
        word_frequency: 转换后的词频(3/4次方)
        words_counts: 原始词频计数数组
    """
    # 直接从字典值创建numpy数组，避免列表推导式
    words_counts = np.fromiter(vocab.values(), dtype=np.float32, count=len(vocab))

    # 一次性计算归一化频率和3/4次方
    # 使用np.sum的out参数避免临时数组分配
    sum_counts = np.sum(words_counts)
    word_frequency = np.divide(words_counts, sum_counts, out=words_counts.copy())
    np.power(word_frequency, 0.75, out=word_frequency)  # 3/4 = 0.75

    return word_frequency, words_counts


def set_random_seed(seed=1, cuda=False):
    """全局设置随机种子，保证可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def save_pt(obj, path):
    """保存 torch 对象"""
    torch.save(obj, path)

def load_pt(path):
    """加载 torch 对象"""
    return torch.load(path)

def log_args(args):
    """记录实验参数"""
    with open(args.test_record, "a+") as f:
        f.write(f'\n{datetime.datetime.now()}\n')
        f.write('------------------args------------------\n')
        for arg, value in vars(args).items():
            f.write(f'{arg} : {value}\n')
        f.write('------------------- end -------------------\n')
def build_corpus(words, ent_id, rel_id):
    """语料库构建函数"""
    # 1. 构建初始词汇表
    # 使用生成器表达式替代双重循环
    # 相当于将每条单个的路径都连起来得到整个path数据集的词库，里面是每个词的原始id。相当于text
    corpus_word = [str(word) for path in words for word in path]
    # 使用Counter直接获取词频统计
    vocab_counter = collections.Counter(corpus_word)
    # 获取前vocab_size-1个最常见词, 建立词典，最后一位留给不常用或者没出现过的单词
    vocab_size = len(ent_id) + len(rel_id)
    vocab = dict(vocab_counter.most_common(vocab_size))
    # 3. 补充所有实体和关系
    for e in ent_id:
        vocab.setdefault(str(e), 1)  # 频次给个最小值

    for r in rel_id:
        vocab.setdefault(str(r + len(ent_id)), 1)
    # with open(os.path.join("../data/iBKH/", "corpus_word.txt"), "w") as f:
    #     json.dump(corpus_word, f, ensure_ascii=False)


    # 计算UNK词频
    unk_count = sum(vocab_counter.values()) - sum(vocab.values())
    vocab['<unk>'] = unk_count
    word_freqs, _ = word_freq(vocab)

    # 3. 构建字典
    dictionary = {str(word): int(idx) for idx, word in tqdm(enumerate(vocab.keys()),
                                                  total=len(vocab),
                                                  desc="Building dictionary")}

    # 4. 构建反向字典
    reverse_dictionary = {int(v): str(k) for k, v in dictionary.items()}

    # 5. 转换数据路径
    # 预构建字典查找函数
    get_index = lambda w: dictionary.get(str(w), dictionary['<unk>'])

    # 使用列表推导式替代双重循环,得到的是词典里面词的id
    data_path = [[get_index(word) for word in path] for path in tqdm(words, desc="Processing data_path")]

    # 6. 构建实体索引列表和关系索引列表
    # 预计算实体和关系索引
    ent_dic = []
    rel_dic = []
    addition = []

    ent_id_values = {str(e) for e in ent_id}
    rel_shifted_values = {str(r + len(ent_id)) for r in rel_id}

    for word, idx in tqdm(dictionary.items(), desc="Generating rel_dic & ent_dic"):
        if word in rel_shifted_values:
            rel_dic.append(idx)
            print('rel_dic', word, dictionary[word])
        elif word in ent_id_values:
            ent_dic.append(idx)
        else:
            addition.append(idx)
            print('addition', word, dictionary[word])


    print(f'ent_dic length: {len(ent_dic)}')
    print(f'rel_dic length: {len(rel_dic)}')
    print(f'addition length: {len(addition)}')
    return data_path, dictionary, reverse_dictionary, rel_dic, ent_dic, word_freqs

def main():
    args = parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
    print('CUDA available:', torch.cuda.is_available())

    # 固定随机种子
    set_random_seed(1, torch.cuda.is_available())


    # ==== 1. 载入实体/关系映射 ====
    entity2id = read_entity_from_id(args.data)
    relation2id = read_relation_from_id(args.data)
    ent_num, rel_num = len(entity2id), len(relation2id)
    with open(os.path.join(args.data, "entity2id.txt"), "w") as f:
        json.dump(entity2id, f, ensure_ascii=False)
    print('Number of entities:', ent_num, 'relations:', rel_num)


    # ==== 2. 生成训练数据 ====
    train_data, paths = load_saved_data(args) ###train_data=[[e1_id,r1_id,e2_id],...],paths=[[e1_id,r1_id,e2_id,r2_id,e3_id],...]
    print('path generated')
    # ==== 3. 生成词典数据 ====
    if os.path.exists(os.path.join(args.data, "path_corpus.pt")):
        package = torch.load(os.path.join(args.data, "path_corpus.pt"))
        path_corpus = package["path_corpus"]
        dictionary = package["dictionary"]
        reverse_dictionary = package["reverse_dictionary"]
        ent_dic = package["ent_dic"]
        rel_dic = package["rel_dic"]
        word_freqs = package["word_freqs"]
    else:
        path_store_train_reltoent = path_rel2ent(paths, ent_num)
        path_corpus, dictionary, reverse_dictionary, rel_dic, ent_dic, word_freqs = build_corpus(
            path_store_train_reltoent, entity2id.values(), relation2id.values()
        )
        save_dict = {
            "path_corpus": path_corpus,
            "dictionary": dictionary,
            "reverse_dictionary": reverse_dictionary,
            "ent_dic": ent_dic,
            "rel_dic": rel_dic,
            "word_freqs": word_freqs
        }
        torch.save(save_dict, os.path.join(args.data, "path_corpus.pt"))
    print('path_corpus has been prepared!')
    print(f'rel_dic length: {len(rel_dic)}',rel_dic)
    print(f'ent_dic length: {len(ent_dic)}')
    with open(os.path.join(args.data, "ent_dic.txt"), "w") as f:
        json.dump(ent_dic, f, ensure_ascii=False)
    with open(os.path.join(args.data, "rel_dic.txt"), "w") as f:
        json.dump(rel_dic, f, ensure_ascii=False)
    with open(os.path.join(args.data, "dictionary.txt"), "w") as f:
        json.dump(dictionary, f, ensure_ascii=False)
    with open(os.path.join(args.data, "reverse_dictionary.txt"), "w") as f:
        json.dump(reverse_dictionary, f, ensure_ascii=False)

    # ==== 4. 处理训练数据 ====
    if os.path.exists(os.path.join(args.data, "input_word_kg2vec.pt")):
        input_word = torch.load(os.path.join(args.data, "input_word_kg2vec.pt"))
    else:
        input_word = datatoword(train_data, dictionary, ent_num)
        save_pt(input_word, os.path.join(args.data, "input_word_kg2vec.pt"))
    print('input_word has been processed!')

    # ==== 5. 准备 DataLoader ====
    print('Processing train labels...')
    np.random.shuffle(path_corpus)
    print('path_corpus length',len(path_corpus)) #28415091

    if os.path.exists(os.path.join(args.data, "dataset_kg2vec_training.pt")):
        dataset = torch.load(os.path.join(args.data, "dataset_kg2vec_training.pt"))
    else:
        dataset = WordEmbeddingDataset_KG2vec(path_corpus,
                                              torch.cuda.is_available(),
                                              args.sampled_num)
        save_pt(dataset, os.path.join(args.data, "dataset_kg2vec_training.pt"))
    print('dataset for kg2vec training has been generated!')
    print("dataset length:",len(dataset))

    subset_size = int(len(dataset) * 0.001)
    sampler = RandomSubsetSampler(dataset, subset_size)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size_kg2vec,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )

    # ==== 6. 日志记录 ====
    log_args(args)

    # ==== 7. 训练模型 ====
    # 调用多卡版本的训练函数
    embedding_weights = kg2vec_fast(
        args.data, ent_num, rel_num, args.EMBEDDING_SIZE,
        dataloader, args.walk_num, args.walk_num_edge,
        reverse_dictionary, args.lr, args.epoch_num,
        ent_dic, rel_dic, input_word, args.gpu
    )

    ent_embed, rel_embed = embedding_weights

    save_pt(ent_embed, os.path.join(args.data, "ent_embeddings_kg2vec.pt"))
    save_pt(rel_embed, os.path.join(args.data, "rel_embeddings_kg2vec.pt"))

    print(f"Embedding shapes - Entities: {np.array(ent_embed).shape}, Relations: {np.array(rel_embed).shape}")



if __name__ == "__main__":

    #
    # mp.set_start_method('spawn', force=True)

    main()
