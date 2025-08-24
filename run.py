import os
import time
import datetime
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import argparse
import collections
import random
from itertools import islice
import numpy as np
import torch
import torch.utils.data as tud
from create_batch import build_graph, generate_paths
from preprocessing import read_entity_from_id, read_relation_from_id, build_data
from generate_path import recover_entities_for_guu_paths
from WordEmbeddingDataset import WordEmbeddingDatasetOptimized
from torch.utils.data.distributed import DistributedSampler
from dfs_edge import candidate_choose
from preprocessing import load_item_pop, construct_graph
from MCN_sampling import MCNS_gpu
from EmbeddingModel import EmbeddingModel_Optimized

# ====== 这里是你原来的工具函数/数据处理函数 ======
# parse_args(), set_random_seed(), read_entity_from_id(), read_relation_from_id(),
# load_saved_data(), build_corpus(), datatoword(), save_pt(),
# WordEmbeddingDataset_KG2vec, EmbeddingModel_Optimized, load_item_pop(),
# construct_graph(), candidate_choose(), MCNS_gpu()
# ====================================================
def parse_args():
    parser = argparse.ArgumentParser(description="KG2Vec Training Arguments")

    # Network arguments
    parser.add_argument("-data", "--data", default="../data/iBKH/",
                        help="data directory")
    parser.add_argument("-pre_emb", "--pretrained_emb", type=bool, default=False,
                        help="Use pretrained embeddings")
    parser.add_argument("-emb_size", "--EMBEDDING_SIZE", type=int, default=50,
                        help="Size of embeddings (if pretrained not used)")
    parser.add_argument("-skip_window", "--skip_window", type=int, default=2,
                        help="length of context window")
    parser.add_argument("-EPOCHS", "--epoch_num", type=int, default=4,
                        help="number of training epochs of kg2vec")
    parser.add_argument("-sampled_num", "--sampled_num", type=int, default=1,
                        help="number of negative samples")
    parser.add_argument("-walk_num", "--walk_num", type=int, default=2,
                        help="number of walk steps")#fb:10,wn:6
    parser.add_argument("-walk_num_edge", "--walk_num_edge", type=int, default=1,
                        help="number of edges through a walk")#fb:6,wn:3
    parser.add_argument("-lr", "--lr", type=float, default=1e-3,
                        help="learning rate") #FB: 1e-3
    parser.add_argument("-BATCH_SIZE", "--batch_size_kg2vec", type=int, default=64,
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

    parser.add_argument("--local_rank", type=int, default=1)

    return parser.parse_args()

def load_data(args):
    train_data, validation_data, test_data, entity2id, relation2id = build_data(
        args.data, is_unweighted=False, directed=True)
    if os.path.exists(os.path.join(args.data, "training_graph.tsv")):
        graph = torch.load(os.path.join(args.data, "training_graph.tsv"))
    else:
        graph = build_graph(os.path.join(args.data, "training_triplets.tsv"), entity2id, relation2id)
        torch.load(graph, os.path.join(args.data, "training_graph.tsv"))
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

def data_form(path_data, ent_size):
    """筛选合适长度的路径，并将关系编码转变为与实体编码一致"""
    print('data processing')

    # 将关系统一加 ent_size
    # path_reltoent = [
    #     [int(j) + ent_size if i % 2 else j for i, j in enumerate(path)]
    #     for path in path_data
    # ]
    path_reltoent = []
    for path in tqdm(path_data, desc="Converting paths"):
        new_path = [int(j) + ent_size if i % 2 else j for i, j in enumerate(path)]
        path_reltoent.append(new_path)

    return path_reltoent

def build_corpus(words, ent_id, rel_id):
    """语料库构建函数"""
    # 1. 构建初始词汇表
    # 使用生成器表达式替代双重循环
    corpus_word = [word for path in words for word in path] # 相当于将每条单个的路径都连起来得到整个path数据集的词库，里面是每个词的原始id。相当于text

    # 使用Counter直接获取词频统计
    vocab_counter = collections.Counter(corpus_word)
    # 获取前vocab_size-1个最常见词, 建立词典，最后一位留给不常用或者没出现过的单词
    vocab_size = len(ent_id) + len(rel_id) + 1
    vocab = dict(vocab_counter.most_common(vocab_size - 1))

    # 计算UNK词频
    unk_count = sum(vocab_counter.values()) - sum(vocab.values())
    vocab['<unk>'] = unk_count
    word_freqs, _ = word_freq(vocab)

    # 2. 构建字典和反向字典
    # 直接构建字典，避免中间列表
    dictionary = {word: idx for idx, word in tqdm(enumerate(vocab.keys()),
                                                  total=len(vocab),
                                                  desc="Building dictionary")}
    last_idx = len(dictionary) - 1

    # 3. 添加实体和关系到字典
    # 使用集合操作加速查找
    existing_keys = set(dictionary.keys())

    # 添加未收录的实体
    for e in ent_id:
        if e not in existing_keys:
            last_idx += 1
            dictionary[e] = last_idx
    for e in tqdm(ent_id, desc="Processing entities for dictionary"):
        if e not in existing_keys:
            last_idx += 1
            dictionary[e] = last_idx

    # 添加未收录的关系
    for r in tqdm(rel_id, desc="Processing relations for dictionary"):
        r_shifted = str(int(r) + len(ent_id))
        if r_shifted not in existing_keys:
            last_idx += 1
            dictionary[r_shifted] = last_idx

    # 4. 转换数据路径
    # 预构建字典查找函数

    get_index = lambda w: dictionary.get(w, dictionary['<unk>'])  # 0对应UNK

    # 使用列表推导式替代双重循环,得到的是词典里面词的id
    data_path = [[get_index(word) for word in path] for path in tqdm(words, desc="Processing data_path")]

    # 5. 构建反向字典和分类字典
    reverse_dictionary = {v: k for k, v in tqdm(dictionary.items(),desc="Reversing dictionary")}

    # 预计算实体和关系索引
    ent_dic = []
    rel_dic = []
    ent_id_values = set(ent_id)
    rel_shifted_values = {str(int(r) + len(ent_id)) for r in rel_id}

    for word, idx in tqdm(dictionary.items(),desc="Generating rel_dic & ent_dic"):
        if word == '<unk>':
            ent_dic.append(idx)
        elif word in rel_shifted_values:
            rel_dic.append(idx)
        elif word in ent_id_values:
            ent_dic.append(idx)

    print(f'rel_dic length: {len(rel_dic)}')
    return data_path, dictionary, reverse_dictionary, rel_dic, ent_dic, word_freqs

def datatoword(data, dictionary, ent_size):
    """将实体和关系转变为词典的id"""
    unk_index = dictionary['<unk>']

    # 向量化处理，避免 Python 循环开销
    data = np.array(data, dtype=np.int64)
    rel_shifted = data[:, 1] + ent_size

    words = np.stack([
        np.vectorize(lambda x: dictionary.get(x, unk_index))(data[:, 0]),
        np.vectorize(lambda x: dictionary.get(str(x), unk_index))(rel_shifted),
        np.vectorize(lambda x: dictionary.get(x, unk_index))(data[:, 2]),
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

def kg2vec_fast(
        data_path, ent_size, rel_size, EMBEDDING_SIZE, dataloader, walk_num, walk_num_edge,
        reverse_dictionary, LEARNING_RATE, NUM_EPOCHS, ent_dic, rel_dic, input_word, device
):
    model = EmbeddingModel_Optimized(ent_size, rel_size, EMBEDDING_SIZE, device,
                                     reverse_dictionary, ent_dic).to(device)
    model = DDP(model, device_ids=[device.index], output_device=device.index)

    # 预加载必要数据到 GPU
    q_1_dict, q_2_dict, mask, mask_rel, rr, dr = load_item_pop(input_word, ent_dic, rel_dic)
    G = construct_graph(input_word)
    candidates = candidate_choose(G, mask, mask_rel, rr, dr, walk_num, walk_num_edge, rel_dic)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08)
    scaler = GradScaler()

    N_steps = 2
    N_negs = 1
    loss_log_path = os.path.join(data_path, 'loss_log.txt')

    if dist.get_rank() == 0:
        with open(loss_log_path, 'w') as f:
            f.write('epoch,avg_loss,epoch_time\n')

    for epoch in tqdm(range(NUM_EPOCHS), desc="Training"):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        neg_labels = None
        dataloader.sampler.set_epoch(epoch)  # 对 DistributedSampler 必须

        for batch_idx, (input_labels, pos_labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            input_labels = input_labels.to(device, non_blocking=True)
            pos_labels = pos_labels.to(device, non_blocking=True)

            start_given = neg_labels if batch_idx > 0 else None
            neg_labels = MCNS_gpu(
                model, candidates, start_given, q_1_dict, q_2_dict,
                N_steps, input_labels, ent_dic, reverse_dictionary, device
            )

            optimizer.zero_grad(set_to_none=True)
            with autocast():  # 混合精度
                loss = model(
                    input_labels,
                    pos_labels,
                    neg_labels.view(pos_labels.size(0), N_negs),
                    ent_dic,
                    reverse_dictionary
                ).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(dataloader)
        if dist.get_rank() == 0:
            print(f"Epoch: {epoch + 1:02d}, Loss: {avg_loss:.5f}, Time: {epoch_time:.2f}s")
            with open(loss_log_path, 'a') as f:
                f.write(f'{epoch + 1},{avg_loss:.5f},{epoch_time:.2f}\n')

    # 只在 rank 0 返回权重
    if dist.get_rank() == 0:
        return model.module.input_embeddings()
    else:
        return None
def kg2vec(data_path, ent_size, rel_size, EMBEDDING_SIZE, dataloader,
           LEARNING_RATE, NUM_EPOCHS, ent_dic, reverse_dictionary):

    device = torch.device(f'cuda:{gpu}')
    model = EmbeddingModel_Optimized(ent_size, rel_size, EMBEDDING_SIZE, device,
                                     reverse_dictionary, ent_dic).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08
    )

    # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    print('begining training')

    loss_log_path = data_path + '/loss_log.txt'
    with open(loss_log_path, 'w') as f:
        f.write('epoch,avg_loss,epoch_time\n')

    for epoch in tqdm(range(NUM_EPOCHS), desc="Training"):
        total_loss = 0.0
        start_time = time.time()

        for batch_idx, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):

            input_labels = input_labels.to(device, non_blocking=True)
            pos_labels = pos_labels.to(device, non_blocking=True)
            neg_labels = neg_labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            loss = model(input_labels,
                         pos_labels,
                         neg_labels.view(pos_labels.size(0), N_negs),
                         ).mean()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 100 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"{datetime.datetime.now()} Batch:[{batch_idx}/{len(dataloader)}] Loss:{avg_loss:.5f}")

        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch: {epoch + 1:02d}, Loss: {avg_loss:.5f}, Time: {epoch_time:.2f}s")

        with open(loss_log_path, 'a') as f:
            f.write(f'{epoch + 1},{avg_loss:.5f},{epoch_time:.2f}\n')

    return model.input_embeddings()

def main():
    args = parse_args()

    local_rank = args.local_rank

    # ==== 初始化分布式 ====
    dist.init_process_group(backend='nccl')
    # local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    # 固定随机种子
    set_random_seed(1, torch.cuda.is_available())

    # ==== 1. 载入实体/关系映射 ====
    entity2id = read_entity_from_id(args.data)
    relation2id = read_relation_from_id(args.data)
    ent_num, rel_num = len(entity2id), len(relation2id)
    print('Number of entities:', ent_num)
    print('Number of relations:', rel_num)

    # ==== 2. 数据准备 ====
    train_data, paths = load_saved_data(args)
    if os.path.exists(os.path.join(args.data, "path_corpus.pt")):
        package = torch.load(os.path.join(args.data, "path_corpus.pt"))
        path_corpus = package["path_corpus"]
        dictionary = package["dictionary"]
        reverse_dictionary = package["reverse_dictionary"]
        ent_dic = package["reverse_dictionary"]
        rel_dic = package["reverse_dictionary"]
        word_freqs = package["word_freqs"]
    else:
        # 生成词典
        path_store_train_reltoent = data_form(paths, ent_num)
        path_corpus, dictionary, reverse_dictionary, rel_dic, ent_dic, word_freqs = build_corpus(
            path_store_train_reltoent, entity2id.values(), relation2id.values()
        )
        save_dict = {
            "path_corpus": path_corpus,
            "dictionary": dictionary,
            "reverse_dictionary": reverse_dictionary,
            "ent_dic": ent_dic,
            "rel_dic": rel_dic
        }
        torch.save(save_dict, os.path.join(args.data, "path_corpus.pt"))

    # ==== 3. 处理训练数据 ====
    if os.path.exists(os.path.join(args.data, "input_word_kg2vec.pt")):
        input_word = torch.load(os.path.join(args.data, "input_word_kg2vec.pt"))
    else:
        input_word = datatoword(train_data, dictionary, ent_num)
        save_pt(input_word, os.path.join(args.data, "input_word_kg2vec.pt"))
    print('input_word has been processed!')

    # ==== 4. DataLoader ====
    if os.path.exists(os.path.join(args.data, "dataset_kg2vec_training.pt")):
        dataset = torch.load(os.path.join(args.data, "dataset_kg2vec_training.pt"))
    else:
        dataset = WordEmbeddingDatasetOptimized(path_corpus, dictionary, torch.cuda.is_available(),
                                                args.sampled_num, reverse_dictionary, rel_dic, ent_dic,
                                                word_freqs)
        save_pt(dataset, os.path.join(args.data, "dataset_kg2vec_training.pt"))

    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size_kg2vec, sampler=sampler, num_workers=2)

    # ==== 5. 日志记录 ====
    log_args(args)

    # ==== 6. 训练模型 ====
    # embedding_weights = kg2vec_fast(
    #     args.data, ent_num + 1, rel_num, args.EMBEDDING_SIZE,
    #     dataloader, args.walk_num, args.walk_num_edge,
    #     reverse_dictionary, args.lr, args.epoch_num,
    #     ent_dic, rel_dic, input_word, device
    # )
    embedding_weights = kg2vec(
        args.data, ent_num + 1, rel_num, args.EMBEDDING_SIZE,
        dataloader, args.lr, args.epoch_num,
        ent_dic,reverse_dictionary
    )

    if dist.get_rank() == 0:
        ent_embed, rel_embed = embedding_weights
        save_pt(ent_embed, os.path.join(args.data, "ent_embeddings_kg2vec.pt"))
        save_pt(rel_embed, os.path.join(args.data, "rel_embeddings_kg2vec.pt"))
        print(f"Embedding shapes - Entities: {np.array(ent_embed).shape}, Relations: {np.array(rel_embed).shape}")


if __name__ == "__main__":
    main()