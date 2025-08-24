import os
import networkx as nx
import numpy as np
from collections import defaultdict
from tqdm import tqdm   # pip install tqdm

def read_entity_from_id(file):
    """ entity2id:{ent:id} (lowercased, reindexed) """
    entity2id = {}

    with open(f"{file}entities.txt", "r") as f:
        for line in f:
            entity = line.strip()
            if entity:  # 跳过空行
                entity2id[entity.lower()] = None  # 去重（大小写合并）

    # 重新编号，保证连续
    entity2id = {ent: idx for idx, ent in enumerate(entity2id.keys())}
    return entity2id
def read_relation_from_id(file):
    """ relation2id:{rel:id} """
    relation2id = {}

    with open(f"{file}relations.txt", "r") as f:
        for idx, line in enumerate(f):
            relation = line.strip()
            if relation:  # 跳过空行
                relation2id[relation] = idx
    # relation_lower = {k.lower(): v for k, v in relation2id.items()}
    return relation2id
def parse_line(line):###定义三元组解析函数，解析出头尾实体和关系
    line = line.strip().split()
    e1, relation, e2 = line[0].strip(), line[1].strip(), line[2].strip()
    return e1, relation, e2

def load_data(filename, entity2id, relation2id, is_unweighted=False, directed=True):
    """加载知识图谱数据并构建三元组

    Args:
        filename: 数据文件路径
        entity2id: 实体到ID的映射字典
        relation2id: 关系到ID的映射字典
        is_unweigted: 是否是无权图
        directed: 是否是有向图

    Returns:
        triples_data: 三元组数据列表
    """
    with open(filename) as f:
        lines = f.readlines()

    triples_data = []

    for line in lines:

        e1, relation, e2 = parse_line(line)
        e1_id, e2_id = entity2id[e1.lower()], entity2id[e2.lower()]
        rel_id = relation2id[relation]
        triples_data.append((e1_id, rel_id, e2_id))

    return triples_data
def build_data(path, is_unweighted=False, directed=True):
    """Build knowledge graph data structures and calculate Bernoulli sampling probabilities.
    Args:
        path: Directory path containing the dataset files
        is_unweighted: Whether to create unweighted adjacency matrix
        directed: Whether to create directed adjacency matrix
    Returns:
        Tuple containing:
        - Training, validation, and test data with adjacency matrices
        - Entity and relation mappings
        - Bernoulli sampling probabilities
        - Unique entities in training set
    """
    # Load entity and relation mappings
    entity2id = read_entity_from_id(path)
    relation2id = read_relation_from_id(path)

    # Load and process data files
    data_files = [
        ('training_triplets.tsv', 'train'),
        ('validation_triplets.tsv', 'valid'),
        ('testing_triplets.tsv', 'test')
    ]

    results = {}
    for filename, key in data_files:
        triples = load_data(
            os.path.join(path, filename),
            entity2id,
            relation2id,
            is_unweighted,
            directed
        )
        results[key] = triples


    return results['train'],results['valid'],results['test'], entity2id, relation2id

def construct_graph(triples):
    print("Construct the graph for training")
    G = nx.Graph()  # 建立一个空的无向图 G

    for edge in tqdm(triples, desc="Adding edges to graph"):
        x, y, z = edge[0], edge[2], edge[1]
        G.add_edge(x, y, name=z)

    return G
def load_item_pop(X_train, ent_dic, rel_dic):
    """
    根据训练三元组计算节点和边缘流行度指标（带进度条）。
    """
    print('----load_item_pop begin----')
    X_train = np.asarray(X_train, dtype=int)

    ee = defaultdict(list)
    re = defaultdict(set)
    er = defaultdict(set)

    # ------- 构建 dd 和 rd (带进度条) -------
    for h, r, t in tqdm(X_train, desc="Building dd & rd"):
        ee[h].append((t, r))
        re[r].add(h)
        re[r].add(t)
        er[h].add(r)

    num_ent, num_rel = len(ent_dic), len(rel_dic)
    node_deg = {k: 1.0 / num_ent for k in ent_dic}
    edge_deg = {k: 1.0 / num_rel for k in rel_dic}

    # ------- 构建 relation-relation 邻居 (带进度条) -------
    rr = defaultdict(set)
    for r, ent_list in tqdm(re.items(), desc="Building rr"):
        for e in ent_list:
            rr[r].update(er[e])

    # rr = {k: list(v) for k, v in rr.items()}

    print(f'edge_deg {len(edge_deg)}')
    return node_deg, edge_deg, ee, re, rr, er

def load_edges(filename):
    # 存储的是边=<user,item>=[(node1,rel1,node2),(node3,rel2,node4),...]
    edges = list()
    with open(filename, 'r') as f:
        for line in f:
            user,rel,item = line.strip().split('\t')#user, item指的是node1,rel和node2
            edges.append((int(user),int(rel),int(item)))
    return edges