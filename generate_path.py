from tqdm import tqdm
import os

def recover_entities_for_guu_paths(ent_nighbors):
    """Recover entity paths from neighbor relationships and store them in a list format.
    Args:
        ent_nighbors: Dictionary containing entity neighbor relationships
    Returns:
        List of paths, where each path is represented as a list of alternating entities and relations
    """
    print('recovering entity path file')
    path_store = []

    # Use items() instead of keys() to avoid repeated dictionary lookups
    for s_ent, neighbors in tqdm(ent_nighbors.items()):
        line_count = 0

        for path_distance, paths in neighbors.items():
            for rel_list, ent_list in paths:
                # Validate path length
                if len(ent_list) != len(rel_list):
                    print('Error: the length of ent_list and rel_list are not equal')
                    exit(1)

                # Build path
                path_single = [s_ent]
                for rel, ent in zip(rel_list, ent_list):
                    path_single.extend([rel, ent])

                path_store.append(path_single)
                line_count += 1

                # Progress reporting
                if line_count % 1000 == 0:
                    print(f'{line_count} ....')

    print('\t\t\t\t ..........over')
    return path_store


def keylist_2_valuelist(keylist, dic, start_index=0):
    """将keylist中的key转换为字典中的value，若不存在则自动分配新ID"""
    value_list = []
    for key in keylist:
        if key not in dic:
            dic[key] = len(dic) + start_index
        value_list.append(dic[key])
    return value_list


def add_tuple2tailset(ent_path, one_path, tuple2tailset):
    """更新(头实体,关系)→尾实体集合的映射"""
    size = len(one_path)
    if len(ent_path) != size + 1:
        print(f'len(ent_path)!=len(one_path)+1: {len(ent_path)} {size}')
        exit(1)

    for i in range(size):
        tuple = (ent_path[i], one_path[i])
        tail = ent_path[i + 1]
        tuple2tailset.setdefault(tuple, set()).add(tail)
    return tuple2tailset


def add_rel2tailset(ent_path, one_path, rel2tailset):
    """更新关系→尾实体集合的映射"""
    size = len(one_path)
    if len(ent_path) != size + 1:
        print(f'len(ent_path)!=len(one_path)+1: {len(ent_path)} {size}')
        exit(1)

    for i in range(size):
        rel = one_path[i]
        tail = ent_path[i + 1]
        rel2tailset.setdefault(rel, set()).add(tail)
    return rel2tailset


def add_ent2relset(ent_path, one_path, ent2relset, maxSetSize):
    """更新实体→关系集合的映射并返回最大关系集合大小"""
    size = len(one_path)
    if len(ent_path) != size + 1:
        print(f'len(ent_path)!=len(one_path)+1: {len(ent_path)} {size}')
        exit(1)

    current_max = maxSetSize
    for i in range(size):
        ent_id = ent_path[i + 1]
        rel_id = one_path[i]
        relset = ent2relset.setdefault(ent_id, set())
        relset.add(rel_id)
        if len(relset) > current_max:
            current_max = len(relset)
    return ent2relset, current_max


def load_guu_data_v2(path_store, maxPathLen):
    """加载并处理路径数据，返回处理后的训练测试数据及各类映射字典"""
    relation_str2id = {}
    ent_str2id = {}
    tuple2tailset = {}
    rel2tailset = {}
    ent2relset = {}
    max_relset_size = 0

    train_data = []
    test_data = []

    for file_id, file_paths in enumerate(path_store):
        paths_store = []
        ents_store = []
        masks_store = []

        for path in file_paths:
            # 分离实体和关系
            ent_list = path[::2]
            rel_list = path[1::2]

            if len(ent_list) != len(rel_list) + 1:
                print(f'Invalid path: entities {len(ent_list)}, relations {len(rel_list)}')
                exit(1)

            # 转换为ID
            ent_path = keylist_2_valuelist(ent_list, ent_str2id, 0)
            one_path = [relation_str2id.setdefault(r, len(relation_str2id) + 1) for r in rel_list]

            # 更新各类映射
            tuple2tailset = add_tuple2tailset(ent_path, one_path, tuple2tailset)
            rel2tailset = add_rel2tailset(ent_path, one_path, rel2tailset)
            ent2relset, max_relset_size = add_ent2relset(ent_path, one_path, ent2relset, max_relset_size)

            # 路径padding处理
            valid_size = len(one_path)
            pad_size = max(maxPathLen - valid_size, 0)

            padded_path = [0] * pad_size + one_path[-maxPathLen:]
            padded_ent = ent_path[:1] * (pad_size + 1) + ent_path[-maxPathLen if valid_size > maxPathLen else 1:]
            padded_mask = [0.0] * pad_size + [1.0] * min(valid_size, maxPathLen)

            if file_id == 0:  # 训练数据
                train_data.append((padded_path, padded_mask, padded_ent))
            else:  # 测试数据
                test_data.append((padded_path, padded_mask, padded_ent))

    # 分离训练测试数据
    train_paths, train_masks, train_ents = zip(*train_data) if train_data else ([], [], [])
    test_paths, test_masks, test_ents = zip(*test_data) if test_data else ([], [], [])

    print(f'Load complete: train={len(train_paths)}, test={len(test_paths)}, '
          f'tuple2tailset={len(tuple2tailset)}, max_relset={max_relset_size}')

    return ((train_paths, train_masks, train_ents), (test_paths, test_masks, test_ents)), \
        ent_str2id, relation_str2id, tuple2tailset, rel2tailset, ent2relset, max_relset_size
