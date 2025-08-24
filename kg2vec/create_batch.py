import csv
import random
from collections import defaultdict
from torch.utils.data import Sampler
import random
from tqdm import tqdm


def build_graph(triples_file, entity2id, relation2id):
    """根据三元组文件构建图"""
    graph = defaultdict(list)
    with open(triples_file, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for h, r, t in reader:
            h = h.lower()
            t = t.lower()
            if h not in entity2id or t not in entity2id or r not in relation2id:
                continue
            h_id, r_id, t_id = entity2id[h], relation2id[r], entity2id[t]
            graph[h_id].append((r_id, t_id))

    return graph

def generate_paths(graph, n):
    """
    遍历图生成所有长度为n的路径
    每条路径 alternates: entity, relation, entity, relation, ...
    """
    paths = set()

    def dfs(path, visited, remaining_len):
        if remaining_len == 0:
            paths.add(tuple(path))
            return

        last_ent = path[-1]
        if last_ent not in graph:
            return

        for r_id, next_ent in graph[last_ent]:
            if next_ent in visited:  # 避免环
                continue
            dfs(path + [r_id, next_ent], visited | {next_ent}, remaining_len - 2)

    for start_ent in graph.keys():
        dfs([start_ent], {start_ent}, n - 1)

    return [list(p) for p in paths]

class Personalized:
    def __init__(self, mask, mask_rel, rr, dr, walks_num, walks_num_edge, rel_dic):
        """
        graph: defaultdict(list)，邻接表 {h: [(r, t), ...]}
        """
        self.graph = mask              # {node: [(neighbor, rel), ...]}
        self.rel_ents = mask_rel      # {rel: [ent1, ent2, ...]}
        self.walks_num = walks_num
        self.walks_num_edge = walks_num_edge
        self.rr = rr                  # {rel: [rel1, rel2, ...]}
        self.dr = dr                  # {ent: [rel1, rel2, ...]}
        self.rel_dic = rel_dic

    def _dfs_core(self, start_node, mask_neighbors):
        """DFS 核心逻辑：从 start_node 出发，限制长度为 walks_num"""
        stack = [start_node]
        seen = {start_node}
        walks_node = []
        while stack:
            vertex = stack.pop()
            # 获取 neighbors（只要尾实体，不关心关系）
            neighbors = [t[0] for t in self.graph.get(vertex, [])]
            for w in neighbors:
                if w not in seen:
                    stack.append(w)
                    seen.add(w)
            # 不要加入起点，也不要加入 mask 里的节点
            if vertex != start_node and vertex not in mask_neighbors:
                walks_node.append(vertex)
            if len(walks_node) >= self.walks_num:
                break
        return walks_node

    def dfs(self, start_node):
        mask_neighbors = {t[0] for t in self.graph[start_node]}
        return self._dfs_core(start_node, mask_neighbors)

    def dfs_edge(self, rel):
        rr_edge = set(self.rr[rel])
        cond_list = []
        for d in self.rel_ents[rel]:
            dr_set = set(self.dr[d])
            cond_list.extend(list(dr_set - rr_edge))

        if len(cond_list) >= self.walks_num_edge:
            cond_list = random.sample(cond_list, self.walks_num_edge)
        elif len(set(self.rel_dic) - rr_edge) >= self.walks_num_edge:
            cond_list = random.sample(list(set(self.rel_dic) - rr_edge), self.walks_num_edge)
        return cond_list

    def intermediate(self):
        candidate = defaultdict(list)
        # 边 DFS
        for r in tqdm(self.rel_ents, desc="Processing edges"):
            candidate[r].extend(self.dfs_edge(r))

        print(len(self.graph))

        # with open(os.path.join("../data/iBKH/", "mask.txt"), "w") as f:
        #     json.dump(self.graph, f, ensure_ascii=False)

        # 节点 DFS
        for h in tqdm(self.graph, desc="Processing nodes"):

            candidate[h].extend(self.dfs(h))
        return candidate

def candidate_choose(mask, mask_rel, rr, dr, walk_num, walks_num_edge, rel_dic):

    G = Personalized(mask, mask_rel, rr, dr, walk_num, walks_num_edge, rel_dic)
    candidates = G.intermediate()
    return candidates


class RandomSubsetSampler(Sampler):
    def __init__(self, data_source, subset_size):
        self.data_source = data_source
        self.subset_size = subset_size

    def __iter__(self):
        indices = random.sample(range(len(self.data_source)), self.subset_size)
        return iter(indices)

    def __len__(self):

        return self.subset_size
