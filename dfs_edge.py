from collections import defaultdict
from tqdm import tqdm
import random
class Personalized:
    def __init__(self, nx_G, mask, mask_rel, rr, dr, walks_num, walks_num_edge,rel_dic):
        self.G = nx_G
        self.mask = mask      # {node: [(neighbor, rel), ...]}
        self.rel_ents = mask_rel  # {rel: [ent1,ent2, ...]}
        self.walks_num = walks_num
        self.walks_num_edge = walks_num_edge
        self.rr = rr   # {rel: [rel1,rel2, ...]}
        self.dr = dr   # {ent: [rel1,rel2, ...]}
        self.rel_dic = rel_dic

    def _dfs_core(self, start_node, mask_neighbors):
        """DFS 核心逻辑：从 start_node 出发，限制长度为 walks_num"""
        stack = [start_node]
        seen = {start_node}
        walks_node = []
        while stack:
            vertex = stack.pop()
            # 缓存 neighbors 引用，避免重复属性查找
            neighbors = self.G[vertex]
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
        mask_neighbors = {t[0] for t in self.mask[start_node]}
        return self._dfs_core(start_node, mask_neighbors)

    def dfs_edge(self, rel):

        rr_edge = self.rr[rel]
        cond_list = []
        for d in self.rel_ents[rel]:
            dr_set = self.dr[d]
            cond_list.extend(list(dr_set-rr_edge))

        if len(cond_list) >= self.walks_num_edge:
            cond_list = random.sample(cond_list, self.walks_num_edge)
        elif len(set(self.rel_dic)-rr_edge) >= self.walks_num_edge:
            cond_list = random.sample(list(set(self.rel_dic)-rr_edge), self.walks_num_edge)
        return cond_list


    def intermediate(self):
        candidate = defaultdict(list)
        # 边 DFS - 添加进度条
        for edge in tqdm(self.G.edges(), desc="Processing edges"):
            rel = self.G.edges[edge]["name"]
            candidate[rel].extend(self.dfs_edge(rel))
        # 节点 DFS - 添加进度条
        for node in tqdm(self.G.nodes(), desc="Processing nodes"):
            candidate[node].extend(self.dfs(node))
        return candidate


def candidate_choose(nx_Graph, mask, mask_rel, rr, dr, walk_num, walks_num_edge, rel_dic):
    G = Personalized(nx_Graph, mask, mask_rel, rr, dr, walk_num, walks_num_edge, rel_dic)
    candidates = G.intermediate()
    return candidates


