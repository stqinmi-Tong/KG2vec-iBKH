import torch
import pandas as pd
import numpy as np
import os
import csv
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as fn
from KG_link_pred import transE_l2,DistMult, transR, complEx
def generate_hypothesis(target_entity, candidate_entity_type, relation_type,
                        embedding_folder, method='transE_l2',
                        kg_folder='../data/iBKH',
                        entity_folder='id2ent.csv',
                        relation_folder='id2rel.csv',
                        triplet_folder='../data/iBKH/triplets',
                        without_any_rel=False, topK=100, save_path='output', save=True):
    # load entity vocab
    entities = {}
    for e in ['anatomy', 'disease', 'drug', 'dsp', 'gene',
              'molecule', 'pathway', 'sdsi', 'side_effect',
              'symptom', 'tc']:
        e_df = pd.read_csv(kg_folder + '/entity/' + e + '_vocab.csv', header=0, low_memory=False)
        if e == 'gene':
            e_df = e_df.rename(columns={'symbol': 'name'})
        if e == 'molecule':
            e_df = e_df.rename(columns={'chembl_id': 'name'})

        entities[e] = e_df

    # load embeddings from .pt files (PyTorch tensors)
    ent_embeddings = torch.load(
        os.path.join(embedding_folder, 'ent_embeddings_kg2vec_200.pt'))  # Load entity embeddings
    rel_embeddings = torch.load(
        os.path.join(embedding_folder, 'rel_embeddings_kg2vec_200.pt'))  # Load relation embeddings

    # load entity and relation embedding maps (IDs)
    entity_emb_map = pd.read_csv(os.path.join(kg_folder, entity_folder))
    entity_emb_map.columns = ['id', 'primary']
    rel_emb_map = pd.read_csv(os.path.join(kg_folder, relation_folder))
    rel_emb_map.columns = ['rid', 'relation']

    # get target entity vocab (PD)
    target_entity_vocab = pd.DataFrame()
    for e in entities:
        e_df = entities[e][['primary', 'name']]
        target_entity_vocab = pd.concat([target_entity_vocab, e_df[e_df['name'].isin(target_entity)]])

    # ---- 对齐 primary 名称（去掉空格 + 小写）----
    entity_emb_map['primary'] = entity_emb_map['primary'].str.strip().str.lower()
    target_entity_vocab['primary'] = target_entity_vocab['primary'].str.strip().str.lower()

    # find target entity IDs
    target_entity_vocab = pd.merge(target_entity_vocab, entity_emb_map, on='primary', how='left')
    target_entity_ids = target_entity_vocab['id'].tolist()
    print(target_entity_vocab.head())
    #         primary                            name    id
    # 0  doid:0060892  late onset parkinson's disease  1725
    # 1    doid:14330             parkinson's disease  2227
    print("Target entity IDs:", target_entity_ids) #[1725, 2227]

    # load drug candidate entities
    entities[candidate_entity_type]['primary'] = entities[candidate_entity_type]['primary'].str.strip().str.lower()
    entity_emb_map['primary'] = entity_emb_map['primary'].str.strip().str.lower()
    entities[candidate_entity_type]['primary'] = entities[candidate_entity_type]['primary'].astype(str)
    entity_emb_map['primary'] = entity_emb_map['primary'].astype(str)

    candidate_entities = pd.merge(entities[candidate_entity_type], entity_emb_map, on='primary', how='inner')
    candidate_entity_ids = torch.tensor(candidate_entities.id.tolist()).long()
    print("candidate_entity_ids", candidate_entity_ids)  # [1725, 2227]
    candidate_embs = ent_embeddings[candidate_entity_ids]

    # get target relation embeddings (treats and palliates)
    target_relations = rel_emb_map[rel_emb_map['relation'].isin(relation_type)]
    target_relation_ids = torch.tensor(target_relations.rid.tolist()).long()
    target_relation_embs = [rel_embeddings[rid] for rid in target_relation_ids]

    # rank candidate entities
    scores_per_target_ent = []
    candidate_ids = []
    for rid in range(len(target_relation_embs)):
        rel_emb = target_relation_embs[rid]
        for target_id in target_entity_ids:
            print("target_id:", target_id, type(target_id)) #target_id: 1725 <class 'int'>
            target_emb = ent_embeddings[target_id]

            if method == 'transE_l2':
                score = fn.logsigmoid(transE_l2(candidate_embs, rel_emb, target_emb))
            elif method == 'transR':
                score = fn.logsigmoid(transR(candidate_embs, rel_emb, target_emb))
            elif method == 'complEx':
                score = fn.logsigmoid(complEx(candidate_embs, rel_emb, target_emb))
            elif method == 'DistMult':
                score = fn.logsigmoid(DistMult(candidate_embs, rel_emb, target_emb))
            else:
                print("Method name error!!! Please check name of the knowledge graph embedding method you used.")

            scores_per_target_ent.append(score)
            print("candidate_entity_ids:", candidate_entity_ids)
            candidate_ids.append(candidate_entity_ids)

    scores = torch.cat(scores_per_target_ent)
    candidate_ids = torch.cat(candidate_ids)

    # Sort candidates by score
    idx = torch.flip(torch.argsort(scores), dims=[0])
    print("len(scores_per_target_ent):", len(scores_per_target_ent))
    print("len(candidate_ids):", len(candidate_ids))
    scores = scores[idx].cpu().numpy()
    candidate_ids = candidate_ids[idx].cpu().numpy()

    # De-duplicate
    _, unique_indices = np.unique(candidate_ids, return_index=True)
    ranked_unique_indices = np.sort(unique_indices)
    proposed_candidate_ids = candidate_ids[ranked_unique_indices]
    proposed_scores = scores[ranked_unique_indices]
    proposed_scores_norm = MinMaxScaler().fit_transform(proposed_scores.reshape(-1, 1))

    # Create a DataFrame for the results
    proposed_df = pd.DataFrame()
    proposed_df['id'] = proposed_candidate_ids
    proposed_df['score'] = proposed_scores
    proposed_df['score_norm'] = proposed_scores_norm

    # Merge with candidate entities for more information
    proposed_df = pd.merge(candidate_entities, proposed_df, on='id', how='right')

    # Filter out entities with known relations to PD (treats or palliates)
    rel_meta_type = relation_type[0].split('_')[-1]  # e.g., Treats_DDi => DDi
    triplet_df = pd.read_csv(triplet_folder + '/' + rel_meta_type + '_triplet.csv', header=0, low_memory=False)
    if not without_any_rel:
        triplet_df = triplet_df[triplet_df['Relation'].isin(relation_type)]

    # Only keep triplets that contain the target entity (PD)
    # print("target_entity_ids:", target_entity_ids[:5], type(target_entity_ids[0])) #[1725, 2227] <class 'int'>
    # print("triplet_df Head types:", triplet_df['Head'].dtype)
    # print(triplet_df.head())


    # triplet_df = triplet_df[(triplet_df['Head'].isin(target_entity_ids)) | (triplet_df['Tail'].isin(target_entity_ids))]
    # candidates_known = triplet_df['Head'].tolist() + triplet_df['Tail'].tolist()
    # candidates_known = list(set(candidates_known) - set(target_entity_ids))
    # print('candidates_known',candidates_known)

    # 统一格式：去空格 + 小写
    triplet_df['Head'] = triplet_df['Head'].astype(str).str.strip().str.lower()
    triplet_df['Tail'] = triplet_df['Tail'].astype(str).str.strip().str.lower()
    target_entity_vocab['primary'] = target_entity_vocab['primary'].astype(str).str.strip().str.lower()

    # 用 target_entity_vocab['primary'] 筛选
    triplet_df_filtered = triplet_df[
        triplet_df['Head'].isin(target_entity_vocab['primary']) |
        triplet_df['Tail'].isin(target_entity_vocab['primary'])
        ]

    # 获取已知候选实体（去掉目标实体本身）
    candidates_known = triplet_df_filtered['Head'].tolist() + triplet_df_filtered['Tail'].tolist()
    candidates_known = list(set(candidates_known) - set(target_entity_vocab['primary'].tolist()))
    # print("candidates_known:", candidates_known)  ###['drugbank:db00915', 'drugbank:db01200', 'drugbank:db01037', 'drugbank:db00413', ...

    # Remove candidates with known relations to PD
    print(len(proposed_df))
    proposed_df = proposed_df[~proposed_df['primary'].isin(candidates_known)]
    proposed_df = proposed_df[~proposed_df['name'].isin(target_entity_vocab['name'].tolist())]


    # Limit to top K results
    if topK is not None:
        proposed_df = proposed_df.head(topK)

    # Save results if needed
    if save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        proposed_df.to_csv(save_path + '/prediction_drug_top%s_%s.csv' % (topK, method), index=False)

    return proposed_df

def read_entity_from_id():
    """
    读取实体文件并生成:
      - entity2id: {entity: id}
      - id2ent.csv: 两列 ['id', 'primary']
    """
    entity2id = {}
    # 读取实体，去重（小写合并）
    with open("../data/iBKH/entities.txt", "r") as f:
        for line in f:
            entity = line.strip()
            if entity:  # 跳过空行
                entity2id[entity.lower()] = None
    # 重新编号，保证连续
    entity2id = {ent: idx for idx, ent in enumerate(entity2id.keys())}

    # 生成 id2ent
    id2ent = {idx: ent for ent, idx in entity2id.items()}

    # 写入 CSV
    with open("../data/iBKH/id2ent.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "primary"])  # 表头
        for idx in range(len(id2ent)):  # 保证顺序
            writer.writerow([idx, id2ent[idx]])

    return entity2id, id2ent
def read_relation_from_id():
    """ relation2id:{rel:id} """
    relation2id = {}

    with open("../data/iBKH/relations.txt", "r") as f:
        for idx, line in enumerate(f):
            relation = line.strip()
            if relation:  # 跳过空行
                relation2id[relation] = idx
    id2rel = {idx: rel for rel, idx in relation2id.items()}
    # 写入 CSV
    with open("../data/iBKH/id2rel.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['rid', 'relation'])  # 表头
        for idx in range(len(id2rel)):  # 保证顺序
            writer.writerow([idx, id2rel[idx]])
    return relation2id, id2rel


if __name__ == "__main__":
    # Here we collect a list of PD terms.
    # These PD terms can be obtained in the entity vocabularies in the “data/iBKH/entity” folder.
    PD = ["parkinson's disease", "late onset parkinson's disease"]
    # The task is to predict drug entities that don’t have “treats” and “palliates” relationships with PD in the iBKH
    # but can potentially treat or palliate PD. Therefore, we define a relation type list:
    r_type = ["Treats_DDi", "Palliates_DDi"]
    kg_folder = '../data/iBKH'
    csv_out_ent = "id2ent.csv"
    csv_out_rel = "id2rel.csv"
    # entity2id, id2ent = read_entity_from_id('../data/iBKH',  "id2ent.csv")
    # relation2id, id2rel = read_relation_from_id('../data/iBKH', "id2rel.csv")

    proposed_df = generate_hypothesis(target_entity=PD,
                                      candidate_entity_type='drug',
                                      relation_type=r_type,
                                      embedding_folder='../data/iBKH/embeddings',
                                      method='transE_l2',
                                      kg_folder='../data/iBKH',
                                      entity_folder='id2ent.csv',
                                      relation_folder='id2rel.csv',
                                      triplet_folder='../data/iBKH/triplets',
                                      topK=100, save_path='../output',
                                      save=True, without_any_rel=False)


# Running the above code will result in an output CSV file within the “output” folder,
# which stores top-100 ranked repurposable drug candidates for PD based on the TransE model.



# This will generate two output files for each model:
# “iBKH_[model name]_entity.npy”, containing the low dimension embeddings of entities in iBKH;
# “iBKH_[model name]_relation.npy”, containing the low dimension embeddings of relations in iBKH.
# These embeddings can be used in downstream BKD tasks.

