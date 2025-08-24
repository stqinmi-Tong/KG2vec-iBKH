import pandas as pd
import numpy as np
import pickle
import torch as th
import torch.nn.functional as fn
import os
import sys
import KG_processing as KG_processing
sys.path.append('.')

import glob




# Extract triplets from raw data of iBKH.
# 1. Set up the input and output file paths.

# Input iBKH-KG data path
kg_folder = '/home/shent/.conda/envs/GCNConv/iBKH-KD-protocol/data/iBKH/'

# Output path
triplet_path = '/home/shent/.conda/envs/GCNConv/iBKH-KD-protocol/data/iBKH/triplets/'
if not os.path.exists(triplet_path):
    os.makedirs(triplet_path)

# Output data file path
output_path = '/home/shent/.conda/envs/GCNConv/iBKH-KD-protocol/data/DGL-KE/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

#
# # 2. Extract triplets of different entity pair types by running following codes
# # This will result in a set of CSV files in the “iBKH-KD-protocol/data/triplets/”, storing triplets regarding each entity pair type.
# KG_processing.DDi_triplets(kg_folder, triplet_path)
# KG_processing.DG_triplets(kg_folder, triplet_path)
# KG_processing.DPwy_triplets(kg_folder, triplet_path)
# KG_processing.DSE_triplets(kg_folder, triplet_path)
# KG_processing.DiDi_triplets(kg_folder, triplet_path)
# KG_processing.DiG_triplets(kg_folder, triplet_path)
# KG_processing.DiPwy_triplets(kg_folder, triplet_path)
# KG_processing.DiSy_triplets(kg_folder, triplet_path)
# KG_processing.GG_triplets(kg_folder, triplet_path)
# KG_processing.GPwy_triplets(kg_folder, triplet_path)
# KG_processing.DD_triplets(kg_folder, triplet_path)
#
# # 3. Combine the triplets to generate a TSV file based on the DGL-KE input requirement.
# # The variable “included_pair_type” specifies a list of triplet types that we plan to use for analysis.
# # The generated data files can be found in the folder “iBKH-KD-protocol/data/dataset/”,
# # including “training_triplets.txt”, “validation_triplets.tsv”, and “testing_triplets.tsv”,
# # which will be used for training and evaluating the knowledge graph embedding models,
# # as well as “whole_triplets.tsv”, which will be used for training the final models.
#
# # Specify triplet types you want to use
# included_pair_type = [
#     'DDi', 'DG', 'DPwy', 'DSE', 'DiDi', 'DiG',
#     'DiPwy', 'DiSy', 'GG', 'GPwy', 'DD'
# ]
#
# # Combine triplets
# KG_processing.generate_triplet_set(triplet_path=triplet_path)
#
# # 4. Generate DGL-KE required input triplet file
# KG_processing.generate_DGL_training_set(
#     triplet_path=triplet_path,
#     output_path=output_path
# )
#
# KG_processing.generate_DGL_data_set(
#         triplet_path=triplet_path, output_path=output_path,
#         train_val_test_ratio=[.8, .1, .1])


# 5. Generate Kg2vec required entities.csv, relation.csv
def merge_first_columns(input_folder, output_csv, output_file):
    """
    从指定文件夹中读取多个CSV文件，提取第一列（去除表头），拼接为一列并保存到新的CSV文件中
    """
    all_values = []

    # 遍历文件夹中的所有 CSV 文件
    for file in glob.glob(os.path.join(input_folder, "*.csv")):
        df = pd.read_csv(file, usecols=[0])  # 只读第一列
        values = df.iloc[:, 0].tolist()      # 转为list
        all_values.extend(values[1:])        # 去掉第一行表头（假设所有文件第一列都有表头）

    # 保存为新 CSV（单列）
    result = pd.DataFrame(all_values, columns=["Merged_Column"])
    result.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ 合并完成，结果已保存到: {output_csv}")

    # 去重并保持原顺序
    df = pd.read_csv(output_csv, usecols=[0])
    unique_entities = df.iloc[:, 0].dropna().drop_duplicates()

    # 保存到 txt 文件
    with open(output_file, "w") as f:
        for ent in unique_entities:
            f.write(str(ent) + "\n")

    print(f"✅ 已提取 {len(unique_entities)} 个唯一实体，保存到 {output_file}")

def extract_unique_second_column(input_tsv, output_txt):

    df = pd.read_csv(input_tsv, sep="\t", header=None, usecols=[1])  # 只取第二列
    unique_items = df.iloc[:, 0].dropna().drop_duplicates()

    unique_items.to_csv(output_txt, index=False, header=False)

    print(f"✅ 从 {input_tsv} 提取 {len(unique_items)} 个唯一元素，保存到 {output_txt}")
def update_entities(entities_file, triples_file):
    # 1. 读取已存在的实体集合
    with open(entities_file, "r") as f:
        entities = set(line.strip() for line in f if line.strip())

    # 2. 读取 triples.tsv，提取 head 和 tail
    new_entities = set()
    with open(triples_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:  # 确保至少有 head, relation, tail
                head, _, tail = parts[0], parts[1], parts[2]
                if head not in entities:
                    new_entities.add(head)
                if tail not in entities:
                    new_entities.add(tail)

    # 3. 追加写入到 entities.txt
    if new_entities:
        with open(entities_file, "a") as f:
            for e in sorted(new_entities):  # 排序写入，方便查找
                f.write(e + "\n")

    print(f"新增 {len(new_entities)} 个实体，已更新 {entities_file}")


if __name__ == "__main__":

    # input_folder = "/home/shent/.conda/envs/GCNConv/iBKH-KD-protocol/data/iBKH/entity"          # 存放CSV文件的文件夹路径
    # output_ent_csv = "/home/shent/.conda/envs/GCNConv/iBKH-KD-protocol/data/iBKH/entities.csv"  # 输出文件名
    # ouput_ent_txt = "/home/shent/.conda/envs/GCNConv/iBKH-KD-protocol/data/iBKH/entities.txt"  # 输出文件名
    # merge_first_columns(input_folder, output_ent_csv, ouput_ent_txt)
    # input_tsv = "/home/shent/.conda/envs/GCNConv/iBKH-KD-protocol/data/DGL-KE/whole_triplets.tsv"
    # output_rel_txt = "/home/shent/.conda/envs/GCNConv/iBKH-KD-protocol/data/iBKH/relations.txt"
    # extract_unique_second_column(input_tsv, output_rel_txt)
    update_entities("/home/shent/.conda/envs/GCNConv/iBKH-KD-protocol/data/iBKH/entities.txt",
                    "/home/shent/.conda/envs/GCNConv/iBKH-KD-protocol/data/DGL-KE/whole_triplets.tsv")