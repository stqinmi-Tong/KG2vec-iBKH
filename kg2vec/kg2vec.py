import time, datetime
import torch
from tqdm import tqdm
from create_batch import candidate_choose
from preprocessing import load_item_pop
from MCN_sampling import MCNS_r
from EmbeddingModel import EmbeddingModel_Optimized

def kg2vec_fast(
    data_path, ent_size, rel_size, EMBEDDING_SIZE, dataloader, walk_num, walk_num_edge,
    reverse_dictionary, LEARNING_RATE, NUM_EPOCHS, ent_dic, rel_dic, train_data, gpu
):
    device = torch.device(f'cuda:{gpu}')
    model = EmbeddingModel_Optimized(ent_size, rel_size, EMBEDDING_SIZE, device,
                                     reverse_dictionary, ent_dic).to(device)

    # 预加载必要数据到 GPU
    q_1_dict, q_2_dict, mask, mask_rel, rr, dr = load_item_pop(train_data, ent_dic, rel_dic)
    print('---load_item_pop finished---')

    candidates = candidate_choose(mask, mask_rel, rr, dr, walk_num, walk_num_edge,rel_dic)
    # candidate[node/edge]={[n1,n2,...]/[r1,r2,...]}
    print('---candidate_choose finished---')

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08
    )

    N_steps = 2
    N_negs = 1
    print('Beginning training')

    loss_log_path = data_path + '/loss_log.txt'
    with open(loss_log_path, 'w') as f:
        f.write('epoch,avg_loss,epoch_time\n')

    for epoch in tqdm(range(NUM_EPOCHS), desc="Training"):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        neg_labels = None

        # 使用 DataLoader pin_memory 提速
        for batch_idx, (input_labels, pos_labels) in enumerate(dataloader):

            input_labels = input_labels.to(device, non_blocking=True)
            pos_labels = pos_labels.to(device, non_blocking=True)

            # 负采样（批量处理，减少 Python 循环开销）
            start_given = neg_labels if batch_idx > 0 else None
            neg_labels = MCNS_gpu(
                model, candidates, start_given, q_1_dict, q_2_dict,
                N_steps, input_labels, ent_dic, reverse_dictionary, device
            )
            # print('input_labels', input_labels.size())
            # print('neg_labels', neg_labels.size())

            optimizer.zero_grad(set_to_none=True)
            loss = model(
                input_labels,
                pos_labels,
                neg_labels.view(pos_labels.size(0), N_negs)
            ).mean()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 20 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"{datetime.datetime.now()} Batch:[{batch_idx}/{len(dataloader)}] Loss:{avg_loss:.5f}")

        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch: {epoch + 1:02d}, Loss: {avg_loss:.5f}, Time: {epoch_time:.2f}s")

        with open(loss_log_path, 'a') as f:
            f.write(f'{epoch + 1},{avg_loss:.5f},{epoch_time:.2f}\n')

    return model.input_embeddings()


def kg2vec(data_path, ent_size, rel_size, EMBEDDING_SIZE, dataloader,
           LEARNING_RATE, NUM_EPOCHS, ent_dic, reverse_dictionary, gpu):

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

