## Download iBKH dataset

1. Download the entities and relations files according to the readme file under the ./data/iBKH/entity and ./data/iBKH/relation.

2. Prepare the triplets files running the following code:

   ```markup
   python ./funcs/KG_generation_run.py 
   ```

The generated data files can be found in the folder “iBKH-KD-protocol/data/iBKH/”, including “training_triplets.tsv”, “validation_triplets.tsv”, and “testing_triplets.tsv”, which will be used for training and evaluating the knowledge graph embedding models, as well as “whole_triplets.tsv”, which will be used for training the final models.

## KG2vec Training for iBKH

Train the KG2vec by:

```markup
python ./kg2vec/kg2vec_run.py
```

This will generate two output files for each model: “ent_embeddings_kg2vec.pt”, containing the low-dimensional embeddings of entities in iBKH and “rel_embeddings_kg2vec”, containing the low-dimensional embeddings of relations in iBKH. These embeddings can be used in downstream BKD tasks.&#x20;

## Conduct biomedical knowledge discovery (BKD)

This section introduces the implementation of BKD based on knowledge graph embeddings learned from iBKH.

Here, we showcase a case study of drug repurposing hypothesis generation for Parkinson’s disease (PD).

We conduct biomedical knowledge discovery task using the learned embeddings:

```markup
python ./funcs/link_pred_run.py
```

Running the above code will result in an output CSV file within the “output” folder, which stores the top-50 ranked repurposable drug candidates for PD based on the TransE model.

## Example of our BKD Results:

This table shows the top-3 predicted results, which don’t have “treats” and “palliates” relationships with PD in the iBKH but has the highest probability of potentially treating or palliating PD.&#x20;

| primary          | name                         | &#x20;         ...  | id      | score          | score_norm  |
| :--------------- | :--------------------------- | :------------------ | :------ | :------------- | :---------- |
| cid:cid100003939 | LiOH                         | &#x20;         ...  | 2149717 | -1.9311718e-05 | 1.0         |
| kegg:d05008      | Metoclopramide hydrochloride | &#x20;         ...  | 2129139 | -2.396078e-05  | 0.9994735   |
| kegg:d04479      | Hyoscyamine hydrobromide     | &#x20;          ... | 2128747 | -2.5391257e-05 | 0.9993115 |

