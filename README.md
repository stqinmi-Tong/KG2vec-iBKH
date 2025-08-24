## Download iBKH dataset

1. Download the entities and relations files according to the readme file under the ./data/iBKH/entity and ./data/iBKH/relation.

2. Prepare the triplets files run the following code:



   ```markup
   python ./funcs/KG_generation_run.py 
   ```

The generated data files can be found in the folder “iBKH-KD-protocol/data/iBKH/”, including “training_triplets.tsv”, “validation_triplets.tsv”, and “testing_triplets.tsv”, which will be used for training and evaluating the knowledge graph embedding models, as well as “whole_triplets.tsv”, which will be used for training the final models.

## KG2vec Training for iBKH

Train the KG2vec by:

```markup
python ./kg2vec/kg2vec_run.py
```

This will generate two output files for each model: “ent_embeddings_kg2vec.pt”, containing the low dimension embeddings of entities in iBKH and “rel_embeddings_kg2vec”, containing the low dimension embeddings of relations in iBKH. These embeddings can be used in downstream BKD tasks.&#x20;



## Conduct biomedical knowledge discovery

Finally we conduct biomedical knowledge discovery task using the learned embeddings:

```markup
python ./funcs/link_pred_run.py
```

Running the above code will result in an output CSV file within the “output” folder, which stores top-50 ranked repurposable drug candidates for PD based on the TransE model.
