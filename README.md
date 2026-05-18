# Integrating Reliability in Knowledge Graph Completion

This repository contains the code and results of my project for the Graph Mining and Applications course of the Data Science master's degree @ Sapienza.

**Author**: Géraldine Valérie Maurer, maurer.1996887@studenti.uniroma1.it

## Structure

The repository is structured as follows:

 * [main.ipynb](./main.ipynb): main notebook with all the experiments
 * [functions.py](./functions.py): Python module with the ReliK algorithm implementation and functions to run the experiments
 * [results](./results): folder containing the results obtained by the three models (TransE, RotatE, ComplEx), with the `predictions.pkl` and `combined_predictions.pkl` files representing the result of approach 1, and `predictions2.pkl`, `combined_predictions2.pkl` containing the results for approach 2.
 * [data](./data): folder with mid2name information and pickle files containing extracted mappings between head/tail entities and the types associated with the relations they appear in in the training set.
 * [trained models](./trained_models): folder with the TransE, RotatE and ComplEx models trained with Pykeen on the Google Colab T4 GPU.
 * [Train_Models.ipynb](./Train_Models.ipynb): notebook used on Google Colab to train the models and pre-compute local ReliK scores for a subset of tails that appear most frequently in training set triples (these scores were used as a cache dictionary in approach 2 of the project)
 * [README.md](./README.md): current file
 * [LICENSE](./LICENSE): MIT License
 * [.gitignore](./.gitignore)


## Dataset

The dataset used is FB15k-237 and was downloaded from Kaggle at the link https://www.kaggle.com/datasets/latebloomer/fb15k-237.

It was built from Freebase and contains triples in the form $(head, relation, tail)$, usually with Freebase identifiers for the head and tail. Entities represent real-world things such as people, locations, organizations, films and concepts, while relations are links between entities, such as nationality, genre, or place of birth.

The following description is associated with the data:

*FB15k-237 is a link prediction dataset created from FB15k. While FB15k consists of 1,345 relations, 14,951 entities, and 592,213 triples, many triples are inverses that cause leakage from the training to testing and validation splits. FB15k-237 was created by Toutanova and Chen (2015) to ensure that the testing and evaluation datasets do not have inverse relation test leakage. In summary, FB15k-237 dataset contains 310,116 triples with 14,541 entities and 237 relation types.*

Additional information about the entity labels and descriptions was found on Huggingface at the link https://huggingface.co/datasets/KGraph/FB15k-237/tree/main/data.


## Methods

The goal of the project is to study **how reliability can be used to re-rank triples in tail prediction**.

Two approaches were used for the re-ranking of triples using ReliK scores. Here is the first one:

1. Given a $(h,r)$ pair, let the embedding model predict a list of tail candidates.

2. Keep the top $C$ candidates, where $C=50$ in the code.

3. Compute the ReliK scores of the triples $(h,r,t_j)$ for $j=1,\dots,C$.

4. Fit the model scores to the range $[0,1]$ so that they are comparable to the ReliK scores, for example using min-max normalization or the sigmoid function.

5. Combine the embedding scores and ReliK scores for each triple as follows:

$$
\operatorname{Combined}(x_{hrt_j})
=
\lambda \cdot \operatorname{ENC}_{\text{norm}}(x_{hrt_j})
+
(1-\lambda)\cdot \operatorname{ReliK}_{\text{Apx}}(x_{hrt_j})
$$

6. Finally, re-rank the candidates according to the new combined scores.

The idea is that, despite not knowing whether $x_{hrt_j}$ is positive, the ReliK score tells us how highly the triple ranks among known negative triples. Therefore, it can serve as an additional indicator of where the triple should appear in the final ranking.

If the embedding model is locally reliable, a positive triple in the ranking should have a higher ReliK score, increasing its combined score. Conversely, a negative triple is expected to have a lower ReliK score, because the algorithm is ranking it among other negative triples.

The second approach used in the project is the following:

1. Given a $(h,r)$ pair, let the embedding model predict a list of tail candidates.

2. Keep the top $C$ candidates, where $C=20$ in the code.

3. For each candidate triple $(h,r,t_j)$, with $j=1,\dots,C$, sample $p$ times from the positive neighbourhood of the node corresponding to tail $t_j$, where $p=30$ in the code.

4. For each positive triple in $t_j$'s neighbourhood, compute its ReliK score.

5. Aggregate the ReliK scores by $t_j$ using the median or mean.

6. Fit the model scores to the range $[0,1]$ so that they are comparable to the ReliK scores, for example using min-max normalization or the sigmoid function.

7. Combine the embedding scores and local ReliK scores for each triple as follows:

$$
\operatorname{Combined}(x_{hrt_j})
=
\lambda \cdot \operatorname{ENC}(x_{hrt_j})
+
(1-\lambda)\cdot \operatorname{ReliK}_{\text{Apx}}\left(\left(N^+(t_j)\right)_p\right)
$$

8. Finally, re-rank the candidates according to the new combined scores.

The idea is that, if a tail node's neighbourhood has several triples with high ReliK scores, it is more likely that positive triples in this subgraph have reliable embedding scores. Therefore, the aggregated positive neighbourhood score will improve the ranking of that candidate.

The opposite will happen for neighbourhoods with low ReliK scores.

## References

[1] Maximilian K. Egger, Wenyue Ma, Davide Mottin, Panagiotis Karras, Ilaria Bordino, Francesco Gullo and Aris Anagnostopoulos. *ReliK:  A Reliability Measure for Knowledge Graph
Embeddings*. SEBD 2024: 32nd Symposium on Advanced Database System, June 23-26, 2024 - Villasimius, Sardinia, Italy. https://ceur-ws.org/Vol-3741/paper04.pdf

[2] Kristina Toutanova, Danqi Chen, Patrick Pantel, Hoifung Poon, Pallavi Choudhury, and Michael Gamon. *Representing text for joint embedding of text and knowledge bases*. In Proceedings of EMNLP 2015.

[3] Kristina Toutanova and Danqi Chen. *Observed versus latent features for knowledge base and text inference*. In Proceedings of the 3rd Workshop on Continuous Vector Space Models and Their Compositionality 2015.

[4] Antoine Bordes, Nicolas Usunier, Alberto Garcia Duran, Jason Weston, and Oksana Yakhnenko. *Translating embeddings for modeling multirelational data*. In Advances in Neural Information Processing Systems (NIPS) 2013.

[5] Evgeniy Gabrilovich, Michael Ringgaard, and Amarnag Subramanya. *FACC1: Freebase annotation of ClueWeb corpora, Version 1* (release date 2013-06-26, format version 1, correction level 0). http://lemurproject.org/clueweb12/FACC1/

[6] https://github.com/AU-DIS/ReliK

[7] http://lemurproject.org/clueweb12/

[8] https://www.kaggle.com/datasets/latebloomer/fb15k-237