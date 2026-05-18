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