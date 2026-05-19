from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.predict import predict_target
import torch
import pandas as pd
import random
import numpy as np
from collections import defaultdict

import time


SEED = 42

# Set all random seeds to 42
random.seed = SEED
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




# Function that creates the mid to name dictionary
def build_mid_name_dict(filepath):
    mid_to_name = {}
    # Read the file
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            parts = line.split() # split on whitespace
            mid = parts[0]
            name = " ".join(parts[1:]) # in case names themselves contain spaces
            mid_to_name[mid] = name
    return mid_to_name


# Function that extracts the types of the entities
def extract_types(text_triples):
    head_types = defaultdict(set)
    tail_types = defaultdict(set)
    for h, r, t in text_triples:
        paths = r.split('.')
        first = paths[0].strip('/').split('/')
        last = paths[-1].strip('/').split('/')
        #print("First: ",first)
        #print("Last: ", last)
        head_types[h].add(first[1])
        tail_types[t].add(last[2])
    return head_types, tail_types


# Validate tail 
def check_tail_type_valid(relation, tail, tail_types):
    #print("Relation: ",relation)
    #print("Tail: ",tail)
    # Extract the tail type
    paths = relation.split('.')
    first = paths[0].strip('/').split('/')
    last = paths[-1].strip('/').split('/')
    #print(first)
    #print(last)
    tail_type = last[2]
    # Check if the tail type is valid for the relation
    if tail_type in tail_types[tail]:
        return True
    else:
        return False


def sample_negative_triples(head, tail, num_samples, all_triples, entities, relations, SEED=42):
  '''
  Inputs:
  - head (int): head ID of the triple
  - tail (int): tail ID of the triple
  - num_samples (int): number of samples
  - all_triples (torch.tensor): tensor with all triples, containing entity and relation IDs
  - entities (list): list of entity IDs
  - relations (list): list of relation IDs
  Outputs:
  - negative_head_samples (torch.tensor): tensor with k triples sampled from the negative neighbourhood of the head
  - negative_tail_samples (torch.tensor): tensor with k triples sampled from the negative neighbourhood of the tail
  '''

  # Extract all the positive triples
  head_all_triples = all_triples[all_triples[:,0]==head] # head
  tail_all_triples = all_triples[all_triples[:,2]==tail] # tail

  # Number of all positive triples
  num_positive_head_triples = len(head_all_triples) # head
  num_positive_tail_triples = len(tail_all_triples) # tail
  # Number of all possible triples
  num_possible_triples = len(entities) * len(relations)

  # Size of the negative neighbourhood
  size_neg_nb_head = num_possible_triples - num_positive_head_triples # head
  size_neg_nb_tail = num_possible_triples - num_positive_tail_triples # tail

  # Number of entities and relations
  E = len(entities)
  R = len(relations)

  # Get the indexes of all positive triples
  positive_head_indexes = head_all_triples[:, 1] * E + head_all_triples[:, 2]  # head
  positive_tail_indexes = tail_all_triples[:, 0] * R + tail_all_triples[:, 1]  # tail

  # All possible indexes
  all_indexes = torch.arange(num_possible_triples, device=all_triples.device)

  # Create a positive mask for both head and tail
  mask_head = torch.ones(num_possible_triples, dtype=torch.bool, device=all_triples.device) # head
  mask_tail = torch.ones(num_possible_triples, dtype=torch.bool, device=all_triples.device) # tail

  # Mask positive triples as false
  mask_head[positive_head_indexes] = False  # head
  mask_tail[positive_tail_indexes] = False  # tail

  # Apply the Mask
  negative_head_indexes = all_indexes[mask_head] # head
  negative_tail_indexes = all_indexes[mask_tail] # tail

  # Randomly permute the negative indexes
  torch.manual_seed(SEED)
  torch.cuda.manual_seed_all(SEED)
  perm_head = torch.randperm(negative_head_indexes.size(0), device=negative_head_indexes.device)[:num_samples]
  perm_tail = torch.randperm(negative_tail_indexes.size(0), device=negative_tail_indexes.device)[:num_samples]

  # Get the sampled indexes
  sample_indexes_head = negative_head_indexes[perm_head]
  sample_indexes_tail = negative_tail_indexes[perm_tail]

  # Reconstruct the triples
  # Recover relation and tail for the triples with fixed head
  relations_head = sample_indexes_head // E
  tails = sample_indexes_head % E

  # Recover head and relation for the triples with fixed tail
  heads = sample_indexes_tail // R
  relations_tail = sample_indexes_tail % R

  # Stack into (num_samples, 3) tensor with [head, relation, tail] triples
  negative_head_samples = torch.stack([torch.full_like(relations_head, head), relations_head, tails], dim=1) # head
  negative_tail_samples = torch.stack([heads, relations_tail, torch.full_like(relations_tail, tail)], dim=1) # tail

  return negative_head_samples, negative_tail_samples, size_neg_nb_head, size_neg_nb_tail



def approximate_relik(triple, train_tf, model, num_samples, SEED=42):
  '''
  Function that computes the approximations for the KGE reliability score ReliK: ReliK_lb and ReliK_apx
  Inputs:
  - train_tf (TriplesFactory): pykeen training object with triples and IDs
  - triple (list/tensor of 3 elements): triple with head, relation, tail ID to compute the relik score of
  - model (pykeen model): pykeen model used to score the triples
  - num_samples (int): number of samples to use in the negative sampling (k in the algorithm)
  - SEED: random seed
  Outputs:
  - relik_lb (float): lower bound of the reliability score
  - relik_apx (float): approximated reliability score
  '''


  # Entities
  entities = list(train_tf.entity_id_to_label.keys())

  # Relations
  relations = list(train_tf.relation_id_to_label.keys())

  # All triples
  all_triples = train_tf.mapped_triples

  # Setup
  head = triple[0]
  tail = triple[2]
  relation = triple[1]

  # Sample k=num_samples times from the negative neighborhoods of the head and tail
  negative_head_samples, negative_tail_samples, size_neg_nb_head, size_neg_nb_tail = sample_negative_triples(head, tail, num_samples, all_triples, entities, relations, SEED=SEED)

  # Score of the true triple
  true_triple = torch.tensor([[head, relation, tail]], device=model.device)
  true_score = model.score_hrt(true_triple).item()

  # Head negative neighbourhood scores
  head_scores = model.score_hrt(negative_head_samples.to(model.device))
  # Get the rank of the triple among the sampled negative head neighbourhood triples
  rank_head = 1 + (head_scores > true_score).sum().item()

  # Tail negative neighbourhood scores
  tail_scores = model.score_hrt(negative_tail_samples.to(model.device))
  # Get the rank of the triple among the sampled negative tail neighbourhood triples
  rank_tail = 1 + (tail_scores > true_score).sum().item()

  #print(f"Head rank: {rank_head}")
  #print(f"Tail rank: {rank_tail}")

  # Compute relik lower bound
  relik_lb = 0.5 * (1/(rank_head + size_neg_nb_head - num_samples) + 1/(rank_tail + size_neg_nb_tail - num_samples))

  # Compute relik approximation
  relik_apx = 0.5 * (1/(rank_head * size_neg_nb_head / num_samples) + 1/(rank_tail * size_neg_nb_tail / num_samples))

  return relik_lb, relik_apx







def make_prediction(model, training, head=None, relation=None, tail=None):

  # Check if there is only one missing element in the triple
  if [head, relation, tail].count(None)!=1:
    raise ValueError('Exactly two elements of the triple need to be defined (not None) in order to make a prediction.')

  # Case 1: missing head
  if head is None:
    pred = predict_target(
      model = model,
      relation = training.relation_id_to_label[relation],
      tail = training.entity_id_to_label[tail],
      triples_factory = training,
    )

  # Case 2: missing relation
  if relation is None:
    pred = predict_target(
      model = model,
      head = training.entity_id_to_label[head],
      tail = training.entity_id_to_label[tail],
      triples_factory = training,
    )

  # Case 3: missing tail
  if tail is None:
    pred = predict_target(
      model = model,
      head = training.entity_id_to_label[head],
      relation = training.relation_id_to_label[relation],
      triples_factory = training,
    )

  return pred



def compute_scores(model, training, num_candidates, num_samples, tail_types, head=None, relation=None, tail=None, verbose=False, SEED=42):

  # Check if there is only one missing element in the triple
  if [head, relation, tail].count(None)!=1:
    raise ValueError('Exactly two elements of the triple need to be defined (not None) in order to make a prediction.')

  if verbose:
    print(f'Starting computation of model and ReliK scores, considering the top {num_candidates} predicted candidates')
    print(f'and sampling {num_samples} negative triples in the ReliK score computations.\n')

  # Predict the missing element
  pred = make_prediction(model, training, head, relation, tail)

  # Extract the IDs, scores, labels of the predicted elements
  df = pred.df.copy()
  IDs = df.filter(like="_id").iloc[:, 0]
  model_scores = df["score"]
  labels = df.filter(like="_label").iloc[:, 0]

  #print(df.columns)

  # Filter out invalid tails
  for row in range(len(labels)):
    relation_str = training.relation_id_to_label[relation]
    tail_str = df['tail_label'][row]
    #print(f"Relation: {relation_str}, Tail: {tail_str}")
    if not check_tail_type_valid(relation_str, tail_str, tail_types):
      df.drop(row, inplace=True)

  num_candidates = min(num_candidates, len(df))

  #print('IDs:')
  #print(IDs)
  #print(normalization)

  # Build id to entity/relation dictionaries
  id_to_entity = training.entity_id_to_label
  id_to_relation = training.relation_id_to_label

  # Entities, relations
  entities, relations = list(id_to_entity.keys()), list(id_to_relation.keys())

  # Get the target's label
  target = 'head' if head is None else 'relation' if relation is None else 'tail'

  if verbose:
    print("Candidates:")
    print(df.iloc[:num_candidates,:])
    #print(f"Target={target}")

  # For the first num_candidates triples, approximate the ReliK score of their neighbourhood
  relik_scores = np.zeros(num_candidates)

  for i in range(num_candidates):

    if target=='head':
      triple = [int(IDs.iloc[i]), relation, tail]
    if target=='relation':
      triple = [head, int(IDs.iloc[i]), tail]
    if target=='tail':
      triple = [head, relation, int(IDs.iloc[i])]

    if verbose:
      print(f'\nApproximating the ReliK score for triple {[id_to_entity[triple[0]], id_to_relation[triple[1]], id_to_entity[triple[2]]]}')
      print(f'Triple IDs = {triple}')

    # Compute relik scores
    _, relik_scores[i] = approximate_relik(triple, training, model, num_samples, SEED=SEED)
    if verbose:
      print(f"Relik score = {relik_scores[i]}")

  # Min-max model normalization
  min_s = model_scores.min()
  max_s = model_scores.max()
  min_max_model = (model_scores-min_s) / (max_s-min_s)

  # Min-max ReliK normalization
  min_r = relik_scores.min()
  max_r = relik_scores.max()
  min_max_relik = (relik_scores-min_r) / (max_r-min_r) if max_r-min_r!=0 else np.ones(len(relik_scores))

  # Sigmoid model normalization
  sigmoid_model = 1/(1+np.exp(-model_scores))

  # Take the data frame of predictions with the first num_candidates candidates
  prediction = df[:num_candidates].copy()
  # Add relik and combined score columns to the prediction df
  prediction.loc[:,'min_max_model'] = min_max_model[:num_candidates].to_numpy()
  prediction.loc[:,'min_max_relik'] = min_max_relik
  prediction.loc[:,'sigmoid_model'] = sigmoid_model[:num_candidates].to_numpy()
  prediction.loc[:,'relik'] = relik_scores

  return prediction






def approximate_local_relik(tail, model, training, num_nb, num_samples, aggregation='median'):

  # Get the triples that share the predicted element with the candidate triple
  similar_triples = training.mapped_triples[training.mapped_triples[:,2]==tail]

  # Get the min between num_nb and the number of similar triples
  num_nb = min(num_nb, len(similar_triples))

  # Sample num_nb times from similar_triples
  similar_triples = random.sample(list(similar_triples), num_nb)

  # Compute the ReliK score for all similar triples
  relik_scores = np.zeros(len(similar_triples))

  for i, similar_triple in enumerate(similar_triples):
    _, relik_apx = approximate_relik(similar_triple, training, model, num_samples, SEED=SEED)

    relik_scores[i] = relik_apx

  # Return the median ReliK score of the similar triples
  if aggregation=='median':
    return np.median(relik_scores)
  elif aggregation=='mean':
    return np.mean(relik_scores)
  else:
    raise ValueError('aggregation must be "median" or "mean"')




def approximate_local_relik2(triple, model, training, num_nb, num_samples, target='head', tail=None, verbose=False):

  # Make sure the value of the target variable is valid
  if target not in ['head', 'relation', 'tail']:
    raise ValueError('target must be "head", "relation" or "tail"')

  # Target to position in triple dictionary
  target_to_pos = {'head':0, 'relation':1, 'tail':2}

  # Retrieve the location of the target in the triple
  target_pos = target_to_pos[target]

  # Get the triples that share the predicted element with the candidate triple
  similar_triples = training.mapped_triples[training.mapped_triples[:,target_pos]==triple[target_pos]]

  if verbose:
    #print(f"Target position: {target_pos}")
    print(f"Number of similar triples: {len(similar_triples)}")
    #print(f"Similar triples: {similar_triples}")

  # Get the min between num_nb and the number of similar triples
  num_nb = min(num_nb, len(similar_triples))

  # Sample num_nb times from similar_triples
  similar_triples = random.sample(list(similar_triples), num_nb)

  # Compute the ReliK score for all similar triples
  relik_scores = np.zeros(len(similar_triples))
  for i, similar_triple in enumerate(similar_triples):
    _, relik_apx = approximate_relik(similar_triple, training, model, num_samples, SEED=SEED)
    if verbose:
      print(f"{i}: relik={relik_apx}")
    relik_scores[i] = relik_apx

  # Return the median ReliK score of the similar triples
  return np.median(relik_scores)





# this function started as a general element prediction function but was adapted for tail prediction
# so some checks in it are not needed

def compute_scores2(model, training, num_candidates, num_nb, num_samples, tail_types, cache, head=None, relation=None, tail=None, verbose=False):

  # Check if there is only one missing element in the triple
  if [head, relation, tail].count(None)!=1:
    raise ValueError('Exactly two elements of the triple need to be defined (not None) in order to make a prediction.')

  if verbose:
    print(f'Starting computation of model and ReliK scores, considering the top {num_candidates} predicted candidates, {num_nb} similar neighbours')
    print(f'and sampling {num_samples} negative triples in the ReliK score computations.\n')

  # Predict the missing element
  pred = make_prediction(model, training, head, relation, tail)

  # Extract the IDs, scores, labels of the predicted elements
  df = pred.df
  IDs = df.filter(like="_id").iloc[:, 0]
  model_scores = df["score"]
  labels = df.filter(like="_label").iloc[:, 0]

  # Filter out invalid tails
  for row in range(len(labels)):
    relation_str = training.relation_id_to_label[relation]
    tail_str = df['tail_label'][row]
    #print(f"Relation: {relation_str}, Tail: {tail_str}")
    if not check_tail_type_valid(relation_str, tail_str, tail_types):
      df.drop(row, inplace=True)

  num_candidates = min(num_candidates, len(df))

  #print('IDs:')
  #print(IDs)

  # Build id to entity/relation dictionaries
  id_to_entity = training.entity_id_to_label
  id_to_relation = training.relation_id_to_label

  # Entities, relations
  entities, relations = list(id_to_entity.keys()), list(id_to_relation.keys())

  # Get the target's label
  target = 'head' if head is None else 'relation' if relation is None else 'tail'

  if verbose:
    print("Candidates:")
    print(df.iloc[:num_candidates,:])
    #print(f"Target={target}")

  # For the first num_candidates triples, approximate the ReliK score of their neighbourhood
  relik_scores = np.zeros(num_candidates)

  for i in range(num_candidates):

    if target=='head':
      triple = [int(IDs.iloc[i]), relation, tail]
    if target=='relation':
      triple = [head, int(IDs.iloc[i]), tail]
    if target=='tail':
      triple = [head, relation, int(IDs.iloc[i])]

    if verbose:
      print(f'\nApproximating the ReliK scores for positive triples similar to the triple {[id_to_entity[triple[0]], id_to_relation[triple[1]], id_to_entity[triple[2]]]}')
      print(f'Triple IDs = {triple}')

    # FOR TAIL PREDICTION
    # Get the tail of the triple
    tail = int(triple[2])

    # Check the cache
    if cache and tail in cache.keys():
      relik_scores[i] = cache[tail]
    else:
      # Compute relik scores
      relik_scores[i] = approximate_local_relik(tail, model, training, num_nb, num_samples)
      cache[tail] = relik_scores[i]
      #relik_scores[i] = approximate_local_relik2(triple, model, training, num_nb, num_samples, target, verbose=verbose)

  # Min-max model normalization
  min_s = model_scores.min()
  max_s = model_scores.max()
  min_max_model = (model_scores-min_s) / (max_s-min_s) if max_s-min_s!=0 else np.ones(len(model_scores))

  # Min-max ReliK normalization
  min_r = relik_scores.min()
  max_r = relik_scores.max()
  min_max_relik = (relik_scores-min_r) / (max_r-min_r) if max_r-min_r!=0 else np.ones(len(relik_scores))

  # Sigmoid model normalization
  sigmoid_model = 1/(1+np.exp(-model_scores))

  # Take the data frame of predictions with the first num_candidates candidates
  prediction = df[:num_candidates].copy()
  # Add relik and combined score columns to the prediction df
  prediction.loc[:,'min_max_model'] = min_max_model[:num_candidates].to_numpy()
  prediction.loc[:,'min_max_relik'] = min_max_relik
  prediction.loc[:,'sigmoid_model'] = sigmoid_model[:num_candidates].to_numpy()
  prediction.loc[:,'relik'] = relik_scores

  return prediction




def combine_scores(model_scores, relik_scores, lambda_):
  return model_scores * lambda_ + relik_scores * (1-lambda_)


def filter_triples(predictions, threshold):  
  # Iterate over the rows
  for row in predictions.iterrows():
    if len(predictions)>1 and row[1]['relik']<threshold:
      #print(predictions)
      predictions.drop(row[0], inplace=True)

# Check if the true head is in the predictions
def hit_at_k(pred, true_head, k):
  return int(true_head in pred[:k])

# Evaluate combined scores using convex combination
def evaluate_combined_scores(predictions, true_head, lambda_, threshold=1e-4, filter=True, model_norm='min_max_model', relik='relik'):

  if filter:
    # Filter out scores with low ReliK scores
    filter_triples(predictions, threshold)

  # Combine model scores and ReliK scores
  predictions['combined'] = combine_scores(predictions[model_norm], predictions[relik], lambda_)

  # Sort predictions by combined score
  predictions.sort_values(by=['combined'], ascending=False, inplace=True)

  # Evaluate the hits@1 of the combined model scores
  pred = list(predictions.filter(like="_id").iloc[:, 0])

  # See if the right prediction is in the top 1, 3, 10 candidates
  hit_at_1 = hit_at_k(pred, true_head, 1)
  hit_at_3 = hit_at_k(pred, true_head, 3)
  hit_at_5 = hit_at_k(pred, true_head, 5)
  hit_at_10 = hit_at_k(pred, true_head, 10)

  return {
      'predictions': predictions,
      'hit_at_1': hit_at_1,
      'hit_at_3': hit_at_3,
      'hit_at_5': hit_at_5,
      'hit_at_10': hit_at_10
  }


#Function that describes the hits@k by lambda data frame via summary statistics
def describe_results(df):

    summary_dicts = []
    
    # Extraxt the max Hits@k for every k=1,3,5,10 and find the associated lambda
    for k in [1,3,5,10]:
        # Summary statistics
        mean_hits_at_k = df['Hits@'+str(k)].mean()
        median_hits_at_k = df['Hits@'+str(k)].median()
        std_hits_at_k = df['Hits@'+str(k)].std()
        # Max hits@k by lambda
        max_hits_at_k = df['Hits@'+str(k)].max()
        first_max_hits_at_k_lambda = df[df['Hits@'+str(k)]==max_hits_at_k]['Lambda'].iloc[0]
        last_max_hits_at_k_lambda = df[df['Hits@'+str(k)]==max_hits_at_k]['Lambda'].iloc[-1]
        # Min hits@k by lambda
        min_hits_at_k = df['Hits@'+str(k)].min()
        first_min_hits_at_k_lambda = df[df['Hits@'+str(k)]==min_hits_at_k]['Lambda'].iloc[0]
        last_min_hits_at_k_lambda = df[df['Hits@'+str(k)]==min_hits_at_k]['Lambda'].iloc[-1]
        
        summary_dicts.append({
            'k': k, 
            'max': max_hits_at_k,
            'max_first_lambda': first_max_hits_at_k_lambda,
            'max_last_lambda': last_max_hits_at_k_lambda,
            'min': min_hits_at_k,
            'min_first_lambda': first_min_hits_at_k_lambda,
            'min_last_lambda': last_min_hits_at_k_lambda,
            'mean':mean_hits_at_k,
            'median':median_hits_at_k,
            'std':std_hits_at_k
            })

    res = pd.DataFrame(summary_dicts, columns=['k', 'max', 'max_first_lambda', 'max_last_lambda', 'min', 'min_first_lambda', 'min_last_lambda', 'mean', 'median', 'std'])
    return res