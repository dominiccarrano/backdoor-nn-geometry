import pandas as pd
import os
import torch
import numpy as np
import argparse
from trojai_utils import *

def batch_embeddings(reviews, N, batch_size, tokenizer, embedding, cls_first, embedding_dim=768):
    embeddings = torch.zeros((N, 1, embedding_dim))
    for i in range(N // batch_size):
        review_batch = reviews[i*batch_size:(i+1)*batch_size]
        embedding_batch = get_embeddings(tokenizer, embedding, review_batch, cls_token_is_first=cls_first)
        embeddings[i*batch_size:(i+1)*batch_size, :, :] = embedding_batch
    return embeddings

# Get args
parser = argparse.ArgumentParser(description="Generate embeddings")
parser.add_argument('--embedding-type', type=str,
                    help='Model architecture (one of "BERT", "DistilBERT", "GPT-2")')
parser.add_argument('--n', type=int, default=1000,
                    help='Number of embeddings of each sentiment to generate')
parser.add_argument('--batch-size', type=int, default=50,
                    help='Size of batches to feed into the language model for embedding generation')
args = parser.parse_args()

# Load in the data
base_huggingface_path = "your path with the huggingface transformer files"
base_data_path = "your file path with the reviews datasets"
sentiment_data = pd.read_csv(os.path.join(base_data_path, "train_datasets.csv"))

# Split by sentiment
pos_data = sentiment_data[sentiment_data.sentiment==True].sample(args.n)
neg_data = sentiment_data[sentiment_data.sentiment==False].sample(args.n)

# Get random samples 
pos_reviews = list(np.asarray(pos_data.reviewText, dtype=str))
pos_labels = torch.ones(args.n)

neg_reviews = list(np.asarray(neg_data.reviewText, dtype=str))
neg_labels = torch.zeros(args.n)

# Make embeddings
cls_first = (args.embedding_type == "DistilBERT") or (args.embedding_type == "BERT")
tokenizer, embedding = get_LM(args.embedding_type, base_huggingface_path)
pos_embeddings = batch_embeddings(pos_reviews, args.n, args.batch_size, tokenizer, embedding, cls_first)
neg_embeddings = batch_embeddings(neg_reviews, args.n, args.batch_size, tokenizer, embedding, cls_first)

# Save results
base_embedding_path = "your path to save embeddings to"
torch.save(pos_embeddings, os.path.join(base_embedding_path, args.embedding_type, "pos_embeddings{}.pt".format(args.n)))
torch.save(neg_embeddings, os.path.join(base_embedding_path, args.embedding_type, "neg_embeddings{}.pt".format(args.n)))
torch.save(pos_labels, os.path.join(base_embedding_path, args.embedding_type, "pos_labels{}.pt".format(args.n)))
torch.save(neg_labels, os.path.join(base_embedding_path, args.embedding_type, "neg_labels{}.pt".format(args.n)))