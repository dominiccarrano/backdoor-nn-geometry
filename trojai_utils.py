import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import DataLoader, sampler, TensorDataset, ConcatDataset
from scipy.stats import skew, kurtosis

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_adversarial_dataset(clean_dataset, attack, batch_size):
    return attack.__call__(clean_dataset, batch_size)

def flip_success(dataset, target_class, model):
    flip_rate = 0
    for _, (x, _) in enumerate(DataLoader(dataset, batch_size=32)):
        pred = torch.argmax(model(x), dim=-1)
        flip_rate += torch.sum(pred == target_class)
    return flip_rate.item() / len(dataset)

def moment_features(density):
    """
    Input
    - density: tensor of shape [n_samples]
    
    Output
    - moments: tensor of shape [4]
    """    
    if len(density) == 1:
        moments = torch.tensor([torch.mean(density), 0, skew(density.numpy()), kurtosis(density.numpy())])
    else: 
        moments = torch.tensor([torch.mean(density), torch.var(density), skew(density.numpy()), kurtosis(density.numpy())])
    return moments

def quantile_features(density, q_vals):
    """
    Input
    - density: tensor of shape [n_samples]
    - q_vals: list of numbers between 0 and 1 with the quantiles to use
    
    Output
    - quartile_sigs: tensor of shape [len(q_vals)]
    """    
    q_vals = torch.tensor(q_vals, dtype=density.dtype)
    quantiles = torch.quantile(density, q_vals) 
    return quantiles

def load_clean_examples(index, base_data_path):
    metadata = pd.read_csv(os.path.join(base_data_path, "METADATA.csv"))
    model_metadata = metadata.iloc[index]
    clean_data_dir = os.path.join(base_data_path, "models",
                                     model_metadata["model_name"],
                                     "clean_example_data")
    poisoned_reviews = []
    for filename in os.listdir(clean_data_dir):
        with open(os.path.join(clean_data_dir, filename), 'r') as f:
            poisoned_reviews.append(f.readlines()[0])
    return poisoned_reviews

def load_poisoned_examples(index, base_data_path):
    metadata = pd.read_csv(os.path.join(base_data_path, "METADATA.csv"))
    model_metadata = metadata.iloc[index]
    poisoned_data_dir = os.path.join(base_data_path, "models",
                                     model_metadata["model_name"],
                                     "poisoned_example_data")
    poisoned_reviews = []
    for filename in os.listdir(poisoned_data_dir):
        with open(os.path.join(poisoned_data_dir, filename), 'r') as f:
            poisoned_reviews.append(f.readlines()[0])
    return poisoned_reviews

def load_model(index, base_data_path):
    metadata = pd.read_csv(os.path.join(base_data_path, "METADATA.csv"))
    model_metadata = metadata.iloc[index]
    model = torch.load(os.path.join(base_data_path, "models", model_metadata["model_name"], "model.pt"), map_location=device)
    return model, model_metadata

def get_LM(embedding_type, base_data_path):
    """Return a Language Model's (LM's) tokenizer and embedding map."""
    if embedding_type == "BERT":
        embedding_path = "embeddings/BERT-bert-base-uncased.pt"
        tokenizer_path = "tokenizers/BERT-bert-base-uncased.pt"
    elif embedding_type == "GPT-2":
        embedding_path = "embeddings/GPT-2-gpt2.pt"
        tokenizer_path = "tokenizers/GPT-2-gpt2.pt"
    elif embedding_type == "DistilBERT":
        embedding_path = "embeddings/DistilBERT-distilbert-base-uncased.pt"
        tokenizer_path = "tokenizers/DistilBERT-distilbert-base-uncased.pt"
    else:
        raise ValueError("Expected one of 'BERT', 'GPT-2', or 'DistilBERT' for argument 'embedding_type'.")
    tokenizer = torch.load(os.path.join(base_data_path, tokenizer_path), map_location=device)
    embedding = torch.load(os.path.join(base_data_path, embedding_path), map_location=device)
    return tokenizer, embedding

# Credit: https://github.com/usnistgov/trojai-example/blob/40a2c80651793d9532edf2d29066934f1de500b0/inference_example_data.py
def get_embeddings(tokenizer, embedding, text, cls_token_is_first):
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path]

    # Tokenize; extract ids and attention mask
    results = tokenizer(text, max_length=max_input_length-2, padding=True, truncation=True, return_tensors="pt")
    results = results.to(device)

    input_ids = results.data['input_ids']
    attention_mask = results.data['attention_mask']

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    embedding = embedding.to(device)

    # Convert to embedding
    with torch.no_grad():
        embeddings = embedding(input_ids, attention_mask=attention_mask)[0]

        # Ignore all but the first embedding since this is sentiment classification
        if cls_token_is_first:
            # BERT-like models (use the first token as the text summary)
            embeddings = embeddings[:, 0, :]
            embeddings = embeddings.cpu().detach().numpy()
        else:
            # GPT-2 (use the last token as the text summary)
            embeddings = embeddings.cpu().detach().numpy()
            attn_mask = attention_mask.detach().cpu().detach().numpy()
            emb_list = list()
            for i in range(attn_mask.shape[0]):
                idx = int(np.argwhere(attn_mask[i, :] == 1)[-1])
                emb_list.append(embeddings[i, idx, :])
            embeddings = np.stack(emb_list, axis=0)

        # Add sequence dimension (needed for RNNs) of length 1 since we only use a
        # single summary token for downstream tasks
        embeddings = torch.unsqueeze(torch.from_numpy(embeddings), dim=1)
    return embeddings
