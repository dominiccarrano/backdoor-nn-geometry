"""boundary_thickness was adapted from code at https://github.com/nsfzyzz/boundary_thickness."""
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import os
import multiprocessing
import torch.nn.functional as F
from torch.utils.data import DataLoader, sampler, TensorDataset, Subset
from attack_functions import *
import argparse

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class ParallelDataset(torch.utils.data.Dataset):
    def __init__(self, xr_dataset, xs_dataset):
        super(ParallelDataset, self).__init__()
        self.xr_dataset = xr_dataset
        self.xs_dataset = xs_dataset

    def __len__(self):
        return min(len(self.xr_dataset), len(self.xs_dataset))

    def __getitem__(self, index):
        img1, label1 = self.xr_dataset.__getitem__(index)
        img2, label2 = self.xs_dataset.__getitem__(index)
        return img1, label1, img2, label2

def boundary_tilting(xr_dataset, xs_dataset, xr_adv_dataset, model, batch_size, reduce_clean=False):    
    eps_for_division = 1e-10 # avoid divide by zero
    model.eval()
    
    # Build loaders
    clean_dataset = ParallelDataset(xr_dataset, xs_dataset)
    clean_loader = DataLoader(clean_dataset, batch_size=batch_size, shuffle=True)
    adv_dataset = ParallelDataset(xr_dataset, xr_adv_dataset)
    adv_loader = DataLoader(adv_dataset, batch_size=batch_size, shuffle=True)
    similarities = []
    
    # Compute clean and adversarial directions to transform xr's into xs's
    for (xr, _, xs, _), (xr, _, xr_adv, _) in zip(clean_loader, adv_loader):
        xr, xs, xr_adv = xr.to(device), xs.to(device), xr_adv.to(device)
        
        shortest_len = min(xr.size(0), xs.size(0), xr_adv.size(0))
        xr, xs, xr_adv = xr[:shortest_len], xs[:shortest_len], xr_adv[:shortest_len]
        
        # Mask out any points that the attack failed to move across the decision boundary
        xr_pred = torch.argmax(F.softmax(model(xr.float()), dim=-1), dim=-1)
        xr_adv_pred = torch.argmax(F.softmax(model(xr_adv.float()), dim=-1), dim=-1)
        mask = (xr_pred != xr_adv_pred) # only use points where the adversarial attack succeeded
        
        # No usable points in this batch; skip it
        if torch.sum(mask) == 0:
            continue
        
        # Compute masked directions
        xr, xr_adv, xs = xr[mask != 0].float(), xr_adv[mask != 0].float(), xs[mask != 0].float()
        if reduce_clean:
            xr_mean = torch.mean(xr, dim=0).unsqueeze(dim=0)
            xs_mean = torch.mean(xs, dim=0).unsqueeze(dim=0)
            clean_directions = xr_mean - xs_mean
            clean_directions = clean_directions.repeat(xr.size(0), 1, 1)
        else:
            clean_directions = xr - xs
        adv_directions = xr - xr_adv
    
        # Compute cosine similarities between each sample's clean direction and adversarial direction
        clean_directions = torch.flatten(clean_directions, start_dim=1, end_dim=-1)
        adv_directions = torch.flatten(adv_directions, start_dim=1, end_dim=-1)
        
        curr_sims = torch.einsum("ia,ia->i", clean_directions, adv_directions)
        curr_sims /= (torch.norm(clean_directions, dim=1) * torch.norm(adv_directions, dim=1) + eps_for_division)
        similarities.append(curr_sims)
    return torch.cat(similarities)
    
def boundary_thickness(xr_dataset, xs_dataset, model, alpha_beta_list, num_points, batch_size):
    model.eval()
    shortest_len = min(len(xr_dataset), len(xs_dataset))
    xr_dataset, xs_dataset = Subset(xr_dataset, torch.arange(shortest_len)), Subset(xs_dataset, torch.arange(shortest_len))
    dataset = ParallelDataset(xr_dataset, xs_dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    thickness_values = torch.zeros(len(alpha_beta_list), shortest_len)
    
    t = torch.linspace(0.0, 1.0, num_points)
    t = t.to(device)
    t = t.float()
    one_minus_t = 1 - t    
    for j, (xr, xr_label, xs, xs_label) in enumerate(loader):
        xr, xr_label, xs, xs_label = xr.float().to(device), xr_label.long().to(device), xs.float().to(device), xs_label.long().to(device)
        batch_size = xr.size(0) # used to handle when last batch isn't a complete one
        
        # Sample points along the line segment between the current batch's xr and xs pairs
        sampled_points = torch.einsum("i,a...->ia...", t, xr) + torch.einsum("i,a...->ia...", one_minus_t, xs)
        sampled_points = torch.transpose(sampled_points, 0, 1) # [batch_size, num_points, xr.size()]
        sampled_points = sampled_points.reshape(batch_size * num_points, *xr.size()[1:]) # [batch_size * num_points, xr.size()]
        
        # Resize labels for torch.gather()
        xr_label = xr_label.unsqueeze(dim=1).unsqueeze(dim=2)
        xs_label = xs_label.unsqueeze(dim=1).unsqueeze(dim=2)
        xr_label = xr_label.repeat(1, num_points, 1)
        xs_label = xs_label.repeat(1, num_points, 1) 
        
        # Compute the difference in the model's posterior class probabilities
        logits_sampled = model(sampled_points) # [batch_size * num_points, num_classes]
        class_probs = F.softmax(logits_sampled, dim=-1) # [batch_size * num_points, num_classes]
        class_probs = class_probs.view(batch_size, num_points, class_probs.size(-1))
        g_ij = torch.gather(class_probs, dim=2, index=xr_label) - torch.gather(class_probs, dim=2, index=xs_label)
        g_ij = g_ij.view(batch_size, num_points) # [batch_size, num_points]
        
        # Use difference in probabilities to compute thickness
        data_dimensions = [i+1 for i in range(len(xr[0].size()))]
        dist = torch.norm(xr - xs, p=2, dim=data_dimensions).squeeze() # [batch_size] 
        for i, (alpha, beta) in enumerate(alpha_beta_list):
            # Only use (xr, xs) pairs that cross the decision boundary
            mask = torch.logical_and(torch.min(g_ij, dim=1).values < 0, 
                                     torch.max(g_ij, dim=1).values > 0).squeeze()
            masked_dist = mask * dist
            
            # Integrate and multiply by distance
            line_segment_fraction = torch.logical_and((alpha <= g_ij), (g_ij <= beta)) # [batch_size, num_points]
            curr_batch_thicknesses = masked_dist * torch.sum(line_segment_fraction, dim=1) / num_points # [batch_Size]
            thickness_values[i, j*batch_size:(j+1)*batch_size] = curr_batch_thicknesses

    return thickness_values