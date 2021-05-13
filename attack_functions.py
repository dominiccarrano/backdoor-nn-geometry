"""Adapted from code at https://github.com/nsfzyzz/boundary_thickness."""
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, sampler, TensorDataset, ConcatDataset

class Attacks(object):
    """
    An abstract class representing attacks.
    Arguments:
        name (string): name of the attack.
        model (nn.Module): a model to attack.
    .. note:: device("cpu" or "cuda") will be automatically determined by a given model.
    """
    def __init__(self, name, model):
        self.attack = name
        self.model = model.train()
        self.model_name = str(model).split("(")[0]
        self.device = torch.device("cuda" if next(model.parameters()).is_cuda else "cpu")

    # Whole structure of the model will be NOT displayed for pretty print.
    def __str__(self):
        info = self.__dict__.copy()
        del info['model']
        del info['attack']
        return self.attack + "(" + ', '.join('{}={}'.format(key, val) for key, val in info.items()) + ")"

class PGDAdversarialDataset(Attacks):
    """
    Parameters
    ==========
    models: list of torch.nn.Module
        List of models you want to simultaneously fool.
    eps: float
        The attack range, i.e. size of the Lp norm ball to project the iterates back onto.
    step_size: float
        The step size for PGD.
    iters: int
        The number of PGD iterations.
    p: float
        The index p of the Lp norm ball to project the iterates back onto.
    universal: bool
        Whether to do a UAP attack or generate a separate adversarial direction per sample.
    n_restarts: int
        The number of random restarts to use.
    """
    def __init__(self, models, eps, step_size, iters, p, universal, n_restarts=2):
        super(PGDAdversarialDataset, self).__init__("PGDAdversarialDataset", models[0])
        self.models = models
        self.eps = eps
        self.step_size = step_size
        self.iters = iters
        self.eps_for_division = 1e-10
        self.p = p
        self.n_restarts = n_restarts
        self.universal = universal
    
    def __call__(self, dataset, batch_size): 
        for model in self.models:
            model.train() # For torch.autograd.grad        
        loss = nn.CrossEntropyLoss()
        data_shape = [len(dataset)] + list(dataset[0][0].shape)
        
        # Learn best trigger
        lowest_loss = float("inf")
        best_trigger = None
        
        for restart_id in range(self.n_restarts):
            # Trigger setup, initialize to random normal
            if self.universal:
                trigger_init = .1 * torch.randn(1, *data_shape[1:], device=self.device)
            else:
                trigger_init = .1 * torch.randn(*data_shape, device=self.device)
            trigger = Variable(trigger_init, requires_grad=True)

            # Mini-batch PGD
            for epoch in range(1, self.iters+1):
                for j, (x, y) in enumerate(DataLoader(dataset, batch_size=batch_size, shuffle=False)):
                    x, y = x.to(self.device).float(), y.to(self.device).long()
                    samples_in_batch = x.size(0)
                    
                    trigger.requires_grad_()
                    y_perturbed = torch.remainder(y + 1, 2) # CAUTION: This only works for binary classification!

                    # Use the same adv. perturbation for all samples if a universal attack; else only optimize
                    # over the adv. perturbations for the samples in the current batch (others will get zero grad)
                    if self.universal:
                        x_perturbed = x + trigger.repeat(samples_in_batch, 1, 1)
                    else:
                        x_perturbed = x + trigger[j*batch_size:(j+1)*batch_size]

                    # Backward pass
                    current_loss = 0
                    for model in self.models:
                        model.zero_grad()                    
                        current_loss += loss(model(x_perturbed), y_perturbed).to(self.device)
                    grad = torch.autograd.grad(current_loss, [trigger])[0]
                    grad_norm = torch.norm(grad, p=self.p, dim=[d for d in range(1, len(trigger.size()))])
                    grad_norm = grad_norm.unsqueeze(dim=-1).unsqueeze(dim=-1)
                    grad_norm = grad_norm.repeat(1, 1, grad.size(-1))

                    # PGD step
                    trigger = trigger.detach() - self.step_size * grad.detach() / (grad_norm + self.eps_for_division)
                    trigger_norm = torch.norm(trigger, p=self.p, dim=[d for d in range(1, len(trigger.size()))])
                    trigger_norm = trigger_norm.unsqueeze(dim=-1).unsqueeze(dim=-1)
                    trigger_norm = trigger_norm.repeat(1, 1, trigger.size(-1))
                    factor = self.eps / (trigger_norm + self.eps_for_division)
                    factor = torch.min(factor, torch.ones_like(trigger_norm)) 
                    trigger = trigger * factor                

            # Update best result
            if current_loss.item() < lowest_loss:
                lowest_loss = current_loss.item()
                best_trigger = trigger
        
        # Group into dataset
        adversarial_dataset = []
        for j, (x, y) in enumerate(DataLoader(dataset, batch_size=batch_size, shuffle=False)):
            x, y = x.to(self.device).float(), y.to(self.device).long()
            samples_in_batch = x.size(0)
            
            if self.universal:
                x_perturbed = x + best_trigger.repeat(samples_in_batch, 1, 1)
            else:
                x_perturbed = x + best_trigger[j*batch_size:(j+1)*batch_size]
            y_perturbed = torch.remainder(y + 1, 2) # CAUTION: This only works for binary classification!
            adversarial_dataset.append(TensorDataset(x_perturbed, y_perturbed))
        
        return ConcatDataset(adversarial_dataset), lowest_loss