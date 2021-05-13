import numpy as np
import pandas as pd
import torch
import os
import argparse
from torch.utils.data import DataLoader, sampler, TensorDataset, ConcatDataset
from attack_functions import *
from trojai_utils import *
from boundary_geometry import *

parser = argparse.ArgumentParser(description="TrojAI Round 5 script for boundary thickness and tilting")
parser.add_argument('--N', type=int, help="number of embeddings of each class to use")
parser.add_argument('--embedding-type', type=str, 
                    choices = ['GPT-2', 'BERT', 'DistilBERT'], 
                    help='use which embedding')
parser.add_argument('--architecture-type', type=str, 
                    choices = ['GruLinear', 'LstmLinear'], 
                    help='use which architecture')
parser.add_argument('--batch-size', type=int, 
                    help='Batch size for the adversarial attacks')
parser.add_argument('--eps', type=float,
                    help='PGD attack strength')
parser.add_argument('--iters', type=int,
                    help='PGD attack iterations')
args = parser.parse_args()

# For Round 5 (change as needed based on your file system's structure)
THICK_NAMES = ["clean", "adv+to-", "adv-to+", "uap+to-", "uap-to+"]
TILT_NAMES = ["adv_adv+to-", "adv_adv-to+", "uap_uap+to-", "uap_uap-to+"]
BASE_EMBEDDINGS_PATH = "your embedding path"
RESULTS_PATH_TRAIN = "your train results path"
RESULTS_PATH_TEST = "your test results path"
RESULTS_PATH_HOLDOUT = "your holdout results path"
METADATA_TRAIN = pd.read_csv("place where training set's METADATA.csv is")
METADATA_TEST = pd.read_csv("place where test set's METADATA.csv is")
METADATA_HOLDOUT = pd.read_csv("place where holdout set's METADATA.csv is")
TRAIN_BASE_PATH = "point me to round5-train-dataset"
TEST_BASE_PATH = "point me to round5-test-dataset"
HOLDOUT_BASE_PATH = "point me to round5-holdout-dataset"

# Round 5 reference models (50 per (embedding, architecture) type)
REF_IDS = {
    "BERT": {"LstmLinear": [14, 68, 73, 74, 98, 110, 123, 138, 163, 168, 196, 234, 240, 256, 263, 274, 299, 303, 318, 320, 349, 364, 389, 395, 405, 422, 446, 450, 463, 503, 512, 517, 524, 526, 533, 542, 563, 576, 599, 605, 617, 643, 646, 706, 707, 709, 710, 716, 719, 720], 
             "GruLinear":  [20, 22, 30, 47, 67, 69, 79, 87, 92, 93, 97, 109, 112, 122, 152, 157, 165, 171, 175, 178, 181, 183, 185, 187, 190, 220, 230, 266, 273, 279, 294, 315, 322, 334, 336, 342, 354, 404, 415, 421, 431, 474, 477, 491, 497, 499, 502, 506, 511, 519]},
    "DistilBERT": {"LstmLinear": [2, 12, 83, 86, 104, 105, 127, 131, 134, 135, 141, 156, 159, 201, 243, 244, 254, 272, 288, 310, 321, 332, 374, 377, 387, 398, 399, 416, 427, 445, 449, 460, 464, 483, 510, 523, 532, 537, 541, 543, 551, 570, 583, 588, 631, 648, 669, 670, 673, 678], 
                   "GruLinear":  [8, 17, 39, 41, 42, 45, 49, 55, 63, 76, 90, 96, 103, 149, 153, 176, 177, 179, 184, 193, 204, 208, 213, 231, 239, 245, 265, 270, 306, 347, 348, 350, 365, 371, 384, 391, 396, 419, 423, 425, 467, 468, 476, 487, 500, 516, 527, 529, 531, 548]},
    "GPT-2": {"LstmLinear": [13, 18, 29, 48, 61, 72, 80, 88, 95, 100, 108, 114, 121, 132, 151, 158, 161, 162, 197, 198, 226, 228, 258, 264, 285, 304, 312, 317, 325, 333, 337, 345, 351, 368, 373, 386, 401, 403, 418, 426, 433, 461, 466, 472, 479, 493, 507, 508, 514, 530], 
              "GruLinear":  [3, 7, 28, 32, 36, 52, 59, 71, 82, 89, 124, 126, 128, 148, 154, 191, 205, 206, 207, 224, 236, 237, 241, 246, 251, 253, 259, 260, 278, 284, 287, 289, 301, 335, 356, 360, 362, 366, 367, 378, 409, 411, 438, 471, 478, 485, 509, 513, 546, 547]}
}
UAP_MIN_SUCCESS_RATE = .80

dtype = torch.float32
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def filter_dataset(models, ds):
    # Keep all datapoints that are correct for at least 70% the models
    filtered_ds = []
    for _, (x, y) in enumerate(DataLoader(ds, batch_size=args.batch_size)):
        successes = torch.zeros(len(y), device=device)
        for model in models:
            pred = torch.argmax(model(x), dim=-1)
            successes += (pred == y)
        correct_pred_ids = successes >= (0.7 * len(models))
        filtered_ds.append(TensorDataset(x[correct_pred_ids], y[correct_pred_ids]))
    return ConcatDataset(filtered_ds)

def make_perturbed_datasets(models, pos_ds, neg_ds, batch_size, attack_type, eps, iters, step_size):    
    # Run the attack
    attack = PGDAdversarialDataset(models, eps=eps, step_size=step_size, iters=iters, p=2, universal=(attack_type=="uap"))
    attacked_pos_ds, pos_loss_final = make_adversarial_dataset(pos_ds, attack, batch_size)
    attacked_neg_ds, neg_loss_final = make_adversarial_dataset(neg_ds, attack, batch_size)
    
    # Verify success
    mean_psr, mean_nsr = 0, 0
    for model in models:
        psr = flip_success(attacked_pos_ds, 0, model) # + == 1, so want it to flip to 0
        nsr = flip_success(attacked_neg_ds, 1, model) # - == 0, so want it to flip to 1
        mean_psr, mean_nsr = (mean_psr + psr / len(models)), (mean_nsr + nsr / len(models))
        if not (psr > UAP_MIN_SUCCESS_RATE and nsr > UAP_MIN_SUCCESS_RATE):
            print("psr {}, nsr {} failed to pass threshold {}".format(psr, nsr, UAP_MIN_SUCCESS_RATE))
            raise RuntimeError()
    print(mean_psr, mean_nsr)
    return attacked_pos_ds, attacked_neg_ds, pos_loss_final, neg_loss_final

def compute_geometry(pos_ds, neg_ds, batch_size, eps, iters, step_size):
    # Get reference model's datasets
    ref_model_ids = REF_IDS[args.embedding_type][args.architecture_type]
    ref_models = [load_model(ref_model_id, TRAIN_BASE_PATH)[0] for ref_model_id in ref_model_ids]
    ref_filt_pos_ds, ref_filt_neg_ds = filter_dataset(ref_models, pos_ds), filter_dataset(ref_models, neg_ds)
    print("\t ref model filter dataset lengths:", len(ref_filt_pos_ds), len(ref_filt_neg_ds))
    ref_adv_pos_ds, ref_adv_neg_ds, _, _ = make_perturbed_datasets(ref_models, ref_filt_pos_ds, ref_filt_neg_ds, 
                                                             batch_size, "adv", eps, iters, step_size)
    ref_uap_pos_ds, ref_uap_neg_ds, _, _ = make_perturbed_datasets(ref_models, ref_filt_pos_ds, ref_filt_neg_ds, 
                                                             batch_size, "uap", eps, iters, step_size)
    
    # Compute features
    for which in ["clean", "poisoned"]:
        for metadata, base_path, results_path in zip([METADATA_TRAIN, METADATA_TEST, METADATA_HOLDOUT], [TRAIN_BASE_PATH, TEST_BASE_PATH, HOLDOUT_BASE_PATH], [RESULTS_PATH_TRAIN, RESULTS_PATH_TEST, RESULTS_PATH_HOLDOUT]):
            model_ids = metadata.index[(metadata.embedding==args.embedding_type)
                                     & (metadata.model_architecture==args.architecture_type)
                                     & (metadata.poisoned==(which=="poisoned"))].tolist()
    
            # Iterate over models
            for i, model_id in enumerate(model_ids):
                try:
                    # Load model and only keep samples it correctly classifies
                    model, _ = load_model(model_id, base_path)
                    filt_pos_ds, filt_neg_ds = filter_dataset([model], pos_ds), filter_dataset([model], neg_ds)
                    print("\t model {} len(filt_pos_ds): {}, len(filt_neg_ds): {}".format(model_id, 
                                                                                          len(filt_pos_ds), 
                                                                                          len(filt_neg_ds)))

                    # Make adv and UAP datasets
                    adv_pos_ds, adv_neg_ds, adv_pos_loss_final, adv_neg_loss_final = make_perturbed_datasets([model], 
                                                                                                              filt_pos_ds, 
                                                                                                              filt_neg_ds,
                                                                                                              batch_size, 
                                                                                                              "adv", 
                                                                                                              eps, 
                                                                                                              iters, 
                                                                                                              step_size)
                    uap_pos_ds, uap_neg_ds, uap_pos_loss_final, uap_neg_loss_final = make_perturbed_datasets([model], 
                                                                                                              filt_pos_ds, 
                                                                                                              filt_neg_ds,
                                                                                                              batch_size, 
                                                                                                              "uap", 
                                                                                                              eps, 
                                                                                                              iters, 
                                                                                                              step_size)

                    # Compute boundary thickness
                    xr_ds_thick = [filt_pos_ds, filt_pos_ds, filt_neg_ds, filt_pos_ds, filt_neg_ds]
                    xs_ds_thick = [filt_neg_ds, adv_pos_ds,  adv_neg_ds,  uap_pos_ds, uap_neg_ds]
                    for xr_ds, xs_ds, file_suffix in zip(xr_ds_thick, xs_ds_thick, THICK_NAMES):
                        # NOTE: batch_size in boundary_thickness has no effect on the statistical accuracy of the 
                        # computation, it only affects how many inputs go through the DNN at a time. We have to 
                        # set it to a low value (32, for our TrojAI experiments) since we sample 1000 points along 
                        # the line segment between each pair of inputs, implying 32 * 1000 points are going through
                        # the DNN at a time; feel free to adjust based on how powerful your GPUs are
                        thick = boundary_thickness(xr_ds, xs_ds, model, [(0, 0.75), (0, 1)], batch_size=32, num_points=1000)
                        torch.save(thick, os.path.join(results_path, 
                                                       args.embedding_type, 
                                                       args.architecture_type, 
                                                       which + file_suffix + "_thickness{}.pt".format(model_id)))

                    # Compute boundary tilting
                    xr_ds_tilt =     [filt_pos_ds,    filt_neg_ds,    filt_pos_ds,    filt_neg_ds]
                    xr_adv_ds_tilt = [adv_pos_ds,     adv_neg_ds,     uap_pos_ds,     uap_neg_ds]
                    xs_ds_tilt =     [ref_adv_pos_ds, ref_adv_neg_ds, ref_uap_pos_ds, ref_uap_neg_ds]        
                    for xr_ds, xs_ds, xr_adv_ds, file_suffix in zip(xr_ds_tilt, xs_ds_tilt, xr_adv_ds_tilt, TILT_NAMES):
                        tilt = boundary_tilting(xr_ds, xs_ds, xr_adv_ds, model, batch_size=args.batch_size, reduce_clean=False)
                        torch.save(tilt, os.path.join(results_path, 
                                                      args.embedding_type,
                                                      args.architecture_type, 
                                                      which + file_suffix + "_tilting{}.pt".format(model_id)))

                except Exception:
                    print("Failed for model_id {}".format(model_id))

                # Print progress
                print("{0} of {1} {2} models done".format(i, len(model_ids), which))

def get_dataset(embeddings, labels):
    embeddings = embeddings.to("cuda")
    labels = labels.to("cuda")
    dataset = torch.utils.data.TensorDataset(embeddings, labels)
    return dataset

# Load in embeddings to use
pos_embeddings = torch.load(os.path.join(BASE_EMBEDDINGS_PATH, args.embedding_type, "pos_embeddings{}.pt".format(args.N)))
pos_labels = torch.load(os.path.join(BASE_EMBEDDINGS_PATH, args.embedding_type, "pos_labels{}.pt".format(args.N)))
pos_ds = get_dataset(pos_embeddings, pos_labels)

neg_embeddings = torch.load(os.path.join(BASE_EMBEDDINGS_PATH, args.embedding_type, "neg_embeddings{}.pt".format(args.N)))
neg_labels = torch.load(os.path.join(BASE_EMBEDDINGS_PATH, args.embedding_type, "neg_labels{}.pt".format(args.N)))
neg_ds = get_dataset(neg_embeddings, neg_labels)

# Compute and save features
step_size = 2 * args.eps / args.iters
compute_geometry(pos_ds, neg_ds, args.batch_size, args.eps, args.iters, step_size)