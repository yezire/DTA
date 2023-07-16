from model import CreaTDA_og
import numpy as np
import torch
import os
import argparse
import random
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import row_normalize,get_re_args,set_seed
from tqdm import tqdm

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
os.chdir(BASE_DIR)
os.makedirs("models", exist_ok=True)
torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(True)


def retrain(args, TDAtrain, verbose=True):
    protein_disease = np.zeros((num_protein, num_disease))
    mask = np.zeros((num_protein, num_disease))
    for ele in TDAtrain:
        protein_disease[ele[0], ele[1]] = ele[2]
        mask[ele[0], ele[1]] = 1
    disease_protein = protein_disease.T

    disease_protein_normalize = row_normalize(
        disease_protein, False).to(device)
    protein_disease_normalize = row_normalize(
        protein_disease, False).to(device)
    protein_disease = torch.Tensor(protein_disease).to(device)
    mask = torch.Tensor(mask).to(device)

    if args.model == 'CreaTDA_og':
        model = CreaTDA_og(args, num_drug, num_disease,
                           num_protein, num_sideeffect)

    model.to(device)
    no_decay = ["bias"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = torch.optim.Adam(
        optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', factor=0.8, patience=2)

    ground_truth_train = [ele[2] for ele in TDAtrain]
    best_train_aupr = 0
    best_train_auc = 0
    for i in tqdm(range(args.num_steps),desc='retrain step'):
        model.train()
        model.zero_grad()
        if args.model == 'CreaTDA_og':
            tloss, tdaloss, results = model(drug_drug_normalize, drug_chemical_normalize, drug_disease_normalize,
                                            drug_sideeffect_normalize, protein_protein_normalize, protein_sequence_normalize,
                                            protein_disease_normalize, disease_drug_normalize, disease_protein_normalize,
                                            sideeffect_drug_normalize, drug_protein_normalize, protein_drug_normalize,
                                            drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein,
                                            protein_sequence, protein_disease, drug_protein, mask)

        tloss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.n)
        optimizer.step()
        if i % 25 == 0 and verbose == True:
            print('step', i, 'total and tda loss',
                  tloss.item(), tdaloss.item())
            model.eval()
            pred_list_train = [results[ele[0], ele[1]] for ele in TDAtrain]
            train_auc = roc_auc_score(ground_truth_train, pred_list_train)
            train_aupr = average_precision_score(
                ground_truth_train, pred_list_train)
            scheduler.step(train_aupr)
            if train_aupr > best_train_aupr:
                best_train_aupr = train_aupr
                best_train_auc = train_auc
                torch.save(model.state_dict(),
                           "models/CreaTDA_{}_retrain.pth".format(args.model))
            print('train auc aupr', train_auc, train_aupr)

    return best_train_auc, best_train_aupr


if __name__ == '__main__':
    args = get_re_args()
    set_seed(args)
    device = torch.device("cuda:{}".format(args.device)
                          ) if args.device >= 0 else torch.device("cpu")
    inter_path = '../Data/InteractionData/'
    sim_path = '../Data/SimilarityData/'
    print('loading networks ...')
    drug_drug = np.loadtxt(inter_path+'mat_drug_drug.txt')
    # First [0:708] are drugs, the rest are compounds retrieved from ZINC15 database
    true_drug = 708
    drug_chemical = np.loadtxt(sim_path+'Similarity_Matrix_Drugs.txt')
    drug_chemical = drug_chemical[:true_drug, :true_drug]
    drug_disease = np.loadtxt(inter_path+'mat_drug_disease.txt')
    drug_sideeffect = np.loadtxt(inter_path+'mat_drug_se.txt')
    disease_drug = drug_disease.T
    sideeffect_drug = drug_sideeffect.T

    protein_protein = np.loadtxt(inter_path+'mat_protein_protein.txt')
    protein_sequence = np.loadtxt(
        sim_path+'Similarity_Matrix_Proteins.txt')
    protein_drug = np.loadtxt(inter_path+'mat_protein_drug.txt')
    drug_protein = protein_drug.T

    print('normalize network for mean pooling aggregation')
    drug_drug_normalize = row_normalize(drug_drug, True).to(device)
    drug_chemical_normalize = row_normalize(drug_chemical, True).to(device)
    drug_protein_normalize = row_normalize(drug_protein, False).to(device)
    drug_disease_normalize = row_normalize(drug_disease, False).to(device)
    drug_sideeffect_normalize = row_normalize(
        drug_sideeffect, False).to(device)

    protein_protein_normalize = row_normalize(protein_protein, True).to(device)
    protein_sequence_normalize = row_normalize(
        protein_sequence, True).to(device)
    protein_drug_normalize = row_normalize(protein_drug, False).to(device)

    disease_drug_normalize = row_normalize(disease_drug, False).to(device)

    sideeffect_drug_normalize = row_normalize(
        sideeffect_drug, False).to(device)

    # define computation graph
    num_drug = len(drug_drug_normalize)
    num_protein = len(protein_protein_normalize)
    num_disease = len(disease_drug_normalize)
    num_sideeffect = len(sideeffect_drug_normalize)

    drug_drug = torch.Tensor(drug_drug).to(device)
    drug_chemical = torch.Tensor(drug_chemical).to(device)
    drug_disease = torch.Tensor(drug_disease).to(device)
    drug_sideeffect = torch.Tensor(drug_sideeffect).to(device)
    protein_protein = torch.Tensor(protein_protein).to(device)
    protein_sequence = torch.Tensor(protein_sequence).to(device)
    protein_drug = torch.Tensor(protein_drug).to(device)
    drug_protein = torch.Tensor(drug_protein).to(device)

    # prepare drug_protein and mask
    tda_o = np.loadtxt(inter_path+'mat_protein_disease.txt')
    whole_positive_index = []
    whole_negative_index = []
    for i in range(np.shape(tda_o)[0]):
        for j in range(np.shape(tda_o)[1]):
            if int(tda_o[i][j]) == 1:
                whole_positive_index.append([i, j])
            elif int(tda_o[i][j]) == 0:
                whole_negative_index.append([i, j])
    negative_sample_index = np.arange(len(whole_negative_index))
    data_set = np.zeros((len(negative_sample_index) +
                        len(whole_positive_index), 3), dtype=int)
    count = 0
    for i in whole_positive_index:
        data_set[count][0] = i[0]
        data_set[count][1] = i[1]
        data_set[count][2] = 1
        count += 1
    for i in negative_sample_index:
        data_set[count][0] = whole_negative_index[i][0]
        data_set[count][1] = whole_negative_index[i][1]
        data_set[count][2] = 0
        count += 1
    print('Retraning Network')
    retrain(args=args, TDAtrain=data_set)
