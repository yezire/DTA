from model import CreaTDA_og
import numpy as np
import torch
import os
import argparse
import random
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import row_normalize, get_pre_args, set_seed
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


def predict(args, TDAtrain):
    set_seed(args)
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
    model_path = 'models/CreaTDA_{}_retrain.pth'.format(args.model)
    if args.model == 'CreaTDA_og':
        model = CreaTDA_og(args, num_drug, num_disease,
                           num_protein, num_sideeffect)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    ground_truth_train = [ele[2] for ele in TDAtrain]
    best_train_aupr = 0
    best_train_auc = 0

    model.eval()
    if args.model == 'CreaTDA_og':
        tloss, tdaloss, results = model(drug_drug_normalize, drug_chemical_normalize, drug_disease_normalize,
                                        drug_sideeffect_normalize, protein_protein_normalize, protein_sequence_normalize,
                                        protein_disease_normalize, disease_drug_normalize, disease_protein_normalize,
                                        sideeffect_drug_normalize, drug_protein_normalize, protein_drug_normalize,
                                        drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein,
                                        protein_sequence, protein_disease, drug_protein, mask)
    pred_list_train = [results[ele[0], ele[1]] for ele in TDAtrain]
    train_auc = roc_auc_score(ground_truth_train, pred_list_train)
    train_aupr = average_precision_score(ground_truth_train, pred_list_train)
    print('auc aupr', train_auc, train_aupr)

    return results


def get_topk(args, output, tda):
    # 1. 3-sigma rule for significance for each column (disease)
    output_mean = np.mean(output, axis=0)  # mean over proteins
    output_std = np.std(output, axis=0)
    sig_thresh = output_mean + 2 * output_std
    # print(output_mean.shape, output_std.shape, tda.shape)
    list_of_indices = []
    for i in range(output.shape[1]):
        column = output[:, i]
        sig_indices = np.where(column >= sig_thresh[i])[0]  # array of indices
        if len(sig_indices) > 0:
            list_of_indices.extend([(idx, i) for idx in sig_indices])
    new_output = np.zeros_like(output)
    for tup in list_of_indices:
        new_output[tup[0], tup[1]] = output[tup[0], tup[1]]
    new_output = new_output * (1-tda)
    tensor_output = torch.tensor(new_output.flatten())
    top_k = torch.topk(tensor_output, args.top).indices.numpy()
    real_indices = np.unravel_index(top_k, tda.shape)
    # print(real_indices[0])
    real_indices = np.array(real_indices).T
    return real_indices


def plot_bias_figs(output, tda, indices):
    # plot max scores
    num_known_diseases = np.sum(tda, axis=1)   # num_proteins

    max_score_proteins = np.max(output, axis=1)

    protein_score_association_corr = spearmanr(
        num_known_diseases, max_score_proteins)

    fig, ax = plt.subplots()
    ax.scatter(num_known_diseases, max_score_proteins, c='dodgerblue', s=1)
    fig.tight_layout()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(11)
    fig.savefig("bias_figs/max_scores_{}.png".format(args.model),
                bbox_inches='tight', pad_inches=0.1, dpi=400)
    plt.close(fig)
    print("{}: protein max scores and known associated diseases.\n correlation: {:.3f} p-value: {:.3f}".format(
        args.model, protein_score_association_corr[0], protein_score_association_corr[1]))


if __name__ == '__main__':
    args = get_pre_args()
    set_seed(args)
    device = torch.device("cuda:{}".format(args.device)
                          ) if args.device >= 0 else torch.device("cpu")

    if not (args.model == 'DTINet' or args.model == 'GTN' or args.model == 'HGT' or args.model == 'RGCN'):
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

        protein_protein_normalize = row_normalize(
            protein_protein, True).to(device)
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
        print('Predicting ...')
        unfiltered_output = predict(args=args, TDAtrain=data_set)
        np.save("topk_indices/output_raw_{}.npy".format(args.model),
                unfiltered_output, allow_pickle=True)

    # TODO Here
    unfiltered_output = np.load(
        "topk_indices/output_raw_{}.npy".format(args.model), allow_pickle=True)
    # only those positions on which ground-truth=0
    output = np.where(tda_o != 1, unfiltered_output, 0)

    # compute top indices and correlation
    indices = get_topk(args, unfiltered_output, tda_o)
    print("correlation between top {} ".format(args.top))

    # TODO get bias figs
    plot_bias_figs(output, tda_o, indices)

    protein_dict = {}
    drug_dict = {}
    with open('../Data/protein_dict_map.txt', 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                key, value = line.split(':')
                protein_dict[key] = value

    with open('../Data/drug_dict_map.txt', 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                key, value = line.split(':')
                drug_dict[key] = value

    # write to file
    os.chdir(inter_path)
    result_path = os.path.abspath(
        "../../Output/results/{}_results.tsv".format(args.model))
    with open("protein.txt", "r") as protein_txt, open("disease.txt", "r") as disease_txt, open(result_path, "w+") as results:
        proteins = protein_txt.readlines()
        diseases = disease_txt.readlines()
        results.write('protein_idx'+'\t'+"disease_idx"+'\t'+'disease'+'\t' +
                      'protein_number'+'\t'+'protein_name'+'\t'+'output'+'\t'+'co'+'\n')
        for i in range(len(indices)):
            protein_idx = indices[i][0]
            disease_idx = indices[i][1]
            # print(disease_idx)
            disease = diseases[disease_idx].strip()
            protein_number = proteins[protein_idx].strip()
            protein_name = protein_dict[protein_number]
            # print(output.shape)
            pred = output[protein_idx, disease_idx]
            # results.write(str(protein_idx)+'\t'+str(disease_idx)+'\t'+ disease + '\t'+ protein_number+'\t'+protein_name+'\t'+ str(pred) + str(label) +'\n')
            results.write('{}\t{}\t{}\t{}\t{}\t{:.3f}\n'.format(
                protein_idx, disease_idx, disease, protein_number, protein_name, pred))
