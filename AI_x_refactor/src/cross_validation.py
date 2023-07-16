from utils import row_normalize, get_cv_args, set_seed
from model import CreaTDA_og
import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
os.chdir(BASE_DIR)

torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(True)

os.makedirs('tda_disease', exist_ok=True)
os.makedirs('rand_mask', exist_ok=True)


def train_and_evaluate(args, TDAtrain, TDAvalid, TDAtest, verbose=True):

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
    ground_truth_valid = [ele[2] for ele in TDAvalid]
    ground_truth_test = [ele[2] for ele in TDAtest]

    best_valid_aupr = 0
    best_valid_auc = 0
    test_aupr = 0
    test_auc = 0
    for i in tqdm(range(args.num_steps),desc='train_and_evaluate steps'):
        model.train()
        model.zero_grad()
        if args.model == 'CreaTDA_og':
            tloss, tdaloss, results = model(drug_drug_normalize, drug_chemical_normalize, drug_disease_normalize,
                                            drug_sideeffect_normalize, protein_protein_normalize, protein_sequence_normalize,
                                            protein_disease_normalize, disease_drug_normalize, disease_protein_normalize,
                                            sideeffect_drug_normalize, drug_protein_normalize, protein_drug_normalize,
                                            drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein,
                                            protein_sequence, protein_disease, drug_protein, mask)
        # print(results)
        tloss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.n)
        optimizer.step()
        if i % 25 == 0 and verbose == True:
            print('step', i, 'total and tda loss',
                  tloss.item(), tdaloss.item())
            model.eval()
            pred_list_valid = [results[ele[0], ele[1]] for ele in TDAvalid]
            valid_auc = roc_auc_score(ground_truth_valid, pred_list_valid)
            valid_aupr = average_precision_score(
                ground_truth_valid, pred_list_valid)

            pred_list_train = [results[ele[0], ele[1]] for ele in TDAtrain]
            train_auc = roc_auc_score(ground_truth_train, pred_list_train)
            train_aupr = average_precision_score(
                ground_truth_train, pred_list_train)
            scheduler.step(valid_aupr)
            if i >= 50:
                if valid_aupr >= best_valid_aupr:
                    best_valid_aupr = valid_aupr
                    best_valid_auc = valid_auc
                    pred_list_test = [results[ele[0], ele[1]]
                                      for ele in TDAtest]
                    test_auc = roc_auc_score(ground_truth_test, pred_list_test)
                    test_aupr = average_precision_score(
                        ground_truth_test, pred_list_test)
                print('train auc aupr', train_auc, train_aupr, 'valid auc aupr',
                      valid_auc, valid_aupr, 'test auc aupr', test_auc, test_aupr)
            else:
                print('train auc aupr', train_auc, train_aupr,
                      'valid auc aupr', valid_auc, valid_aupr)

    return best_valid_auc, best_valid_aupr, test_auc, test_aupr


if __name__ == '__main__':
    args = get_cv_args()
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
    test_auc_round = []
    test_aupr_round = []
    val_auc_round = []
    val_aupr_round = []

    tda_o = np.loadtxt(inter_path+'mat_protein_disease.txt')

    whole_positive_index = []
    whole_negative_index = []
    for i in range(np.shape(tda_o)[0]):
        for j in range(np.shape(tda_o)[1]):
            if int(tda_o[i][j]) == 1:
                whole_positive_index.append([i, j])
            elif int(tda_o[i][j]) == 0:
                whole_negative_index.append([i, j])

    for r in tqdm(range(args.round),desc='cross validation round'):
        print('sample round', r+1)
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

        val_auc_fold = []
        val_aupr_fold = []
        test_auc_fold = []
        test_aupr_fold = []
        rs = np.random.randint(0, 1000, 1)[0]
        print(rs)
        if args.mask == 'random':
            skf = StratifiedKFold(n_splits=args.n_folds,
                                  shuffle=True, random_state=rs)
            for train_index, test_index in skf.split(np.arange(len(data_set)), data_set[:, 2]):
                TDAtrain, TDAtest = data_set[train_index], data_set[test_index]
                TDAtrain, TDAvalid = train_test_split(
                    TDAtrain, test_size=args.test_size, random_state=rs)
                print(len(TDAtrain), len(TDAvalid), len(TDAtest))
                v_auc, v_aupr, t_auc, t_aupr = train_and_evaluate(
                    args=args, TDAtrain=TDAtrain, TDAvalid=TDAvalid, TDAtest=TDAtest)
                val_auc_fold.append(v_auc)
                val_aupr_fold.append(v_aupr)
                test_auc_fold.append(t_auc)
                test_aupr_fold.append(t_aupr)
            # break
            val_auc_round.append(np.mean(val_auc_fold))
            val_aupr_round.append(np.mean(val_aupr_fold))
            test_auc_round.append(np.mean(test_auc_fold))
            test_aupr_round.append(np.mean(test_aupr_fold))
            np.savetxt('rand_mask/val_auc_{}'.format(args.model), val_auc_round)
            np.savetxt('rand_mask/val_aupr_{}'.format(args.model),
                       val_aupr_round)
            np.savetxt('rand_mask/test_auc_{}'.format(args.model),
                       test_auc_round)
            np.savetxt('rand_mask/test_aupr_{}'.format(args.model),
                       test_aupr_round)
