import global_var
from utils import row_normalize, get_cv_args, set_seed
from model import CreaTDA_og
import numpy as np
import torch
import os

# global_var._init()
# inter_path = '../Data/InteractionData/'
# sim_path = '../Data/SimilarityData/'
# print('loading networks ...')
# global_var.set_value('drug_drug', np.loadtxt(inter_path+'mat_drug_drug.txt'))
# true_drug = 708
# drug_chemical = np.loadtxt(sim_path+'Similarity_Matrix_Drugs.txt')
# global_var.set_value('drug_chemical', drug_chemical[:true_drug, :true_drug])
# global_var.set_value('drug_disease', np.loadtxt(
#     inter_path+'mat_drug_disease.txt'))
# global_var.set_value('drug_drug', np.loadtxt(inter_path+'mat_drug_drug.txt'))
# global_var.set_value('drug_drug', np.loadtxt(inter_path+'mat_drug_drug.txt'))
# global_var.set_value('drug_drug', np.loadtxt(inter_path+'mat_drug_drug.txt'))
# global_var.set_value('drug_drug', np.loadtxt(inter_path+'mat_drug_drug.txt'))
# # First [0:708] are drugs, the rest are compounds retrieved from ZINC15 database

# drug_disease = np.loadtxt(inter_path+'mat_drug_disease.txt')
# drug_sideeffect = np.loadtxt(inter_path+'mat_drug_se.txt')
# disease_drug = drug_disease.T
# sideeffect_drug = drug_sideeffect.T

# protein_protein = np.loadtxt(inter_path+'mat_protein_protein.txt')
# protein_sequence = np.loadtxt(
#     sim_path+'Similarity_Matrix_Proteins.txt')
# protein_drug = np.loadtxt(inter_path+'mat_protein_drug.txt')
# drug_protein = protein_drug.T
