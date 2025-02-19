
import ipdb
from models.GHGNN.ghgnn import GH_GNN
from models.GHGNN.ghgnn_old import GH_GNN_old
import pandas as pd
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser


# python src/models/01_predict_IDACs_GH_GNN.py --version combined

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--version", help="GH-GNN version to use", type=str)

    args = parser.parse_args()
    version = args.version

    assert version in ['combined', 'organic', 'organic_old']

    df = pd.read_csv('data/processed/kdb_vle.csv')

    # Predict IDACs with GH-GNN
    ln_IDAC_12_lst, ln_IDAC_21_lst, K1_12_lst, K2_12_lst, K1_21_lst, K2_21_lst = [], [], [], [], [], []
    jaccard_distance = []
    for i, row in tqdm(df.iterrows(), desc='Predicting with GH-GNN', total=df.shape[0]):
        smiles_1, smiles_2, T = row['SMILES_1'], row['SMILES_2'], row['T_K']

        if version == 'combined':
            T = T + 273.15 # due to moved training of combined GHGNN
        
        if version == 'organic_old':
            ghgnn_12 = GH_GNN_old(smiles_1, smiles_2)
            ghgnn_21 = GH_GNN_old(smiles_2, smiles_1)
        else:
            ghgnn_12 = GH_GNN(smiles_1, smiles_2, version)
            ghgnn_21 = GH_GNN(smiles_2, smiles_1, version)
        
        K1_12, K2_12 = ghgnn_12.predict(T, constants=True)
        K1_21, K2_21 = ghgnn_21.predict(T, constants=True)
        
        K1_12_lst.append(K1_12)
        K2_12_lst.append(K2_12)
        K1_21_lst.append(K1_21)
        K2_21_lst.append(K2_21)

        ln_IDAC_12 = K1_12 + K2_12/T
        ln_IDAC_21 = K1_21 + K2_21/T

        ln_IDAC_12_lst.append(ln_IDAC_12)
        ln_IDAC_21_lst.append(ln_IDAC_21)

        # Check Jaccard distance metric for measuring applicability domain
        _, jaccard_similarity_1 = ghgnn_12.get_AD('tanimoto')
        _, jaccard_similarity_2 = ghgnn_21.get_AD('tanimoto')

        if jaccard_similarity_1 > 1 and jaccard_similarity_2 > 1:
            jaccard_distance.append(0)
        else:
            jaccard_distance.append(max(1-jaccard_similarity_1, 1-jaccard_similarity_2))


    df['pred_ln_IDAC_12'], df['pred_ln_IDAC_21'] = ln_IDAC_12_lst, ln_IDAC_21_lst
    df['K1_12'], df['K2_12'], df['K1_21'], df['K2_21'] = K1_12_lst, K2_12_lst, K1_21_lst, K2_21_lst
    df['Jaccard_distance'] = jaccard_distance


    df.to_csv(f'models/GHGNN/kdb_vle_IDACs_pred_{version}.csv', index=False)

    
