import pandas as pd
import ipdb
from tqdm import tqdm
from models.GHGNN.ghgnn_old import GH_GNN_old

# python src/models/07_predict_IDACs_for_ternary.py

if __name__ == "__main__":
    
    df = pd.read_csv('data/processed/ternary_vle.csv')

    # Predict IDACs with GH-GNN
    ln_IDAC_12_lst, ln_IDAC_13_lst = [], []
    ln_IDAC_21_lst, ln_IDAC_23_lst = [], []
    ln_IDAC_31_lst, ln_IDAC_32_lst = [], []

    K1_12_lst, K2_12_lst, K1_13_lst, K2_13_lst = [], [], [], []
    K1_21_lst, K2_21_lst, K1_23_lst, K2_23_lst = [], [], [], []
    K1_31_lst, K2_31_lst, K1_32_lst, K2_32_lst = [], [], [], []
    
    jaccard_distance = []
    for i, row in tqdm(df.iterrows(), desc='Predicting with GH-GNN', total=df.shape[0]):
        smiles_1, smiles_2, smiles_3, T = row['SMILES_1'], row['SMILES_2'], row['SMILES_3'], row['T_K']

        ghgnn_12 = GH_GNN_old(smiles_1, smiles_2)
        ghgnn_13 = GH_GNN_old(smiles_1, smiles_3)
        ghgnn_21 = GH_GNN_old(smiles_2, smiles_1)
        ghgnn_23 = GH_GNN_old(smiles_2, smiles_3)
        ghgnn_31 = GH_GNN_old(smiles_3, smiles_1)
        ghgnn_32 = GH_GNN_old(smiles_3, smiles_2)
        
        K1_12, K2_12 = ghgnn_12.predict(T, constants=True)
        K1_13, K2_13 = ghgnn_13.predict(T, constants=True)
        K1_21, K2_21 = ghgnn_21.predict(T, constants=True)
        K1_23, K2_23 = ghgnn_23.predict(T, constants=True)
        K1_31, K2_31 = ghgnn_31.predict(T, constants=True)
        K1_32, K2_32 = ghgnn_32.predict(T, constants=True)
        
        K1_12_lst.append(K1_12); K2_12_lst.append(K2_12); K1_13_lst.append(K1_13); K2_13_lst.append(K2_13)
        K1_21_lst.append(K1_21); K2_21_lst.append(K2_21); K1_23_lst.append(K1_23); K2_23_lst.append(K2_23)
        K1_31_lst.append(K1_31); K2_31_lst.append(K2_31); K1_32_lst.append(K1_32); K2_32_lst.append(K2_32)

        ln_IDAC_12 = K1_12 + K2_12/T
        ln_IDAC_13 = K1_13 + K2_13/T
        ln_IDAC_21 = K1_21 + K2_21/T
        ln_IDAC_23 = K1_23 + K2_23/T
        ln_IDAC_31 = K1_31 + K2_31/T
        ln_IDAC_32 = K1_32 + K2_32/T

        ln_IDAC_12_lst.append(ln_IDAC_12); ln_IDAC_13_lst.append(ln_IDAC_13)
        ln_IDAC_21_lst.append(ln_IDAC_21); ln_IDAC_23_lst.append(ln_IDAC_23)
        ln_IDAC_31_lst.append(ln_IDAC_31); ln_IDAC_32_lst.append(ln_IDAC_32)

        # Check Jaccard distance metric for measuring applicability domain
        _, jaccard_similarity_1 = ghgnn_12.get_AD('tanimoto')
        _, jaccard_similarity_2 = ghgnn_13.get_AD('tanimoto')
        _, jaccard_similarity_3 = ghgnn_21.get_AD('tanimoto')
        _, jaccard_similarity_4 = ghgnn_23.get_AD('tanimoto')
        _, jaccard_similarity_5 = ghgnn_31.get_AD('tanimoto')
        _, jaccard_similarity_6 = ghgnn_32.get_AD('tanimoto')

        if jaccard_similarity_1>1 and jaccard_similarity_2>1 and jaccard_similarity_3>1 and jaccard_similarity_4>1 and jaccard_similarity_5>1 and jaccard_similarity_6>1:
            jaccard_distance.append(0)
        else:
            jaccard_distance.append(max(1-jaccard_similarity_1, 1-jaccard_similarity_2, 1-jaccard_similarity_3, 1-jaccard_similarity_4, 1-jaccard_similarity_5, 1-jaccard_similarity_6))


    df['pred_ln_IDAC_12'], df['pred_ln_IDAC_13'] = ln_IDAC_12_lst, ln_IDAC_13_lst
    df['pred_ln_IDAC_21'], df['pred_ln_IDAC_23'] = ln_IDAC_21_lst, ln_IDAC_23_lst
    df['pred_ln_IDAC_31'], df['pred_ln_IDAC_32'] = ln_IDAC_31_lst, ln_IDAC_32_lst
    df['K1_12'], df['K2_12'] = K1_12_lst, K2_12_lst
    df['K1_13'], df['K2_13'] = K1_13_lst, K2_13_lst
    df['K1_21'], df['K2_21'] = K1_21_lst, K2_21_lst
    df['K1_23'], df['K2_23'] = K1_23_lst, K2_23_lst
    df['K1_31'], df['K2_31'] = K1_31_lst, K2_31_lst
    df['K1_32'], df['K2_32'] = K1_32_lst, K2_32_lst
    df['Jaccard_distance'] = jaccard_distance

    df.to_csv('models/GHGNN/ternary_vle_IDACs_pred.csv', index=False)