
import ipdb
import pandas as pd
from src.utils.utils import margules_binary, percentage_within_threshold
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import numpy as np
from argparse import ArgumentParser

# python src/models/02_predict_gammas_GH_GNN_Margules.py --version combined --jaccard_threshold 0.6

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--version", help="GH-GNN version to use", type=str)
    parser.add_argument("--jaccard_threshold", help="Jaccard distance threshold to be considered in the predictions", type=str)

    args = parser.parse_args()
    version = args.version
    jaccard_threshold = float(args.jaccard_threshold)

    assert version in ['combined', 'organic', 'organic_old']
    df_org = pd.read_csv(f'models/GHGNN/kdb_vle_IDACs_pred_{version}.csv')

    # Filter out systems not within Jaccard distance threshold
    df = df_org[df_org['Jaccard_distance'] <= jaccard_threshold].copy()
    print(f'Points discarded due to Jaccard threshold: {df_org.shape[0] - df.shape[0]}')

    # Eliminated suspected wrong data series
    df_wrong = pd.read_csv('data/external/suspected_wrong_KDB.csv')
    systems_to_be_eliminated = df_wrong['ID'].tolist()
    df = df[~df['ID'].isin(systems_to_be_eliminated)]
    df_updated = df.copy()

    # Flip IDACs of suspected incorrectly placed component_1 and component_2 in KDB
    df_filp = pd.read_csv('data/external/suspected_invertion_compounds_KDB.csv')
    systems_to_be_flipped = df_filp['ID'].tolist()
    
    n_points_flipped = 0
    for idx, row in df.iterrows():
        if row['ID'] in systems_to_be_flipped:
            row_flip = df_filp[df_filp['ID'] == row['ID']]
            
            df_updated.at[idx, 'Compound_1'] = df.at[idx, 'Compound_2']
            df_updated.at[idx, 'Compound_2'] = df.at[idx, 'Compound_1']

            df_updated.at[idx, 'Coefficient A_1'] = df.at[idx, 'Coefficient A_2']
            df_updated.at[idx, 'Coefficient B_1'] = df.at[idx, 'Coefficient B_2']
            df_updated.at[idx, 'Coefficient C_1'] = df.at[idx, 'Coefficient C_2']
            df_updated.at[idx, 'Coefficient D_1'] = df.at[idx, 'Coefficient D_2']
            df_updated.at[idx, 'Coefficient A_2'] = df.at[idx, 'Coefficient A_1']
            df_updated.at[idx, 'Coefficient B_2'] = df.at[idx, 'Coefficient B_1']
            df_updated.at[idx, 'Coefficient C_2'] = df.at[idx, 'Coefficient C_1']
            df_updated.at[idx, 'Coefficient D_2'] = df.at[idx, 'Coefficient D_1']

            df_updated.at[idx, 'T range, from_1'] = df.at[idx, 'T range, from_2']
            df_updated.at[idx, 'T range, to_1'] = df.at[idx, 'T range, to_2']
            df_updated.at[idx, 'T range, from_2'] = df.at[idx, 'T range, from_1']
            df_updated.at[idx, 'T range, to_2'] = df.at[idx, 'T range, to_1']

            df_updated.at[idx, 'SMILES_1'] = df.at[idx, 'SMILES_2']
            df_updated.at[idx, 'Class_1'] = df.at[idx, 'Class_2']
            df_updated.at[idx, 'Subclass_1'] = df.at[idx, 'Subclass_2']
            df_updated.at[idx, 'Component_ID_1'] = df.at[idx, 'Component_ID_2']
            df_updated.at[idx, 'SMILES_2'] = df.at[idx, 'SMILES_1']
            df_updated.at[idx, 'Class_2'] = df.at[idx, 'Class_1']
            df_updated.at[idx, 'Subclass_2'] = df.at[idx, 'Subclass_1']
            df_updated.at[idx, 'Component_ID_2'] = df.at[idx, 'Component_ID_1']

            df_updated.at[idx, 'P_sat_1'] = df.at[idx, 'P_sat_2']
            df_updated.at[idx, 'gamma_1'] = df.at[idx, 'gamma_2']
            df_updated.at[idx, 'P_sat_2'] = df.at[idx, 'P_sat_1']
            df_updated.at[idx, 'gamma_2'] = df.at[idx, 'gamma_1']

            df_updated.at[idx, 'pred_ln_IDAC_12'] = df.at[idx, 'pred_ln_IDAC_21']
            df_updated.at[idx, 'pred_ln_IDAC_21'] = df.at[idx, 'pred_ln_IDAC_12']

            df_updated.at[idx, 'K1_12'] = df.at[idx, 'K1_21']
            df_updated.at[idx, 'K2_12'] = df.at[idx, 'K2_21']
            df_updated.at[idx, 'K1_21'] = df.at[idx, 'K1_12']
            df_updated.at[idx, 'K2_21'] = df.at[idx, 'K2_12']
            n_points_flipped += 1
    print(f'Systems flipped due to highly probable misplacing from KDB: {n_points_flipped}')
    df = df_updated.copy()


    gamma_1_lst, gamma_2_lst = [], []
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        ln_IDAC_12, ln_IDAC_21 = row['pred_ln_IDAC_12'], row['pred_ln_IDAC_21']
        x_1, x_2 = row['x'], 1-row['x']
        gamma_1, gamma_2 = margules_binary(ln_IDAC_12, ln_IDAC_21, x_1, x_2)

        gamma_1_lst.append(gamma_1)
        gamma_2_lst.append(gamma_2)
    
    df['pred_gamma_1'] = gamma_1_lst
    df['pred_gamma_2'] = gamma_2_lst

    df.to_csv(f'models/GHGNN/kdb_vle_pred_{version}.csv', index=False)

    df_with_gammas = df[(~df['gamma_1'].isnull()) & (~df['gamma_2'].isnull())] # Filter out systems without experimental gammas
    df_with_gammas = df_with_gammas[~df_with_gammas['gamma_1'].isin([np.inf, -np.inf]) & ~df_with_gammas['gamma_2'].isin([np.inf, -np.inf])] # Filter out systems with inf experimental gammas
    df_with_gammas = df_with_gammas[~(df_with_gammas['gamma_1'] == 0) & ~(df_with_gammas['gamma_2'] == 0)]

    
    for j in [1, 2]:
        print('-'*80)
        gamma_true = df_with_gammas[f'gamma_{j}'].to_numpy()
        gamma_pred = df_with_gammas[f'pred_gamma_{j}'].to_numpy()

        threshold = 0.3
        print(f'MAE gamma {j}: {mean_absolute_error(gamma_true, gamma_pred)}')
        print(f'MAPE gamma {j}: {mean_absolute_percentage_error(gamma_true, gamma_pred)*100}')
        print(f'R2 gamma {j}: {r2_score(gamma_true, gamma_pred)}')
        print(f'% AE <=  {threshold} gamma {j}: {percentage_within_threshold(gamma_true, gamma_pred, threshold)}')

        ln_gamma_true = np.log(gamma_true)
        ln_gamma_pred = np.log(gamma_pred)

        print(f'MAE ln(gamma) {j}: {mean_absolute_error(ln_gamma_true, ln_gamma_pred)}')
        print(f'MAPE ln(gamma) {j}: {mean_absolute_percentage_error(ln_gamma_true, ln_gamma_pred)*100}')
        print(f'R2 ln(gamma) {j}: {r2_score(ln_gamma_true, ln_gamma_pred)}')
        print(f'% AE <=  {threshold} ln(gamma) {j}: {percentage_within_threshold(ln_gamma_true, ln_gamma_pred, threshold)}')

    