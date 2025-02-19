import pandas as pd
import ipdb
from argparse import ArgumentParser
from rdkit import Chem
from tqdm import tqdm
from src.utils.utils import create_folder, get_binary_VLE_isothermal, get_binary_VLE_isothermal_unifac, percentage_within_threshold
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

# python src/models/06_predict_all.py --version organic_old --jaccard_threshold 0.6 --model GH_GNN_Margules

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--version", help="GH-GNN version to use", type=str)
    parser.add_argument("--jaccard_threshold", help="Jaccard distance threshold to be considered in the predictions", type=str)
    parser.add_argument("--model", help="Model to be used in the predictions", type=str)

    args = parser.parse_args()
    version = args.version
    jaccard_threshold = float(args.jaccard_threshold)
    model = args.model

    assert version in ['combined', 'organic', 'organic_old']
    assert model in ['GH_GNN_Margules', 'UNIFAC_Do']

    print(f'\n====> RUN WITH {model}\n')

    df = pd.read_csv(f'models/GHGNN/kdb_vle_pred_{version}_with_unifac_do.csv')

    # Filter out systems not within Jaccard distance threshold
    df = df[df['Jaccard_distance'] <= jaccard_threshold]

    if model == 'UNIFAC_Do':
        # Take only systems that are feasible to UNIFAC-Do
        df = df[~df['pred_gamma_1_unifac_do'].isna()]
        print(f'Number of points feasible with unifac_do: {df.shape[0]}')
    df = df.copy()

    # Get how many of the binary mixtures where observed during training of the GH-GNN model
    df_train_ghgnn = pd.read_csv('data/external/molecular_train.csv')
    df_train_ghgnn['Solute_SMILES'] = df_train_ghgnn['Solute_SMILES'].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))
    df_train_ghgnn['Solvent_SMILES'] = df_train_ghgnn['Solvent_SMILES'].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))
    training_combinations = (df_train_ghgnn['Solute_SMILES'] + '_' + df_train_ghgnn['Solvent_SMILES']).unique().tolist()
    training_compounds = list(set(df_train_ghgnn['Solute_SMILES'].tolist() + df_train_ghgnn['Solvent_SMILES'].tolist()))

    standard_smiles_1 = df['SMILES_1'].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))
    standard_smiles_2 = df['SMILES_2'].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))
    compounds = list(set(standard_smiles_1.unique().tolist() + standard_smiles_2.unique().tolist()))

    task_on_idacs = []
    for smi_1, smi_2 in zip(standard_smiles_1, standard_smiles_2):
        comb_12 = smi_1 + '_' + smi_2
        comb_21 = smi_2 + '_' + smi_1
        if comb_12 in training_combinations and comb_21 in training_combinations:
            task_on_idacs.append('observed')
        elif smi_1 in training_compounds and smi_2 in training_compounds:
            task_on_idacs.append('interpolation')
        else:
            task_on_idacs.append('extrapolation')

    df['task_on_idacs'] = task_on_idacs
    grouped = df.groupby(['SMILES_1', 'SMILES_2'])
    grouped_dfs = [group for _, group in grouped]

    count_dict = {'interpolation': 0, 'observed': 0, 'extrapolation': 0}
    for df_s in grouped_dfs:
        task = df_s['task_on_idacs'].iloc[0]
        count_dict[task] += 1
    print(f'Number of systems: {len(grouped_dfs)}')
    print(f'Number of systems with observed IDACs: {count_dict["observed"]}')
    print(f'Number of sytems with interpolated IDACs: {count_dict["interpolation"]}')
    print(f'Number of systems with extrapolated IDACs: {count_dict["extrapolation"]}')
    print('-'*150)

    observed_compounds = 0
    for comp in compounds:
        if comp in training_compounds:
            observed_compounds += 1
    print(f'Number of compounds: {len(compounds)}')
    print(f'Number of observed compounds: {observed_compounds}')

    y_1_lst = list()
    for i, row in tqdm(df.iterrows(), total=df.shape[0], desc='Predicting specific points'):
        T = row['T_K']
        c_1 = {
                'x':row['x'],
                'Pvap_constants':(row['Coefficient A_1'], 
                                  row['Coefficient B_1'],
                                  row['Coefficient C_1'],
                                  row['Coefficient D_1']),
                'gamma':row['pred_gamma_1_unifac_do'],
                'GH_constants':(row['K1_12'],
                                row['K2_12'])
            }
        c_2 = {
                'x':1-row['x'],
                'Pvap_constants':(row['Coefficient A_2'], 
                                  row['Coefficient B_2'],
                                  row['Coefficient C_2'],
                                  row['Coefficient D_2']),
                'gamma':row['pred_gamma_2_unifac_do'],
                'GH_constants':(row['K1_21'],
                                row['K2_21'])
            }
        
        if model == 'UNIFAC_Do':
            (x_1, y_1), (x_2, y_2), P = get_binary_VLE_isothermal_unifac(T, c_1, c_2, specific=True)
        else:
            (x_1, y_1), (x_2, y_2), P = get_binary_VLE_isothermal(T, c_1, c_2, specific=True)
        y_1_lst.append(y_1)
    df['y_pred'] = y_1_lst

    folder_preds = f"models/{model}/{version}"
    create_folder(folder_preds)
    df.to_csv(f'{folder_preds}/all_pred_{version}.csv', index=False)

    # Get how many binary classes are present
    df['Binary_class'] = ['_'.join(sorted([row['Class_1'], row['Class_2']])) for _, row in df.iterrows()]
    unique_binary_classes = df['Binary_class'].unique().tolist()
    print(f'Number of binary classes: {df["Binary_class"].nunique()}')

    threshold = 0.03
    y_1_true = df['y'].to_numpy()
    y_1_pred = df['y_pred'].to_numpy()
    
    print(f'MAE y_1: {mean_absolute_error(y_1_true, y_1_pred)}')
    print(f'MAPE y_1: {mean_absolute_percentage_error(y_1_true, y_1_pred)*100}')
    print(f'R2 y_1: {r2_score(y_1_true, y_1_pred)}')
    print(f'% AE <=  {threshold} y_1: {percentage_within_threshold(y_1_true, y_1_pred, threshold)}')

    n_compounds = len(set(df['Compound_1'].unique().tolist() + df['Compound_2'].unique().tolist()))
    n_combinations = (df['Compound_1'] + '_' + df['Compound_2']).nunique()
    print(f'Number of isothermal data points: {df.shape[0]}')
    print(f'Number of isothermal combinations: {n_combinations}')
    print(f'T range: {df["T_K"].min()} - {df["T_K"].max()}')
    
    # Get perfromance metrics according to observed, interpolation and extrapolation
    if model == 'GH_GNN_Margules':
        for task in ['observed', 'interpolation', 'extrapolation']:
            print('*'*100)
            print(f'Task {task}')
            df_spec = df[df['task_on_idacs'] == task]

            y_1_true = df_spec['y'].to_numpy()
            y_1_pred = df_spec['y_pred'].to_numpy()
            
            print(f'Number of points: {df_spec.shape[0]}')
            print(f'Number of binary classes: {df_spec["Binary_class"].nunique()}')
            print(f'MAE y_1: {mean_absolute_error(y_1_true, y_1_pred)}')
            print(f'MAPE y_1: {mean_absolute_percentage_error(y_1_true, y_1_pred)*100}')
            print(f'R2 y_1: {r2_score(y_1_true, y_1_pred)}')
            print(f'% AE <=  {threshold} y_1: {percentage_within_threshold(y_1_true, y_1_pred, threshold)}')

