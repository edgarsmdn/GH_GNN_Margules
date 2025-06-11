import pandas as pd
import ipdb
from src.utils.utils import plot_binary_VLE, create_folder, percentage_within_threshold, plot_heatmap_performance_MAE, plot_cumulative_errors, get_binary_VLE_isothermal_unifac, get_binary_VLE_isothermal, get_errors_classes
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from rdkit import Chem
from argparse import ArgumentParser
import numpy as np
from src.utils.utils import fredenslund_consistency_test

# python src/models/04_predict_isothermal.py --version organic_old --jaccard_threshold 0.6 --model GH_GNN_Margules --consistent

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--version", help="GH-GNN version to use", type=str)
    parser.add_argument("--jaccard_threshold", help="Jaccard distance threshold to be considered in the predictions", type=str)
    parser.add_argument("--model", help="Model to be used in the predictions", type=str)
    parser.add_argument("--consistent", help="Whether to run the predictions only with thermodynamically consistent data", action='store_true')

    args = parser.parse_args()
    version = args.version
    jaccard_threshold = float(args.jaccard_threshold)
    model = args.model
    consistent = args.consistent

    assert version in ['combined', 'organic_old']
    assert model in ['GH_GNN_Margules', 'UNIFAC_Do']

    print(f'\n====> RUN WITH {model}\n')

    df = pd.read_csv(f'models/GHGNN/kdb_vle_pred_{version}_with_unifac_do.csv')
    # Filter out systems not within Jaccard distance threshold
    df = df[df['Jaccard_distance'] <= jaccard_threshold]
    df_isothermal = df[df['Type'] == 'Isothermal']

    # Consistency
    if consistent:
        legendre_order = 5
        df_consistency = fredenslund_consistency_test(df_isothermal, legendre_order=legendre_order, vle_type='isothermal')
        merged_df = pd.merge(df_isothermal, df_consistency[['Compound_1', 'Compound_2', 'T_K', 'is_consistent']], 
                            on=['Compound_1', 'Compound_2', 'T_K'], 
                            how='left')
        df_isothermal = df_isothermal.copy()
        df_isothermal['is_consistent'] = merged_df['is_consistent'].tolist()
        df_isothermal = df_isothermal[df_isothermal['is_consistent'] == True]

        # df_isothermal = df_isothermal[~df_isothermal['pred_gamma_1_unifac_do'].isna()]

    if model == 'UNIFAC_Do':
        # Take only systems that are feasible to UNIFAC-Do
        df_isothermal = df_isothermal[~df_isothermal['pred_gamma_1_unifac_do'].isna()]
        print(f'Number of points feasible with unifac_do: {df_isothermal.shape[0]}')
    df_isothermal = df_isothermal.copy()

    # Get how many of the binary mixtures where observed during training of the GH-GNN model
    df_train_ghgnn = pd.read_csv('data/external/molecular_train.csv')
    df_train_ghgnn['Solute_SMILES'] = df_train_ghgnn['Solute_SMILES'].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))
    df_train_ghgnn['Solvent_SMILES'] = df_train_ghgnn['Solvent_SMILES'].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))
    training_combinations = (df_train_ghgnn['Solute_SMILES'] + '_' + df_train_ghgnn['Solvent_SMILES']).unique().tolist()
    training_compounds = list(set(df_train_ghgnn['Solute_SMILES'].tolist() + df_train_ghgnn['Solvent_SMILES'].tolist()))

    standard_smiles_1 = df_isothermal['SMILES_1'].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))
    standard_smiles_2 = df_isothermal['SMILES_2'].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))
    compounds = list(set(standard_smiles_1.unique().tolist() + standard_smiles_2.unique().tolist()))

    task_on_idacs = []
    extrapolated_compounds = []
    for smi_1, smi_2 in zip(standard_smiles_1, standard_smiles_2):
        comb_12 = smi_1 + '_' + smi_2
        comb_21 = smi_2 + '_' + smi_1
        if comb_12 in training_combinations and comb_21 in training_combinations:
            task_on_idacs.append('observed')
        elif smi_1 in training_compounds and smi_2 in training_compounds:
            task_on_idacs.append('interpolation')
        else:
            task_on_idacs.append('extrapolation')
            if smi_1 not in training_compounds and smi_1 not in extrapolated_compounds:
                extrapolated_compounds.append(smi_1)
            if smi_2 not in training_compounds and smi_2 not in extrapolated_compounds:
                extrapolated_compounds.append(smi_2)
    print(f'\n Extrapolated compounds: {len(extrapolated_compounds)}\n')

    df_isothermal['task_on_idacs'] = task_on_idacs
    grouped = df_isothermal.groupby(['SMILES_1', 'SMILES_2'])
    grouped_dfs = [group for _, group in grouped]

    count_dict = {'interpolation': 0, 'observed': 0, 'extrapolation': 0}
    for df in grouped_dfs:
        task = df['task_on_idacs'].iloc[0]
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

    unique_binary_sys = (df_isothermal['Compound_1'] + '_' + df_isothermal['Compound_2']).unique().tolist()

    for bin_sys in tqdm(unique_binary_sys, desc='Predicting isothermal VLE diagrams'):
        c1, c2 = bin_sys.split('_')
        df_spec = df_isothermal[(df_isothermal['Compound_1'] == c1) & (df_isothermal['Compound_2'] == c2)]
        unique_Ts = df_spec['T_K'].unique().tolist()
        for T in unique_Ts:
            df_subspec = df_spec[df_spec['T_K'] == T]

            c_1 = {
                'Name':df_subspec['Compound_1'].iloc[0],
                'Class':df_subspec['Class_1'].iloc[0],
                'Subclass':df_subspec['Subclass_1'].iloc[0],
                'inchikey':df_subspec['inchikey_1'].iloc[0],
                'Pvap_constants':(df_subspec['Coefficient A_1'].iloc[0], 
                                  df_subspec['Coefficient B_1'].iloc[0],
                                  df_subspec['Coefficient C_1'].iloc[0],
                                  df_subspec['Coefficient D_1'].iloc[0]),
                'GH_constants':(df_subspec['K1_12'].iloc[0],
                                df_subspec['K2_12'].iloc[0])
            }
            c_2 = {
                'Name':df_subspec['Compound_2'].iloc[0],
                'Class':df_subspec['Class_2'].iloc[0],
                'Subclass':df_subspec['Subclass_2'].iloc[0],
                'inchikey':df_subspec['inchikey_2'].iloc[0],
                'Pvap_constants':(df_subspec['Coefficient A_2'].iloc[0], 
                                  df_subspec['Coefficient B_2'].iloc[0],
                                  df_subspec['Coefficient C_2'].iloc[0],
                                  df_subspec['Coefficient D_2'].iloc[0]),
                'GH_constants':(df_subspec['K1_21'].iloc[0],
                                df_subspec['K2_21'].iloc[0])
            }
            if model == 'UNIFAC_Do':
                (x_1, y_1), (x_2, y_2), P = get_binary_VLE_isothermal_unifac(T, c_1, c_2)
            else:
                (x_1, y_1), (x_2, y_2), P = get_binary_VLE_isothermal(T, c_1, c_2, version=version)

            # Add predicted and experimental VLE data of component 1 for plotting
            c_1['exp_ids'] = df_subspec['ID'].to_numpy()
            c_1['x_exp'] = df_subspec['x'].to_numpy()
            c_1['y_exp'] = df_subspec['y'].to_numpy()
            c_1['x_pred'] = x_1
            c_1['y_pred'] = y_1

            fig = plot_binary_VLE(c_1, model)
            sorted_classes = sorted([c_1['Class'], c_2['Class']])

            folder = f"visualization/Isothermal/{version}/{model}/{sorted_classes[0]}_{sorted_classes[1]}"
            create_folder(folder)
            fig.savefig(f"{folder}/{c_1['Name']}_{c_2['Name']}_{str(np.round(T, 4))}.png")

            # Save data
            df_subspec.to_csv(f"{folder}/{c_1['Name']}_{c_2['Name']}_{str(np.round(T, 4))}_experimental.csv", index=False)
            df_system = pd.DataFrame({'x1_pred':x_1, 'y1_pred':y_1})
            df_system['T_K'] = T
            df_system['P_kPa'] = P
            df_system.to_csv(f"{folder}/{c_1['Name']}_{c_2['Name']}_{str(np.round(T, 4))}_predicted.csv", index=False)

    y_1_lst = list()
    P_lst = list()
    for i, row in tqdm(df_isothermal.iterrows(), total=df_isothermal.shape[0], desc='Predicting specific isothermal points'):
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
            (x_1, y_1), (x_2, y_2), P = get_binary_VLE_isothermal(T, c_1, c_2, specific=True, version=version)
        y_1_lst.append(y_1)
        P_lst.append(P)
    df_isothermal['y_pred'] = y_1_lst
    df_isothermal['P_pred'] = P_lst

    folder_preds = f"models/{model}/{version}"
    create_folder(folder_preds)
    df_isothermal.to_csv(f'{folder_preds}/isothermal_pred_{version}.csv', index=False)
    

    threshold = 0.03
    y_1_true = df_isothermal['y'].to_numpy()
    y_1_pred = df_isothermal['y_pred'].to_numpy()
    
    print(f'MAE y_1: {mean_absolute_error(y_1_true, y_1_pred)}')
    print(f'MAPE y_1: {mean_absolute_percentage_error(y_1_true, y_1_pred)*100}')
    print(f'R2 y_1: {r2_score(y_1_true, y_1_pred)}')
    print(f'% AE <=  {threshold} y_1: {percentage_within_threshold(y_1_true, y_1_pred, threshold)}')

    # Print statistics of the complete isothermal dataset
    n_compounds = len(set(df_isothermal['Compound_1'].unique().tolist() + df_isothermal['Compound_2'].unique().tolist()))
    n_combinations = (df_isothermal['Compound_1'] + '_' + df_isothermal['Compound_2']).nunique()
    print(f'Number of isothermal data points: {df_isothermal.shape[0]}')
    print(f'Number of isothermal combinations: {n_combinations}')
    print(f'T range: {df_isothermal["T_K"].min()} - {df_isothermal["T_K"].max()}')

    # Analyze according to chemical classes
    df_errors = get_errors_classes(df_isothermal, 'isothermal')
    cum_fig = plot_cumulative_errors(df_errors, model)
    cum_fig.savefig(f"visualization/Isothermal/{version}/{model}/Cumulative_MAE_classes_isothermal.png", dpi=300, format='png')
    df_errors.to_csv(f'{folder_preds}/isothermal_binary_class_performance.csv', index=False)
    fig = plot_heatmap_performance_MAE(df_errors)
    fig.savefig(f"visualization/Isothermal/{version}/{model}/MAE_matrix_{version}_Jaccard{jaccard_threshold}.png", dpi=300, format='png')
    print(f'Number of binary classes: {df_errors["Binary_class"].shape[0]}')

    # Heatmap of MAE in predicted P
    df_errors_P = get_errors_classes(df_isothermal, 'isothermal', true_col='P_kPa', pred_col='P_pred')
    fig = plot_heatmap_performance_MAE(df_errors_P, vmax=7)
    fig.savefig(f"visualization/Isothermal/{version}/{model}/MAE_matrix_{version}_Jaccard{jaccard_threshold}_Pressure.png", dpi=300, format='png')

    P_true = df_isothermal['P_kPa'].to_numpy()
    P_pred = df_isothermal['P_pred'].to_numpy()

    threshold_P = 2
    print('*'*100)
    print('Metrics for predicted pressure')
    print(f'MAE P: {mean_absolute_error(P_true, P_pred)}')
    print(f'MAPE P: {mean_absolute_percentage_error(P_true, P_pred)*100}')
    print(f'R2 P: {r2_score(P_true, P_pred)}')
    print(f'% AE <=  {threshold_P} P: {percentage_within_threshold(P_true, P_pred, threshold_P)}')

    
    # Get perfromance metrics according to observed, interpolation and extrapolation
    if model == 'GH_GNN_Margules':
        for task in ['observed', 'interpolation', 'extrapolation']:
            print('*'*100)
            print(f'Task {task}')
            df_spec = df_isothermal[df_isothermal['task_on_idacs'] == task]

            y_1_true = df_spec['y'].to_numpy()
            y_1_pred = df_spec['y_pred'].to_numpy()
            
            print(f'Number of points: {df_spec.shape[0]}')
            print(f'MAE y_1: {mean_absolute_error(y_1_true, y_1_pred)}')
            print(f'MAPE y_1: {mean_absolute_percentage_error(y_1_true, y_1_pred)*100}')
            print(f'R2 y_1: {r2_score(y_1_true, y_1_pred)}')
            print(f'% AE <=  {threshold} y_1: {percentage_within_threshold(y_1_true, y_1_pred, threshold)}')

            df_errors = get_errors_classes(df_spec, 'isothermal')
            cum_fig = plot_cumulative_errors(df_errors, model)
            cum_fig.savefig(f"visualization/Isothermal/{version}/{model}/Cumulative_MAE_classes_isothermal_{task}.png", dpi=300, format='png')
            df_errors.to_csv(f'{folder_preds}/isothermal_binary_class_performance_{task}.csv', index=False)
            fig = plot_heatmap_performance_MAE(df_errors)
            fig.savefig(f"visualization/Isothermal/{version}/{model}/MAE_matrix_{version}_Jaccard{jaccard_threshold}_{task}.png", dpi=300, format='png')
            print(f'Number of binary classes: {df_errors["Binary_class"].shape[0]}')

            

