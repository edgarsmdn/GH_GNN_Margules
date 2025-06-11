import pandas as pd
import ipdb
from argparse import ArgumentParser
from rdkit import Chem
from tqdm import tqdm
from src.utils.utils import plot_binary_VLE, create_folder, get_binary_VLE_isobaric, get_binary_VLE_isobaric_unifac, percentage_within_threshold, get_errors_classes, plot_cumulative_errors, plot_heatmap_performance_MAE
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from src.utils.utils import fredenslund_consistency_test

# python src/models/05_predict_isobaric.py --version organic_old --jaccard_threshold 0.6 --model GH_GNN_Margules --consistent

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

    assert version in ['combined', 'organic', 'organic_old']
    assert model in ['GH_GNN_Margules', 'UNIFAC_Do']

    print(f'\n====> RUN WITH {model}\n')

    df = pd.read_csv(f'models/GHGNN/kdb_vle_pred_{version}_with_unifac_do.csv')
    # Filter out systems not within Jaccard distance threshold
    df = df[df['Jaccard_distance'] <= jaccard_threshold]
    df_isobaric = df[df['Type'] == 'Isobaric']

    # Consistency
    if consistent:
        legendre_order = 5
        df_consistency = fredenslund_consistency_test(df_isobaric, legendre_order=legendre_order, vle_type='isobaric')
        merged_df = pd.merge(df_isobaric, df_consistency[['Compound_1', 'Compound_2', 'P_kPa', 'is_consistent']], 
                            on=['Compound_1', 'Compound_2', 'P_kPa'], 
                            how='left')
        df_isobaric = df_isobaric.copy()
        df_isobaric['is_consistent'] = merged_df['is_consistent'].tolist()
        df_isobaric = df_isobaric[df_isobaric['is_consistent'] == True]

        # df_isobaric = df_isobaric[~df_isobaric['pred_gamma_1_unifac_do'].isna()]

    if model == 'UNIFAC_Do':
        # Take only systems that are feasible to UNIFAC-Do
        df_isobaric = df_isobaric[~df_isobaric['pred_gamma_1_unifac_do'].isna()]
        print(f'Number of points feasible with unifac_do: {df_isobaric.shape[0]}')
    df_isobaric = df_isobaric.copy()

    # Get how many of the binary mixtures where observed during training of the GH-GNN model
    df_train_ghgnn = pd.read_csv('data/external/molecular_train.csv')
    df_train_ghgnn['Solute_SMILES'] = df_train_ghgnn['Solute_SMILES'].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))
    df_train_ghgnn['Solvent_SMILES'] = df_train_ghgnn['Solvent_SMILES'].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))
    training_combinations = (df_train_ghgnn['Solute_SMILES'] + '_' + df_train_ghgnn['Solvent_SMILES']).unique().tolist()
    training_compounds = list(set(df_train_ghgnn['Solute_SMILES'].tolist() + df_train_ghgnn['Solvent_SMILES'].tolist()))

    standard_smiles_1 = df_isobaric['SMILES_1'].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))
    standard_smiles_2 = df_isobaric['SMILES_2'].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))
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

    df_isobaric['task_on_idacs'] = task_on_idacs
    grouped = df_isobaric.groupby(['SMILES_1', 'SMILES_2'])
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

    # General data statistics
    
    observed_compounds = 0
    for comp in compounds:
        if comp in training_compounds:
            observed_compounds += 1
    print(f'Number of compounds: {len(compounds)}')
    print(f'Number of observed compounds: {observed_compounds}')
    print('-'*150)

    unique_binary_sys = (df_isobaric['Compound_1'] + '_' + df_isobaric['Compound_2']).unique().tolist()

    for bin_sys in tqdm(unique_binary_sys, desc='Predicting isobaric VLE diagrams'):
        c1, c2 = bin_sys.split('_')
        df_spec = df_isobaric[(df_isobaric['Compound_1'] == c1) & (df_isobaric['Compound_2'] == c2)]
        unique_Ps = df_spec['P_kPa'].unique().tolist()
        for P in unique_Ps:
            df_subspec = df_spec[df_spec['P_kPa'] == P]

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
            T1_min = df_subspec['T range, from_1'].iloc[0]
            T2_min = df_subspec['T range, from_2'].iloc[0]
            T1_max = df_subspec['T range, to_1'].iloc[0]
            T2_max = df_subspec['T range, to_2'].iloc[0]
            T_bounds = (max(T1_min, T2_min), min(T1_max, T2_max)) # Get the most constrained T bounds for feasible P_vap correlation
            if model == 'UNIFAC_Do':
                (x_1, y_1), (x_2, y_2), T = get_binary_VLE_isobaric_unifac(P, c_1, c_2, T_bounds, SLSQP=False)
            else:
                (x_1, y_1), (x_2, y_2), T = get_binary_VLE_isobaric(P, c_1, c_2, T_bounds)

            # Add predicted and experimental VLE data of component 1 for plotting
            c_1['exp_ids'] = df_subspec['ID'].to_numpy()
            c_1['x_exp'] = df_subspec['x'].to_numpy()
            c_1['y_exp'] = df_subspec['y'].to_numpy()
            c_1['x_pred'] = x_1
            c_1['y_pred'] = y_1

            fig = plot_binary_VLE(c_1, model)
            sorted_classes = sorted([c_1['Class'], c_2['Class']])

            folder = f"visualization/Isobaric/{version}/{model}/{sorted_classes[0]}_{sorted_classes[1]}"
            create_folder(folder)
            fig.savefig(f"{folder}/{c_1['Name']}_{c_2['Name']}_{str(np.round(P, 4))}.png")

            # Save data
            df_subspec.to_csv(f"{folder}/{c_1['Name']}_{c_2['Name']}_{str(np.round(P, 4))}_experimental.csv", index=False)
            df_system = pd.DataFrame({'x1_pred':x_1, 'y1_pred':y_1})
            df_system['T_K'] = T
            df_system['P_kPa'] = P
            df_system.to_csv(f"{folder}/{c_1['Name']}_{c_2['Name']}_{str(np.round(P, 4))}_predicted.csv", index=False)

    y_1_lst = list()
    T_lst = list()
    for i, row in tqdm(df_isobaric.iterrows(), total=df_isobaric.shape[0], desc='Predicting specific isobaric points'):
        P = row['P_kPa']
        c_1 = {
                'x':row['x'],
                'Pvap_constants':(row['Coefficient A_1'], 
                                  row['Coefficient B_1'],
                                  row['Coefficient C_1'],
                                  row['Coefficient D_1']),
                'inchikey':row['inchikey_1'],
                'GH_constants':(row['K1_12'],
                                row['K2_12'])
            }
        c_2 = {
                'x':1-row['x'],
                'Pvap_constants':(row['Coefficient A_2'], 
                                  row['Coefficient B_2'],
                                  row['Coefficient C_2'],
                                  row['Coefficient D_2']),
                'inchikey':row['inchikey_2'],
                'GH_constants':(row['K1_21'],
                                row['K2_21'])
            }
        
        T1_min = row['T range, from_1']
        T2_min = row['T range, from_2']
        T1_max = row['T range, to_1']
        T2_max = row['T range, to_2']
        T_bounds = (max(T1_min, T2_min), min(T1_max, T2_max)) # Get the most constrained T bounds for feasible P_vap correlation
        if model == 'UNIFAC_Do':
            (x_1, y_1), (x_2, y_2), T = get_binary_VLE_isobaric_unifac(P, c_1, c_2, T_bounds, specific=True, SLSQP=False)
        else:
            (x_1, y_1), (x_2, y_2), T = get_binary_VLE_isobaric(P, c_1, c_2, T_bounds, specific=True)
        y_1_lst.append(y_1)
        T_lst.append(T)
    df_isobaric['y_pred'] = y_1_lst
    df_isobaric['T_pred'] = T_lst

    folder_preds = f"models/{model}/{version}"
    create_folder(folder_preds)
    df_isobaric.to_csv(f'{folder_preds}/isobaric_pred_{version}.csv', index=False)

    threshold = 0.03
    y_1_true = df_isobaric['y'].to_numpy()
    y_1_pred = df_isobaric['y_pred'].to_numpy()
    
    print(f'MAE y_1: {mean_absolute_error(y_1_true, y_1_pred)}')
    print(f'MAPE y_1: {mean_absolute_percentage_error(y_1_true, y_1_pred)*100}')
    print(f'R2 y_1: {r2_score(y_1_true, y_1_pred)}')
    print(f'% AE <=  {threshold} y_1: {percentage_within_threshold(y_1_true, y_1_pred, threshold)}')

    # Print statistics of the complete isothermal dataset
    n_compounds = len(set(df_isobaric['Compound_1'].unique().tolist() + df_isobaric['Compound_2'].unique().tolist()))
    n_combinations = (df_isobaric['Compound_1'] + '_' + df_isobaric['Compound_2']).nunique()
    print(f'Number of isobaric data points: {df_isobaric.shape[0]}')
    print(f'Number of isobaric combinations: {n_combinations}')
    print(f'P range: {df_isobaric["P_kPa"].min()} - {df_isobaric["P_kPa"].max()}')

    # Analyze according to chemical classes
    df_errors = get_errors_classes(df_isobaric, 'isobaric')
    cum_fig = plot_cumulative_errors(df_errors, model)
    cum_fig.savefig(f"visualization/Isobaric/{version}/{model}/Cumulative_MAE_classes_isobaric.png", dpi=300, format='png')
    df_errors.to_csv(f'{folder_preds}/isobaric_binary_class_performance.csv', index=False)
    fig = plot_heatmap_performance_MAE(df_errors)
    fig.savefig(f"visualization/Isobaric/{version}/{model}/MAE_matrix_{version}_Jaccard{jaccard_threshold}.png", dpi=300, format='png')
    print(f'Number of binary classes: {df_errors["Binary_class"].shape[0]}')
    
    # Heatmap of MAE in predicted P
    df_errors_T = get_errors_classes(df_isobaric, 'isobaric', true_col='T_K', pred_col='T_pred')
    fig = plot_heatmap_performance_MAE(df_errors_T, vmax=7)
    fig.savefig(f"visualization/Isothermal/{version}/{model}/MAE_matrix_{version}_Jaccard{jaccard_threshold}_Temperature.png", dpi=300, format='png')

    T_true = df_isobaric['T_K'].to_numpy()
    T_pred = df_isobaric['T_pred'].to_numpy()

    threshold_P = 2
    print('*'*100)
    print('Metrics for predicted temperature')
    print(f'MAE T: {mean_absolute_error(T_true, T_pred)}')
    print(f'MAPE T: {mean_absolute_percentage_error(T_true, T_pred)*100}')
    print(f'R2 T: {r2_score(T_true, T_pred)}')
    print(f'% AE <=  {threshold_P} T: {percentage_within_threshold(T_true, T_pred, threshold_P)}')

    

    # Get perfromance metrics according to observed, interpolation and extrapolation
    if model == 'GH_GNN_Margules':
        for task in ['observed', 'interpolation', 'extrapolation']:
            print('*'*100)
            print(f'Task {task}')
            df_spec = df_isobaric[df_isobaric['task_on_idacs'] == task]

            y_1_true = df_spec['y'].to_numpy()
            y_1_pred = df_spec['y_pred'].to_numpy()
            
            print(f'Number of points: {df_spec.shape[0]}')
            print(f'MAE y_1: {mean_absolute_error(y_1_true, y_1_pred)}')
            print(f'MAPE y_1: {mean_absolute_percentage_error(y_1_true, y_1_pred)*100}')
            print(f'R2 y_1: {r2_score(y_1_true, y_1_pred)}')
            print(f'% AE <=  {threshold} y_1: {percentage_within_threshold(y_1_true, y_1_pred, threshold)}')

            df_errors = get_errors_classes(df_spec, 'isobaric')
            cum_fig = plot_cumulative_errors(df_errors, model)
            cum_fig.savefig(f"visualization/Isobaric/{version}/{model}/Cumulative_MAE_classes_isobaric_{task}.png", dpi=300, format='png')
            df_errors.to_csv(f'{folder_preds}/isobaric_binary_class_performance_{task}.csv', index=False)
            fig = plot_heatmap_performance_MAE(df_errors)
            fig.savefig(f"visualization/Isobaric/{version}/{model}/MAE_matrix_{version}_Jaccard{jaccard_threshold}_{task}.png", dpi=300, format='png')
            print(f'Number of binary classes: {df_errors["Binary_class"].shape[0]}')