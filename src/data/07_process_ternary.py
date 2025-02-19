import pandas as pd
import ipdb
from tqdm import tqdm
from src.utils.utils import name_to_smiles_OPSIN, load_json_files, KDB_correlation_Pvap, find_name_from_smiles
from rdkit import Chem

# python src/data/07_process_ternary.py

if __name__ == "__main__":

    df = pd.read_csv('data/raw/ternary_vle.csv')

    compounds_1 = df['Compound_1'].unique()
    compounds_2 = df['Compound_2'].unique()
    compounds_3 = df['Compound_3'].unique()
    unique_compounds = set(compounds_1) | set(compounds_2) | set(compounds_3)

    pure_comp_dict = load_json_files('data/raw/KDB/pure_component_properties')

    # Find SMILES
    smiles_dict = {}
    p_vap_dict = {}
    for name in tqdm(unique_compounds):
        smiles = name_to_smiles_OPSIN(name)
        smiles_dict[name] = smiles
        if name.upper() in pure_comp_dict:
            p_vap_info = pure_comp_dict[name.upper()]['Vapor Pressure']
        else:
            name_in_kdb = find_name_from_smiles(smiles, pure_comp_dict)
            try:
                p_vap_info = pure_comp_dict[name_in_kdb]['Vapor Pressure']
            except:
                p_vap_info = None
        p_vap_dict[name] = p_vap_info
        
    compounds_without_kdb_data = [name for name in p_vap_dict if p_vap_dict[name] is None]
    

    # Filter out systems with compounds without KBD data
    mask = df['Compound_1'].isin(compounds_without_kdb_data) | df['Compound_2'].isin(compounds_without_kdb_data) | df['Compound_3'].isin(compounds_without_kdb_data)
    print(f'Eliminated points due to lack of KDB data: {mask.sum()}')
    df = df[~mask].copy()
      
    # Include SMILES
    for i in range(1,4):
        df[f'SMILES_{i}'] = df[f'Compound_{i}'].map(smiles_dict)

    
    As, Bs, Cs, Ds, T_mins, T_maxs, P_vaps, gammas ={1:[], 2:[], 3:[]},{1:[], 2:[], 3:[]},{1:[], 2:[], 3:[]},{1:[], 2:[], 3:[]},{1:[], 2:[], 3:[]},{1:[], 2:[], 3:[]},{1:[], 2:[], 3:[]}, {1:[], 2:[], 3:[]}
    for i, row in df.iterrows():
        # Include P_vap
        for j in range(1,4):
            A = float(p_vap_dict[row[f'Compound_{j}']]['Coefficient A'])
            B = float(p_vap_dict[row[f'Compound_{j}']]['Coefficient B'])
            C = float(p_vap_dict[row[f'Compound_{j}']]['Coefficient C'])
            D = float(p_vap_dict[row[f'Compound_{j}']]['Coefficient D'])
            T_min = float(p_vap_dict[row[f'Compound_{j}']]['T range, from'][:-2])
            T_max = float(p_vap_dict[row[f'Compound_{j}']]['T range, to'][:-2])
            P_vap = KDB_correlation_Pvap(row['T_K'],A,B,C,D)
            if j == 3:
                y_3 = 1 - row[f'y_1'] - row[f'y_2']
                x_3 = 1 - row[f'x_1'] - row[f'x_2']
                gamma = (row['P_kPa']*y_3) / (P_vap * x_3)
            else:
                gamma = (row['P_kPa']*row[f'y_{j}']) / (P_vap * row[f'x_{j}'])

            As[j].append(A)
            Bs[j].append(B)
            Cs[j].append(C)
            Ds[j].append(D)
            T_mins[j].append(T_min)
            T_maxs[j].append(T_max)
            P_vaps[j].append(P_vap)
            gammas[j].append(gamma)
    
    for j in range(1,4):
        df[f'Coefficient A_{j}'] = As[j]
        df[f'Coefficient B_{j}'] = Bs[j]
        df[f'Coefficient C_{j}'] = Cs[j]
        df[f'Coefficient D_{j}'] = Ds[j]
        df[f'T range, from_{j}'] = T_mins[j]
        df[f'T range, to_{j}'] = T_maxs[j]
        df[f'P_vap_{j}'] = P_vaps[j]
        df[f'gamma_{j}'] = gammas[j]
        
    # Filter out datapoints where Antoine is outside of the T range
    mask = ((df['T_K'] >= df['T range, from_1']) & (df['T_K'] <= df['T range, to_1']) &
        (df['T_K'] >= df['T range, from_2']) & (df['T_K'] <= df['T range, to_2']) &
        (df['T_K'] >= df['T range, from_3']) & (df['T_K'] <= df['T range, to_3']))
    print(f'Eliminated points due to being outside of the applicability of Vapor pressure correlation: {df.shape[0] - mask.sum()}')
    df = df[mask]

    # include x_3 and y_3
    df['x_3'] = 1 - df['x_1'] - df['x_2']
    df['y_3'] = 1 - df['y_1'] - df['y_2']

    # include inchikeys
    df['inchikey_1'] = [Chem.inchi.MolToInchiKey(Chem.MolFromSmiles(sm)) for sm in df['SMILES_1'].tolist()]
    df['inchikey_2'] = [Chem.inchi.MolToInchiKey(Chem.MolFromSmiles(sm)) for sm in df['SMILES_2'].tolist()]
    df['inchikey_3'] = [Chem.inchi.MolToInchiKey(Chem.MolFromSmiles(sm)) for sm in df['SMILES_3'].tolist()]
    
    df.to_csv('data/processed/ternary_vle.csv', index=False)