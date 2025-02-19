

import pandas as pd
from src.utils.utils import load_json_files, KDB_correlation_Pvap, check_infeasible_mol_for_ghgnn
import ipdb

# python src/data/05_get_feasible_systems.py

if __name__ == "__main__":

    df = pd.read_csv('data/interim/collected_raw_vle_data.csv')
    pure_comp_dict = load_json_files('data/raw/KDB/pure_component_properties')

    # Filter out systems at high pressures (where ideal vapor is not good assumption any more), 5 bar based on the work of SPT-NRTL
    df = df[df['P_kPa'] < 500]

    # Filter out systems with x or y > 1
    df = df[(df['x'] <= 1) & (df['y'] <= 1)]

    # Filter out systems without available Antoine parameters for the given range of temperatures
    unique_comps = list(set(df['Compound_1'].unique().tolist() + df['Compound_2'].unique().tolist()))
    vap_p = list()
    for comp in unique_comps:
        # synonym of 1,1,2,2-TETRACHLORODIFLUOROETHANE is present in the database as 1,2-DIFLUOROTETRACHLOROETHANE
        if comp == '1,1,2,2-TETRACHLORODIFLUOROETHANE':
            comp = '1,2-DIFLUOROTETRACHLOROETHANE'
        try:
            vap_p_info = pure_comp_dict[comp]['Vapor Pressure']
            name = pure_comp_dict[comp]['Name']
            smiles = pure_comp_dict[comp]['SMILES']
            c_class = pure_comp_dict[comp]['Class']
            c_subclass = pure_comp_dict[comp]['Class']
            c_id = pure_comp_dict[comp]['ID']

            # synonym of 1,1,2,2-TETRACHLORODIFLUOROETHANE is present in the database as 1,2-DIFLUOROTETRACHLOROETHANE
            if comp == '1,2-DIFLUOROTETRACHLOROETHANE':
                name = '1,1,2,2-TETRACHLORODIFLUOROETHANE'

            if isinstance(vap_p_info, dict):
                vap_p_info['Name'] = name
                vap_p_info['SMILES'] = smiles
                vap_p_info['Class'] = c_class
                vap_p_info['Subclass'] = c_subclass
                vap_p_info['Component_ID'] = c_id
                vap_p.append(vap_p_info)
            else:
                print(f'Compound {comp} from VLE data does not have Vapor pressure info in the pure compound database')
        except:
            print(f'Compound {comp} from VLE data is not available in the pure compound database')
            pass
    df_vap_p = pd.DataFrame(vap_p)

    for T_column in ['T range, from', 'T range, to']:
        all_end_with_K = df_vap_p[T_column].str.endswith('K').all()
        if all_end_with_K:
            df_vap_p[T_column] = df_vap_p[T_column].apply(lambda s: s[:-2])

    assert df_vap_p.shape[0] == df_vap_p['SMILES'].nunique() # check that molecules are unique
    
    # convert to floats
    for col in ['Coefficient A', 'Coefficient B', 'Coefficient C', 'Coefficient D', 'Coefficient E', 'Coefficient F', 'Coefficient G', 'T range, from', 'T range, to']:
        df_vap_p[col] = df_vap_p[col].astype(float)

    df_vap_p.to_csv('data/interim/unique_compounds_less1MPa.csv', index=False)

    unique_compounds_with_vap_p = df_vap_p['Name'].unique().tolist()

    compounds_without_vap_p = set(unique_comps) - set(unique_compounds_with_vap_p)

    mask1 = df['Compound_1'].isin(compounds_without_vap_p)
    mask2 = df['Compound_2'].isin(compounds_without_vap_p)
    combined_mask = mask1 | mask2
    df = df[~combined_mask]

    unique_comps_filtered = list(set(df['Compound_1'].unique().tolist() + df['Compound_2'].unique().tolist()))
    assert set(unique_comps_filtered) == set(unique_compounds_with_vap_p) # check that all compounds with available vap_p info are the same in the vle data

    # include all vap_p info in the data file
    df = df.merge(df_vap_p, how='left', left_on='Compound_1', right_on='Name')
    df = df.merge(df_vap_p, how='left', left_on='Compound_2', right_on='Name', suffixes=('_1', '_2'))
    df.drop('Name_1', axis=1, inplace=True)
    df.drop('Name_2', axis=1, inplace=True)

    # Filter out datapoints where Antoine is outside of the T range
    df = df[
        (df['T_K'] >= df['T range, from_1']) & (df['T_K'] <= df['T range, to_1']) &
        (df['T_K'] >= df['T range, from_2']) & (df['T_K'] <= df['T range, to_2'])
    ]

    # save VLE data with all x,y,T,P information complete to calculate experimental \gammas
    mask = df[['x', 'y', 'T_K', 'P_kPa']].notnull().all(axis=1)
    df_complete = df[mask].copy()

    # -- Filter out infeasible systems for GH-GNN
    mask3 = df['SMILES_1'].apply(check_infeasible_mol_for_ghgnn) 
    mask4 = df['SMILES_2'].apply(check_infeasible_mol_for_ghgnn)
    combined_mask = mask3 | mask4
    df_complete = df_complete[~combined_mask]

    
    # -- Calculate experimental gammas
    T = df_complete['T_K'].to_numpy()
    P = df_complete['P_kPa'].to_numpy()
    for cp in [1, 2]:
        A = df_complete[f'Coefficient A_{cp}'].to_numpy()
        B = df_complete[f'Coefficient B_{cp}'].to_numpy()
        C = df_complete[f'Coefficient C_{cp}'].to_numpy()
        D = df_complete[f'Coefficient D_{cp}'].to_numpy()
        
        df_complete[f'P_sat_{cp}'] = KDB_correlation_Pvap(T,A,B,C,D)

        if cp == 1:
            x = df_complete['x'].to_numpy()
            y = df_complete['y'].to_numpy()
        else:
            x = 1 - df_complete['x'].to_numpy()
            y = 1 - df_complete['y'].to_numpy()
        df_complete[f'gamma_{cp}'] = (y * P) / (x * df_complete[f'P_sat_{cp}'].to_numpy())

    df_complete.to_csv('data/processed/kdb_vle.csv', index=False)
    print(f' ---> Number of datapoints complete: {df_complete.shape[0]}')
