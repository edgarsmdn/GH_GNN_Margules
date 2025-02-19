import ipdb
import pandas as pd
from src.utils.utils import get_file_info, extract_system_info, replace_symbols, sanitize_element, trim_df
from tqdm import tqdm
import numpy as np

# python src/data/04_get_complete_vle_data.py

if __name__ == "__main__":

    raw_data_folder = 'data/raw/KDB/binary_vle'

    final_id = 4602

    df_lst = list()
    for i in tqdm(range(1, final_id+1)):
        success, title, method = get_file_info(f'{raw_data_folder}/{i}_info.txt')
        if success:
            system_type, compound1, compound2 = extract_system_info(title)
            vle_data = pd.read_csv(f'{raw_data_folder}/{i}.csv')

            n_points = vle_data.shape[0]

            # Standarize units to P in kPa and T in K
            T_units_old, P_units_old = vle_data.columns.tolist()[:2]
            try:
                T_old = np.array([sanitize_element(elem) for elem in vle_data[T_units_old].to_numpy()], dtype=float)
                P_old = np.array([sanitize_element(elem) for elem in vle_data[P_units_old].to_numpy()], dtype=float)
                T_err_old = np.array([sanitize_element(elem) for elem in vle_data['T Err'].to_numpy()])
                P_err_old = np.array([sanitize_element(elem) for elem in vle_data['P Err'].to_numpy()])
                x_old = np.array([sanitize_element(elem) for elem in vle_data['X'].to_numpy()], dtype=float)
                y_old = np.array([sanitize_element(elem) for elem in vle_data['Y'].to_numpy()], dtype=float)
                x_err_old = np.array([sanitize_element(elem) for elem in vle_data['X Err'].to_numpy()])
                y_err_old = np.array([sanitize_element(elem) for elem in vle_data['Y Err'].to_numpy()])
            except Exception as e:
                print('\n', e)
                continue
            
            if not any(isinstance(s, str) for s in T_err_old):
                T_err_old = np.array([0]*n_points)
            else:
                if any('%' not in s for s in T_err_old):
                    T_err_old = np.array([float(val.replace('+-', '')) if isinstance(val, str) else 0 for val in T_err_old])
                elif any('%' in s for s in T_err_old):
                    # convert relative to absolute
                    T_err_old = np.array([float(replace_symbols(val,{'+-':'', '%':''})) if isinstance(val, str) else 0 for val in T_err_old])
                    T_err_old = T_err_old * T_old
                
            
            if not any(isinstance(s, str) for s in P_err_old):
                P_err_old = np.array([0]*n_points)
            else:
                
                if any('%' not in s for s in P_err_old):
                    P_err_old = np.array([float(val.replace('+-', '')) if isinstance(val, str) else 0 for val in P_err_old])
                elif any('%' in s for s in P_err_old):
                    # convert relative to absolute
                    P_err_old = np.array([float(replace_symbols(val,{'+-':'', '%':''})) if isinstance(val, str) else 0 for val in P_err_old])
                    P_err_old = P_err_old * P_old
                
            if T_units_old == 'T, deg.C':
                T = T_old + 273.15
                T_err = T_err_old
            elif T_units_old == 'T, K':
                T = T_old
                T_err = T_err_old
            elif T_units_old == 'T, deg.F':
                T =  (T_old - 32) * 5/9 + 273.15
                T_err = T_err_old * 5/9
            elif T_units_old == 'T, deg.R':
                T =  T_old * 5/9
                T_err = T_err_old * 5/9
                

            if P_units_old == 'P, psi':
                P = P_old * 6.89475729
                P_err = P_err_old * 6.89475729
            elif P_units_old == 'P, atm':
                P = P_old * 101.325
                P_err = P_err_old * 101.325
            elif P_units_old == 'P, MPa':
                P = P_old * 1000
                P_err = P_err_old * 1000
            elif P_units_old == 'P, mmHg':
                P = P_old * 0.13332237
                P_err = P_err_old * 0.13332237
            elif P_units_old == 'P, kPa':
                P = P_old
                P_err = P_err_old
            elif P_units_old == 'P, Torr':
                P = P_old * 0.13332237
                P_err = P_err_old * 0.13332237
            elif P_units_old == 'P, bar':
                P = P_old * 100
                P_err = P_err_old * 100
            elif P_units_old == 'P, Pa':
                P = P_old * 0.001
                P_err = P_err_old * 0.001

            vle_data['T_K'] = T
            vle_data['P_kPa'] = P
            vle_data['T_err_K'] = T_err
            vle_data['P_err_kPa'] = P_err

            # clean error composition data
            if not any(isinstance(s, str) for s in x_err_old):
                x_err_old = np.array([0]*n_points)
            else:
                if any('%' not in s for s in x_err_old):
                    x_err_old = np.array([float(val.replace('+-', '')) if isinstance(val, str) else 0 for val in x_err_old])
                elif any('%' in s for s in x_err_old):
                    # convert relative to absolute
                    x_err_old = np.array([float(replace_symbols(val,{'+-':'', '%':''})) if isinstance(val, str) else 0 for val in x_err_old])
                    x_err_old = x_err_old * x_old
            
            if not any(isinstance(s, str) for s in y_err_old):
                y_err_old = np.array([0]*n_points)
            else:
                if any('%' not in s for s in y_err_old):
                    y_err_old = np.array([float(val.replace('+-', '')) if isinstance(val, str) else 0 for val in y_err_old])
                elif any('%' in s for s in y_err_old):
                    # convert relative to absolute
                    y_err_old = np.array([float(replace_symbols(val,{'+-':'', '%':''})) if isinstance(val, str) else 0 for val in y_err_old])
                    y_err_old = y_err_old * y_old
            
            
            df_vle_standard = pd.DataFrame({
                'ID':[i]*n_points,
                'Type':[system_type]*n_points,
                'Compound_1':[compound1]*n_points,
                'Compound_2':[compound2]*n_points,
                'Method':[method]*n_points,
                'T_K':T,
                'P_kPa':P,
                'T_err_K':T_err,
                'P_err_kPa':P_err,
                'x':x_old,
                'y':y_old,
                'x_err':x_err_old,
                'y_err':y_err_old,
            })

            df_lst.append(df_vle_standard)

    df_VLE = pd.concat(df_lst)
    df_VLE = df_VLE.applymap(trim_df)

    # Standarize name of same compound, change METHYL-ISO-BUTYL KETONE to METHYL ISOBUTYL KETONE
    df_VLE['Compound_1'] = df_VLE['Compound_1'].replace('METHYL-ISO-BUTYL KETONE', 'METHYL ISOBUTYL KETONE')
    df_VLE['Compound_2'] = df_VLE['Compound_2'].replace('METHYL-ISO-BUTYL KETONE', 'METHYL ISOBUTYL KETONE')

    # convert to floats
    for col in ['T_K', 'P_kPa', 'T_err_K', 'P_err_kPa', 'x', 'y', 'x_err', 'y_err']:
        df_VLE[col] = df_VLE[col].astype(float)

    df_VLE.to_csv('data/interim/collected_raw_vle_data.csv', index=False)