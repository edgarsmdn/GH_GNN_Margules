import pandas as pd
import ipdb
from tqdm import tqdm
from src.utils.utils import get_ternary_VLE_isobaric, get_ternary_VLE_isothermal, get_ternary_VLE_isobaric_unifac, get_ternary_VLE_isothermal_unifac, percentage_within_threshold
from argparse import ArgumentParser
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import numpy as np

# python src/models/08_predict_ternary_vles.py --type_vle isobaric --model UNIFAC_Do

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--type_vle", help="Type of VLE to calculate", type=str)
    parser.add_argument("--model", help="Model to be used in the predictions", type=str)
    
    args = parser.parse_args()
    type_vle = args.type_vle
    model = args.model

    assert type_vle in ['isothermal', 'isobaric']
    assert model in ['GH_GNN_Margules', 'UNIFAC_Do']

    print('-'*150)
    print(f'Predicting ternary {type_vle} VLEs with {model}')
    
    df = pd.read_csv('models/GHGNN/ternary_vle_IDACs_pred.csv')
    df = df[df['Type'] == type_vle.capitalize()].copy()
    
    y_1_lst, y_2_lst, y_3_lst = [],[],[]
    for i, row in tqdm(df.iterrows(), total=df.shape[0], desc='Predicting specific points'):
        P = row['P_kPa']
        T = row['T_K']
        c_1 = {
                'x':row['x_1'],
                'Pvap_constants':(row['Coefficient A_1'], 
                                  row['Coefficient B_1'],
                                  row['Coefficient C_1'],
                                  row['Coefficient D_1']),
                'inchikey':row['inchikey_1'],
                'GH_constants':(row['K1_12'],
                                row['K2_12'],
                                row['K1_13'],
                                row['K2_13'],
                                )
            }
        c_2 = {
                'x':row['x_2'],
                'Pvap_constants':(row['Coefficient A_2'], 
                                  row['Coefficient B_2'],
                                  row['Coefficient C_2'],
                                  row['Coefficient D_2']),
                'inchikey':row['inchikey_2'],
                'GH_constants':(row['K1_21'],
                                row['K2_21'],
                                row['K1_23'],
                                row['K2_23'],
                                )
            }
        c_3 = {
                'x':row['x_3'],
                'Pvap_constants':(row['Coefficient A_3'], 
                                  row['Coefficient B_3'],
                                  row['Coefficient C_3'],
                                  row['Coefficient D_3']),
                'inchikey':row['inchikey_3'],
                'GH_constants':(row['K1_31'],
                                row['K2_31'],
                                row['K1_32'],
                                row['K2_32'],
                                )
            }
        
        T1_min = row['T range, from_1']
        T2_min = row['T range, from_2']
        T3_min = row['T range, from_3']
        T1_max = row['T range, to_1']
        T2_max = row['T range, to_2']
        T3_max = row['T range, to_3']
        T_bounds = (max(T1_min, T2_min, T3_min), min(T1_max, T2_max, T3_max)) # Get the most constrained T bounds for feasible P_vap correlation
        if model == 'UNIFAC_Do':
            if type_vle == 'isothermal':
                (x_1, y_1), (x_2, y_2), (x_3, y_3), P = get_ternary_VLE_isothermal_unifac(T, c_1, c_2, c_3, specific=True)
            elif type_vle == 'isobaric':
                (x_1, y_1), (x_2, y_2), (x_3, y_3), T = get_ternary_VLE_isobaric_unifac(P, c_1, c_2, c_3, T_bounds, specific=True)
        elif model == 'GH_GNN_Margules':
            if type_vle == 'isothermal':
                (x_1, y_1), (x_2, y_2), (x_3, y_3), P = get_ternary_VLE_isothermal(T, c_1, c_2, c_3, specific=True)
            elif type_vle == 'isobaric':
                (x_1, y_1), (x_2, y_2), (x_3, y_3), T = get_ternary_VLE_isobaric(P, c_1, c_2, c_3, T_bounds, specific=True)
        y_1_lst.append(y_1)
        y_2_lst.append(y_2)
        y_3_lst.append(y_3)
    df['y_1_pred'] = y_1_lst
    df['y_2_pred'] = y_2_lst
    df['y_3_pred'] = y_3_lst

    # Get performance metrics
    threshold = 0.03
    y_1_true = df['y_1'].to_numpy()
    y_1_pred = df['y_1_pred'].to_numpy()
    y_2_true = df['y_2'].to_numpy()
    y_2_pred = df['y_2_pred'].to_numpy()

    y_true = np.concatenate([y_1_true, y_2_true])
    y_pred = np.concatenate([y_1_pred, y_2_pred])
    
    print(f'MAE y: {mean_absolute_error(y_true, y_pred)}')
    print(f'MAPE y: {mean_absolute_percentage_error(y_true, y_pred)*100}')
    print(f'R2 y: {r2_score(y_true, y_pred)}')
    print(f'% AE <=  {threshold} y: {percentage_within_threshold(y_true, y_pred, threshold)}')

    for sys in df['ID'].unique().tolist():
        df_spec = df[df['ID'] == sys]
        y_1_true = df_spec['y_1'].to_numpy()
        y_1_pred = df_spec['y_1_pred'].to_numpy()
        y_2_true = df_spec['y_2'].to_numpy()
        y_2_pred = df_spec['y_2_pred'].to_numpy()
        y_true = np.concatenate([y_1_true, y_2_true])
        y_pred = np.concatenate([y_1_pred, y_2_pred])
        print(f'  ---> System {sys} ---> MAE y: {mean_absolute_error(y_true, y_pred)}')

    # Save predictions
    df.to_csv(f'models/{model}/organic_old/ternary_{type_vle}_pred.csv', index=False)