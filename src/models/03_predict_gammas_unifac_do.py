import ipdb
import pandas as pd
from argparse import ArgumentParser
import pickle
from rdkit import Chem
from thermo.unifac import UNIFAC

# python src/models/03_predict_gammas_unifac_do.py --version combined

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--version", help="GH-GNN version to use for merging the predictions", type=str)

    args = parser.parse_args()
    version = args.version

    with open('data/external/unifacdo_fragmentations.pkl', 'rb') as file:
        unifac_do_fragments = pickle.load(file)
    
    df = pd.read_csv(f'models/GHGNN/kdb_vle_pred_{version}.csv')
    Ts = df['T_K'].tolist()
    x1s = df['x'].tolist()
    smiles_1 = df['SMILES_1'].tolist()
    smiles_2 = df['SMILES_2'].tolist()
    inchikeys_1 = [Chem.inchi.MolToInchiKey(Chem.MolFromSmiles(sm)) for sm in smiles_1]
    inchikeys_2 = [Chem.inchi.MolToInchiKey(Chem.MolFromSmiles(sm)) for sm in smiles_2]

    unique_inchikeys = list(set(inchikeys_1 + inchikeys_2))
    print(f'Unique inchikeys: {len(unique_inchikeys)}')
    failed = 0
    for inchikey in unique_inchikeys:
        frag = unifac_do_fragments[inchikey]['successful_fragmentation']
        if frag == 0:
            failed += 1
    print(f'Could not be fragmented with UNIFAC Unique inchikeys: {failed}')

    gammas_1, gammas_2 = [], []
    for sm1, sm2, inch_1, inch_2, T, x1 in zip(smiles_1, smiles_2, inchikeys_1, inchikeys_2, Ts, x1s):
        frag_1 = unifac_do_fragments[inch_1]
        frag_2 = unifac_do_fragments[inch_2]

        succ_1 = frag_1['successful_fragmentation']
        succ_2 = frag_2['successful_fragmentation']
        
        if succ_1 == 1 and succ_2 == 1:
            sub_1 = frag_1['subgroup_count']
            sub_2 = frag_2['subgroup_count']
            try:
                GE = UNIFAC.from_subgroups(chemgroups=[sub_1, sub_2], T=T, xs=[x1, 1-x1], version=1)
                gamma_1, gamma_2 = GE.gammas()
            except:
                gamma_1 = ''
                gamma_2 = ''
            gammas_1.append(gamma_1)
            gammas_2.append(gamma_2)
        else:
            gammas_1.append('')
            gammas_2.append('')

    df['inchikey_1'] = inchikeys_1
    df['inchikey_2'] = inchikeys_2
    df['pred_gamma_1_unifac_do'] = gammas_1
    df['pred_gamma_2_unifac_do'] = gammas_2
    df.to_csv(f'models/GHGNN/kdb_vle_pred_{version}_with_unifac_do.csv', index=False)

    n_compounds = len(set(df['Compound_1'].unique().tolist() + df['Compound_2'].unique().tolist()))
    n_combinations = (df['Compound_1'] + '_' + df['Compound_2']).nunique()
    n_isobaric = df[df['Type'] == 'Isobaric']['ID'].nunique()
    n_isothermal = df[df['Type'] == 'Isothermal']['ID'].nunique()
    n_random = df[df['Type'] == 'Random']['ID'].nunique()
    
    print(f'Number of points: {df.shape[0]}')
    print(f'Number of compounds: {n_compounds}')
    print(f'Unique combinations: {n_combinations}')
    print(f'Number of  isobaric subsets: {n_isobaric}')
    print(f'Number of  isothermal subsets: {n_isothermal}')
    print(f'Number of  random subsets: {n_random}')
    print(f'P range (kPa): {df["P_kPa"].min()} - {df["P_kPa"].max()}')
    print(f'T range (K): {df["T_K"].min()} - {df["T_K"].max()}')
            


    
    