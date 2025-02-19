from tqdm import tqdm
import json
import ipdb
import glob
import os
import pandas as pd


from src.utils.utils import cas_to_smiles, name_to_smiles_OPSIN


# python src/data/03_include_SMILES_pure_compounds.py

if __name__ == "__main__":

    pure_compound_folder = 'data/raw/KDB/pure_component_properties'

    json_files = glob.glob(pure_compound_folder + '/*.json')
    n_no_smiles = 0

    names_of_missing_compounds = []
    ids_of_missing_compounds = []
    for file in tqdm(json_files):
        with open(f'{file}', 'r') as f:
            pure_compound_data = json.load(f)
        cas = pure_compound_data['CAS No.']
        name = pure_compound_data['Name']
        id_comp = pure_compound_data['ID']

        # If CAS and name are not available, eliminate the redundant pure compound
        if not isinstance(cas, str) and not isinstance(name, str):
            os.remove(file)
        else:
            if 'SMILES' not in pure_compound_data or pure_compound_data['SMILES'] is None and isinstance(cas, str):
                # Try to get SMILES from CAS-RN from PubChem
                smiles = cas_to_smiles(cas)
                pure_compound_data['SMILES'] = smiles
                with open(f'{file}', 'w') as f:
                    json.dump(pure_compound_data, f, indent=4)

                if smiles is None:
                    # Try to get SMILES from name from OPSIN
                    smiles = name_to_smiles_OPSIN(name)
                    pure_compound_data['SMILES'] = smiles
                    with open(f'{file}', 'w') as f:
                        json.dump(pure_compound_data, f, indent=4)

                    if smiles is None:
                        names_of_missing_compounds.append(name)
                        ids_of_missing_compounds.append(id_comp)
                        n_no_smiles += 1
            elif not isinstance(cas, str):
                # If CAS-RN is not available, try to get SMILES from name from OPSIN
                smiles = name_to_smiles_OPSIN(name)
                pure_compound_data['SMILES'] = smiles
                with open(f'{file}', 'w') as f:
                    json.dump(pure_compound_data, f, indent=4)

                if smiles is None:
                    names_of_missing_compounds.append(name)
                    ids_of_missing_compounds.append(id_comp)
                    n_no_smiles += 1
    print(f'Number of compounds without SMILES {n_no_smiles}')
    df_missing = pd.DataFrame({
        'ID':ids_of_missing_compounds,
        'Name':names_of_missing_compounds
    })
    df_missing.to_csv('data/interim/missing_smiles.csv', index=False)

    
