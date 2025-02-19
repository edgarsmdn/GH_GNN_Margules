
import pandas as pd

# python src/data/06_get_general_statistics_data.py

if __name__ == "__main__":

    df = pd.read_csv('models/GHGNN/kdb_vle_pred_organic_old_with_unifac_do.csv')

    jaccard_threshold = 0.6
    # Filter out systems not within Jaccard distance threshold
    df = df[df['Jaccard_distance'] <= jaccard_threshold]

    n_compounds = len(set(df['Compound_1'].unique().tolist() + df['Compound_2'].unique().tolist()))
    sorted_combinations = df.apply(lambda row: '/'.join(sorted([row['Compound_1'], row['Compound_2']])), axis=1)
    n_combinations = sorted_combinations.nunique()
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