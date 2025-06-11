
import ipdb
import pandas as pd
from src.utils.utils import fredenslund_consistency_test


# python src/data/00_thermodynamic_consistency.py

if __name__ == "__main__":

    version = 'organic_old'
    jaccard_threshold = 0.6
    legendre_order = 5

    df = pd.read_csv(f'models/GHGNN/kdb_vle_pred_{version}_with_unifac_do.csv')
    df = df[df['Jaccard_distance'] <= jaccard_threshold]

    df_isobaric = df[df['Type'] == 'Isobaric']
    df_isothermal = df[df['Type'] == 'Isothermal']

    df_consistency_isobaric = fredenslund_consistency_test(df_isobaric, legendre_order=legendre_order, vle_type='isobaric')
    df_consistency_isobaric.to_csv('data/interim/consistency_isobaric.csv', index=False)

    counts = df_consistency_isobaric['is_consistent'].value_counts()
    percentages = df_consistency_isobaric['is_consistent'].value_counts(normalize=True) * 100
    print('Isobaric')
    print(counts)
    print(percentages)

    print('*'*150)

    df_consistency_isothermal = fredenslund_consistency_test(df_isothermal, legendre_order=legendre_order, vle_type='isothermal')
    df_consistency_isothermal.to_csv('data/interim/consistency_isothermal.csv', index=False)

    counts = df_consistency_isothermal['is_consistent'].value_counts()
    percentages = df_consistency_isothermal['is_consistent'].value_counts(normalize=True) * 100
    print('Isothermal')
    print(counts)
    print(percentages)



