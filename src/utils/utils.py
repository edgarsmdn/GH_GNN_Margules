
import ipdb
from bs4 import BeautifulSoup
import requests
import pandas as pd
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json
import numpy as np
from rdkit import Chem
import matplotlib.pyplot as plt
import seaborn as sns
from thermo.unifac import UNIFAC
import pickle
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.optimize import minimize, minimize_scalar
from models.GHGNN.ghgnn_old import atom_features, bond_features
import mpltern

def cas_to_smiles(cas_rn):
        """
        Converts a CAS Registry Number (CAS RN) to its corresponding SMILES representation 
        by querying PubChem.

        Parameters:
            cas_rn (str): The CAS Registry Number of the compound.

        Returns:
            str or None: The SMILES representation of the compound if found, otherwise None.
        """
        try:
            # Query PubChem to retrieve compound information
            url_canonical = f"https://pubchem.ncbi.nlm.nih.gov/compound/{cas_rn}#section=Canonical-SMILES"

            driver = webdriver.Chrome()
            driver.get(url_canonical)
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "Canonical-SMILES")))
            html_content = driver.page_source
            soup = BeautifulSoup(html_content, 'html.parser')
            driver.quit()

            # Find Canonical SMILES
            canonical_smiles_section = soup.find('section', id='Canonical-SMILES')
            canonical_smiles_div = canonical_smiles_section.find('div', class_='break-words space-y-1')
            canonical_smiles = canonical_smiles_div.get_text(strip=True)

            isomeric_smiles_section = soup.find('section', id='Isomeric-SMILES')
            if isomeric_smiles_section is not None:
                isomeric_smiles_div = isomeric_smiles_section.find('div', class_='break-words space-y-1')
                isomeric_smiles = isomeric_smiles_div.get_text(strip=True)
                smiles = isomeric_smiles
            else:
                smiles = canonical_smiles
        except Exception as e:
            smiles = None
            # print(f"An error occurred: {str(e)}")
        return smiles

def name_to_smiles_OPSIN(name):
    """
    Converts a chemical name to its corresponding SMILES representation using the OPSIN web service.

    Parameters:
        name (str): The IUPAC or common name of the compound.

    Returns:
        str or None: The SMILES representation of the compound if found, otherwise None.
    """

    url = f'https://opsin.ch.cam.ac.uk/opsin/{name}'
    webpage = requests.get(url)
    content = json.loads(webpage.content.decode("utf-8"))
    if content['status'] == 'SUCCESS':
        smiles = content['smiles']
    else:
        smiles = None
    return smiles

def get_file_info(file_path):
    """
    Extracts success status, title, and method information from a given file.

    Parameters:
        file_path (str): The path to the file to be read.

    Returns:
        tuple: A tuple containing:
            - success_status (bool): Whether the operation was successful.
            - title (str or None): The extracted title if present, otherwise None.
            - method (str or None): The extracted method if present, otherwise None.
    """
    success_status = False
    title = None
    method = None
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('Success:'):
                success_status = line.split(':')[1].strip().lower() == 'true'
            elif line.startswith('Title:'):
                title = ':'.join(line.split(':')[1:])[1:-1]
            elif line.startswith('Method:'):
                method = line.split(':')[1].strip()
    
    return success_status, title, method

def extract_system_info(string):
    """
    Extracts system type and compound information from a given string.

    Parameters:
        string (str): The input string containing system information.

    Returns:
        tuple: A tuple containing:
            - system_type (str): The type of system extracted from the string.
            - compound1 (str): The first compound extracted.
            - compound2 (str): The second compound extracted.
    """

    system_type = string.split(' ')[0]
    compound1 = string.split(':')[1].split('+')[0][1:-1]
    # check if at is in the title
    if 'at' in string.split(':')[1].split('+')[1]:
        compound2 = string.split(':')[1].split('+')[1].split('at')[0][1:-1]
    else:
        compound2 = string.split(':')[1].split('+')[1][1:]

    return system_type, compound1, compound2

def replace_symbols(text, replacements):
    """
    Replaces multiple symbols in a given string based on a provided mapping.

    Parameters:
        text (str): The input string where replacements will be made.
        replacements (dict): A dictionary mapping old symbols (keys) to new symbols (values).

    Returns:
        str: The modified string with replacements applied.
    """
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def sanitize_element(element):
    """
    Sanitizes an element by checking for invalid patterns and modifying specific cases.

    Parameters:
        element (str): The input element to be sanitized.

    Returns:
        str or float: The sanitized element. Returns NaN if the element is just a dot.
                    Removes trailing '+' signs. Adjusts specific percentage formats.

    Raises:
        Exception: If a double or triple point is detected in the data.
    """
    if isinstance(element, str):
        if '..' in element:
            raise Exception('Double point detected in data')
        elif '...' in element:
            raise Exception('Triple point detected in data')
        elif element.count('.') > 1:
            raise Exception('Double point detected in data')
        elif element == '.':
            return np.nan
        elif element[-1] == '+':
            return element[:-1]
        elif element[-1] == '%' and element[:2] == '0-':
            return element[2:]
    return element

def trim_df(x):
    """
    Trims leading and trailing whitespace from a string if the input is a string.

    Parameters:
        x (any): The input value to be processed.

    Returns:
        any: The trimmed string if the input is a string, otherwise returns the input unchanged.
    """
    if isinstance(x, str):
        return x.strip()
    else:
        return x
    
def load_json_files(folder_path):
    """
    Loads all JSON files from a specified folder into a dictionary.

    Parameters:
        folder_path (str): The path to the folder containing JSON files.

    Returns:
        dict: A dictionary where keys are the 'Name' field from each JSON file,
            and values are the corresponding JSON data.
    """

    data_dict = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                info = json.load(file)
                data_dict[info['Name']] = info
    return data_dict

def KDB_correlation_Pvap(T, A, B, C, D):
    """
    Calculates the vapor pressure (Pvp) using the KDB correlation.

    The equation used is:
        ln(Pvp) = A * ln(T) + B / T + C + D * T^2

    Parameters:
        T (float or ndarray): Temperature in Kelvin.
        A (float): Empirical coefficient.
        B (float): Empirical coefficient.
        C (float): Empirical coefficient.
        D (float): Empirical coefficient.

    Returns:
        float or ndarray: The calculated vapor pressure (Pvp) in kPa.
    """
    return np.exp(A*np.log(T) + B/T + C + D*T**2)


def check_infeasible_mol_for_ghgnn(smiles):
    """
    Checks whether a given SMILES string is infeasible for use in the GH-GNN model.

    Parameters:
        smiles (str): The SMILES representation of the molecule.

    Returns:
        bool: True if the molecule is infeasible (e.g., invalid SMILES or processing error 
                due to featurization used to develop the GH-GNN), 
            False if it is feasible.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)

        atoms  = mol.GetAtoms()
        bonds  = mol.GetBonds()

        [atom_features(atom) for atom in atoms]
        [bond_features(bond) for bond in bonds]
        return False
    except:
        return True
    

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
def find_name_from_smiles(smiles_to_find, pure_comp_dict):
    """
    Finds the compound name corresponding to a given SMILES string.

    Parameters:
        smiles_to_find (str): The SMILES string to search for.
        pure_comp_dict (dict): A dictionary where keys are compound names and values are compound information 
                                (including SMILES strings).

    Returns:
        str or None: The compound name if a match is found, otherwise None.
    """
    for compound_name, compound_info_set in pure_comp_dict.items():
        try:
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(compound_info_set['SMILES']))
        except:
            smiles = None
        if smiles == Chem.MolToSmiles(Chem.MolFromSmiles(smiles_to_find)):
            return compound_name
    return None

def margules_binary(ln_IDAC_12, ln_IDAC_21, x_1, x_2):
    """
    Calculates the activity coefficients (ln(gamma)) for a binary mixture using the Margules model.

    Parameters:
        ln_IDAC_12 (float): The natural logarithm of the activity coefficient of 1 infinitely diluted in 2.
        ln_IDAC_21 (float): The natural logarithm of the activity coefficient of 2 infinitely diluted in 1.
        x_1 (float): Mole fraction of component 1 in the mixture.
        x_2 (float): Mole fraction of component 2 in the mixture.

    Returns:
        tuple: A tuple containing the activity coefficients for components 1 and 2, respectively.
    """
    w_12 = ln_IDAC_12
    w_21 = ln_IDAC_21

    gE = x_1*x_2*(x_1*w_21 + x_2*w_12)

    ln_gamma_1 = 2*w_21*x_1*x_2 + x_2**2 * w_12 - 2*gE
    ln_gamma_2 = 2*w_12*x_2*x_1 + x_1**2 * w_21 - 2*gE

    return np.exp(ln_gamma_1), np.exp(ln_gamma_2)

def percentage_within_threshold(y_true, y_pred, threshold):
    """
    Calculate the percentage of points where the absolute error between y_true and y_pred
    is less than or equal to the given threshold.

    Parameters:
    - y_true: array-like, true target values.
    - y_pred: array-like, predicted values.
    - threshold: float, threshold value for the absolute error.

    Returns:
    - percentage: float, percentage of points within the threshold.
    """
    # Calculate absolute error
    absolute_errors = np.abs(y_true - y_pred)
    
    # Count the number of points within the threshold
    within_threshold_count = np.sum(absolute_errors <= threshold)
    
    # Calculate the total number of points
    total_points = len(y_true)
    
    # Calculate the percentage of points within the threshold
    percentage = (within_threshold_count / total_points) * 100
    
    return percentage

def plot_binary_VLE(c_1, label):
    """
    Plots a binary VLE (Vapor-Liquid Equilibrium) curve, including experimental and predicted data points.

    Parameters:
        c_1 (dict): A dictionary containing the data to be plotted, with the following keys:
            - 'exp_ids': Experimental identifiers.
            - 'x_exp': Molar liquid fraction of the experimental data.
            - 'y_exp': Molar vapor fraction of the experimental data.
            - 'x_pred': Molar liquid fraction of the predicted data.
            - 'y_pred': Molar vapor fraction of the predicted data.
            - 'Name': The name of the compound.
        label (str): The label for the predicted curve (e.g., 'GH_GNN_Margules' or 'UNIFAC_Do').

    Returns:
        matplotlib.figure.Figure: The generated plot figure.
    """
    if label == 'GH_GNN_Margules':
        label = 'GH-GNN-Margules'
    elif label == 'UNIFAC_Do':
        label = 'UNIFAC (Do)'

    exp_ids = c_1['exp_ids']
    x_exp = c_1['x_exp']
    y_exp = c_1['y_exp']
    x_pred = c_1['x_pred']
    y_pred = c_1['y_pred']

    unique_ids = set(exp_ids)
    markers = ['o', 's', 'D', '^', 'v', '>', '<', '*', 'x', '+']  # Define markers for each ID

    fig = plt.figure(figsize=(6,5))
    plt.plot([0,1], [0,1], 'k--', label='x=y')
    for idx, unique_id in enumerate(unique_ids):
        mask = exp_ids == unique_id
        plt.scatter(x_exp[mask], y_exp[mask], marker=markers[idx], label=f'Experimental ID: {unique_id}')
    plt.plot(x_pred, y_pred, '-', label=label)

    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel(f'Molar liquid fraction {c_1["Name"]}', fontsize=12)
    plt.ylabel(f'Molar vapor fraction {c_1["Name"]}', fontsize=12)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.close(fig)
    return fig

def create_folder(directory):
    """
    Create a folder/directory if it doesn't exist.

    Parameters:
    - directory: str, path of the directory to be created.
    """
    # Check if the directory already exists
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_heatmap_performance_MAE(df_errors):
    """
    Plots a heatmap of the mean absolute error (MAE) for different binary class combinations.

    Parameters:
        df_errors (pandas.DataFrame): A DataFrame containing error data with the following columns:
            - 'Binary_class': A column containing the binary class combinations (e.g., 'Class_A_Class_B').
            - 'MAE': The mean absolute error for each class combination.
            - 'n_points': The number of data points corresponding to each class combination.

    Returns:
        matplotlib.figure.Figure: The generated heatmap figure with annotations for the number of data points.
    """
    df_errors[['Class_A', 'Class_B']] = df_errors['Binary_class'].str.split('_', expand=True)
    heatmap_data = df_errors.pivot(index='Class_A', columns='Class_B', values='MAE')
    heatmap_data = heatmap_data.fillna(0)
    annotation_values = df_errors.pivot(index='Class_A', columns='Class_B', values='n_points').fillna(0).astype(int).values
    
    fig = plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(heatmap_data, annot=False, cmap='Greys', fmt=".0f", linewidths=0.5, linecolor='gray', vmin=0, vmax=0.1)
    # Add annotation values to the heatmap
    for i, (_, row) in enumerate(heatmap_data.iterrows()):
        for j, val in enumerate(row):
            # Determine color based on MAE value
            if val > 0.07:  # Change threshold as needed
                color = 'white'
            else:
                color = 'black'
            val = annotation_values[i, j]
            if val == 0:
                val=''
            heatmap.text(j+0.5, i+0.5, val, ha='center', va='center', color=color)
    plt.xlabel('') 
    plt.ylabel('')
    # Add vertical label to colorbar
    colorbar = heatmap.collections[0].colorbar
    colorbar.set_label('Mean absolute error', rotation=270, labelpad=-37, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    colorbar.ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.close(fig)
    return fig

def plot_cumulative_errors(df_errors, model):
    """
    Plots the cumulative errors of the predicted absolute errors (AE) for a model or multiple models.

    Parameters:
        df_errors (pandas.DataFrame): A DataFrame containing the MAE (mean absolute error) values.
            - If a single model is provided, it should contain a 'MAE' column.
            - If multiple models are provided, the DataFrame should contain columns for each model (e.g., 'MAE_GH_GNN_Margules', 'MAE_UNIFAC_Do').
        model (str or list): The model(s) to plot:
            - If a string, it plots the cumulative error for the specified model.
            - If a list, it compares the cumulative errors of multiple models.

    Returns:
        matplotlib.figure.Figure: The generated plot figure showing cumulative error vs. percentage of systems.
    """

    if isinstance(model, str):
        maes_ghgnn_margules = df_errors['MAE'].to_numpy()
        cumulative_percentage = np.arange(1, len(maes_ghgnn_margules) + 1) / len(maes_ghgnn_margules) * 100
        fig = plt.figure(figsize=(8, 5))
        plt.grid(True, which='major', color='gray', linestyle='dashed', alpha=0.3)
        # plt.plot(errors_unifac, cumulative_percentage, label='UNIFAC-IL', color='#33A5C3', alpha=0.7, lw=4)
        plt.plot(maes_ghgnn_margules, cumulative_percentage, label=model, color='#78004B', alpha=0.7, lw=4)
        plt.xlabel('Predicted $y_i$ absolute error (AE)', fontsize=18)
        plt.ylabel('Percentage of binary systems (%)', fontsize=18)
        plt.legend(fontsize=18)
        plt.xlim(0,0.1)
        plt.ylim(0,100)
        plt.xticks(fontsize=16)  # Set x-axis tick labels font size
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.close(fig)
    elif isinstance(model, list):
        fig = plt.figure(figsize=(8, 5))
        plt.grid(True, which='major', color='gray', linestyle='dashed', alpha=0.3)
        colors = ['#78004B', '#33A5C3']
        colors = ['#2c3e50', '#1f77b4']
        for i, m in enumerate(model):
            color = colors[i]
            if m =='GH_GNN_Margules':
                label = 'GH-GNN + Margules'
            elif m == 'UNIFAC_Do':
                label = 'UNIFAC-Dortmund'
            df_errors = df_errors.sort_values(by=f'MAE_{m}')
            maes = df_errors[f'MAE_{m}'].to_numpy()
            cumulative_percentage = np.arange(1, len(maes) + 1) / len(maes) * 100
            plt.plot(maes, cumulative_percentage, label=label, color=color, alpha=0.7, lw=4)
        plt.xlabel('Predicted $y_i$ absolute error (AE)', fontsize=18)
        plt.ylabel('Percentage of binary systems (%)', fontsize=18)
        plt.legend(fontsize=18)
        plt.xlim(0,0.1)
        plt.ylim(0,100)
        plt.xticks(fontsize=16)  # Set x-axis tick labels font size
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.close(fig)
    return fig

with open('data/external/unifacdo_fragmentations.pkl', 'rb') as file:
        unifac_do_fragments = pickle.load(file)

def get_binary_VLE_isothermal_unifac(T, c_1, c_2, specific=False):
    """
    Calculate the binary vapor-liquid equilibrium (VLE) for an isothermal system using the UNIFAC model.

    Parameters:
        T (float): The temperature (in Kelvin) at which to calculate the VLE.
        c_1 (dict): A dictionary containing information for component 1. Must include:
            - 'x': Mole fraction in the liquid phase (only needed if 'specific=True').
            - 'gamma': Activity coefficient for component 1 (only needed if 'specific=True').
            - 'Pvap_constants': Constants (A, B, C, D) for the Antoine equation for component 1's vapor pressure.
        c_2 (dict): A dictionary containing information for component 2. Must include:
            - 'x': Mole fraction in the liquid phase (only needed if 'specific=True').
            - 'gamma': Activity coefficient for component 2 (only needed if 'specific=True').
            - 'Pvap_constants': Constants (A, B, C, D) for the Antoine equation for component 2's vapor pressure.
        specific (bool): If True, gamma values and mole fractions are already provided in `c_1` and `c_2`.
                        If False, the function will compute gamma values using the UNIFAC model based on the InChI keys
                        of the components.

    Returns:
        tuple: A tuple containing:
            - A tuple (x_1, y_1) for component 1, where x_1 is the mole fraction and y_1 is the vapor phase mole fraction.
            - A tuple (x_2, y_2) for component 2, where x_2 is the mole fraction and y_2 is the vapor phase mole fraction.
            - The total pressure P (sum of partial pressures).
    """

    if specific:
        x_1 = c_1['x']
        x_2 = c_2['x']
        gamma_1 = c_1['gamma']
        gamma_2 = c_2['gamma']
    else:
        inchikey_1 = c_1['inchikey']
        inchikey_2 = c_2['inchikey']
        sub_1 = unifac_do_fragments[inchikey_1]['subgroup_count']
        sub_2 = unifac_do_fragments[inchikey_2]['subgroup_count']
        
        x_1 = np.linspace(0,1,100)
        x_2 = 1-x_1
        
        gamma_1, gamma_2 = np.zeros(x_1.shape[0]), np.zeros(x_1.shape[0])
        for i, x1 in enumerate(x_1):
            GE = UNIFAC.from_subgroups(chemgroups=[sub_1, sub_2], T=T, xs=[x1, 1-x1], version=1)
            g_1, g_2 = GE.gammas()
            gamma_1[i] = g_1
            gamma_2[i] = g_2

    A1, B1, C1, D1 = c_1['Pvap_constants']
    A2, B2, C2, D2 = c_2['Pvap_constants']

    P_vap_1 = KDB_correlation_Pvap(T, A1, B1, C1, D1)
    P_vap_2 = KDB_correlation_Pvap(T, A2, B2, C2, D2)

    p_1 = x_1 * gamma_1 * P_vap_1
    p_2 = x_2 * gamma_2 * P_vap_2
    P = p_1 + p_2

    y_1 = p_1/P
    y_2 = p_2/P

    return (x_1, y_1), (x_2, y_2), P


def get_binary_VLE_isothermal(T, c_1, c_2, specific=False, version=None):
    """
    Calculate the binary vapor-liquid equilibrium (VLE) for an isothermal system using the Margules model and GH-GNN
      parameters for activity coefficients.

    Parameters:
        T (float): The temperature (in Kelvin) at which to calculate the VLE.
        c_1 (dict): A dictionary containing information for component 1. Must include:
            - 'x': Mole fraction in the liquid phase (only needed if 'specific=True').
            - 'Pvap_constants': Constants (A, B, C, D) for the Antoine equation for component 1's vapor pressure.
            - 'GH_constants': Constants (K1, K2) for component 1's GH-GNN parameters.
        c_2 (dict): A dictionary containing information for component 2. Must include:
            - 'x': Mole fraction in the liquid phase (only needed if 'specific=True').
            - 'Pvap_constants': Constants (A, B, C, D) for the Antoine equation for component 2's vapor pressure.
            - 'GH_constants': Constants (K1, K2) for component 2's GH-GNN parameters.
        specific (bool): If True, mole fractions (`x`) are provided in `c_1` and `c_2`.
                        If False, the function computes `x_1` and `x_2` over a range from 0 to 1.
        version (str or None): If 'combined', uses a specific correlation version for the calculation of ln_IDAC values.
                                Defaults to None, using the standard correlation.

    Returns:
        tuple: A tuple containing:
            - A tuple (x_1, y_1) for component 1, where x_1 is the mole fraction and y_1 is the vapor phase mole fraction.
            - A tuple (x_2, y_2) for component 2, where x_2 is the mole fraction and y_2 is the vapor phase mole fraction.
            - The total pressure P (sum of partial pressures).
    """

    if specific:
        x_1 = c_1['x']
        x_2 = c_2['x']
    else:
        x_1 = np.linspace(0,1,100)
        x_2 = 1-x_1

    A1, B1, C1, D1 = c_1['Pvap_constants']
    A2, B2, C2, D2 = c_2['Pvap_constants']

    K1_1, K2_1 = c_1['GH_constants']
    K1_2, K2_2 = c_2['GH_constants']

    if version == 'combined':
        ln_IDAC_12 = K1_1 + K2_1/(T + 273.15) # because of missmatch int raining combined GH-GNN
        ln_IDAC_21 = K1_2 + K2_2/(T + 273.15) # because of missmatch int raining combined GH-GNN
    else:
        ln_IDAC_12 = K1_1 + K2_1/T
        ln_IDAC_21 = K1_2 + K2_2/T

    gamma_1, gamma_2 = margules_binary(ln_IDAC_12, ln_IDAC_21, x_1, x_2)

    P_vap_1 = KDB_correlation_Pvap(T, A1, B1, C1, D1)
    P_vap_2 = KDB_correlation_Pvap(T, A2, B2, C2, D2)

    p_1 = x_1 * gamma_1 * P_vap_1
    p_2 = x_2 * gamma_2 * P_vap_2
    P = p_1 + p_2

    y_1 = p_1/P
    y_2 = p_2/P

    return (x_1, y_1), (x_2, y_2), P

def get_errors_classes(df, data_type, models=['']):
    """
    Calculate error metrics (Mean Absolute Error and R-squared) for each binary class in a dataset.

    Parameters:
        df (DataFrame): A pandas DataFrame containing the data with the true and predicted values for each class.
        data_type (str): The type of data being analyzed. Can be 'isothermal', 'isobaric', or other (e.g., 'all').
        models (list of str): A list of model names (or empty string for the default model). Each entry corresponds to 
                            the name of a model to compare. If no models are provided, defaults to the base model.

    Returns:
        DataFrame: A DataFrame containing the error metrics for each binary class. The columns include:
            - 'Binary_class': The unique binary class identifier (sorted combination of Class_1 and Class_2).
            - 'MAE': Mean Absolute Error for the model(s).
            - 'R2': R-squared for the model(s).
            - 'n_points': The number of data points in each binary class.
            - 'n_systems' (optional): The number of unique systems in each binary class (based on the 'data_type').
    """

    df = df.copy()
    df['Binary_class'] = ['_'.join(sorted([row['Class_1'], row['Class_2']])) for _, row in df.iterrows()]
    unique_binary_classes = df['Binary_class'].unique().tolist()
    
    df_errors = pd.DataFrame({
        'Binary_class':unique_binary_classes
    })
    for model in models:
        if model != '':
            model = '_' + model
        bin_class_MAE = []
        bin_class_R2 = []
        n_points_bin_class = []
        n_systems = []
        for bin_class in unique_binary_classes:
            df_super_spec = df[df['Binary_class'] == bin_class].copy()
            if data_type == 'isothermal':
                df_super_spec['system_key'] = df_super_spec['Compound_1'] + '_' + df_super_spec['Compound_2'] + '_' + df_super_spec['T_K'].astype(str)
                n_systems.append(df_super_spec['system_key'].nunique())
            elif data_type == 'isobaric':
                df_super_spec['system_key'] = df_super_spec['Compound_1'] + '_' + df_super_spec['Compound_2'] + '_' + df_super_spec['P_kPa'].astype(str)
                n_systems.append(df_super_spec['system_key'].nunique())
            y_1_true = df_super_spec['y'].to_numpy()
            y_1_pred = df_super_spec[f'y_pred{model}'].to_numpy()

            bin_class_MAE.append(mean_absolute_error(y_1_true, y_1_pred))
            bin_class_R2.append(r2_score(y_1_true, y_1_pred))
            n_points_bin_class.append(df_super_spec.shape[0])
            

        df_errors[f'MAE{model}'] = bin_class_MAE
        df_errors[f'R2{model}'] = bin_class_R2
        df_errors['n_points'] = n_points_bin_class
        if data_type != 'all':
            df_errors['n_systems'] = n_systems
    df_errors = df_errors.sort_values(by=f'MAE{model}')
    return df_errors

def get_binary_VLE_isobaric(P, c_1, c_2, T_bounds, specific=False, SLSQP=False):
    """
    Calculate the binary vapor-liquid equilibrium (VLE) for an isobaric system.

    Parameters:
        P (float): The system pressure (in kPa).
        c_1 (dict): The data for component 1, including vapor pressure constants and vapor pressure constants.
        c_2 (dict): The data for component 2, including vapor pressure constants and vapor pressure constants.
        T_bounds (tuple): The temperature bounds (in Kelvin) for the system.
        specific (bool, optional): If True, the molar fractions (x_1, x_2) are taken directly from the components. 
                                    If False, they are computed over a range from 0 to 1 for component 1.
        SLSQP (bool, optional): If True, the SLSQP optimization method is used for temperature estimation. 
                                If False, the bounded method is used.

    Returns:
        tuple: A tuple containing the following elements:
            - (x_1, y_1): Molar fraction and vapor mole fraction for component 1.
            - (x_2, y_2): Molar fraction and vapor mole fraction for component 2.
            - Ts: The temperature values corresponding to the binary VLE calculation.

        If `specific` is True, the returned values are scalars for a single set of inputs. Otherwise, arrays of 
        values for the entire range of mole fractions are returned.
    """

    if specific:
        x_1 = np.array([c_1['x']])
        x_2 = np.array([c_2['x']])
    else:
        x_1 = np.linspace(0,1,100)
        x_2 = 1-x_1
    A1, B1, C1, D1 = c_1['Pvap_constants']
    A2, B2, C2, D2 = c_2['Pvap_constants']

    K1_1, K2_1 = c_1['GH_constants']
    K1_2, K2_2 = c_2['GH_constants']

    # Optimization
    p_1s = np.zeros(x_1.shape[0])
    p_2s = np.zeros(x_1.shape[0])
    Ts = np.zeros(x_1.shape[0])
    tol = 1e-5
    T0s = np.linspace(T_bounds[0], T_bounds[1], 10)
    for i in range(x_1.shape[0]):
    
        def error_in_P(T, P_true=P):
            P_vap_1 = KDB_correlation_Pvap(T, A1, B1, C1, D1)
            P_vap_2 = KDB_correlation_Pvap(T, A2, B2, C2, D2)
            
            ln_IDAC_12 = K1_1 + K2_1/T
            ln_IDAC_21 = K1_2 + K2_2/T
            
            gamma_1, gamma_2 = margules_binary(ln_IDAC_12, ln_IDAC_21, x_1[i], x_2[i])
            
            p_1 = x_1[i] * gamma_1 * P_vap_1
            p_2 = x_2[i] * gamma_2 * P_vap_2
            P_calc = p_1 + p_2
            
            return np.abs(P_calc - P_true)
        
        if SLSQP:
            error_Ps = []
            candi_Ts = []
            for T0 in T0s:
                results = minimize(error_in_P, bounds=[T_bounds], method='SLSQP', x0=T0, tol=tol, options={'maxiter':2000, 'ftol':tol})
                if results.success:
                    error_Ps.append(results.fun)
                    candi_Ts.append(results.x)
            idx_best = np.argmin(error_Ps)
            T = candi_Ts[idx_best]
        else:
            results = minimize_scalar(error_in_P, bounds=T_bounds, method='bounded', options={'maxiter':2000})
            if results.success:
                T = results.x
        
        P_vap_1 = KDB_correlation_Pvap(T, A1, B1, C1, D1)
        P_vap_2 = KDB_correlation_Pvap(T, A2, B2, C2, D2)
        
        ln_IDAC_12 = K1_1 + K2_1/T
        ln_IDAC_21 = K1_2 + K2_2/T
        
        gamma_1, gamma_2 = margules_binary(ln_IDAC_12, ln_IDAC_21, x_1[i], x_2[i])
        
        p_1s[i] = x_1[i]*gamma_1*P_vap_1
        p_2s[i] = x_2[i]*gamma_2*P_vap_2
        Ts[i] = T
        
    y_1 = p_1s/P
    y_2 = p_2s/P

    if specific:
        x_1, y_1, x_2, y_2, Ts = x_1[0], y_1[0], x_2[0], y_2[0], Ts[0]

    return (x_1, y_1), (x_2, y_2), Ts

def get_binary_VLE_isobaric_unifac(P, c_1, c_2, T_bounds, specific=False, SLSQP=False):
    """
    Calculate the binary vapor-liquid equilibrium (VLE) for an isobaric system using the UNIFAC model for activity coefficients.

    Parameters:
        P (float): The system pressure (in kPa).
        c_1 (dict): The data for component 1, including vapor pressure constants and inchikey.
        c_2 (dict): The data for component 2, including vapor pressure constants and inchikey.
        T_bounds (tuple): The temperature bounds (in Kelvin) for the system.
        specific (bool, optional): If True, the molar fractions (x_1, x_2) are taken directly from the components. 
                                    If False, they are computed over a range from 0 to 1 for component 1.
        SLSQP (bool, optional): If True, the SLSQP optimization method is used for temperature estimation. 
                                If False, the bounded method is used.

    Returns:
        tuple: A tuple containing the following elements:
            - (x_1, y_1): Molar fraction and vapor mole fraction for component 1.
            - (x_2, y_2): Molar fraction and vapor mole fraction for component 2.
            - Ts: The temperature values corresponding to the binary VLE calculation.

        If `specific` is True, the returned values are scalars for a single set of inputs. Otherwise, arrays of 
        values for the entire range of mole fractions are returned.
    """
    if specific:
        x_1 = np.array([c_1['x']])
        x_2 = np.array([c_2['x']])
    else:
        x_1 = np.linspace(0,1,100)
        x_2 = 1-x_1
    
    A1, B1, C1, D1 = c_1['Pvap_constants']
    A2, B2, C2, D2 = c_2['Pvap_constants']

    inchikey_1 = c_1['inchikey']
    inchikey_2 = c_2['inchikey']

    sub_1 = unifac_do_fragments[inchikey_1]['subgroup_count']
    sub_2 = unifac_do_fragments[inchikey_2]['subgroup_count']

    # Optimization
    p_1s = np.zeros(x_1.shape[0])
    p_2s = np.zeros(x_1.shape[0])
    Ts = np.zeros(x_1.shape[0])
    tol = 1e-5
    T0s = np.linspace(T_bounds[0], T_bounds[1], 10)
    for i in range(x_1.shape[0]):

        def error_in_P(T, P_true=P):
            P_vap_1 = KDB_correlation_Pvap(T, A1, B1, C1, D1)
            P_vap_2 = KDB_correlation_Pvap(T, A2, B2, C2, D2)
            
            GE = UNIFAC.from_subgroups(chemgroups=[sub_1, sub_2], T=T, xs=[x_1[i], 1-x_1[i]], version=1)
            gamma_1, gamma_2 = GE.gammas()
            
            p_1 = x_1[i] * gamma_1 * P_vap_1
            p_2 = x_2[i] * gamma_2 * P_vap_2
            P_calc = p_1 + p_2
            
            return np.abs(P_calc - P_true)
        
        if SLSQP:
            error_Ps = []
            candi_Ts = []
            for T0 in T0s:
                results = minimize(error_in_P, bounds=[T_bounds], method='SLSQP', x0=T0, tol=tol, options={'maxiter':2000})
                if results.success:
                    error_Ps.append(results.fun)
                    candi_Ts.append(results.x)
            idx_best = np.argmin(error_Ps)
            T = candi_Ts[idx_best]
        else:
            results = minimize_scalar(error_in_P, bounds=T_bounds, method='bounded', options={'maxiter':2000})
            if results.success:
                T = results.x
        
        P_vap_1 = KDB_correlation_Pvap(T, A1, B1, C1, D1)
        P_vap_2 = KDB_correlation_Pvap(T, A2, B2, C2, D2)

        GE = UNIFAC.from_subgroups(chemgroups=[sub_1, sub_2], T=T, xs=[x_1[i], 1-x_1[i]], version=1)
        gamma_1, gamma_2 = GE.gammas()
        
        p_1s[i] = x_1[i]*gamma_1*P_vap_1
        p_2s[i] = x_2[i]*gamma_2*P_vap_2
        Ts[i] = T
        
    y_1 = p_1s/P
    y_2 = p_2s/P

    if specific:
        x_1, y_1, x_2, y_2, Ts = x_1[0], y_1[0], x_2[0], y_2[0], Ts[0]

    return (x_1, y_1), (x_2, y_2), Ts

def margules_ternary(ln_IDAC_12, ln_IDAC_13, ln_IDAC_21, ln_IDAC_23, ln_IDAC_31, ln_IDAC_32, x_1, x_2, x_3):
    """
    Calculate the activity coefficients for a ternary system using the Margules model.

    Parameters:
        ln_IDAC_12 (float): Natural logarithm of the activity coefficient parameter between components 1 and 2.
        ln_IDAC_13 (float): Natural logarithm of the activity coefficient parameter between components 1 and 3.
        ln_IDAC_21 (float): Natural logarithm of the activity coefficient parameter between components 2 and 1.
        ln_IDAC_23 (float): Natural logarithm of the activity coefficient parameter between components 2 and 3.
        ln_IDAC_31 (float): Natural logarithm of the activity coefficient parameter between components 3 and 1.
        ln_IDAC_32 (float): Natural logarithm of the activity coefficient parameter between components 3 and 2.
        x_1 (float): Mole fraction of component 1 in the mixture.
        x_2 (float): Mole fraction of component 2 in the mixture.
        x_3 (float): Mole fraction of component 3 in the mixture.

    Returns:
        tuple: Activity coefficients for components 1, 2, and 3 (gamma_1, gamma_2, gamma_3).
    """

    w_12 = ln_IDAC_12
    w_13 = ln_IDAC_13
    w_21 = ln_IDAC_21
    w_23 = ln_IDAC_23
    w_31 = ln_IDAC_31
    w_32 = ln_IDAC_32
    w_123 = 0
    c_123 = 0.5*(w_12 + w_13 + w_21 + w_23 + w_31 + w_32) - w_123

    part_12 = x_1*x_2*(x_2*w_12 + x_1*w_21)
    part_13 = x_1*x_3*(x_3*w_13 + x_1*w_31)
    part_23 = x_2*x_3*(x_3*w_23 + x_2*w_32)

    gE = part_12 + part_13 + part_23 + x_1*x_2*x_3*c_123

    gamma_1 = np.exp((
                2*(x_1*x_2*w_12 + x_1*x_3*w_31) + x_2**2*w_12 + 
                x_3**2*w_13 + x_2*x_3*c_123 - 2*gE))
    gamma_2 = np.exp((
                2*(x_2*x_3*w_32 + x_2*x_1*w_12) + x_3**2*w_23 + 
                x_1**2*w_21 + x_3*x_1*c_123 - 2*gE))
        
    gamma_3 = np.exp((
            2*(x_3*x_1*w_13 + x_3*x_2*w_23) + x_1**2*w_31 + 
            x_2**2*w_32 + x_1*x_2*c_123 - 2*gE))

    return gamma_1, gamma_2, gamma_3

def get_ternary_VLE_isobaric(P, c_1, c_2, c_3, T_bounds, specific=False):
    """
    Calculate the ternary vapor-liquid equilibrium (VLE) for an isobaric system using the Margules model for activity coefficients.

    Parameters:
        P (float): The system pressure (in kPa).
        c_1 (dict): The data for component 1, including vapor pressure constants, mole fraction, and GH-GNN constants.
        c_2 (dict): The data for component 2, including vapor pressure constants, mole fraction, and GH-GNN constants.
        c_3 (dict): The data for component 3, including vapor pressure constants, mole fraction, and GH-GNN constants.
        T_bounds (tuple): The temperature bounds (in Kelvin) for the system.
        specific (bool, optional): If True, the mole fractions (x_1, x_2, x_3) are taken directly from the components. 
                                    If False, they are computed over a range of values.

    Returns:
        tuple: A tuple containing the following elements:
            - (x_1, y_1): Molar fraction and vapor mole fraction for component 1.
            - (x_2, y_2): Molar fraction and vapor mole fraction for component 2.
            - (x_3, y_3): Molar fraction and vapor mole fraction for component 3.
            - Ts: The temperature values corresponding to the ternary VLE calculation.

        If `specific` is True, the returned values are scalars for a single set of inputs. Otherwise, arrays of 
        values for the entire range of mole fractions are returned.
    """

    if specific:
        x_1 = np.array([c_1['x']])
        x_2 = np.array([c_2['x']])
        x_3 = np.array([c_3['x']])
    else:
        pass
    A1, B1, C1, D1 = c_1['Pvap_constants']
    A2, B2, C2, D2 = c_2['Pvap_constants']
    A3, B3, C3, D3 = c_3['Pvap_constants']

    K1_12, K2_12, K1_13, K2_13 = c_1['GH_constants']
    K1_21, K2_21, K1_23, K2_23 = c_2['GH_constants']
    K1_31, K2_31, K1_32, K2_32 = c_3['GH_constants']

    # Optimization
    p_1s, p_2s, p_3s, Ts = np.zeros(x_1.shape[0]), np.zeros(x_1.shape[0]), np.zeros(x_1.shape[0]), np.zeros(x_1.shape[0])
    for i in range(x_1.shape[0]):
    
        def error_in_P(T, P_true=P):
            P_vap_1 = KDB_correlation_Pvap(T, A1, B1, C1, D1)
            P_vap_2 = KDB_correlation_Pvap(T, A2, B2, C2, D2)
            P_vap_3 = KDB_correlation_Pvap(T, A3, B3, C3, D3)
            
            ln_IDAC_12 = K1_12 + K2_12/T
            ln_IDAC_13 = K1_13 + K2_13/T
            ln_IDAC_21 = K1_21 + K2_21/T
            ln_IDAC_23 = K1_23 + K2_23/T
            ln_IDAC_31 = K1_31 + K2_31/T
            ln_IDAC_32 = K1_32 + K2_32/T
            
            gamma_1, gamma_2, gamma_3 = margules_ternary(ln_IDAC_12, ln_IDAC_13, ln_IDAC_21, ln_IDAC_23, ln_IDAC_31, ln_IDAC_32, x_1[i], x_2[i], x_3[i])
            
            p_1 = x_1[i] * gamma_1 * P_vap_1
            p_2 = x_2[i] * gamma_2 * P_vap_2
            p_3 = x_3[i] * gamma_3 * P_vap_3
            P_calc = p_1 + p_2 + p_3
            
            return np.abs(P_calc - P_true)
        
        results = minimize_scalar(error_in_P, bounds=T_bounds, method='bounded', options={'maxiter':2000})
        if results.success:
            T = results.x
        
        P_vap_1 = KDB_correlation_Pvap(T, A1, B1, C1, D1)
        P_vap_2 = KDB_correlation_Pvap(T, A2, B2, C2, D2)
        P_vap_3 = KDB_correlation_Pvap(T, A3, B3, C3, D3)
        
        ln_IDAC_12 = K1_12 + K2_12/T
        ln_IDAC_13 = K1_13 + K2_13/T
        ln_IDAC_21 = K1_21 + K2_21/T
        ln_IDAC_23 = K1_23 + K2_23/T
        ln_IDAC_31 = K1_31 + K2_31/T
        ln_IDAC_32 = K1_32 + K2_32/T
        
        gamma_1, gamma_2, gamma_3 = margules_ternary(ln_IDAC_12, ln_IDAC_13, ln_IDAC_21, ln_IDAC_23, ln_IDAC_31, ln_IDAC_32, x_1[i], x_2[i], x_3[i])
        
        p_1s[i] = x_1[i]*gamma_1*P_vap_1
        p_2s[i] = x_2[i]*gamma_2*P_vap_2
        p_3s[i] = x_3[i]*gamma_3*P_vap_3
        Ts[i] = T
        
    y_1 = p_1s/P
    y_2 = p_2s/P
    y_3 = p_3s/P

    if specific:
        (x_1, y_1), (x_2, y_2), (x_3, y_3), Ts = (x_1[0], y_1[0]), (x_2[0], y_2[0]), (x_3[0], y_3[0]), Ts[0]

    return (x_1, y_1), (x_2, y_2), (x_3, y_3), Ts


def get_ternary_VLE_isothermal(T, c_1, c_2, c_3, specific=False):
    """
    Calculate the ternary vapor-liquid equilibrium (VLE) for an isothermal system using the Margules model for activity coefficients.

    Parameters:
        T (float): The system temperature (in Kelvin).
        c_1 (dict): The data for component 1, including vapor pressure constants, mole fraction, and GH-GNN constants.
        c_2 (dict): The data for component 2, including vapor pressure constants, mole fraction, and GH-GNN constants.
        c_3 (dict): The data for component 3, including vapor pressure constants, mole fraction, and GH-GNN constants.
        specific (bool, optional): If True, the mole fractions (x_1, x_2, x_3) are taken directly from the components. 
                                    If False, they are computed over a range of values.

    Returns:
        tuple: A tuple containing the following elements:
            - (x_1, y_1): Molar fraction and vapor mole fraction for component 1.
            - (x_2, y_2): Molar fraction and vapor mole fraction for component 2.
            - (x_3, y_3): Molar fraction and vapor mole fraction for component 3.
            - P: The total system pressure (in kPa).
        
        If `specific` is True, the returned values are scalars for a single set of inputs. Otherwise, arrays of 
        values for the entire range of mole fractions are returned.
    """

    if specific:
        x_1 = c_1['x']
        x_2 = c_2['x']
        x_3 = c_3['x']
    else:
        pass

    A1, B1, C1, D1 = c_1['Pvap_constants']
    A2, B2, C2, D2 = c_2['Pvap_constants']
    A3, B3, C3, D3 = c_3['Pvap_constants']

    K1_12, K2_12, K1_13, K2_13 = c_1['GH_constants']
    K1_21, K2_21, K1_23, K2_23 = c_2['GH_constants']
    K1_31, K2_31, K1_32, K2_32 = c_3['GH_constants']

    ln_IDAC_12 = K1_12 + K2_12/T
    ln_IDAC_13 = K1_13 + K2_13/T
    ln_IDAC_21 = K1_21 + K2_21/T
    ln_IDAC_23 = K1_23 + K2_23/T
    ln_IDAC_31 = K1_31 + K2_31/T
    ln_IDAC_32 = K1_32 + K2_32/T

    gamma_1, gamma_2, gamma_3 = margules_ternary(ln_IDAC_12, ln_IDAC_13, ln_IDAC_21, ln_IDAC_23, ln_IDAC_31, ln_IDAC_32, x_1, x_2, x_3)

    P_vap_1 = KDB_correlation_Pvap(T, A1, B1, C1, D1)
    P_vap_2 = KDB_correlation_Pvap(T, A2, B2, C2, D2)
    P_vap_3 = KDB_correlation_Pvap(T, A3, B3, C3, D3)

    p_1 = x_1 * gamma_1 * P_vap_1
    p_2 = x_2 * gamma_2 * P_vap_2
    p_3 = x_3 * gamma_3 * P_vap_3
    P = p_1 + p_2 + p_3

    y_1 = p_1/P
    y_2 = p_2/P
    y_3 = p_3/P

    return (x_1, y_1), (x_2, y_2), (x_3, y_3), P

def get_ternary_VLE_isobaric_unifac(P, c_1, c_2, c_3, T_bounds, specific=False):

    """
    Calculate the ternary vapor-liquid equilibrium (VLE) for an isobaric system using the UNIFAC model for activity coefficients.

    Parameters:
        P (float): The system pressure (in kPa).
        c_1 (dict): The data for component 1, including vapor pressure constants, mole fraction, and other relevant constants.
        c_2 (dict): The data for component 2, including vapor pressure constants, mole fraction, and other relevant constants.
        c_3 (dict): The data for component 3, including vapor pressure constants, mole fraction, and other relevant constants.
        T_bounds (tuple): The bounds for the temperature range (min, max) to search for the equilibrium temperature.
        specific (bool, optional): If True, the mole fractions (x_1, x_2, x_3) are taken directly from the components. 
                                    If False, they are computed over a range of values.

    Returns:
        tuple: A tuple containing the following elements:
            - (x_1, y_1): Molar fraction and vapor mole fraction for component 1.
            - (x_2, y_2): Molar fraction and vapor mole fraction for component 2.
            - (x_3, y_3): Molar fraction and vapor mole fraction for component 3.
            - Ts: The calculated temperatures corresponding to the pressure (in Kelvin).

        If `specific` is True, the returned values are scalars for a single set of inputs. Otherwise, arrays of 
        values for the entire range of mole fractions are returned.
    """

    
    if specific:
        x_1 = np.array([c_1['x']])
        x_2 = np.array([c_2['x']])
        x_3 = np.array([c_3['x']])
    else:
        pass
    A1, B1, C1, D1 = c_1['Pvap_constants']
    A2, B2, C2, D2 = c_2['Pvap_constants']
    A3, B3, C3, D3 = c_3['Pvap_constants']

    inchikey_1 = c_1['inchikey']
    inchikey_2 = c_2['inchikey']
    inchikey_3 = c_3['inchikey']

    sub_1 = unifac_do_fragments[inchikey_1]['subgroup_count']
    sub_2 = unifac_do_fragments[inchikey_2]['subgroup_count']
    sub_3 = unifac_do_fragments[inchikey_3]['subgroup_count']

    # Optimization
    p_1s, p_2s, p_3s, Ts = np.zeros(x_1.shape[0]), np.zeros(x_1.shape[0]), np.zeros(x_1.shape[0]), np.zeros(x_1.shape[0])
    for i in range(x_1.shape[0]):

        def error_in_P(T, P_true=P):
            P_vap_1 = KDB_correlation_Pvap(T, A1, B1, C1, D1)
            P_vap_2 = KDB_correlation_Pvap(T, A2, B2, C2, D2)
            P_vap_3 = KDB_correlation_Pvap(T, A3, B3, C3, D3)
            
            GE = UNIFAC.from_subgroups(chemgroups=[sub_1, sub_2, sub_3], T=T, xs=[x_1[i], x_2[i], x_3[i]], version=1)
            gamma_1, gamma_2, gamma_3 = GE.gammas()
            
            p_1 = x_1[i] * gamma_1 * P_vap_1
            p_2 = x_2[i] * gamma_2 * P_vap_2
            p_3 = x_3[i] * gamma_3 * P_vap_3
            P_calc = p_1 + p_2 + p_3
            
            return np.abs(P_calc - P_true)
        
        results = minimize_scalar(error_in_P, bounds=T_bounds, method='bounded', options={'maxiter':2000})
        if results.success:
            T = results.x
        
        P_vap_1 = KDB_correlation_Pvap(T, A1, B1, C1, D1)
        P_vap_2 = KDB_correlation_Pvap(T, A2, B2, C2, D2)
        P_vap_3 = KDB_correlation_Pvap(T, A3, B3, C3, D3)

        GE = UNIFAC.from_subgroups(chemgroups=[sub_1, sub_2, sub_3], T=T, xs=[x_1[i], x_2[i], x_3[i]], version=1)
        gamma_1, gamma_2, gamma_3 = GE.gammas()
        
        p_1s[i] = x_1[i]*gamma_1*P_vap_1
        p_2s[i] = x_2[i]*gamma_2*P_vap_2
        p_3s[i] = x_3[i]*gamma_3*P_vap_3
        Ts[i] = T
        
    y_1 = p_1s/P
    y_2 = p_2s/P
    y_3 = p_3s/P

    if specific:
        (x_1, y_1), (x_2, y_2), (x_3, y_3), Ts = (x_1[0], y_1[0]), (x_2[0], y_2[0]), (x_3[0], y_3[0]), Ts[0]

    return (x_1, y_1), (x_2, y_2), (x_3, y_3), Ts


def get_ternary_VLE_isothermal_unifac(T, c_1, c_2, c_3, specific=False):
    """
    Calculate the ternary vapor-liquid equilibrium (VLE) for an isothermal system using the UNIFAC model for activity coefficients.

    Parameters:
        T (float): The system temperature (in Kelvin).
        c_1 (dict): The data for component 1, including vapor pressure constants, mole fraction, and other relevant constants.
        c_2 (dict): The data for component 2, including vapor pressure constants, mole fraction, and other relevant constants.
        c_3 (dict): The data for component 3, including vapor pressure constants, mole fraction, and other relevant constants.
        specific (bool, optional): If True, the mole fractions (x_1, x_2, x_3) are taken directly from the components. 
                                    If False, they are computed over a range of values.

    Returns:
        tuple: A tuple containing the following elements:
            - (x_1, y_1): Molar fraction and vapor mole fraction for component 1.
            - (x_2, y_2): Molar fraction and vapor mole fraction for component 2.
            - (x_3, y_3): Molar fraction and vapor mole fraction for component 3.
            - P: The calculated total system pressure.

        If `specific` is True, the returned values are scalars for a single set of inputs. Otherwise, arrays of 
        values for the entire range of mole fractions are returned.
    """

    if specific:
        x_1 = np.array([c_1['x']])
        x_2 = np.array([c_2['x']])
        x_3 = np.array([c_3['x']])
    else:
        pass

    inchikey_1 = c_1['inchikey']
    inchikey_2 = c_2['inchikey']
    inchikey_3 = c_3['inchikey']
    sub_1 = unifac_do_fragments[inchikey_1]['subgroup_count']
    sub_2 = unifac_do_fragments[inchikey_2]['subgroup_count']
    sub_3 = unifac_do_fragments[inchikey_3]['subgroup_count']

    gamma_1, gamma_2, gamma_3 = np.zeros(x_1.shape[0]), np.zeros(x_1.shape[0]), np.zeros(x_1.shape[0])
    for i, (x1, x2, x3) in enumerate(zip(x_1, x_2, x_3)):
        GE = UNIFAC.from_subgroups(chemgroups=[sub_1, sub_2, sub_3], T=T, xs=[x1, x2, x3], version=1)
        g_1, g_2, g_3 = GE.gammas()
        gamma_1[i] = g_1
        gamma_2[i] = g_2
        gamma_3[i] = g_3

    A1, B1, C1, D1 = c_1['Pvap_constants']
    A2, B2, C2, D2 = c_2['Pvap_constants']
    A3, B3, C3, D3 = c_3['Pvap_constants']

    P_vap_1 = KDB_correlation_Pvap(T, A1, B1, C1, D1)
    P_vap_2 = KDB_correlation_Pvap(T, A2, B2, C2, D2)
    P_vap_3 = KDB_correlation_Pvap(T, A3, B3, C3, D3)

    p_1 = x_1 * gamma_1 * P_vap_1
    p_2 = x_2 * gamma_2 * P_vap_2
    p_3 = x_3 * gamma_3 * P_vap_3
    P = p_1 + p_2 + p_3

    y_1 = p_1/P
    y_2 = p_2/P
    y_3 = p_3/P

    if specific:
        (x_1, y_1), (x_2, y_2), (x_3, y_3), P = (x_1[0], y_1[0]), (x_2[0], y_2[0]), (x_3[0], y_3[0]), P[0]

    return (x_1, y_1), (x_2, y_2), (x_3, y_3), P

def plot_ternary_VLE(data, label_top, label_left, label_right):
    """
    Plots a ternary diagram for Vapor-Liquid Equilibrium (VLE) data from multiple methods.

    Parameters:
    -----------
    data : list of tuples
        A list containing the VLE data from different methods. Each element in the list
        should be a tuple containing the name of the method and the corresponding ternary
        coordinates (top, left, right) for each data point. The order of methods should
        match the color and marker assignments in the function.
        
    label_top : str
        Label for the top axis of the ternary plot.

    label_left : str
        Label for the left axis of the ternary plot.

    label_right : str
        Label for the right axis of the ternary plot.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated ternary plot as a `matplotlib` Figure object.
    
    Notes:
    ------
    The ternary diagram is color-coded based on the data method:
        - Experimental data: Color #007675
        - GH-GNN data: Color #78004B
        - UNIFAC data: Color #33A5C3
    Markers used for plotting are circles for experimental data and triangles for model data.
    Dashed, solid, and dotted lines are drawn between the experimental data and the respective
    model data for each method.

    The plot includes a legend, grid lines, and labels for the ternary axes.
    """

    colors = ['#007675', '#78004B', '#33A5C3'] # exp, ghgnn, unifac
    markers = ['o', '^']
    fontsize = 10

    fig = plt.figure(figsize=(6, 5))
    ax = plt.subplot(projection="ternary")
    for i, data_method in enumerate(data):
        color = colors[i]
        for j, (name, top, left, right) in enumerate(data_method):
            marker = markers[j]
            ax.scatter(top, left, right, color=color, marker=marker, label=name, alpha=0.8, edgecolors='k', s=25, linewidths=0.3)
    
    n_points = top.shape[0]
    exp_data, ghgnn_data, unifac_data = data
    y_exp_data, x_exp_data = exp_data
    y_ghgnn_data, y_unifac_data = ghgnn_data[0], unifac_data[0]
    lw = 0.8
    for n in range(n_points):
        tops = [x_exp_data[1][n], y_exp_data[1][n]]
        lefts = [x_exp_data[2][n], y_exp_data[2][n]] 
        rights = [x_exp_data[3][n], y_exp_data[3][n]] 
        ax.plot(tops, lefts, rights, ':', color=colors[0], lw=lw)

        tops = [x_exp_data[1][n], y_ghgnn_data[1][n]]
        lefts = [x_exp_data[2][n], y_ghgnn_data[2][n]] 
        rights = [x_exp_data[3][n], y_ghgnn_data[3][n]] 
        ax.plot(tops, lefts, rights, '-', color=colors[1], lw=lw)

        tops = [x_exp_data[1][n], y_unifac_data[1][n]]
        lefts = [x_exp_data[2][n], y_unifac_data[2][n]] 
        rights = [x_exp_data[3][n], y_unifac_data[3][n]] 
        ax.plot(tops, lefts, rights, '--', color=colors[2], lw=lw)

    ax.set_tlabel(label_top, fontsize=fontsize)
    ax.set_llabel(label_left, fontsize=fontsize)
    ax.set_rlabel(label_right, fontsize=fontsize)
    ax.grid()
    plt.legend(fontsize=fontsize-2, bbox_to_anchor=(0.5, 0), loc='lower center', borderaxespad=-10, ncols=2)
    plt.tight_layout()
    plt.close(fig)
    return fig
    
        
    


