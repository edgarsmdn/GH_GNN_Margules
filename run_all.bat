@echo off

REM ./run_all.bat
REM This file runs all files needed for the analysis in the paper

REM Data processing
python src/data/03_include_SMILES_pure_compounds.py
python src/data/04_get_complete_vle_data.py
python src/data/05_get_feasible_systems.py
python src/data/07_process_ternary.py

REM Predictions of IDACs using GH-GNN
python src/models/01_predict_IDACs_GH_GNN.py --version organic_old

REM Prediction of activity coefficients by GH-GNN-Margules and UNIFAC-Dortmund
python src/models/02_predict_gammas_GH_GNN_Margules.py --version organic_old --jaccard_threshold 0.6
python src/models/03_predict_gammas_unifac_do.py --version organic_old

REM Get statistics of the dataset
python src/data/06_get_general_statistics_data.py

REM Predictions and analysis for binary VLEs
python src/models/04_predict_isothermal.py --version organic_old --jaccard_threshold 0.6 --model GH_GNN_Margules
python src/models/05_predict_isobaric.py --version organic_old --jaccard_threshold 0.6 --model GH_GNN_Margules
python src/models/06_predict_all.py --version organic_old --jaccard_threshold 0.6 --model GH_GNN_Margules

python src/models/04_predict_isothermal.py --version organic_old --jaccard_threshold 0.6 --model UNIFAC_Do
python src/models/05_predict_isobaric.py --version organic_old --jaccard_threshold 0.6 --model UNIFAC_Do
python src/models/06_predict_all.py --version organic_old --jaccard_threshold 0.6 --model UNIFAC_Do

REM Predictions and analysis for binary VLEs that pass Fredenslund's test
python src/models/04_predict_isothermal.py --version organic_old --jaccard_threshold 0.6 --model GH_GNN_Margules --consistent
python src/models/04_predict_isothermal.py --version organic_old --jaccard_threshold 0.6 --model UNIFAC_Do --consistent
python src/models/05_predict_isobaric.py --version organic_old --jaccard_threshold 0.6 --model GH_GNN_Margules --consistent
python src/models/05_predict_isobaric.py --version organic_old --jaccard_threshold 0.6 --model UNIFAC_Do --consistent

REM Predictions and analysis for ternary VLEs
python src/models/07_predict_IDACs_for_ternary.py
python src/models/08_predict_ternary_vles.py --type_vle isothermal --model GH_GNN_Margules
python src/models/08_predict_ternary_vles.py --type_vle isobaric --model GH_GNN_Margules

python src/models/08_predict_ternary_vles.py --type_vle isothermal --model UNIFAC_Do
python src/models/08_predict_ternary_vles.py --type_vle isobaric --model UNIFAC_Do

REM Additional visualizations
python src/visualization/comparison_ghgnnMargules_unifacDo.py --version organic_old --type_system isothermal
python src/visualization/comparison_ghgnnMargules_unifacDo.py --version organic_old --type_system isobaric
python src/visualization/comparison_ghgnnMargules_unifacDo.py --version organic_old --type_system all
python src/visualization/comparison_isothermal_isobaric.py --version organic_old
python src/visualization/plot_isobaric_unifac_vs_ghgnn.py --version organic_old --type_system isobaric
python src/visualization/plot_isothermal_vles_not_feasible_unifac.py --version organic_old
python src/visualization/plot_ternary_vles.py --type_vle isothermal --grouped_according_to x_1
python src/visualization/plot_ternary_vles.py --type_vle isobaric --grouped_according_to x_1





