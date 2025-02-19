# Graph Neural Networks embedded into Margules model for vapor-liquid equilibria prediction


## Overview

This repository contains the code and data used for the analysis presented in the paper [Graph Neural Networks Embedded into Margules Model for Vapor-Liquid Equilibria Prediction](pending_url). The work combines graph neural networks (GNNs) trained exclusively on infinite dilution data with the extended Margules model to predict the vapor-liquid equilibrium (VLE) of binary and ternary mixtures.

## Repository Structure

```
├── data/                   # Datasets used in the study
├── models/                 # Implementation of GNN-based models and UNIFAC-Dortmund calls along with their predictions
├── src/                    # Source code for data processing and model prediction + utilities
├── requirements.txt        # Required Python dependencies
├── README.md               # This document
├── LICENSE                 # License information
└── run_all.bat             # Batch script to execute all steps of the analysis
```

## Installation

### Prerequisites

Ensure you have Python installed (recommended version: 3.8 or higher). It is advised to create a virtual environment before installing dependencies.

```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

Then, clone this repository and enter the working folder:

```
git clone https://github.com/edgarsmdn/GH_GNN_Margules.git
cd GH_GNN_Margules
```

### Install Dependencies

```
pip install -r requirements.txt
```

or simply do

```
conda create --name venv --file requirements.txt
```

## Usage

The `run_all.bat` script automates all steps of the analysis presented in the paper + additional VLE diagrams for every mixture and additional comparisons among methods at different conditions (isothermal, isobaric, random). This includes:

- Data processing
- Prediction of infinite dilution activity coefficients (IDACs) using GH-GNN
- Prediction of activity coefficients using GH-GNN-Margules and UNIFAC-Dortmund
- Statistical analysis of the dataset
- Predictions and evaluations for binary and ternary VLE systems
- Generating visualizations

To execute the full pipeline, simply run:

```
./run_all.bat
```

## Citation

If you use this repository in your research, please cite our paper:

```
@article{YourCitation2024,
  author    = {Your Name and Co-Authors},
  title     = {Graph Neural Networks Embedded into Margules Model for Vapor-Liquid Equilibria Prediction},
  journal   = {Journal Name},
  year      = {2024},
  volume    = {XX},
  pages     = {XX--XX},
  doi       = {DOI}
}
```

## License

This project is licensed under the [MIT License]() - see the LICENSE file for details.

## Contact

For questions or collaborations, please feel free to reach out.