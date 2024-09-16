# MonoKAN
 
MonoKAN is a novel Artificial Neural Network (ANN) architecture designed to enhance interpretability and ensure certified partial monotonicity, building upon the Kolmogorov-Arnold Network (KAN) framework. This repository provides the implementation of MonoKAN, as introduced in the accompanying research paper, along with data loaders, scripts for model training, and experiment notebooks.

## Overview

**MonoKAN** addresses the challenge of making neural networks both interpretable and partially monotonic. While traditional Multi-layer Perceptrons (MLPs) have achieved remarkable success, their lack of transparency, especially in safety-critical and regulated domains, limits their applicability. MonoKAN integrates **cubic Hermite splines** and **positive-weighted linear combinations** to enforce monotonicity constraints and improve model explainability, while maintaining high predictive performance. 

The main contributions of MonoKAN are:
- Guaranteed **certified partial monotonicity**.
- Enhanced **interpretability** through the use of cubic Hermite splines.
- **Improved predictive performance** on several benchmarks, outperforming existing monotonic MLP approaches.

## Repository Structure

├── data/                  # Folder containing CSV data files used in the experiments.

├── loaders/               # Code for loading and preprocessing datasets.

├── Scripts/               # Core implementation of the MonoKAN model.

├── Notebooks/             # Jupyter notebooks showcasing experiments and results.

├── README.md              # This README file.

└── requirements.txt       # Python dependencies.

### Folder Descriptions

- **`data/`**: Contains the CSV datasets used in the experiments. You can replace or add datasets here to evaluate MonoKAN on different tasks.
  
- **`loaders/`**: Contains data loading and preprocessing scripts.
  
- **`Scripts/`**: Core implementation of the MonoKAN architecture, including training and evaluation code.

- **`Notebooks/`**: Jupyter notebooks containing the results of the experiments. These notebooks demonstrate how to load the data, train the MonoKAN model, and evaluate its performance on the provided benchmarks.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/monokan.git
   cd monokan

## Results and Experiments
The Jupyter notebooks in ./Notebooks/ demonstrate how to load the datasets, run the model, and evaluate its performance. Simply run the notebooks in your preferred environment to reproduce the results from the paper or apply MonoKAN to your own data.

## License
This repository is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

This repository is based on the paper "MonoKAN: Certified Monotonic Kolmogorov-Arnold Network". The implementation of the Kolmogorov-Arnold Network (KAN) used in MonoKAN is based on the original [pyKAN](https://github.com/KindXiaoming/pykan) repository. 

If you have any questions or suggestions, feel free to open an issue or submit a pull request.
