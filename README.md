### Master of Science Eidal and Vesetrud (2025)

<p align="center"><h1 align="center">
Beyond Feature Engineering: </br>An Investigation into a Wind Power Prediction Neural Network
</h1></p>

<p align="center">
This repository contains a selection of the Python code developed as part of a Master's thesis in Computer Science, submitted in June 2025 at the Norwegian University of Science and Technology (NTNU). The work was conducted in collaboration with Aneo AS. Due to the presence of sensitive information and adherence to Non-Disclosure Agreements (NDAs), data, model weights, and specific portions of the code are not included in this public repository.
</p>

<p align="center">
	<img src="https://img.shields.io/github/license/simeeid/mscEidalVesetrud?style=default&logo=opensourceinitiative&logoColor=white&color=7ba1d3" alt="license">
	<img src="https://img.shields.io/github/last-commit/simeeid/mscEidalVesetrud?style=default&logo=git&logoColor=white&color=7ba1d3" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/simeeid/mscEidalVesetrud?style=default&color=7ba1d3" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/simeeid/mscEidalVesetrud?style=default&color=7ba1d3" alt="repo-language-count">
</p>
<p align="center"><!-- default option, no dependency badges. -->
</p>
<p align="center">
	<!-- default option, no dependency badges. -->
</p>
<br>

##  Table of Contents

- [ Thesis Objectives](#thesis-objectives)
- [ Repository Content](#repository-content-methods-for-interpretable-deep-learning)
- [ Project Structure](#project-structure)
  - [ Project Index](#project-index)
- [ Setup Information](#setup-information)
  - [ Prerequisites](#prerequisites)
  - [ Installation](#installation)
- [ License](#license)


## Thesis Objectives

The central aim of this thesis is to investigate the internal representations of a neural network model used for wind power prediction. Specifically, it identifies how these representations align with, and reveal novel insights beyond, expert-engineered features utilized in a traditional gradient boosting machine baseline.

## Repository Content: Methods for Interpretable Deep Learning

This repository is structured into three main folders, each containing the code for a specific method used to interpret the neural network model's internal representations:

### `TCAV` (Testing with Concept Activation Vectors)
This folder contains the implementation of the Testing with Concept Activation Vectors (TCAV) method. This method, originally introduced by Kim et al. (2018) for classification tasks, is adapted here for regression to interpret the Convolutional Neural Network (CNN) by quantifying the linear accessibility ($R^2$ score) of expert-engineered baseline features within the CNN. This method was crucial for comparing the neural network's learned representations with predefined meteorological features.
* **Key Findings:** Analysis using TCAV demonstrated that features related to average wind speed consistently showed high linear accessibility ($R^2 \approx 1$) across all layers of the network, while $R^2$ generally decreased in deeper layers for most other features. The research also highlighted inconsistencies in TCAV's sensitivity measure.
* **Original Paper:** [Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)](https://proceedings.mlr.press/v80/kim18d.html) (Kim et al., 2018)

### `Complete Concepts` (Completeness-aware Concept-Based Explanations)
This folder presents the code for the adapted Completeness-aware Concept-Based Explanations (Complete Concepts) method. Originally proposed by Yeh et al. (2020), this technique was extended for regression tasks in this thesis to discover novel, humanly interpretable concepts within the neural network's internal representations. This allowed for the exploration of new insights beyond traditional feature engineering.
* **Key Findings:** The Complete Concepts model achieved a high completeness score of 0.978 (MAE ratio). Correlation analysis of concept nodes identified both known patterns (strong correlations) and potentially novel ones (weak-to-moderate correlations). Visual and perturbation analyses revealed influential concept nodes often related to specific wind conditions, such as high wind speed in the northeast quadrant or high wind variability.
* **Original Paper:** [On Completeness-aware Concept-Based Explanations in Deep Neural Networks](https://proceedings.neurips.cc/paper/2020/hash/ecb287ff763c169694f682af52c1f309-Abstract.html) (Yeh et al., 220)

### `Activation Maximization`
This folder includes the implementation of Activation Maximization. This method is used to reveal the optimal learned conditions by generating synthetic input data that maximally activate specific neurons or layers within the neural network. It contributed to uncovering novel insights from the trained model.
* **Key Findings:** Activation Maximization revealed plausible optimal input conditions for wind power prediction, such as consistent southwesterly winds at 120 meters and optimal seasonal conditions in late October. The method also sometimes produced physically implausible patterns, highlighting challenges in interpretation.

## Baseline Model

This repository also includes a folder detailing the **LightGBM baseline model**. This model employs 212 expert-engineered features derived from Numerical Weather Prediction (NWP) data and hourly wind power production.

* **Inspired By:** [Improving Renewable Energy Forecasting with a Grid of Numerical Weather Predictions](https://ieeexplore.ieee.org/abstract/document/7903735?casa_token=IdHk_TZIBzcAAAAA:fJL9mPpLHDj1VTPbVVWV1eQ-GdsblyNT02sKLEAHvBy_bfaQHwPI7-6OmzHix7oRS0gK0-Wru7E) (Andrade and Bessa, 2017)

---

##  Project Structure

```sh
└── mscEidalVesetrud/
    ├── LICENSE
    ├── README.md
    ├── TCAV
    │   ├── __fix_relative_imports.py
    │   ├── cav_calculation.py
    │   ├── compute_r2_on_true_model.py
    │   ├── lag_lead_time_point_diff.py
    │   ├── r2_histogram.py
    │   ├── random_r2_violinplots.py
    │   ├── sign_test.py
    │   ├── test_model.py
    │   └── vizualize_coefficient_of_determination.py
    ├── activation_maximization
    │   ├── __fix_relative_imports.py
    │   └── activation_maximization.py
    ├── baseline
    │   ├── __fix_relative_imports.py
    │   ├── baseline_test_set_run.ipynb
    │   ├── cross_validation.py
    │   ├── data_prep
    │   └── lightgbm_model.py
    ├── complete_concepts
    │   ├── KDE
    │   ├── __fix_relative_imports.py
    │   ├── complete_concepts.py
    │   └── wind_visualizations
    ├── data_preprocessing
    │   ├── __fix_relative_imports.py
    │   ├── cross_validation.py
    │   ├── data_reading.py
    │   ├── prepare_load_dataset.py
    │   ├── prod_preprocessing.py
    │   └── weather_preprocessing.py
    ├── deep_learning
    │   ├── __fix_relative_imports.py
    │   ├── neural_nets.py
    │   ├── sweep.py
    │   ├── train_model.py
    │   └── visualize.py
    ├── global_constants.py
    ├── requirements.txt
    └── resources
        ├── RoanShadowMap30by30km.png
        ├── RoanShadowMap35by35km.png
        └── wind_turbines_image.png
```


###  Project Index
<details open>
	<summary><b><code>MSCEIDALVESETRUD/</code></b></summary>
	<details> <!-- __root__ Submodule -->
		<summary><b>__root__</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/simeeid/mscEidalVesetrud/blob/master/requirements.txt'>requirements.txt</a></b></td>
				<td><code>❯ Lists the Python dependencies required for this project.</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/simeeid/mscEidalVesetrud/blob/master/global_constants.py'>global_constants.py</a></b></td>
				<td><code>❯ Defines global constants and configurations used throughout the project.</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- deep_learning Submodule -->
		<summary><b>deep_learning</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/simeeid/mscEidalVesetrud/blob/master/deep_learning/visualize.py'>visualize.py</a></b></td>
				<td><code>❯ Contains functions for visualizing wind data.</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/simeeid/mscEidalVesetrud/blob/master/deep_learning/train_model.py'>train_model.py</a></b></td>
				<td><code>❯ Script for training the neural network model.</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/simeeid/mscEidalVesetrud/blob/master/deep_learning/neural_nets.py'>neural_nets.py</a></b></td>
				<td><code>❯ Defines the neural network architectures used in the project.</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/simeeid/mscEidalVesetrud/blob/master/deep_learning/sweep.py'>sweep.py</a></b></td>
				<td><code>❯ Implements hyperparameter sweeping and optimization routines.</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- TCAV Submodule -->
		<summary><b>TCAV</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/simeeid/mscEidalVesetrud/blob/master/TCAV/cav_calculation.py'>cav_calculation.py</a></b></td>
				<td><code>❯ Core logic for calculating Concept Activation Vectors (CAVs).</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/simeeid/mscEidalVesetrud/blob/master/TCAV/sign_test.py'>sign_test.py</a></b></td>
				<td><code>❯ Performs statistical sign tests for result validation.</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/simeeid/mscEidalVesetrud/blob/master/TCAV/compute_r2_on_true_model.py'>compute_r2_on_true_model.py</a></b></td>
				<td><code>❯ Computes R-squared scores of baseline features and logs results.</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/simeeid/mscEidalVesetrud/blob/master/TCAV/test_model.py'>test_model.py</a></b></td>
				<td><code>❯ Contains testing model architecture, ShapeRecognizer3000 predicts the pyramid function in Section 6.1.</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/simeeid/mscEidalVesetrud/blob/master/TCAV/random_r2_violinplots.py'>random_r2_violinplots.py</a></b></td>
				<td><code>❯ Generates violin plots to visualize random R-squared distributions.</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/simeeid/mscEidalVesetrud/blob/master/TCAV/r2_histogram.py'>r2_histogram.py</a></b></td>
				<td><code>❯ Creates distribution plots of R-squared values for basline features. </code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/simeeid/mscEidalVesetrud/blob/master/TCAV/vizualize_coefficient_of_determination.py'>vizualize_coefficient_of_determination.py</a></b></td>
				<td><code>❯ Creates 3D figures of R-squared across model layers and training. </code></td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- activation_maximization Submodule -->
		<summary><b>activation_maximization</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/simeeid/mscEidalVesetrud/blob/master/activation_maximization/activation_maximization.py'>activation_maximization.py</a></b></td>
				<td><code>❯ Implements the activation maximization algorithm.</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- baseline Submodule -->
		<summary><b>baseline</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/simeeid/mscEidalVesetrud/blob/master/baseline/baseline_test_set_run.ipynb'>baseline_test_set_run.ipynb</a></b></td>
				<td><code>❯ Jupyter notebook for running the baseline model on the test set.</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/simeeid/mscEidalVesetrud/blob/master/baseline/lightgbm_model.py'>lightgbm_model.py</a></b></td>
				<td><code>❯ Defines and trains the LightGBM baseline model.</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/simeeid/mscEidalVesetrud/blob/master/baseline/cross_validation.py'>cross_validation.py</a></b></td>
				<td><code>❯ Contains functions for cross-validation within the baseline model.</code></td>
			</tr>
			</table>
			<details>
				<summary><b>data_prep</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/simeeid/mscEidalVesetrud/blob/master/baseline/data_prep/paper_features.py'>paper_features.py</a></b></td>
						<td><code>❯ Generates features as described in the original paper.</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/simeeid/mscEidalVesetrud/blob/master/baseline/data_prep/paper_features_notebook.ipynb'>paper_features_notebook.ipynb</a></b></td>
						<td><code>❯ Jupyter notebook for exploring and generating paper features.</code></td>
					</tr>
					</table>
				</blockquote>
			</details>
		</blockquote>
	</details>
	<details> <!-- complete_concepts Submodule -->
		<summary><b>complete_concepts</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/simeeid/mscEidalVesetrud/blob/master/complete_concepts/complete_concepts.py'>complete_concepts.py</a></b></td>
				<td><code>❯ Implements the Completeness-aware Concept-Based Explanations method for regression.</code></td>
			</tr>
			</table>
			<details>
				<summary><b>wind_visualizations</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/simeeid/mscEidalVesetrud/blob/master/complete_concepts/wind_visualizations/node_6(#1).ipynb'>node_6(#1).ipynb</a></b></td>
						<td><code>❯ Jupyter notebook example for visualizing concept node 6 activations.</code></td>
					</tr>
					</table>
				</blockquote>
			</details>
			<details>
				<summary><b>KDE</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/simeeid/mscEidalVesetrud/blob/master/complete_concepts/KDE/compute_JS_divergence.py'>compute_JS_divergence.py</a></b></td>
						<td><code>❯ Computes Jensen-Shannon divergence for comparing distributions.</code></td>
					</tr>
					</table>
				</blockquote>
			</details>
		</blockquote>
	</details>
	<details> <!-- data_preprocessing Submodule -->
		<summary><b>data_preprocessing</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/simeeid/mscEidalVesetrud/blob/master/data_preprocessing/prepare_load_dataset.py'>prepare_load_dataset.py</a></b></td>
				<td><code>❯ Handles the preparation and loading of the wind and power production datasets.</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/simeeid/mscEidalVesetrud/blob/master/data_preprocessing/prod_preprocessing.py'>prod_preprocessing.py</a></b></td>
				<td><code>❯ Scripts for preprocessing power production data.</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/simeeid/mscEidalVesetrud/blob/master/data_preprocessing/weather_preprocessing.py'>weather_preprocessing.py</a></b></td>
				<td><code>❯ Scripts for preprocessing numerical weather prediction data.</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/simeeid/mscEidalVesetrud/blob/master/data_preprocessing/data_reading.py'>data_reading.py</a></b></td>
				<td><code>❯ Utility script for reading data files.</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/simeeid/mscEidalVesetrud/blob/master/data_preprocessing/cross_validation.py'>cross_validation.py</a></b></td>
				<td><code>❯ Implements the sliding window cross-validation strategy for the sweeps.</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
</details>



---

## Setup Information

Since this repository contains a selection of code without all the necessary data or trained models due to sensitive information and NDAs, the project isn't directly runnable. However, the core functionality of **TCAV**, **Complete Concepts**, and **Activation Maximization** is present for examination.

To explore the code, you'll need to set up your environment:

### Prerequisites

Ensure your runtime environment meets these requirements:

-   **Programming Language:** Python
-   **Package Manager:** Pip

### Installation

To install the project dependencies:

1.  Clone the `mscEidalVesetrud` repository:
    ```sh
    ❯ git clone https://github.com/simeeid/mscEidalVesetrud
    ```

2.  Navigate to the project directory:
    ```sh
    ❯ cd mscEidalVesetrud
    ```

3.  Install the dependencies using `pip`:
    ```sh
    ❯ pip install -r requirements.txt
    ```


---

##  License

This project is protected under the [GNU Affero General Public License v3.0](https://choosealicense.com/licenses/agpl-3.0/) License. For more details, refer to the [LICENSE](LICENSE) file.



