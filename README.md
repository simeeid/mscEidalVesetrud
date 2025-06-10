# Master of Science Eidal and Vesetrud (2025)

# Beyond Feature Engineering: </br>An Investigation into a Wind Power Prediction Neural Network

This repository contains a selection of the Python code developed as part of a Master's thesis in Computer Science, submitted in June 2025 at the Norwegian University of Science and Technology (NTNU). The work was conducted in collaboration with Aneo AS, focusing on critical advancements in wind power forecasting.

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
