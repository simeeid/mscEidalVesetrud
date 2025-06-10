import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import quad
from sklearn.preprocessing import StandardScaler
import warnings

import __fix_relative_imports  # noqa: F401

from mscEidalVesetrud.global_constants import CONTAINING_FOLDER

warnings.filterwarnings("ignore")  # Suppress warnings for numerical stability

df_positive = pd.read_csv(
    f"{CONTAINING_FOLDER}/mscEidalVesetrud/complete_concepts/KDE/new_df_1.csv",
    index_col=0,
)
df_negative = pd.read_csv(
    f"{CONTAINING_FOLDER}/mscEidalVesetrud/complete_concepts/KDE/new_df_2.csv",
    index_col=0,
)

print(df_positive.shape)
print(df_negative.shape)


# Function to compute Jensen-Shannon divergence between two PDFs
def js_divergence(pdf1, pdf2, x_range):
    """
    Compute JS divergence between two PDFs over a common x_range.
    pdf1, pdf2: Functions returning PDF values.
    x_range: Tuple (min, max) for integration range.
    """

    def integrand(x):
        p = pdf1(x)
        q = pdf2(x)
        m = 0.5 * (p + q)
        kl1 = p * np.log2(p / m) if p > 0 else 0
        kl2 = q * np.log2(q / m) if q > 0 else 0
        return 0.5 * (kl1 + kl2)

    js, _ = quad(integrand, x_range[0], x_range[1])
    return js


print("Scaler")
# Standardize data to ensure fair PDF comparisons
scaler = StandardScaler()
df_positive_scaled = pd.DataFrame(
    scaler.fit_transform(df_positive), columns=df_positive.columns
)
df_negative_scaled = pd.DataFrame(
    scaler.transform(df_negative), columns=df_negative.columns
)

print("Divergence")
# Initialize list to store divergence scores
divergences = []

# Loop over each feature (column)
for col in df_positive_scaled.columns:
    # Extract data for positive and negative examples
    pos_data = df_positive_scaled[col].values
    neg_data = df_negative_scaled[col].values

    # Define range for PDF estimation (use combined data range)
    data_range = (
        min(pos_data.min(), neg_data.min()),
        max(pos_data.max(), neg_data.max()),
    )
    if data_range[0] == data_range[1]:  # Skip if no variation
        divergences.append((col, 0.0))
        continue

    # Estimate PDFs using KDE
    try:
        kde_pos = gaussian_kde(pos_data)
        kde_neg = gaussian_kde(neg_data)

        # Compute JS divergence
        js = js_divergence(kde_pos, kde_neg, data_range)
        divergences.append((col, js))
    except:
        # Handle cases where KDE fails (e.g., singular data)
        divergences.append((col, 0.0))

print("Divergences")
# Create dataframe of divergence scores
divergence_df = pd.DataFrame(divergences, columns=["Feature", "JS_Divergence"])

# Sort by divergence and select top 100
top_100_features = divergence_df.sort_values(by="JS_Divergence", ascending=False).head(
    100
)

# Print and save results
print(top_100_features)
divergence_df.to_csv("top_features.csv", index=False)
