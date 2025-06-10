import numpy as np
from scipy.stats import binom


def one_sample_sign_test_target_greater_than_sample_median(
    sample: np.ndarray, target: float
):
    num_greater = np.sum(sample > target)
    num_less = np.sum(sample < target)
    total = num_greater + num_less

    # Say num_greater = 300 and num_less = 700
    # This is then the probability of flipping heads 300 or less out of 1000 flips.
    p_value = binom.cdf(num_greater, total, 0.5)
    return num_less, num_greater, total, p_value


def two_sample_sign_test_sample1_greater_than_sample2(
    sample1: np.ndarray, sample2: np.ndarray
):
    return one_sample_sign_test_target_greater_than_sample_median(sample2 - sample1, 0)
