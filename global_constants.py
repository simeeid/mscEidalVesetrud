import os

CONTAINING_FOLDER = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
MODEL_FOLDER = CONTAINING_FOLDER + "/models"
DATA_FOLDER = CONTAINING_FOLDER + "/data"
WEATHER_DATA_PATH = DATA_FOLDER + "/calc_time_all_data_combined.csv"
PROD_DATA_PATH = DATA_FOLDER + "/prod_avl_downreg.csv"
EARLY_WTGS_AVL_DATA_PATH = DATA_FOLDER + "/roan_available_wtgs.csv"
BASELINE_FEATURES_TRAIN_PATH = DATA_FOLDER + "/all_baseline_features_train_data.csv"
BASELINE_FEATURES_TEST_PATH = DATA_FOLDER + "/all_baseline_features_test_data.csv"


# Wind turbine characteristics
WIND_TURBINE_POWER_MW = 3.6
WIND_TURBINE_COUNT = 71
MAX_POWER_MW = WIND_TURBINE_POWER_MW * WIND_TURBINE_COUNT
MIN_ALLOWABLE_AVAILABILITY = MAX_POWER_MW * 0.5  # 50% of the park


# Data characteristics
START_TIME = "2021-02-08 23:00:00"
TEST_SPLIT_TIME = "2024-04-05 00:00:00"
END_TIME = "2025-04-05 00:00:00"

TRAIN_SIZE = 2 * 12 * 30 * 24  # years*months*days*hours
VAL_SIZE = 1009
# (1008 is 42 days, but that leaves 1 day unused with 8 folds, and 1010 results in 7 folds)
# Only the power of 1009 datapoints in the validation set results in all the data being used

TRAIN_DATA_PATH = f"{DATA_FOLDER}/processed_data"
TEST_DATA_PATH = f"{DATA_FOLDER}/test_data"


# Data shape
GRID_SIZE = 13
TIME_WINDOW = 11


SEEDS = [42, 43, 44, 45, 46]


SHADOW_MAP_30_BY_30_PATH = (
    CONTAINING_FOLDER + "/mscEidalVesetrud/resources/RoanShadowMap30by30km.png"
)
SHADOW_MAP_35_BY_35_PATH = (
    CONTAINING_FOLDER + "/mscEidalVesetrud/resources/RoanShadowMap35by35km.png"
)
WIND_TURBINES_IMAGE = (
    CONTAINING_FOLDER + "/mscEidalVesetrud/resources/wind_turbines_image.png"
)
