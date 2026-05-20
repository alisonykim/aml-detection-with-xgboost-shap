"""
config.py: Central configuration for the AML prioritization pipeline.
"""

from pathlib import Path


## Paths
ROOT_DIR = Path(__file__).parent
OUTPUT_DIR = ROOT_DIR / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

## Reproducibility
SEED = 42

## Dataset
DATASET_ID = 'eexzzm/IBM-Transactions-for-Anti-Money-Laundering-HI-Small-Trans'
TARGET = 'Is Laundering'

## Split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

## Features
FEATURES = [
	'log_amount_paid', 'log_amount_received', 'amount_ratio', 'amount_diff',
	'hour', 'day_of_week', 'is_weekend', 'time_of_day_enc',
	'same_bank', 'from_bank_enc', 'to_bank_enc',
	'payment_format_enc', 'pay_currency_enc', 'rec_currency_enc', 'cross_currency',
	'acct_txn_count_7d', 'acct_amount_sum_7d', 'acct_amount_mean_7d',
	'acct_amount_med_7d', 'acct_amount_max_7d', 'acct_amount_std_7d',
	'acct_txn_count_30d', 'acct_amount_sum_30d', 'acct_amount_mean_30d',
	'acct_amount_med_30d', 'acct_amount_max_30d', 'acct_amount_std_30d'
]

## CHF exchange rates
RATES_TO_CHF = {
	'US Dollar': 0.9729,
	'Euro': 0.9642,
	'UK Pound': 1.1035,
	'Australian Dollar': 0.6504,
	'Canadian Dollar': 0.7309,
	'Yuan': 0.138548,
	'Yen': 0.006808,
	'Rupee': 0.01218,
	'Mexican Peso': 0.048474,
	'Ruble': 0.0162,
	'Brazil Real': 0.186205,
	'Saudi Riyal': 0.25944,
	'Shekel': 0.2742,
	'Swiss Franc': 1.0,
	'Bitcoin': 20246.0
}

## XGBoost hyperparameters
XGB_PARAMS = {
	'n_estimators': 500,
	'learning_rate': 0.05,
	'max_depth': 7,
	'subsample': 0.8,
	'colsample_bytree': 0.8,
	'scale_pos_weight': 1,
	'eval_metric': 'aucpr',
	'early_stopping_rounds': 30,
	'random_state': SEED,
	'verbosity': 1
}

## Hyperparameter sweep
SWEEP_DEPTH_VALUES = [5, 6, 7]
SWEEP_SPW_VALUES = [1, 2, 5, 10, 25, 50, 100, 300, 500, 979, 1500]

## Evaluation
BEST_THRESHOLD = 0.0604 # F1-optimal on test set

## SHAP
SHAP_SAMPLE_SIZE = 2000
SHAP_MAX_DISPLAY = 15

## Visualization
NAVY = '#1B4F72'
LIGHT_BLUE = '#A9CCE3'
ACCENT = '#E74C3C'

PLOT_STYLE = {
	'figure.dpi': 120,
	'axes.spines.top': False,
	'axes.spines.right': False,
	'font.family': 'sans-serif'
}