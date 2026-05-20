"""
models.py: Baseline and XGBoost models for AML detection.

1. Baseline (logistic regression): A simple, interpretable benchmark.
	Uses class_weight='balanced' to correct for class imbalance, which
	is the logistic regression equivalent of XGBoost's scale_pos_weight.

2. Tuning: Joint hyperparameter sweep over max_depth and scale_pos_weight,
	selecting on validation AUC-PR only.

3. XGBoost: Final model training using hyperparameters from the sweep,
	with early stopping monitored on validation AUC-PR.
"""

import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import config


def build_baseline() -> Pipeline:
	"""
	Build a logistic regression baseline pipeline with standard scaling.

	Uses class_weight='balanced' to correct for class imbalance by weighting each class inversely
	proportional to its frequency, ensuring the minority class (laundering) contributes meaningfully
	to the loss function.

	Returns: Unfitted pipeline with StandardScaler and LogisticRegression
	"""
	return Pipeline([
		('scaler', StandardScaler()),
		('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=config.SEED))
	])


def train_baseline(
	X_train: pd.DataFrame,
	y_train: pd.Series,
	X_val: pd.DataFrame,
	y_val: pd.Series
) -> tuple[Pipeline, float, float]:
	"""
	Train the baseline model and evaluate on the validation set.

	Parameters
		X_train: Training features
		y_train: Training labels
		X_val: Validation features
		y_val: Validation labels

	Returns: Fitted pipeline, validation AUC-PR, validation AUC-ROC
	"""
	model = build_baseline()
	model.fit(X_train, y_train)

	val_prob = model.predict_proba(X_val)[:, 1]
	auc_pr = average_precision_score(y_val, val_prob)
	auc_roc = roc_auc_score(y_val, val_prob)

	print(f'Baseline (Logistic Regression): Validation Set')
	print(f'  AUC-PR: {auc_pr:.4f} (no-skill baseline ≈ {y_val.mean():.4f})')
	print(f'  AUC-ROC: {auc_roc:.4f} (random baseline = 0.5000)')

	return model, auc_pr, auc_roc


def sweep(
	X_train: pd.DataFrame,
	y_train: pd.Series,
	X_val: pd.DataFrame,
	y_val: pd.Series,
	depth_values: list[int]=None,
	spw_values: list[int]=None
) -> tuple[int, int, float]:
	"""
	Sweep max_depth and scale_pos_weight jointly on the validation set.

	Parameters
		X_train: Training features
		y_train: Training labels
		X_val: Validation features
		y_val: Validation labels
		depth_values: Candidate max_depth values (defaults to config.SWEEP_DEPTH_VALUES)
		spw_values: Candidate scale_pos_weight values (defaults to config.SWEEP_SPW_VALUES)

	Returns: Best (max_depth, scale_pos_weight, val_auc_pr)

	Note: Always select hyperparameters based on validation performance only. Using test
	performance for selection constitutes leakage and produces optimistic estimates of
	real-world performance.
	"""
	depth_values = depth_values or config.SWEEP_DEPTH_VALUES
	spw_values = spw_values or config.SWEEP_SPW_VALUES

	results = {}
	for depth in tqdm(depth_values, desc='max_depth', unit='depth'):
		for spw in tqdm(spw_values, desc=f'scale_pos_weight (depth={depth})', unit='spw', leave=False):
			m = xgb.XGBClassifier(
				n_estimators=500,
				learning_rate=0.05,
				max_depth=depth,
				subsample=0.8,
				colsample_bytree=0.8,
				scale_pos_weight=spw,
				eval_metric='aucpr',
				early_stopping_rounds=30,
				random_state=config.SEED,
				verbosity=0
			)
			m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
			proba = m.predict_proba(X_val)[:, 1]
			results[(depth, spw)] = average_precision_score(y_val, proba)
			tqdm.write(f'max_depth={depth}, scale_pos_weight={spw}: Val AUC-PR = {results[(depth, spw)]:.4f}')

	best_params = max(results, key=results.get)
	best_depth, best_spw = best_params
	best_auc_pr = results[best_params]

	print(f'\nBest max_depth: {best_depth}')
	print(f'Best scale_pos_weight: {best_spw}')
	print(f'Val AUC-PR: {best_auc_pr:.4f}')

	return best_depth, best_spw, best_auc_pr


def train_xgboost(
	X_train: pd.DataFrame,
	y_train: pd.Series,
	X_val: pd.DataFrame,
	y_val: pd.Series,
	params: dict=None
) -> xgb.XGBClassifier:
	"""
	Train the final XGBoost model with early stopping on validation AUC-PR.

	Parameters
		X_train: Training features
		y_train: Training labels
		X_val: Validation features
		y_val: Validation labels
		params: XGBoost hyperparameters (defaults to config.XBG_PARAMS)

	Returns: Fitted XGBoost classifier

	Note: scale_pos_weight=1 is used by default, which is empirically optimal on this
	dataset despite the ~980:1 class imbalance. See Jupyter notebook aml_xgboost_shap.ipynb
	for the sweep results that motivated this choice.
	"""
	params = params or config.XGB_PARAMS
	model = xgb.XGBClassifier(**params)

	model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

	print(f'\nmax_depth: {model.get_params()["max_depth"]}')
	print(f'scale_pos_weight: {model.get_params()["scale_pos_weight"]}')
	print(f'Best iteration: {model.best_iteration}')

	return model