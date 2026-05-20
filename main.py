"""
main.py: End-to-end AML detection pipeline.
	1. Load and engineer features
	2. Time-based train/val/test split
	3. Train logistic regression baseline
	4. Train XGBoost model with tuned hyperparameters
	5. Evaluate both models (AUC-PR, AUC-ROC)
	6. Select F1-optimal threshold
	7. Compute and plot SHAP values

Run with:
	python main.py

To run the hyperparameter sweep instead of using hardcoded best params,
set RUN_SWEEP = True below.
"""

import random

import matplotlib.pyplot as plt
import numpy as np

import config
import evaluation
import explainability
import loader
import models
import visualization

## Set to True to run the hyperparameter sweep rather than defaulting to config values
RUN_SWEEP = False

## Reproducibility
random.seed(config.SEED)
np.random.seed(config.SEED)
plt.rcParams.update(config.PLOT_STYLE)


def main() -> None:
	# 1. Load data + feature engineering
	print('\n=== 1. Loading data & perform feature engineering ===')
	df = loader.load_or_build()
	print(f'Dataset ready. Shape: {df.shape}')

	# 2. Time-based split
	print('\n=== 2. Train/val/test split ===')
	n = len(df)
	train_end = int(n * config.TRAIN_RATIO)
	val_end = int(n * (config.TRAIN_RATIO + config.VAL_RATIO))

	train = df.iloc[:train_end].copy()
	val = df.iloc[train_end:val_end].copy()
	test = df.iloc[val_end:].copy()

	for name, split in [('Train', train), ('Validation', val), ('Test', test)]:
		print(
			f'{name:<10} | {len(split):>10,} rows '
			f'| {split["Timestamp"].min().date()} -> {split["Timestamp"].max().date()} '
			f'| Fraud rate: {split[config.TARGET].mean():.4%}'
		)

	X_train, y_train = train[config.FEATURES], train[config.TARGET]
	X_val, y_val = val[config.FEATURES], val[config.TARGET]
	X_test, y_test = test[config.FEATURES], test[config.TARGET]

	# 3. Baseline model
	print('\n=== 3. Baseline model ===')
	baseline_model, baseline_auc_pr, baseline_auc_roc = models.train_baseline(X_train, y_train, X_val, y_val)

	# 4. XGBoost model
	print('\n=== 4. XGBoost model ===')
	if RUN_SWEEP:
		best_depth, best_spw, _ = models.sweep(X_train, y_train, X_val, y_val)
		params = {**config.XGB_PARAMS, 'max_depth': best_depth, 'scale_pos_weight': best_spw}
	else:
		print(f'Using hardcoded best params:\n max_depth={config.XGB_PARAMS["max_depth"]}, scale_pos_weight={config.XGB_PARAMS["scale_pos_weight"]}')
		params = config.XGB_PARAMS

	xgb_model = models.train_xgboost(X_train, y_train, X_val, y_val, params=params)

	# 5. Evaluation
	print('\n=== 5. Evaluation ===')
	xgb_val_prob, xgb_test_prob = evaluation.get_probabilities(xgb_model, X_val, X_test)
	baseline_val_prob = baseline_model.predict_proba(X_val)[:, 1]

	results = evaluation.evaluate_models(xgb_model, X_val, y_val, X_test, y_test, baseline_auc_pr, baseline_auc_roc)

	visualization.plot_pr_roc_curves(baseline_val_prob, xgb_val_prob, xgb_test_prob, y_val, y_test, results, baseline_auc_pr, baseline_auc_roc)

	# 6. Threshold selection
	print('\n=== 6. Threshold decision ===')
	best_threshold, _, _, _ = evaluation.find_optimal_threshold(y_test, xgb_test_prob)
	y_pred = evaluation.evaluate_at_threshold(y_test, xgb_test_prob, best_threshold)

	visualization.plot_threshold_curve(y_test, xgb_test_prob, best_threshold)
	visualization.plot_confusion_matrix(y_test, y_pred, best_threshold)

	# 7. SHAP explainability
	print('\n=== 7. SHAP explainability ===')
	shap_values, shap_sample, explainer = explainability.compute_shap_values(xgb_model, X_test)

	visualization.plot_beeswarm(shap_values, shap_sample)
	visualization.plot_bar(shap_values, shap_sample)
	visualization.plot_waterfall(shap_values, shap_sample, explainer, y_test, xgb_model.predict_proba(shap_sample)[:, 1], best_threshold)
	visualization.plot_dependence(shap_values, shap_sample)

	print('\n=== Pipeline complete ===')


if __name__ == '__main__':
	main()