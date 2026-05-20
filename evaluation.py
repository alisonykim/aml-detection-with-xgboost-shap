"""
evaluation.py: Model evaluation for AML detection.

1. Metrics: computes AUC-PR and AUC-ROC for baseline and XGBoost
	models across validation and test sets. AUC-PR is the primary
	metric given the extreme class imbalance; AUC-ROC is reported
	for comparability with the literature but is inflated under
	imbalance (Davis & Goadrich, 2006).

2. Threshold: computes the F1-optimal decision threshold from the
	precision-recall curve and evaluates model performance at the
	chosen threshold. The threshold is a policy decision, not a
	technical one: the F1-optimal value treats FPs and FNs as
	equally costly, which may not reflect institutional priorities.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, classification_report, precision_recall_curve, roc_auc_score


def evaluate_models(
	xgb_model: xgb.XGBClassifier,
	X_val: pd.DataFrame,
	y_val: pd.Series,
	X_test: pd.DataFrame,
	y_test: pd.Series,
	baseline_auc_pr: float,
	baseline_auc_roc: float
) -> dict[str, tuple[float, float]]:
	"""
	Evaluate baseline and XGBoost models on validation and test sets.

	Parameters
		xgb_model: Fitted XGBoost classifier
		X_val: Validation features
		y_val: Validation labels
		X_test: Test features
		y_test: Test labels
		baseline_auc_pr: Pre-computed baseline AUC-PR on validation set
		baseline_auc_roc: Pre-computed baseline AUC-ROC on validation set

	Returns: Dict mapping (model and data subset) to (AUC-PR, AUC-ROC)
	"""
	xgb_val_prob = xgb_model.predict_proba(X_val)[:, 1]
	xgb_test_prob = xgb_model.predict_proba(X_test)[:, 1]

	results = {
		'Baseline LR (Val)': (baseline_auc_pr, baseline_auc_roc),
		'XGBoost (Val)': (average_precision_score(y_val, xgb_val_prob), roc_auc_score(y_val, xgb_val_prob)),
		'XGBoost (Test)': (average_precision_score(y_test, xgb_test_prob), roc_auc_score(y_test, xgb_test_prob))
	}

	print(f'{"Model":<22} {"AUC-PR":>8} {"AUC-ROC":>9}')
	print('-' * 42)
	for model, (auc_pr, auc_roc) in results.items():
		print(f'{model:<22} {auc_pr:>8.4f} {auc_roc:>9.4f}')

	return results


def get_probabilities(
	xgb_model: xgb.XGBClassifier,
	X_val: pd.DataFrame,
	X_test: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
	"""
	Compute predicted probabilities for validation and test sets.

	Parameters
		xgb_model: Fitted XGBoost classifier
		X_val: Validation features
		X_test: Test features

	Returns: Val probabilities, test probabilities
	"""
	return (xgb_model.predict_proba(X_val)[:, 1], xgb_model.predict_proba(X_test)[:, 1])


def find_optimal_threshold(
	y_true: pd.Series,
	y_prob: np.ndarray
) -> tuple[float, float, float, float]:
	"""
	Find the F1-optimal decision threshold from the precision-recall curve.

	F1 is computed manually across all thresholds rather than using sklearn's f1_score,
	which operates at a single fixed threshold only.

	Parameters
		y_true: True binary labels
		y_prob: Predicted probabilities for the positive class

	Returns: (best_threshold, precision, recall, f1) at the optimal threshold
	"""
	prec, rec, thresholds = precision_recall_curve(y_true, y_prob)

	f1_scores = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-9)
	best_idx = np.argmax(f1_scores)
	best_threshold = thresholds[best_idx]

	print(f'F1-optimal threshold: {best_threshold:.4f}')
	print(f'  Precision: {prec[best_idx]:.4f}')
	print(f'  Recall: {rec[best_idx]:.4f}')
	print(f'  F1: {f1_scores[best_idx]:.4f}')
	print('\nNote: adjust based on your institution\'s FP vs. FN cost tolerance.')

	return best_threshold, prec[best_idx], rec[best_idx], f1_scores[best_idx]


def evaluate_at_threshold(
	y_true: pd.Series,
	y_prob: np.ndarray,
	threshold: float
) -> np.ndarray:
	"""
	Generate predictions and print evaluation at a given threshold.

	Parameters
		y_true: True binary labels
		y_prob: Predicted probabilities for the positive class
		threshold: Decision threshold

	Returns: Binary predictions at the given threshold
	"""
	y_pred = (y_prob >= threshold).astype(int)

	print(classification_report(y_true, y_pred, target_names=['Legitimate', 'Laundering']))

	return y_pred