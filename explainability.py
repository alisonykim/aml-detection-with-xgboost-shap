"""
explainability.py: SHAP-based explainability for the XGBoost AML model.

Computes and visualizes SHAP values on a sample of the test set.
SHAP values support auditability of individual risk decisions: a practical
measure toward the transparency requirements of the revDSG, without claiming
full legal compliance.
"""

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

import config


def compute_shap_values(
	model: xgb.XGBClassifier,
	X_test: pd.DataFrame,
	sample_size: int=None
) -> tuple[np.ndarray, pd.DataFrame, shap.TreeExplainer]:
	"""
	Compute SHAP values on a random sample of the test set.

	Parameters
		model: Fitted XGBoost classifier
		X_test: Test features
		sample_size: Number of rows to sample (defaults to config.SHAP_SAMPLE_SIZE)

	Returns: SHAP values array, sampled DataFrame, and the explainer object
	"""
	sample_size = sample_size or config.SHAP_SAMPLE_SIZE
	sample = X_test.sample(n=sample_size, random_state=config.SEED)
	explainer = shap.TreeExplainer(model)
	shap_values = explainer.shap_values(sample)

	print(f'SHAP values computed on {sample_size:,}-row sample of test set.')

	return shap_values, sample, explainer


def plot_beeswarm(
	shap_values: np.ndarray,
	sample: pd.DataFrame,
	feature_names: list[str]=None
) -> None:
	"""
	Plot SHAP beeswarm (summary dot) plot per-feature, per-transaction detail.

	Parameters
		shap_values: SHAP values array from compute_shap_values
		sample: Sampled test features used to compute SHAP values
		feature_names: Feature names for axis labels (defaults to config.FEATURES)
	"""
	feature_names = feature_names or config.FEATURES
	shap.summary_plot(shap_values, sample, feature_names=feature_names, plot_type='dot')


def plot_bar(
	shap_values: np.ndarray,
	sample: pd.DataFrame,
	feature_names: list[str]=None
) -> None:
	"""
	Plot SHAP bar chart: global feature importance by mean |SHAP|.

	Parameters
		shap_values: SHAP values array from compute_shap_values
		sample: Sampled test features used to compute SHAP values
		feature_names: Feature names for axis labels (defaults to config.FEATURES)
	"""
	feature_names = feature_names or config.FEATURES
	shap.summary_plot(shap_values, sample, feature_names=feature_names, plot_type='bar')


def plot_waterfall(
	shap_values: np.ndarray,
	sample: pd.DataFrame,
	explainer: shap.TreeExplainer,
	y_true: pd.Series,
	y_prob: np.ndarray,
	threshold: float,
	feature_names: list[str]=None,
	max_display: int=None
) -> None:
	"""
	Plot SHAP waterfall for a single true positive transaction.

	Finds the first true positive within the SHAP sample and plots a waterfall chart
	showing each feature's contribution to the final risk score.

	Parameters
		shap_values: SHAP values array from compute_shap_values
		sample: Sampled test features used to compute SHAP values
		explainer: SHAP explainer from compute_shap_values
		y_true: True labels for the full test set
		y_prob: Predicted probabilities for the SHAP sample
		threshold: Decision threshold for classifying as laundering
		feature_names: Feature names (defaults to config.FEATURES)
		max_display: Max features to display (defaults to config.SHAP_MAX_DISPLAY)
	"""
	feature_names = feature_names or config.FEATURES
	max_display = max_display or config.SHAP_MAX_DISPLAY

	shap_true_labels = y_true.loc[sample.index]
	true_positives = sample[(shap_true_labels == 1) & (y_prob >= threshold)]

	if len(true_positives) == 0:
		print('No true positives found in SHAP sample. Try increasing sample size or lowering threshold.')
		return

	example_pos = sample.index.get_loc(true_positives.index[0])

	print(f'Transaction index: {true_positives.index[0]}')
	print(f'Risk score: {y_prob[example_pos]:.4f}')
	print(f'True label: Laundering')

	shap.plots.waterfall(
		shap.Explanation(
			values=shap_values[example_pos],
			base_values=explainer.expected_value,
			data=sample.iloc[example_pos].values,
			feature_names=feature_names
		),
		max_display=max_display
	)


def plot_dependence(
	shap_values: np.ndarray,
	sample: pd.DataFrame,
	feature_names: list[str]=None
) -> None:
	"""
	Plot SHAP dependence plot for the most important feature.

	Automatically selects the feature with the highest mean |SHAP| and plots its
	relationship with its SHAP values, colored by the feature SHAP selects as the
	strongest interactor.

	Parameters
		shap_values: SHAP values array from compute_shap_values
		sample: Sampled test features used to compute SHAP values
		feature_names: Feature names (defaults to config.FEATURES)
	"""
	feature_names = feature_names or config.FEATURES
	top_feature = feature_names[np.abs(shap_values).mean(axis=0).argmax()]

	print(f'Top feature by mean |SHAP|: {top_feature}')
	shap.dependence_plot(top_feature, shap_values, sample, feature_names=feature_names)