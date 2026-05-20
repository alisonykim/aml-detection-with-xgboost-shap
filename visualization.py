"""
visualization.py: Plots for the AML pipeline.

1. EDA: Temporal distribution, amount distributions, and class
	distribution plots for exploratory analysis

2. Evaluation: Precision-recall and ROC curves, precision/recall/F1
	vs. threshold, and confusion matrix

3. SHAP: beeswarm, bar, waterfall, and dependence plots; wrappers
	around explainability.py that apply consistent plot styling
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

import config

plt.rcParams.update(config.PLOT_STYLE)


def plot_temporal_distribution(df: pd.DataFrame) -> None:
	"""
	Plot transaction counts by time of day, hour x day heatmap, and day of week as a 1x3 subplot.

	Parameters
		df: EDA DataFrame with 'time_of_day', 'day_of_week', and 'hour' columns
	"""
	day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

	_, axes = plt.subplots(1, 3, figsize=(18, 5))

	# Time of Day
	tod_counts = df['time_of_day'].value_counts().sort_index()
	axes[0].bar(tod_counts.index, tod_counts.values, color=config.NAVY)
	axes[0].set_title('Transactions by Time of Day', fontweight='bold', color=config.NAVY)
	axes[0].set_xlabel('Time of Day Bucket')
	axes[0].set_ylabel('Count')
	axes[0].tick_params(axis='x', rotation=45)

	# Hour x Day heatmap
	pivot = df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
	pivot = pivot.reindex(day_order)
	sns.heatmap(pivot, cmap='YlOrRd', linewidths=0.3, ax=axes[1])
	axes[1].set_title('Transaction Heatmap: Hour x Day of Week', fontweight='bold', color=config.NAVY)
	axes[1].set_xlabel('Hour')
	axes[1].set_ylabel('')

	# Day of Week
	dow_counts = df['day_of_week'].value_counts().reindex(day_order)
	axes[2].bar(dow_counts.index, dow_counts.values, color=config.NAVY)
	axes[2].set_title('Transactions by Day of Week', fontweight='bold', color=config.NAVY)
	axes[2].set_xlabel('Day of Week')
	axes[2].set_ylabel('Count')
	axes[2].tick_params(axis='x', rotation=45)

	plt.suptitle('Temporal Distribution of Transactions', fontweight='bold', y=1.02)
	plt.tight_layout()
	plt.savefig('outputs/eda_temporal_distribution.png')
	plt.close()


def plot_amount_distributions(df: pd.DataFrame) -> None:
	"""
	Plot amount distributions as log10 histogram, by currency, by payment format, and KDE by laundering label

	Parameters
		df: EDA DataFrame with columns 'Amount Paid CHF', 'Payment Currency', 'Payment Format', and 'Is Laundering'
	"""
	_, axes = plt.subplots(2, 2, figsize=(18, 12))

	# Log10 histogram
	np.log10(df['Amount Paid CHF'].clip(lower=1)).plot(kind='hist', bins=80, ax=axes[0, 0], color=config.NAVY)
	axes[0, 0].set_title('Amount Distribution (Log Scale)', fontweight='bold', color=config.NAVY)
	axes[0, 0].set_xlabel('log10(Amount CHF)')
	axes[0, 0].set_ylabel('Count')
	axes[0, 0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'$10^{{{int(x)}}}$'))

	# By currency
	order_currency = df.groupby('Payment Currency')['Amount Paid CHF'].median().sort_values(ascending=False).index
	sns.boxplot(data=df, x='Payment Currency', y='Amount Paid CHF', order=order_currency, ax=axes[0, 1])
	axes[0, 1].set_yscale('log')
	axes[0, 1].set_title('Amount by Currency (Log Scale)', fontweight='bold', color=config.NAVY)
	axes[0, 1].tick_params(axis='x', rotation=45)

	# By payment format
	order_format = df.groupby('Payment Format')['Amount Paid CHF'].median().sort_values(ascending=False).index
	sns.boxplot(data=df, x='Payment Format', y='Amount Paid CHF', order=order_format, ax=axes[1, 0])
	axes[1, 0].set_yscale('log')
	axes[1, 0].set_title('Amount by Payment Format (Log Scale)', fontweight='bold', color=config.NAVY)
	axes[1, 0].tick_params(axis='x', rotation=45)

	# KDE by label
	for label, grp in df.groupby('Is Laundering'):
		np.log10(grp['Amount Paid CHF'].clip(lower=1)).plot(
			kind='kde', ax=axes[1, 1],
			label='Laundering' if label == 1 else 'Legitimate',
			color=config.ACCENT if label == 1 else config.NAVY
		)
	axes[1, 1].set_title('Amount: Laundering vs. Legitimate', fontweight='bold', color=config.NAVY)
	axes[1, 1].set_xlabel('log10(Amount CHF)')
	axes[1, 1].legend()

	plt.suptitle('Amount Paid in CHF: Distribution Analysis', fontweight='bold', y=1.02)
	plt.tight_layout()
	plt.savefig('outputs/eda_amount_distributions.png')
	plt.close()


def plot_class_distribution(
	counts_laundering: pd.Series,
	format_fraud: pd.DataFrame,
	daily: pd.DataFrame
) -> None:
	"""
	Plot class distribution, fraud rate by payment format, and daily transaction volume.

	Parameters
		counts_laundering: Value counts of 'Is Laundering'
		format_fraud: Fraud rate by payment format with columns 'fraud_rate' and 'n'
		daily: Daily aggregates with columns 'total', 'laundering', 'fraud_rate'
	"""
	fig = plt.figure(figsize=(18, 14))
	gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.3)

	ax1 = fig.add_subplot(gs[0, 0])
	ax2 = fig.add_subplot(gs[0, 1])
	ax3a = fig.add_subplot(gs[1, :])
	ax3b = fig.add_subplot(gs[2, :], sharex=ax3a)

	# Class distribution
	bars = ax1.bar(['Legitimate', 'Laundering'], counts_laundering.values, color=[config.NAVY, config.ACCENT], width=0.5)
	ax1.bar_label(bars, fmt='{:,.0f}', padding=4, fontsize=10)
	ax1.set_ylabel('Count Transactions')
	ax1.set_title('Class Distribution', fontweight='bold', color=config.NAVY)
	ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))

	# Fraud rate by payment format
	bars2 = ax2.barh(format_fraud.index, format_fraud['fraud_rate'] * 100, color=config.NAVY)
	ax2.bar_label(bars2, fmt='{:.4f}%', padding=4, fontsize=9)
	ax2.set_xlabel('Fraud Rate (%)')
	ax2.set_title('Fraud Rate by Payment Format', fontweight='bold', color=config.NAVY)

	# Daily volume
	ax3a.fill_between(daily.index, daily['total'], color=config.NAVY, alpha=0.4, label='Total')
	ax3a.set_ylabel('Total Transactions')
	ax3a.set_title('Daily Transaction Volume', fontweight='bold', color=config.NAVY)
	ax3a.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
	plt.setp(ax3a.get_xticklabels(), visible=False)

	ax3a_r = ax3a.twinx()
	ax3a_r.scatter(daily.index, daily['laundering'], color=config.ACCENT, s=20, zorder=5, label='Laundering')
	for date, row in daily[daily['laundering'] > 0].iterrows():
		ax3a_r.annotate(
			f'{int(row["laundering"])}',
			xy=(date, row['laundering']),
			xytext=(0, 6), textcoords='offset points',
			ha='center', fontsize=7, color=config.ACCENT
		)
	ax3a_r.set_ylabel('Fraud Transactions', color=config.ACCENT)
	ax3a_r.tick_params(axis='y', labelcolor=config.ACCENT)
	lines1, labels1 = ax3a.get_legend_handles_labels()
	lines2, labels2 = ax3a_r.get_legend_handles_labels()
	ax3a.legend(lines1+lines2, labels1+labels2, loc='upper right')

	# Daily fraud rate
	ax3b.plot(daily.index, daily['fraud_rate']*100, color=config.ACCENT, linewidth=1)
	ax3b.set_ylabel('Fraud Rate (%)')
	ax3b.set_xlabel('Date')
	ax3b.set_title('Daily Fraud Rate', fontweight='bold', color=config.NAVY)
	ax3b.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.1f}%'))

	plt.show()
	plt.savefig('outputs/eda_class_distribution.png')
	plt.close()


def plot_pr_roc_curves(
	baseline_val_prob: np.ndarray,
	xgb_val_prob: np.ndarray,
	xgb_test_prob: np.ndarray,
	y_val: pd.Series,
	y_test: pd.Series,
	results: dict,
	baseline_auc_pr: float,
	baseline_auc_roc: float
) -> None:
	"""
	Plot precision-recall and ROC curves for all models side by side.

	Parameters
		baseline_val_prob: Baseline predicted probabilities on validation set
		xgb_val_prob: XGBoost predicted probabilities on validation set
		xgb_test_prob: XGBoost predicted probabilities on test set
		y_val: Validation labels
		y_test: Test labels
		results: Results dict from evaluation/metrics.py
		baseline_auc_pr: Baseline AUC-PR on validation set
		baseline_auc_roc: Baseline AUC-ROC on validation set
	"""
	_, axes = plt.subplots(1, 2, figsize=(13, 5))

	# PR curve
	ax = axes[0]
	for proba, y_true, label, color in [
		(baseline_val_prob, y_val, f'Baseline LR (AUC-PR={baseline_auc_pr:.3f})', config.LIGHT_BLUE),
		(xgb_val_prob, y_val, f'XGBoost Val (AUC-PR={results["XGBoost (Val)"][0]:.3f})', config.NAVY),
		(xgb_test_prob, y_test, f'XGBoost Test (AUC-PR={results["XGBoost (Test)"][0]:.3f})', config.ACCENT),
	]:
		prec, rec, _ = precision_recall_curve(y_true, proba)
		ax.plot(rec, prec, label=label, color=color, linewidth=2)
	ax.axhline(y=y_test.mean(), color='grey', linestyle='--', linewidth=1, label='No-Skill Baseline')
	ax.set_xlabel('Recall')
	ax.set_ylabel('Precision')
	ax.set_title('Precision-Recall Curve', fontweight='bold', color=config.NAVY)
	ax.legend(fontsize=8)

	# ROC curve
	ax = axes[1]
	for proba, y_true, label, color in [
		(baseline_val_prob, y_val, f'Baseline LR (AUC-ROC={baseline_auc_roc:.3f})', config.LIGHT_BLUE),
		(xgb_val_prob, y_val, f'XGBoost Val (AUC-ROC={results["XGBoost (Val)"][1]:.3f})', config.NAVY),
		(xgb_test_prob, y_test, f'XGBoost Test (AUC-ROC={results["XGBoost (Test)"][1]:.3f})', config.ACCENT),
	]:
		fpr, tpr, _ = roc_curve(y_true, proba)
		ax.plot(fpr, tpr, label=label, color=color, linewidth=2)
	ax.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=1, label='Random')
	ax.set_xlabel('False Positive Rate')
	ax.set_ylabel('True Positive Rate')
	ax.set_title('ROC Curve', fontweight='bold', color=config.NAVY)
	ax.legend(fontsize=8)

	plt.tight_layout()
	plt.savefig('outputs/eval_pr_roc_curves.png')
	plt.close()


def plot_threshold_curve(
	y_test: pd.Series,
	xgb_test_prob: np.ndarray,
	best_threshold: float
) -> None:
	"""
	Plot precision, recall, and F1 across all decision thresholds.

	Parameters
		y_test: Test labels
		xgb_test_prob: XGBoost predicted probabilities on test set
		best_threshold: F1-optimal threshold to highlight on the plot
	"""
	prec, rec, thresholds = precision_recall_curve(y_test, xgb_test_prob)
	f1_scores = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-9)

	_, ax = plt.subplots(figsize=(10, 5))
	ax.plot(thresholds, prec[:-1], label='Precision', color=config.NAVY, linewidth=2)
	ax.plot(thresholds, rec[:-1], label='Recall', color=config.LIGHT_BLUE, linewidth=2)
	ax.plot(thresholds, f1_scores, label='F1', color=config.ACCENT, linewidth=2, linestyle='--')
	ax.axvline(best_threshold, color='grey', linestyle=':', linewidth=1.5, label=f'Best F1 Threshold = {best_threshold:.3f}')
	ax.set_xlabel('Decision Threshold')
	ax.set_ylabel('Score')
	ax.set_title('Precision / Recall / F1 vs. Threshold (Test Set)', fontweight='bold', color=config.NAVY)
	ax.legend()
	plt.tight_layout()
	plt.savefig('outputs/eval_threshold_curve.png')
	plt.close()


def plot_confusion_matrix(
	y_test: pd.Series,
	y_pred: np.ndarray,
	threshold: float
) -> None:
	"""
	Plot confusion matrix at a given decision threshold.

	Parameters
		y_test: Test labels
		y_pred: Binary predictions at the chosen threshold
		threshold: Decision threshold used to generate predictions
	"""
	cm = confusion_matrix(y_test, y_pred)

	_, ax = plt.subplots(figsize=(5, 4))
	sns.heatmap(
		cm, annot=True, fmt=',d', cmap='Blues',
		xticklabels=['Predicted Legitimate', 'Predicted Launder'],
		yticklabels=['Actual Legitimate', 'Actual Launder'],
		ax=ax
	)
	ax.set_title(f'Confusion Matrix (threshold={threshold:.3f})', fontweight='bold', color=config.NAVY)
	plt.tight_layout()
	plt.savefig('outputs/eval_confusion_matrix.png')
	plt.close()


def plot_beeswarm(
	shap_values: np.ndarray,
	sample: pd.DataFrame,
	feature_names: list[str]=None
) -> None:
	"""
	Plot SHAP per-feature, per-transaction detail.

	Parameters
		shap_values: SHAP values array from explainability.compute_shap_values
		sample: Sampled test features used to compute SHAP values
		feature_names: Feature names for axis labels (defaults to config.FEATURES)
	"""
	feature_names = feature_names or config.FEATURES
	shap.summary_plot(shap_values, sample, feature_names=feature_names, plot_type='dot', show=False)
	plt.savefig('outputs/shap_beeswarm.png')
	plt.close()

def plot_bar(
	shap_values: np.ndarray,
	sample: pd.DataFrame,
	feature_names: list[str]=None
) -> None:
	"""
	Plot global feature importance by mean |SHAP|.

	Parameters
		shap_values: SHAP values array from explainability.compute_shap_values
		sample: Sampled test features used to compute SHAP values
		feature_names: Feature names for axis labels (defaults to config.FEATURES)
	"""
	feature_names = feature_names or config.FEATURES
	shap.summary_plot(shap_values, sample, feature_names=feature_names, plot_type='bar', show=False)
	plt.savefig('outputs/shap_bar.png')
	plt.close()


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
	Finds the first TP within the SHAP sample and plots a waterfall chart showing
	each feature's contribution to the final risk score.

	Parameters
		shap_values: SHAP values array from explainability.compute_shap_values
		sample: Sampled test features used to compute SHAP values
		explainer: SHAP explainer from explainability.compute_shap_values
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

	print(f'Transaction Index: {true_positives.index[0]}')
	print(f'Risk Score: {y_prob[example_pos]:.4f}')
	print(f'True Label: Laundering')

	shap.plots.waterfall(
		shap.Explanation(
			values=shap_values[example_pos],
			base_values=explainer.expected_value,
			data=sample.iloc[example_pos].values,
			feature_names=feature_names,
		),
		max_display=max_display, show=False
	)
	plt.savefig('outputs/shap_waterfall.png')
	plt.close()


def plot_dependence(
	shap_values: np.ndarray,
	sample: pd.DataFrame,
	feature_names: list[str]=None
) -> None:
	"""
	Automatically selects the feature with the highest mean |SHAP| and plots its
	relationship with its SHAP values, colored by the feature SHAP selects as the
	strongest interactor.

	Parameters
		shap_values: SHAP values array from explainability.compute_shap_values
		sample: Sampled test features used to compute SHAP values
		feature_names: Feature names (defaults to config.FEATURES)
	"""
	feature_names = feature_names or config.FEATURES
	top_feature = feature_names[np.abs(shap_values).mean(axis=0).argmax()]

	print(f'Top feature by mean |SHAP|: {top_feature}')
	shap.dependence_plot(top_feature, shap_values, sample, feature_names=feature_names, show=False)
	plt.savefig('outputs/shap_dependence.png')
	plt.close()