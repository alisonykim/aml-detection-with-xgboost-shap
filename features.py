"""
engineering.py: Feature engineering for AML detection.

It is performed in two steps, both on the full dataset before the train/val/test split.

Features are motivated by the three stages of money laundering (placement, layering, integration),
with most signals targeting the layering stage where suspicious patterns are most detectable.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Compute account-level behavioral features using only past transactions.

	For each transaction at time T, only transactions strictly before T are used so that no row sees
	its own data. Features are computed using time-based rolling windows (7D and 30D) on the sending
	account's transaction history.

	Parameters
		df: Raw or minimally processed transaction DataFrame. Must contain 'From Account', 'Timestamp',
			and 'Amount Paid' columns.

	Returns: Original DataFrame with 12 additional rolling feature columns

	Note: Must be called on the full dataset before the train/val/test split.
	"""
	df = df.copy()
	df['Timestamp'] = pd.to_datetime(df['Timestamp'])
	df = df.sort_values('Timestamp').reset_index(drop=True)
	df['_txn_idx'] = df.index

	base = df[['_txn_idx', 'From Account', 'Timestamp', 'Amount Paid']].copy()
	base = base.set_index('Timestamp')

	results = []
	for _, group in tqdm(list(base.groupby('From Account')), desc='Computing rolling features', unit=' accounts'):
		group = group.sort_index()
		r7 = group['Amount Paid'].rolling('7D')
		r30 = group['Amount Paid'].rolling('30D')

		result = pd.DataFrame({
			'_txn_idx': group['_txn_idx'].values,
			'acct_txn_count_7d': r7.count().values,
			'acct_amount_sum_7d': r7.sum().values,
			'acct_amount_mean_7d': r7.mean().values,
			'acct_amount_med_7d': r7.median().values,
			'acct_amount_max_7d': r7.max().values,
			'acct_amount_std_7d': r7.std().fillna(0).values,
			'acct_txn_count_30d': r30.count().values,
			'acct_amount_sum_30d': r30.sum().values,
			'acct_amount_mean_30d': r30.mean().values,
			'acct_amount_med_30d': r30.median().values,
			'acct_amount_max_30d': r30.max().values,
			'acct_amount_std_30d': r30.std().fillna(0).values
		})

		# Shift so each row only sees past transactions, not its own
		result.iloc[:, 1:] = result.iloc[:, 1:].shift(1).fillna(0)
		results.append(result)

	rolling_df = pd.concat(results).reset_index(drop=True)

	df = df.merge(rolling_df, on='_txn_idx', how='left')
	df = df.drop(columns='_txn_idx')

	return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Compute row-wise features from raw transaction columns.

	Parameters
		df: Transaction DataFrame. Must contain fields 'Timestamp', 'Amount Paid', 'Amount Received',
			'Payment Currency', 'Receiving Currency', 'From Bank', 'To Bank', 'Payment Format'.

	Returns: Original DataFrame with additional feature columns:
		Temporal: hour, day_of_week, is_weekend, time_of_day_enc
		Amount: log_amount_paid, log_amount_received, amount_ratio, amount_diff
		Bank/currency: cross_currency, same_bank, from_bank_enc, to_bank_enc
		Categorical: payment_format_enc, pay_currency_enc, rec_currency_enc
	"""
	df = df.copy()

	# Temporal features
	df['hour'] = df['Timestamp'].dt.hour
	df['day_of_week'] = df['Timestamp'].dt.dayofweek
	df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
	df['time_of_day_enc'] = df['hour'].apply(_classify_time_of_day).astype('category').cat.codes

	# Amount features
	df['log_amount_paid'] = np.log1p(df['Amount Paid'])
	df['log_amount_received'] = np.log1p(df['Amount Received'])

	# Ratio: deviation from 1:1 signals layering behavior
	df['amount_ratio'] = (df['Amount Paid'] / (df['Amount Received'] + 1e-9)).clip(0, 100) # Clip to prevent extreme outliers dominating the feature space

	# Diff: large deviations suggest value extraction/injection in a layering chain
	df['amount_diff'] = df['Amount Paid'] - df['Amount Received']

	# Bank and currency flags
	df['cross_currency'] = (df['Payment Currency'] != df['Receiving Currency']).astype(int)

	df['same_bank'] = (df['From Bank'] == df['To Bank']).astype(int)
	df['from_bank_enc'] = df['From Bank'].astype('category').cat.codes
	df['to_bank_enc'] = df['To Bank'].astype('category').cat.codes

	# Categorical encodings
	df['payment_format_enc'] = df['Payment Format'].astype('category').cat.codes
	df['pay_currency_enc'] = df['Payment Currency'].astype('category').cat.codes
	df['rec_currency_enc'] = df['Receiving Currency'].astype('category').cat.codes

	return df


def _classify_time_of_day(hour: int) -> str:
	"""
	Map an hour (0-23) to a labeled time-of-day bucket.

	Parameters
		hour: Hour of day in 24-hour format (0-23)

	Returns: Time of day bucket prefixed with number to ensure ordered label encoding
	"""
	if 5 <= hour < 8:
		return '1_EarlyMorning'
	elif 8 <= hour < 12:
		return '2_Morning'
	elif 12 <= hour < 14:
		return '3_Midday'
	elif 14 <= hour < 18:
		return '4_Afternoon'
	elif 18 <= hour < 21:
		return '5_Evening'
	elif 21 <= hour < 24:
		return '6_Night'
	else:
		return '7_Overnight'