"""
data/loader.py: Data loading and preparation for the AML pipeline.

Handles downloading the IBM Synthetic AML dataset from HuggingFace,
column renaming, and duplicate removal.
"""

import os

import pandas as pd
from datasets import load_dataset

import config
import features

CACHE_PATH = 'df_engineered.parquet'


def load_or_build(force_rebuild: bool=False) -> pd.DataFrame:
	"""
	Load preprocessed DataFrame from cache if available,
	otherwise build from scratch and save to cache.

	Parameters
		force_rebuild: If True, ignore cache and rebuild from source

	Returns: Fully engineered DataFrame
	"""
	if os.path.exists(CACHE_PATH) and not force_rebuild:
		print(f'Loading cached dataset from {CACHE_PATH}...')
		return pd.read_parquet(CACHE_PATH)

	print('Cache not found. Building from scratch...')

	df = load_aml_dataset()
	df = features.add_rolling_features(df)
	df = features.engineer_features(df)

	df.to_parquet(CACHE_PATH, index=False)
	print(f'Saved to {CACHE_PATH}.')

	return df


def load_aml_dataset() -> pd.DataFrame:
	"""
	Load the IBM Synthetic AML dataset from HuggingFace and return a clean pandas DataFrame.

	Returns: Clean dataset with 11 original columns and renamed account fields

	Notes
		Requires a HuggingFace account token for higher rate limits.
		Set HF_TOKEN as an environment variable before running:
			export HF_TOKEN=hf_your_token_here
	"""
	raw = load_dataset(config.DATASET_ID)
	df = raw['train'].to_pandas()

	df = df.rename(columns={
		'Account': 'From Account',
		'Account.1': 'To Account'
	})

	n_before = len(df)
	df = df.drop_duplicates()
	n_dropped = n_before - len(df)

	print(f'Loaded {len(df):,} rows and {df.shape[1]} columns.')
	if n_dropped > 0:
		print(f'Dropped {n_dropped:,} duplicate rows.')

	return df