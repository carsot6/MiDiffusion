#!/usr/bin/env python3
"""Merge existing and new classifications into complete mapping."""

import pandas as pd
from pathlib import Path

out_dir = Path.home() / 'Downloads' / 'gpc_taxonomy_mapping'

# Load existing mapping (1661 productTypes)
existing = pd.read_csv(out_dir / 'productType_mapping_itemcomm.csv')
print(f'Existing mapping: {len(existing)} productTypes')

# Load new classification (821 productTypes)
new = pd.read_csv(out_dir / 'missing_productTypes_classified.csv')
print(f'New classifications: {len(new)} productTypes')

# Check overlap
overlap = set(existing['productType']) & set(new['productType'])
print(f'Overlap: {len(overlap)} (should be 0)')

# Merge
combined = pd.concat([existing, new], ignore_index=True)
print(f'Combined: {len(combined)} productTypes')
print(f'Unique productTypes: {combined.productType.nunique()}')

# Save combined mapping
combined.to_csv(out_dir / 'productType_mapping_complete.csv', index=False)
print(f'\nSaved to productType_mapping_complete.csv')

# Summary
none_count = (combined.category == '__none__').sum()
none_pct = 100 * none_count / len(combined)
print(f'\nOverall __none__: {none_count} ({none_pct:.1f}%)')
print(f'Classified: {len(combined) - none_count} ({100-none_pct:.1f}%)')

# Category distribution
print('\nTop 25 categories:')
print(combined['category'].value_counts().head(25))
