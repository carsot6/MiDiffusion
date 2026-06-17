#!/usr/bin/env python3
"""Analyze __none__ productTypes and propose new categories."""

import pandas as pd
from pathlib import Path

df = pd.read_csv(Path.home() / 'Downloads' / 'gpc_taxonomy_mapping' / 'productType_mapping_complete.csv')
none_df = df[df['category'] == '__none__'].sort_values('num_items', ascending=False)

print(f'Total __none__: {len(none_df)} productTypes\n')
print('ProductType                                    Items')
print('-' * 55)
for _, row in none_df.iterrows():
    pt = row['productType'][:45]
    ni = int(row['num_items']) if pd.notna(row['num_items']) else 0
    print(f"{pt:45s} {ni:5d}")
