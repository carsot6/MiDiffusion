#!/usr/bin/env python3
"""Create item_no to category mapping and upload to GCS."""

import pandas as pd
from pathlib import Path
from google.cloud import storage

out_dir = Path.home() / 'Downloads' / 'gpc_taxonomy_mapping'

# Load combined mapping
df = pd.read_csv(out_dir / 'productType_mapping_complete.csv')
print(f'Loaded {len(df)} productTypes')

# Explode item_nos to create item_no → category mapping
rows = []
for _, row in df.iterrows():
    item_nos = str(row.get('item_nos', '')).split('|')
    for item_no in item_nos:
        if item_no and item_no != 'nan':
            rows.append({
                'item_no': item_no.strip(),
                'category': row['category'],
                'productType': row['productType'],
            })

item_mapping = pd.DataFrame(rows)
print(f'Created {len(item_mapping)} item_no → category rows')
print(f'Unique item_nos: {item_mapping.item_no.nunique()}')

# Save locally
item_mapping.to_csv(out_dir / 'item_to_category.csv', index=False)
print(f'Saved item_to_category.csv')

# Upload to GCS
bucket_name = 'ingka-b2b-da-ifb-test-studior2b'
prefix = 'gpc_taxonomy_mapping'
client = storage.Client(project='ingka-b2bda-iifb-test')
bucket = client.bucket(bucket_name)

files = [
    'productType_mapping_complete.csv',
    'item_to_category.csv',
    'target_categories_v2.json',
    'missing_productTypes_classified.csv',
]

for f in files:
    blob = bucket.blob(f'{prefix}/{f}')
    blob.upload_from_filename(out_dir / f)
    print(f'Uploaded {f}')

print(f'\nAll files at: gs://{bucket_name}/{prefix}/')
