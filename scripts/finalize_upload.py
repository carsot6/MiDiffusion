#!/usr/bin/env python3
"""Check new category distribution and upload to GCS."""

import pandas as pd
from pathlib import Path
from google.cloud import storage

out_dir = Path.home() / 'Downloads' / 'gpc_taxonomy_mapping'
df = pd.read_csv(out_dir / 'productType_mapping_complete.csv')

new_cats = ['lamp_shade', 'kitchen_tap', 'shower_curtain', 'parasol', 'table_linen', 'curtain_hardware', 'outdoor_furniture_set', 'bathroom_fixture']

print("New category distribution:")
for cat in new_cats:
    count = (df['category'] == cat).sum()
    print(f"  {cat}: {count}")

print(f"\nTotal in new categories: {sum((df['category'] == c).sum() for c in new_cats)}")
print(f"Remaining __none__: {(df['category'] == '__none__').sum()}")

# Recreate item_to_category mapping
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
item_mapping.to_csv(out_dir / 'item_to_category.csv', index=False)
print(f"\nitem_to_category.csv: {len(item_mapping)} rows, {item_mapping.item_no.nunique()} unique item_nos")

# Upload to GCS
bucket_name = 'ingka-b2b-da-ifb-test-studior2b'
prefix = 'gpc_taxonomy_mapping'
client = storage.Client(project='ingka-b2bda-iifb-test')
bucket = client.bucket(bucket_name)

files = [
    'productType_mapping_complete.csv',
    'item_to_category.csv',
    'target_categories_v2.json',
]

print("\nUploading to GCS:")
for f in files:
    blob = bucket.blob(f'{prefix}/{f}')
    blob.upload_from_filename(out_dir / f)
    print(f"  ✓ {f}")

print(f"\nAll files at: gs://{bucket_name}/{prefix}/")
