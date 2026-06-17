#!/usr/bin/env python3
"""Add new categories and reclassify __none__ productTypes."""

import json
import re
import time
import requests
from pathlib import Path
import pandas as pd
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Image as VertexImage

# Config
PROJECT_ID = "ingka-b2bda-iifb-test"
LOCATION = "europe-west4"
MODEL_ID = "gemini-2.5-flash"
OUTPUT_DIR = Path.home() / "Downloads" / "gpc_taxonomy_mapping"

# Step 1: Add new categories
cats_file = OUTPUT_DIR / "target_categories_v2.json"
with open(cats_file) as f:
    cats = json.load(f)

new_cats = {
    "lamp_shade": "Lamp shades, pendant lamp shades, chandelier shades - the shade component only",
    "kitchen_tap": "Kitchen mixer taps, kitchen faucets, taps with handspray, pull-out spout taps",
    "shower_curtain": "Shower curtains, bathroom curtains for wet areas",
    "parasol": "Parasols, umbrellas for outdoor use, parasol bases and stands",
    "table_linen": "Tablecloths, table runners, placemats, table textiles",
    "curtain_hardware": "Curtain rods, curtain tracks, curtain wire, rod sets, panel curtain rails",
    "outdoor_furniture_set": "Outdoor furniture sets, conversation sets, patio sets with multiple pieces",
    "bathroom_fixture": "Shower sets, shower trays, riser rails, bathroom hardware fixtures",
}

cats.update(new_cats)
with open(cats_file, 'w') as f:
    json.dump(cats, f, indent=2)
print(f"Updated categories: {len(cats)} total (+{len(new_cats)} new)")

# Step 2: Load current mapping and filter __none__
df = pd.read_csv(OUTPUT_DIR / "productType_mapping_complete.csv")
none_df = df[df['category'] == '__none__'].copy()
print(f"\nReclassifying {len(none_df)} __none__ productTypes...")

# Init Vertex
vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel(MODEL_ID)

CATEGORY_LIST = "\n".join(f"- {k}: {v}" for k, v in cats.items())

def fetch_image(url: str, timeout: int = 10) -> bytes | None:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.content
    except:
        return None

def classify_productType(product_type: str, image_urls: list[str]) -> dict:
    # Fetch up to 4 images
    images = []
    for url in image_urls[:4]:
        img_bytes = fetch_image(url)
        if img_bytes:
            images.append(img_bytes)
        if len(images) >= 4:
            break
    
    prompt = f"""You are classifying IKEA products into categories for a room layout model.

Product type string: "{product_type}"

Categories (pick exactly ONE, or "__none__" if truly unclassifiable - small parts, hardware, consumables):
{CATEGORY_LIST}

Return ONLY valid JSON:
{{"category": "<category_name>", "confidence": <0.0-1.0>, "reason": "<brief reason>"}}
"""
    
    parts = [Part.from_text(prompt)]
    for img_bytes in images:
        try:
            parts.append(Part.from_image(VertexImage.from_bytes(img_bytes)))
        except:
            pass
    
    try:
        response = model.generate_content(parts, generation_config={"temperature": 0.1, "max_output_tokens": 512})
        text = response.text.strip()
        
        if "```json" in text:
            text = re.sub(r'```json\s*', '', text)
            text = re.sub(r'```\s*$', '', text)
        elif "```" in text:
            text = re.sub(r'```\s*', '', text)
        
        json_match = re.search(r'\{[\s\S]*?\}', text)
        if json_match:
            result = json.loads(json_match.group())
            if result.get("confidence", 0) > 1:
                result["confidence"] = result["confidence"] / 5.0
            return result
    except Exception as e:
        print(f"  Error: {e}")
    
    return {"category": "__none__", "confidence": 0.0, "reason": "error"}

# Step 3: Reclassify
results = {}
checkpoint_file = OUTPUT_DIR / "reclassify_checkpoint.json"
if checkpoint_file.exists():
    with open(checkpoint_file) as f:
        results = json.load(f)
    print(f"Resuming from checkpoint: {len(results)} done")

for idx, (_, row) in enumerate(none_df.iterrows()):
    pt = row['productType']
    if pt in results:
        continue
    
    # Get image URLs if available (from item_nos column - we don't have URLs here)
    # We'll use text-only classification
    print(f"[{idx+1}/{len(none_df)}] {pt}")
    result = classify_productType(pt, [])
    results[pt] = result
    print(f"  -> {result['category']} ({result['confidence']:.2f})")
    
    if (idx + 1) % 20 == 0:
        with open(checkpoint_file, "w") as f:
            json.dump(results, f, indent=2)
        print("  Checkpoint saved")
    
    time.sleep(0.3)

# Save final checkpoint
with open(checkpoint_file, "w") as f:
    json.dump(results, f, indent=2)

# Step 4: Update mapping
for pt, result in results.items():
    df.loc[df['productType'] == pt, 'category'] = result['category']
    df.loc[df['productType'] == pt, 'confidence'] = result['confidence']
    df.loc[df['productType'] == pt, 'reason'] = result['reason']

df.to_csv(OUTPUT_DIR / "productType_mapping_complete.csv", index=False)

# Summary
none_count = (df.category == '__none__').sum()
none_pct = 100 * none_count / len(df)
print(f"\n=== DONE ===")
print(f"Total productTypes: {len(df)}")
print(f"__none__ count: {none_count} ({none_pct:.1f}%)")
print(f"\nCategory distribution (top 20):")
print(df['category'].value_counts().head(20))
