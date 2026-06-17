#!/usr/bin/env python3
"""Classify Martin's missing productTypes using Gemini multimodal."""

import json
import time
import requests
import io
import re
from pathlib import Path
from PIL import Image
import pandas as pd
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Image as VertexImage

# Config
PROJECT_ID = "ingka-b2bda-iifb-test"
LOCATION = "europe-west4"
MODEL_ID = "gemini-2.5-flash"
OUTPUT_DIR = Path.home() / "Downloads" / "gpc_taxonomy_mapping"

# Init Vertex
vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel(MODEL_ID)

# Load categories
with open(OUTPUT_DIR / "target_categories_v2.json") as f:
    ALL_CATEGORIES = json.load(f)

CATEGORY_LIST = "\n".join(f"- {k}: {v}" for k, v in ALL_CATEGORIES.items())

def fetch_image(url: str, timeout: int = 10) -> bytes | None:
    """Fetch image bytes from URL."""
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.content
    except Exception as e:
        print(f"  Failed to fetch {url[:60]}...: {e}")
        return None

def classify_productType(product_type: str, image_urls: list[str]) -> dict:
    """Classify a productType using Gemini with images."""
    
    # Fetch up to 6 images
    images = []
    for url in image_urls[:6]:
        img_bytes = fetch_image(url)
        if img_bytes:
            images.append(img_bytes)
        if len(images) >= 6:
            break
    
    # Build prompt
    prompt = f"""You are classifying IKEA products into categories for a room layout model.

Product type string: "{product_type}"

Categories (pick exactly ONE, or "__none__" if truly unclassifiable):
{CATEGORY_LIST}

Return ONLY valid JSON:
{{"category": "<category_name>", "confidence": <0.0-1.0>, "reason": "<brief reason>"}}
"""
    
    # Build content parts
    parts = [Part.from_text(prompt)]
    for img_bytes in images:
        try:
            parts.append(Part.from_image(VertexImage.from_bytes(img_bytes)))
        except Exception as e:
            print(f"  Failed to create image part: {e}")
    
    # Call Gemini
    try:
        response = model.generate_content(
            parts,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 512,
            }
        )
        text = response.text.strip()
        
        # Strip markdown code blocks if present
        if "```json" in text:
            text = re.sub(r'```json\s*', '', text)
            text = re.sub(r'```\s*$', '', text)
        elif "```" in text:
            text = re.sub(r'```\s*', '', text)
        
        # Extract JSON - handle multi-line
        json_match = re.search(r'\{[\s\S]*?\}', text)
        if json_match:
            try:
                result = json.loads(json_match.group())
                # Normalize confidence
                if result.get("confidence", 0) > 1:
                    result["confidence"] = result["confidence"] / 5.0
                return result
            except json.JSONDecodeError:
                print(f"  JSON decode error: {json_match.group()[:100]}")
                return {"category": "__none__", "confidence": 0.0, "reason": "json_decode_error"}
        else:
            print(f"  No JSON in response: {text[:100]}")
            return {"category": "__none__", "confidence": 0.0, "reason": "parse_error"}
    except Exception as e:
        print(f"  Gemini error: {e}")
        return {"category": "__none__", "confidence": 0.0, "reason": str(e)}

def main():
    import sys
    # Load missing productTypes
    df = pd.read_csv("/Users/carlos.soto1/Downloads/missing_productTypes.csv")
    print(f"Loaded {len(df)} missing productTypes", flush=True)
    
    # Check for checkpoint
    checkpoint_file = OUTPUT_DIR / "missing_classification_checkpoint.json"
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            results = json.load(f)
        print(f"Resuming from checkpoint: {len(results)} already done")
    else:
        results = {}
    
    # Classify each
    for idx, row in df.iterrows():
        pt = row["productType"]
        if pt in results:
            continue
        
        urls = row["image_urls"].split("|") if pd.notna(row["image_urls"]) else []
        
        print(f"[{idx+1}/{len(df)}] {pt} ({len(urls)} images)")
        result = classify_productType(pt, urls)
        result["item_nos"] = row["item_nos"]
        result["num_items"] = row["num_items"]
        results[pt] = result
        
        print(f"  -> {result['category']} ({result['confidence']:.2f})")
        
        # Save checkpoint every 10
        if (idx + 1) % 10 == 0:
            with open(checkpoint_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Checkpoint saved")
        
        # Rate limit
        time.sleep(0.5)
    
    # Save final results
    with open(checkpoint_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Convert to DataFrame and save
    rows = []
    for pt, r in results.items():
        rows.append({
            "productType": pt,
            "category": r["category"],
            "confidence": r["confidence"],
            "reason": r["reason"],
            "item_nos": r.get("item_nos", ""),
            "num_items": r.get("num_items", 0),
        })
    
    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_DIR / "missing_productTypes_classified.csv", index=False)
    print(f"\nSaved {len(result_df)} classifications to missing_productTypes_classified.csv")
    
    # Show summary
    print("\nCategory distribution:")
    print(result_df["category"].value_counts().head(20))

if __name__ == "__main__":
    main()
