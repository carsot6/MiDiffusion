#!/usr/bin/env python3
"""Add new categories for Martin's missing productTypes."""

import json
from pathlib import Path

cats_file = Path.home() / 'Downloads' / 'gpc_taxonomy_mapping' / 'target_categories_v2.json'
with open(cats_file) as f:
    cats = json.load(f)

# Add new categories for Martin's missing productTypes
new_cats = {
    "curtain": "Window curtains, drapes, sheer curtains, blackout curtains, room-darkening curtains",
    "window_blind": "Roller blinds, venetian blinds, window shades",
    "furniture_component": "Cabinet doors, drawer fronts, cover panels, glass doors, base frames, furniture parts and accessories",
    "throw_blanket": "Throws, blankets, bed throws, decorative throws",
    "headboard": "Bed headboards",
    "kitchen_worktop": "Worktops, countertops, kitchen work surfaces",
    "kitchen_sink": "Kitchen sinks, inset sinks, undermount sinks",
    "kitchen_appliance_large": "Ovens, hobs, cooktops, dishwashers, refrigerators, freezers, washing machines, large kitchen appliances",
    "wardrobe_system": "Wardrobe components, sliding door combinations, clothes rails, shelves for wardrobes",
    "bathroom_sink": "Wash-basins, countertop wash-basins, bathroom sinks",
    "bathroom_tap": "Wash-basin mixer taps, bathroom faucets, taps with strainers",
    "pendant_lamp": "Pendant lamps, hanging lamps, ceiling-mounted pendant lights",
}

cats.update(new_cats)

with open(cats_file, 'w') as f:
    json.dump(cats, f, indent=2)

print(f"Updated categories: {len(cats)} total")
print("New categories added:")
for k in new_cats:
    print(f"  - {k}")
