#!/usr/bin/env python3
"""Create sample placeholder logos for testing the pipeline."""

import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

ASSETS_DIR = "assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

LOGOS = {
    "logo1": {"color": (255, 50, 50), "bg": (255, 50, 50, 200), "text": "LOGO 1"},
    "logo2": {"color": (50, 150, 255), "bg": (50, 150, 255, 200), "text": "LOGO 2"},
    "logo3": {"color": (50, 255, 100), "bg": (50, 255, 100, 200), "text": "LOGO 3"},
    "logo4": {"color": (255, 200, 50), "bg": (255, 200, 50, 200), "text": "LOGO 4"},
    "logo5": {"color": (200, 50, 255), "bg": (200, 50, 255, 200), "text": "LOGO 5"},
}

for name, config in LOGOS.items():
    # Create a 300x150 logo with rounded rect and text
    img = Image.new("RGBA", (300, 150), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw rounded rectangle
    draw.rounded_rectangle(
        [10, 10, 290, 140],
        radius=20,
        fill=config["bg"],
        outline=(255, 255, 255, 220),
        width=3,
    )

    # Draw text
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
    except (OSError, IOError):
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), config["text"], font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = (300 - tw) // 2
    ty = (150 - th) // 2
    draw.text((tx, ty), config["text"], fill=(255, 255, 255, 255), font=font)

    path = os.path.join(ASSETS_DIR, f"{name}.png")
    img.save(path)
    print(f"Created: {path}")

print(f"\nAll sample logos saved to {ASSETS_DIR}/")
