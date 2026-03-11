"""
Augmentation parameter samplers following the specified distributions.
"""

import numpy as np


class AugmentationSampler:
    """Samples augmentation parameters according to specified distributions."""

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()

    def sample_scale(self) -> float:
        """Logo height as fraction of frame height.

        15-20% → 15%, 20-25% → 20%, 25-30% → 40%, 30-35% → 20%, 35-50% → 5%
        """
        r = self.rng.random()
        if r < 0.15:
            return self.rng.uniform(0.15, 0.20)
        elif r < 0.35:
            return self.rng.uniform(0.20, 0.25)
        elif r < 0.75:
            return self.rng.uniform(0.25, 0.30)
        elif r < 0.95:
            return self.rng.uniform(0.30, 0.35)
        else:
            return self.rng.uniform(0.35, 0.50)

    def sample_opacity(self) -> float:
        """Logo opacity (0-1).

        85-100% → 70%, 70-85% → 20%, 50-70% → 10%
        """
        r = self.rng.random()
        if r < 0.70:
            return self.rng.uniform(0.85, 1.0)
        elif r < 0.90:
            return self.rng.uniform(0.70, 0.85)
        else:
            return self.rng.uniform(0.50, 0.70)

    def sample_jpeg_quality(self) -> int:
        """JPEG compression quality.

        85-95 → 40%, 65-85 → 30%, 45-65 → 20%, 30-45 → 10%
        """
        r = self.rng.random()
        if r < 0.40:
            return int(self.rng.uniform(85, 95))
        elif r < 0.70:
            return int(self.rng.uniform(65, 85))
        elif r < 0.90:
            return int(self.rng.uniform(45, 65))
        else:
            return int(self.rng.uniform(30, 45))

    def sample_rotation(self) -> float:
        """Rotation in degrees.

        0° → 50%, 1-5° → 30%, 5-15° → 15%, 15-30° → 5%
        """
        r = self.rng.random()
        if r < 0.50:
            return 0.0
        elif r < 0.80:
            angle = self.rng.uniform(1, 5)
        elif r < 0.95:
            angle = self.rng.uniform(5, 15)
        else:
            angle = self.rng.uniform(15, 30)
        # Randomly negate
        return angle if self.rng.random() > 0.5 else -angle

    def sample_aspect_distortion(self) -> float:
        """Aspect ratio distortion factor.

        0% → 60%, 5-10% → 25%, 10-20% → 10%, 20-30% → 5%
        """
        r = self.rng.random()
        if r < 0.60:
            return 0.0
        elif r < 0.85:
            return self.rng.uniform(0.05, 0.10)
        elif r < 0.95:
            return self.rng.uniform(0.10, 0.20)
        else:
            return self.rng.uniform(0.20, 0.30)

    def sample_blur_sigma(self) -> float:
        """Gaussian blur sigma.

        none → 60%, 0.5-1px → 25%, 1-2px → 10%, 2-3px → 5%
        """
        r = self.rng.random()
        if r < 0.60:
            return 0.0
        elif r < 0.85:
            return self.rng.uniform(0.5, 1.0)
        elif r < 0.95:
            return self.rng.uniform(1.0, 2.0)
        else:
            return self.rng.uniform(2.0, 3.0)

    def sample_noise_level(self) -> float:
        """Gaussian noise standard deviation.

        none → 50%, 3-8 → 30%, 8-15 → 15%, 15-25 → 5%
        """
        r = self.rng.random()
        if r < 0.50:
            return 0.0
        elif r < 0.80:
            return self.rng.uniform(3, 8)
        elif r < 0.95:
            return self.rng.uniform(8, 15)
        else:
            return self.rng.uniform(15, 25)

    def sample_num_logos(self) -> int:
        """Number of logos per image.

        0 → 12%, 1 → 85%, 2 → 3%
        """
        r = self.rng.random()
        if r < 0.12:
            return 0
        elif r < 0.97:
            return 1
        else:
            return 2

    def sample_placement_type(self) -> str:
        """Placement region.

        bottom_third → 25%, corners → 20%, center → 18%, top_third → 17%, random → 20%
        """
        r = self.rng.random()
        if r < 0.25:
            return "bottom_third"
        elif r < 0.45:
            return "corners"
        elif r < 0.63:
            return "center"
        elif r < 0.80:
            return "top_third"
        else:
            return "random"

    def sample_color_jitter(self) -> dict:
        """Color jitter parameters.

        Brightness ±10%, Contrast ±10%, Saturation ±15%, Hue ±5°
        """
        return {
            "brightness": self.rng.uniform(-0.10, 0.10),
            "contrast": self.rng.uniform(-0.10, 0.10),
            "saturation": self.rng.uniform(-0.15, 0.15),
            "hue": self.rng.uniform(-5, 5),
        }

    def sample_placement_position(self, placement_type: str) -> tuple[float, float]:
        """Sample normalized (cx, cy) based on placement type."""
        if placement_type == "bottom_third":
            cx = self.rng.uniform(0.10, 0.90)
            cy = self.rng.uniform(0.66, 0.95)
        elif placement_type == "corners":
            corner = self.rng.choice(["tl", "tr", "bl", "br"])
            offset_x = self.rng.uniform(0.02, 0.08)
            offset_y = self.rng.uniform(0.02, 0.08)
            if corner == "tl":
                cx, cy = 0.0 + offset_x + 0.10, 0.0 + offset_y + 0.10
            elif corner == "tr":
                cx, cy = 1.0 - offset_x - 0.10, 0.0 + offset_y + 0.10
            elif corner == "bl":
                cx, cy = 0.0 + offset_x + 0.10, 1.0 - offset_y - 0.10
            else:
                cx, cy = 1.0 - offset_x - 0.10, 1.0 - offset_y - 0.10
        elif placement_type == "center":
            cx = self.rng.uniform(0.35, 0.65)
            cy = self.rng.uniform(0.35, 0.65)
        elif placement_type == "top_third":
            cx = self.rng.uniform(0.10, 0.90)
            cy = self.rng.uniform(0.05, 0.33)
        else:  # random
            cx = self.rng.uniform(0.05, 0.95)
            cy = self.rng.uniform(0.05, 0.95)
        return cx, cy

    def should_clip_edge(self) -> bool:
        """~5% of images allow logo partially outside frame."""
        return self.rng.random() < 0.05
