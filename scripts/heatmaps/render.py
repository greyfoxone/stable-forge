import numpy as np
import torch
from matplotlib import cm
from modules.shared import class_debug_log
from modules.shared import debug_log
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


@class_debug_log
class HiddenStateVisualizer:
    def __init__(self, h: list[torch.Tensor]):
        self.cond = h[0].detach().cpu().numpy()
        self.uncond = h[1].detach().cpu().numpy()

        self.fr = FeatureRenderer(
            norm_array=np.concatenate((self.cond, self.uncond)).flatten()
        )

    def render_cond(self, caption):
        return self.fr.grid(self.cond, f"{caption} (conditionend)")

    def render_uncond(self, caption):
        return self.fr.grid(self.uncond, f"{caption} (unconditioned)")

    def render_diff(self, caption):
        diff = self.cond - self.uncond
        return self.fr.grid(diff, f"{caption} (difference)")

    def render_all(self, caption):
        cond_img = self.render_cond(caption)
        cond_img.save(f"tmp/{caption}_cond.png")
        uncond_img = self.render_uncond(caption)
        uncond_img.save(f"tmp/{caption}_uncond.png")
        diff_img = self.render_diff(caption)
        diff_img.save(f"tmp/{caption}_diff.png")
        return [cond_img, uncond_img, diff_img]


@class_debug_log
class FeatureRenderer:
    def __init__(self, norm_array=None):
        self.jet_map = cm.get_cmap("seismic")
        self._norm = self.norm(norm_array)

    def grid(self, features: np.ndarray, caption="") -> Image.Image:
        print(f"{caption} ")
        normalized_features = self.normalize(features, *self._norm)
        heatmap_array = self._generate_heatmap(normalized_features)
        return self.grid_canvas(heatmap_array, caption)

    def norm(self, array: np.ndarray) -> tuple[float, float]:
        if array is None:
            return 0.0, 1.0

        min_val = array.min()
        max_val = array.max()
        return min_val, max_val

    def normalize(
        self, array: np.ndarray, min_val: float, max_val: float
    ) -> np.ndarray:
        if min_val == max_val:
            return np.zeros_like(array)
        return (array - min_val) / (max_val - min_val)

    def _generate_heatmap(self, array: np.ndarray) -> np.ndarray:
        flat = array.reshape(-1)
        heatmap = (self.jet_map(flat)[:, :3] * 255).astype(np.uint8)
        return heatmap.reshape(*array.shape, 3)

    def grid_canvas(
        self, heatmap_array: np.ndarray, caption: str
    ) -> Image.Image:
        if heatmap_array.size == 0:
            return Image.new("RGB", (0, 0))
        H, W, C = (
            heatmap_array.shape[1],
            heatmap_array.shape[2],
            heatmap_array.shape[3],
        )
        cols = int(np.sqrt(heatmap_array.shape[0])) + 1
        rows = (heatmap_array.shape[0] + cols - 1) // cols

        pad_num = rows * cols - heatmap_array.shape[0]
        padded = np.concatenate(
            [
                heatmap_array,
                np.zeros((pad_num, H, W, C), dtype=heatmap_array.dtype),
            ],
            axis=0,
        )

        grid = padded.reshape(rows, cols, H, W, C)
        grid = np.concatenate(
            [np.concatenate(grid[i], axis=1) for i in range(rows)], axis=0
        )
        return Image.fromarray(grid)