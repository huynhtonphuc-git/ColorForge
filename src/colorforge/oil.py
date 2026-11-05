"""Painting oil effect, second control: colour contrast between neighbouring areas, so patches of colour separate like stacked paint. Colour depth itself is film.deepen. Artistic choice, not film data."""
import cv2
import numpy as np


def layer_thickness(logH, um_per_px, sigma=(60.0, 900.0)):
    """Oil colour's second control: colour contrast between neighbouring areas, on the film's layer exposures, so patches of colour separate like stacked paint. Differences between the layers only, so brightness holds.
    Scaled by how coloured an area already is, so whites and greys get none. Add strength x this to logH. Artistic choice (`assumption`). No AI."""
    L = np.asarray(logH, np.float32); d = L - L.mean(-1, keepdims=True); blur = lambda a, s: cv2.GaussianBlur(a, (0, 0), s / um_per_px, borderType=cv2.BORDER_REFLECT); out = np.empty_like(L); soft = np.stack([blur(d[..., k], sigma[0]) for k in range(3)], -1); colour = np.sqrt((soft * soft).sum(-1, keepdims=True)); w = np.clip((colour - 0.03) / 0.15, 0, 1)
    for k in range(3):
        out[..., k] = soft[..., k] - blur(d[..., k], sigma[1])
    return np.clip(out * w, -0.25, 0.25)
