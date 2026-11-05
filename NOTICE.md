# Third-party notices

ColorForge's own code and data are under [LICENSE](LICENSE). These parts keep their own terms.

## Included in this repository

| Part | Files | Licence | Credit |
|---|---|---|---|
| PMRID weights | `third_party/PMRID/models/torch_pretrained.ckp` | Apache-2.0, see `third_party/PMRID/LICENSE` | Megvii Research; Wang et al., ECCV 2020 |
| Real-ESRGAN Compact weights | `third_party/RealESRGAN_compact/realesr-general-x4v3.pth` | BSD-3-Clause, see `third_party/RealESRGAN_compact/LICENSE` | Xintao Wang et al. |
| Digitised curves for Velvia 50, Provia 100F, Ektachrome 100D | `data/stocks/*_sheet.cfstock.json` | MIT | spectral_film_lut, Jan Lohse, https://github.com/JanLohse/spectral_film_lut |

The network definitions in `src/colorforge/denoise.py` and `src/colorforge/enhance.py` are re-implementations of those projects' published architectures, so these notices and licences also apply to those parts. Those definitions were changed from the originals only in form, to load the same published weights. Neither project endorses ColorForge. The AI models' terms are their own, and the ColorForge licence doesn't change them.

## Installed separately, not included

Python packages listed in `requirements.txt` keep their own licences, including PyTorch, OpenVINO, NumPy, SciPy, OpenCV, colour-science, rawpy (LibRaw), lensfunpy (Lensfun, LGPL-3 library; CC BY-SA 3.0 database), wgpu-py, Pillow, imagecodecs, tifffile, ExifRead and Numba.

## Method credits

Film physics draws on the published photographic science literature, including Frieser's emulsion spread model and Nelson's chemical spread model. Film curves are measurements read from manufacturers' published technical data sheets; those documents are not included.
