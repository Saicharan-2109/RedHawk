# Project and website guide

## About the project

Spatial Matrix Vision Lab is an educational computer vision website based on the supplied Streamlit/OpenCV blueprint. It makes three linear-algebra transformations interactive so you can upload an image, change the constants, inspect the matrices and save the results.

The website runs locally through a Python Streamlit server. A browser displays the interface; NumPy performs the matrix calculations; OpenCV handles image conversion, resizing and PNG encoding; Pillow handles image loading, orientation and transparency. No GPU or model download is required.

## Start and stop

1. Open the project folder in a terminal.
2. Run `setup.bat` if dependencies have not been installed.
3. Run `run.bat`.
4. Visit `http://localhost:8501` in your browser.
5. Press **Ctrl+C** in the terminal when finished.

You can run the exact interpreter directly with `.\.venv\Scripts\python.exe -m streamlit run app.py`. This avoids requiring virtual-environment activation or a global `streamlit` command.

## Website workflow

### Select an image

Start with the built-in sample gallery to see results immediately. The architecture sample contains hard boundaries, the geometric sample contains deliberate shapes, and the texture sample contains gradual intensity changes. The sample images are generated locally and need no internet connection.

Switch to the upload option for your own PNG or JPEG. Color images are converted to grayscale for the matrix engine. Transparent regions are placed on white, and EXIF rotation is applied before processing. The website displays the prepared source so you can see what the calculation receives.

### Adjust the controls

The parameter preset selector provides the blueprint defaults, a gentle response, and stronger outlines. Changing a preset fills the individual controls; you can then adjust them further.

| Control | Effect |
| --- | --- |
| `φ₁`, `φ₂`, `φ₃` | Change the autoregressive weights in the first row of the transition matrix. |
| `θ₁` | Changes how the fixed noise component contributes to the image operators. |
| Noise seed | Sets the fourth component of the per-pixel state. |
| Edge gain | Changes the visibility of the absolute difference response. |
| Inverse/integration gain | Changes the visibility of the inverse-operator response. |
| Extension width | Chooses how many pixels to predict to the right of the image. |
| Processing resolution | Caps the longest prepared image dimension; smaller values use less computation. |
| Reset parameters | Restores the supplied blueprint's numerical defaults. |

Large autoregressive weights can make the state unstable. If that happens, lower the `φ` values. The default weights sum to `0.90`, producing a stable matrix. A matrix near singularity can also be rejected because its inverse would be numerically unreliable.

### Compare the outputs

The **difference response** applies `(I - F)` and emphasizes certain horizontal intensity differences. The **inverse response** applies `(I - F)⁻¹`; its appearance depends on the coefficients and gain. The **canvas extension** copies the grayscale source and adds predicted columns on its right.

Each output is an 8-bit grayscale image. Changing gains can clip values to white or black. The extension becomes darker at longer horizons under the stable default recurrence; this follows the matrix dynamics.

### Inspect and export

Use the matrix inspector to examine `F`, `I - F`, `(I - F)⁻¹`, the spectral radius and the inverse condition number. These help explain how a parameter change affects the result.

Expand the grayscale comparison section to see the precise grayscale source and a line chart of the source, difference and inverse responses along a selected row. The chart uses the same gained and clipped 8-bit values shown in the output images.

Download individual PNGs for comparisons or take the complete ZIP bundle for a reproducible experiment. The bundle includes a JSON parameter manifest and NumPy data in addition to images. You can inspect a downloaded NumPy archive with `numpy.load(path, allow_pickle=False)`.

## What this project demonstrates

- Computer vision transformations without a neural network.
- Turning a mathematical prototype into a usable website.
- Normalization, grayscale conversion and bounded display intensities.
- Explicit state construction, linear operators, matrix inversion and matrix powers.
- Stable recurrences and the limits of deterministic spatial forecasting.
- Separation of interface, numerical engine and file handling.

The system does not fit a statistical ARIMA model or reconstruct scene content. Its outputs should be interpreted as mathematical responses to an image rather than guaranteed edge segmentation, denoising, illumination recovery or semantic inpainting.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| `python` or `streamlit` is not recognized | Use `run.bat` or the full `.venv\Scripts\python.exe` command. |
| Missing module error | Run `setup.bat`, or install `requirements.txt` with the virtual-environment interpreter. |
| Port 8501 is already in use | Stop the earlier server, or run with `--server.port 8502` and open that port. |
| Upload is rejected | Use a valid PNG/JPEG below 15 MiB and 40 million pixels, at least three pixels wide. Renaming another file to `.png` does not convert it. |
| Unstable coefficient message | Reduce the `φ` coefficients; reset the defaults if needed. |
| Output is too dark or bright | Adjust the relevant output gain; clipping and stable forecast decay are expected mathematical effects. |
| Browser disconnects | Confirm the Streamlit terminal is still running, then reload the page. |
| Processing takes too long | Lower the processing resolution and extension width. |

## Verification and extending the project

Run the tests with `.\.venv\Scripts\python.exe -m pytest -q` after installing `requirements-dev.txt`. Numerical tests compare the optimized code with a separate literal implementation, and interface tests exercise the Streamlit workflow.

To change the mathematics, start in `spatial_lab/engine.py`. To add upload formats or exports, use `spatial_lab/image_io.py`. To change layout or interaction, edit `app.py`. Keep the core functions independent of Streamlit and add numerical reference cases when changing the transformation equations.
