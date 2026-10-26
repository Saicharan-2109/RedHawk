# Spatial Matrix Vision Lab

A working local website for exploring computer vision with explicit linear algebra. Upload a PNG or JPEG, or choose a built-in sample, and compare three grayscale outputs: a spatial difference response, an inverse-operator baseline response, and a rightward pixel forecast.

The project implements the matrix engine in the supplied blueprints using **Streamlit, NumPy, and OpenCV**. Pillow handles safe image decoding and orientation. It runs on your CPU and needs no API key, account, trained model, or database.

## Quick start after cloning

Install Python 3.12 and Git, then run these commands in PowerShell:

```powershell
git clone https://github.com/Saicharan-2109/RedHawk.git
cd RedHawk
.\setup.bat
.\run.bat
```

Open **http://127.0.0.1:8501/**. The setup script creates a local `.venv` and installs the pinned dependencies. The virtual environment, caches, server logs and local artifacts are excluded from Git. To install the test tools too, use `.\setup.bat -Dev`.

## Run on this computer

Open PowerShell in the project folder:

```powershell
cd "C:\Users\saich\programming Tai lung\bdct cbp"
.\run.bat
```

Open **http://localhost:8501** if the browser does not open automatically. Keep the terminal open while using the website; press **Ctrl+C** to stop it.

The development computer already has an installed `.venv` environment; it is not committed to Git. After cloning, or to install or repair dependencies, run:

```powershell
.\setup.bat
.\run.bat
```

You can also start the app directly without activating the virtual environment:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Manual installation on another computer

Use Python 3.12 (3.12.14 was tested), open a terminal in this folder, and run the commands below. The pinned NumPy version requires Python 3.12 or later.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

On macOS/Linux, the equivalent commands use `python3 -m venv .venv` and `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`.

The installed and verified environment uses Python **3.12.14**, Streamlit **1.66.1**, NumPy **2.5.4**, OpenCV headless **4.14.0.94**, Pillow **12.3.0**, and pytest **9.1.1** for development checks.

`requirements.txt` contains the app dependencies, `requirements-dev.txt` adds test tools, and `requirements-lock.txt` records every package version in the tested Windows/Python 3.12 environment. To recreate that full environment, install with `.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt`.

## What you can do

- Start immediately with three generated samples, including architecture, geometry, and light/texture.
- Upload color or grayscale PNG/JPEG images and view the grayscale processing source.
- Tune the three autoregressive coefficients, the noise coefficient, display gains, seed, processing resolution, and forecast width.
- Apply a gentle-response or strong-outline preset, and inspect a selected row's pixel-response chart.
- View all three results and download each as a PNG.
- Download a ZIP containing the source, outputs, numeric matrices, and a reproducible parameter manifest.
- Inspect the transition, difference, and inverse matrices and numerical stability diagnostics.
- Read the mechanism and project background inside the website.
- Restore the blueprint defaults with one button.

Uploads are limited to 15 MiB and 40 million source pixels, and processing resolution is capped to keep memory and CPU use reasonable. Images narrower than three pixels are rejected. Invalid image data and unstable coefficient combinations receive readable errors.

## How the mechanism works

The normalized grayscale image is read horizontally. For every pixel, the engine forms a four-value state containing the current intensity, two preceding intensities, and a fixed noise seed. It constructs the blueprint's transition matrix `F`, applies `I - F` and `(I - F)⁻¹`, and advances the right-edge state with repeated multiplication by `F` to produce new columns.

The defaults are `φ₁ = 0.55`, `φ₂ = 0.25`, `φ₃ = 0.10`, `θ₁ = -0.40`, noise seed `0.02`, edge gain `4`, inverse gain `0.15`, and a `40`-pixel extension. NumPy evaluates the pixel operations in batches, preserving the blueprint equations while avoiding a Python loop for every pixel. Missing left neighbors repeat the first pixel, avoiding the original prototype's unprocessed black border.

This is an **ARIMA-inspired spatial demonstration with fixed coefficients**. It does not fit a statistical ARIMA model, estimate residuals, or learn image content. The difference response can highlight intensity changes; the inverse response is a local linear transformation and is not a guaranteed denoising or illumination-recovery algorithm. The forecast extends the right boundary and may decay toward black under the stable default matrix; it does not reconstruct objects or fill arbitrary holes.

## Project structure

```text
app.py                       Streamlit interface and interactive workflow
spatial_lab/
  engine.py                  Parameters, operators, transforms, diagnostics
  image_io.py                Image validation, decoding, samples, PNG/ZIP exports
tests/                       Numerical, image pipeline, and app interaction tests
docs/
  ARCHITECTURE.md             Components, data flow, equations, design decisions
  PROJECT_GUIDE.md            Website walkthrough, features, troubleshooting
.streamlit/config.toml       Local server and theme configuration
requirements.txt             Runtime dependencies
requirements-dev.txt         Test dependencies
requirements-lock.txt        Complete tested environment snapshot
setup.bat / run.bat           Windows setup and launch helpers
Dockerfile                   Optional container configuration
```

See [the architecture](docs/ARCHITECTURE.md) for the equations and implementation, and [the project guide](docs/PROJECT_GUIDE.md) for a website walkthrough.

For academic report writing, use [the complete project source brief](docs/PROJECT_REPORT_BRIEF.md), including the actual implementation, mathematical limitations, testing evidence, and the proposed big-data architecture. [The AI report prompt](docs/AI_REPORT_PROMPT.txt) can be pasted into another AI after attaching that brief.

## Run verification

Install the development dependencies with `setup.bat -Dev`, or use the explicit commands below:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Tests compare the optimized engine with an independent pixel-loop calculation and matrix powers, exercise invalid inputs and export contents, and check the Streamlit workflow through its official testing API. See the test files for the precise cases.

## Deployment scope

This is a complete local application. The Streamlit Python process provides the website and computation together. It does not include production authentication, multi-user upload isolation, persistent storage, or public hosting. If you later deploy it to a shared server, review those needs and the server's memory limits first.

An optional Dockerfile is included. With Docker installed, build and run it using:

```powershell
docker build -t spatial-matrix-lab .
docker run --rm -p 127.0.0.1:8501:8501 spatial-matrix-lab
```

The container uses the same website at `http://localhost:8501`. The included container configuration is optional; the verified local workflow uses `.venv`.
