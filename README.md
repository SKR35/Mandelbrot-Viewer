# Mandelbrot-Viewer

**Interactive, high-precision Mandelbrot explorer** built in Python, featuring:
- Smooth real-time zoom and pan
- Rectangle zoom with aspect correction
- Auto-iteration scaling (deeper zoom → more detail)
- Save high-resolution PNG screenshots (2× supersampled)
- Toolbar with color map selector, gamma slider and quick iteration controls
- Multiple **soft gradient palettes** (Plasma, Viridis, SoftSunset, EarthAndSky, etc.)
- Optional Numba acceleration for faster fractal computation

## Create the environment
~~~bash
conda env create -f environment.yml
conda activate mandelbrot-viewer
~~~

## Run the viewer
~~~bash
python mandelbrot_tk.py
~~~