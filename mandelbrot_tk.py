#!/usr/bin/env python3
#Mandelbrot Viewer (Tkinter UI): toolbar, rectangle zoom, save and colormaps
from __future__ import annotations
import time, math, sys, os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

#Optional speed-up
try:
    from numba import njit
    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False

#Pillow for fast image blit
try:
    from PIL import Image, ImageTk
except Exception as e:
    raise SystemExit("Pillow is required. Install with:  pip install pillow") from e

#Colormap via matplotlib (for colors only + no GUI)
try:
    import matplotlib.cm as cm
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

# --- Palettes ---------------------------------------------------------------
CMAPS = [
    "plasma", "viridis", "magma", "inferno", "turbo", "cividis", "twilight",
    "SoftSunset", "EarthAndSky", "Seashore", "Forest", "HotAndCold",
    "Pastel", "Grayscale"
]
DEFAULT_CMAP = "SoftSunset"

#Custom soft palettes: list of hex stops (low -> high)
CUSTOM_STOPS = {
    "SoftSunset":  ["#2b1055","#6a0572","#ff6f91","#ffc15e","#ffe29a"],
    "EarthAndSky": ["#1a2a6c","#28a0b0","#84ffc9","#f0f3bd","#ffd166"],
    "Seashore":    ["#001219","#005f73","#0a9396","#94d2bd","#e9d8a6"],
    "Forest":      ["#0b3d0b","#236e3c","#4caf50","#a8e6cf","#f1f8e9"],
    "HotAndCold":  ["#313695","#4575b4","#74add1","#abd9e9","#fee090","#f46d43","#d73027"],
    "Pastel":      ["#b3e5fc","#c5cae9","#e1bee7","#f8bbd0","#ffe0b2","#dcedc8"],
    "Grayscale":   ["#0a0a0a","#2f2f2f","#5e5e5e","#9a9a9a","#cccccc","#f2f2f2"],
}

#Build a 1024-color lookup table from hex stops (using Matplotlib just for colors)
def _build_lut(stops, n=1024):
    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list("custom", stops, N=n)
    return (cmap(np.linspace(0,1,n))[:,:3] * 255.0).astype(np.uint8)

_CUSTOM_LUTS = {name: _build_lut(stops) for name, stops in CUSTOM_STOPS.items()}
DEFAULT_CMAP = "plasma"
MIN_SCALE = 1e-18  #Deep zoom

@dataclass
class View:
    cx: float = -0.75
    cy: float = 0.0
    scale: float = 1.5     #Half-height of view
    max_iter: int = 1000

    def grid(self, w: int, h: int) -> np.ndarray:
        aspect = h / w
        x_min = self.cx - self.scale / aspect
        x_max = self.cx + self.scale / aspect
        y_min = self.cy - self.scale
        y_max = self.cy + self.scale
        xs = np.linspace(x_min, x_max, w, dtype=np.float64)
        ys = np.linspace(y_min, y_max, h, dtype=np.float64)
        X, Y = np.meshgrid(xs, ys)
        return X + 1j * Y

def mandelbrot_smooth(c: np.ndarray, max_iter: int) -> np.ndarray:
    if HAS_NUMBA:
        return _mandelbrot_smooth_numba(c, max_iter)
    return _mandelbrot_smooth_numpy(c, max_iter)

def _mandelbrot_smooth_numpy(c: np.ndarray, max_iter: int) -> np.ndarray:
    z = np.zeros_like(c, dtype=np.complex128)
    it = np.zeros(c.shape, dtype=np.float64)
    mask = np.ones(c.shape, dtype=bool)
    for n in range(max_iter):
        z[mask] = z[mask]*z[mask] + c[mask]
        escaped = np.greater(np.abs(z), 4.0, where=mask)
        newly = escaped & mask
        if np.any(newly):
            zn = z[newly]
            it[newly] = n + 1 - np.log2(np.log(np.abs(zn)) + 1e-16)
        mask &= ~newly
        if not mask.any():
            break
    it[mask] = float(max_iter)  #Interior plateau
    return it

if HAS_NUMBA:
    @njit(cache=True, fastmath=True)
    def _mandelbrot_smooth_numba(c: np.ndarray, max_iter: int) -> np.ndarray:
        h, w = c.shape
        out = np.zeros((h, w), dtype=np.float64)
        for i in range(h):
            for j in range(w):
                z = 0.0 + 0.0j
                cc = c[i, j]
                hit = False
                for n in range(max_iter):
                    z = z*z + cc
                    if (z.real*z.real + z.imag*z.imag) > 16.0:
                        mag = (z.real*z.real + z.imag*z.imag)**0.5
                        out[i, j] = n + 1.0 - np.log(np.log(mag) + 1e-16)/np.log(2.0)
                        hit = True
                        break
                if not hit:
                    out[i, j] = float(max_iter)
        return out

def escaped_contrast(vals: np.ndarray, max_iter: int) -> Tuple[float, float]:
    escaped = vals < (max_iter - 1e-9)
    if np.any(escaped):
        lo = float(np.percentile(vals[escaped], 0.5))
        hi = float(np.percentile(vals[escaped], 99.5))
        if hi <= lo:
            lo, hi = float(vals.min()), float(vals.max())
    else:
        lo, hi = float(vals.min()), float(vals.max())
    return lo, hi

def map_to_rgb(vals: np.ndarray, cmap_name: str, lo: float, hi: float, gamma: float = 1.35, smoothstep: bool = True) -> np.ndarray:
    """Map scalar field to RGB uint8 with soft transitions.
       - gamma > 1 compresses highlights (taming bright yellows)
       - smoothstep makes gradients silky (reduces banding)
    """
    x = np.clip((vals - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    #Soft shaping first
    if smoothstep:
        x = x * x * (3.0 - 2.0 * x)         #Classic smoothstep
    if gamma and gamma > 0:
        x = np.power(x, gamma)

    if cmap_name in _CUSTOM_LUTS:
        lut = _CUSTOM_LUTS[cmap_name]
        idx = np.minimum((x * (len(lut)-1)).astype(np.int32), len(lut)-1)
        rgb = lut[idx]
    elif HAVE_MPL:
        import matplotlib.cm as cm
        cmap = cm.get_cmap(cmap_name)
        rgb = (cmap(x)[..., :3] * 255.0).astype(np.uint8)
    else:
        #Basic fallback
        r = (0.6 + 0.4*x) * 255
        g = (0.0 + 0.9*x) * 255
        b = (0.6 - 0.5*x) * 255
        rgb = np.dstack([r, g, b]).astype(np.uint8)
    return rgb

def auto_iters(scale: float) -> int:
    depth = max(0.0, -math.log10(max(scale, 1e-18)))
    return int(320 + 240*depth + 90*depth*depth)

class TkMandel:
    def __init__(self, w=1200, h=800):
        self.root = tk.Tk()
        self.root.title("Mandelbrot Viewer — Tk")

        # --- Toolbar
        top = ttk.Frame(self.root, padding=(6,4,6,4))
        top.pack(side=tk.TOP, fill=tk.X)

        self.mode = tk.StringVar(value="zoom")  #"zoom" or "pan"
        ttk.Button(top, text="Zoom", command=lambda: self.mode.set("zoom")).pack(side=tk.LEFT)
        ttk.Button(top, text="Pan", command=lambda: self.mode.set("pan")).pack(side=tk.LEFT)

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        self.auto_iter = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="AutoIter", variable=self.auto_iter, command=self._render).pack(side=tk.LEFT)

        ttk.Button(top, text="Iter +", command=lambda: self._bump_iter(1.25)).pack(side=tk.LEFT, padx=(6,2))
        ttk.Button(top, text="Iter −", command=lambda: self._bump_iter(1/1.25)).pack(side=tk.LEFT, padx=(2,6))

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Button(top, text="Save", command=self._save_dialog).pack(side=tk.LEFT)
        ttk.Button(top, text="Reset", command=self._reset).pack(side=tk.LEFT, padx=(6,0))

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Label(top, text="Colormap:").pack(side=tk.LEFT, padx=(0,4))
        self.cmap_var = tk.StringVar(value=DEFAULT_CMAP)
        ttk.OptionMenu(top, self.cmap_var, DEFAULT_CMAP, *CMAPS, command=lambda _=None: self._render()).pack(side=tk.LEFT)
        
        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Label(top, text="Gamma:").pack(side=tk.LEFT, padx=(0,4))
        self.gamma_var = tk.DoubleVar(value=1.35)  #Softer by default
        gamma_scale = ttk.Scale(top, from_=0.6, to=2.2, value=1.35,
                                command=lambda _=None: self._render(),
                                variable=self.gamma_var, length=120)
        gamma_scale.pack(side=tk.LEFT)

        # --- Canvas + status
        self.canvas = tk.Canvas(self.root, width=w, height=h, highlightthickness=0, bg="#ffffff", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.status = ttk.Label(self.root, anchor="w", font=("Consolas", 10))
        self.status.pack(fill=tk.X)

        #State
        self.w, self.h = w, h
        self.view = View()
        self._imgtk: Optional[ImageTk.PhotoImage] = None

        #Rubber-band rectangle for zoom
        self._rb_start: Optional[Tuple[int,int]] = None
        self._rb_rect_id: Optional[int] = None

        #Pan
        self._pan_start_px: Optional[Tuple[int,int]] = None
        self._pan_start_center: Optional[Tuple[float,float]] = None

        #Bindings
        self.root.bind("<Configure>", self._on_resize)
        # Wheel
        self.canvas.bind("<MouseWheel>", self._on_wheel)       #Windows
        self.canvas.bind("<Button-4>", lambda e: self._zoom_at(0.85, e.x, e.y)) #Linux
        self.canvas.bind("<Button-5>", lambda e: self._zoom_at(1/0.85, e.x, e.y))
        # Mouse buttons
        self.canvas.bind("<ButtonPress-1>", self._on_press_left)
        self.canvas.bind("<B1-Motion>", self._on_drag_left)
        self.canvas.bind("<ButtonRelease-1>", self._on_release_left)
        self.canvas.bind("<Button-3>", lambda e: self._reset())  # right-click reset
        # Keyboard
        self.root.bind("<KeyPress-z>", lambda e: self._zoom_at(0.85, self.w/2, self.h/2))
        self.root.bind("<KeyPress-x>", lambda e: self._zoom_at(1/0.85, self.w/2, self.h/2))
        self.root.bind("<KeyPress-plus>", lambda e: self._bump_iter(1.25))
        self.root.bind("<KeyPress-equal>", lambda e: self._bump_iter(1.25))
        self.root.bind("<KeyPress-minus>", lambda e: self._bump_iter(1/1.25))
        self.root.bind("<KeyPress-c>", lambda e: self._cycle_cmap())
        self.root.bind("<KeyPress-i>", lambda e: self._toggle_autoiter())
        self.root.bind("<KeyPress-r>", lambda e: self._reset())
        self.root.bind("<KeyPress-s>", lambda e: self._save_dialog())
        self.root.bind("<Escape>", lambda e: self.root.quit())

        self._render()

    # ---------- helpers
    def _status_text(self) -> str:
        return (f"Center=({self.view.cx:.6f}, {self.view.cy:.6f})   "
                f"Scale={self.view.scale:.6e}   Iter={self.view.max_iter}   "
                f"AutoIter={'ON' if self.auto_iter.get() else 'OFF'}   "
                f"Size={self.w}x{self.h}")

    def _set_status(self): self.status.config(text=self._status_text())

    def _on_resize(self, event):
        if event.widget is self.root:
            self.w, self.h = max(100, self.canvas.winfo_width()), max(100, self.canvas.winfo_height())
            self._render()

    def _cycle_cmap(self):
        idx = CMAPS.index(self.cmap_var.get())
        self.cmap_var.set(CMAPS[(idx+1) % len(CMAPS)])
        self._render()

    def _toggle_autoiter(self):
        self.auto_iter.set(not self.auto_iter.get())
        self._render()

    # ---------- mouse: left button (zoom-rect or pan)
    def _on_press_left(self, event):
        if self.mode.get() == "pan":
            self._pan_start_px = (event.x, event.y)
            self._pan_start_center = (self.view.cx, self.view.cy)
        else:  #Zoom-rect
            self._rb_start = (event.x, event.y)
            if self._rb_rect_id:
                self.canvas.delete(self._rb_rect_id)
                self._rb_rect_id = None

    def _on_drag_left(self, event):
        if self.mode.get() == "pan" and self._pan_start_px:
            sx, sy = self._pan_start_px
            dx, dy = event.x - sx, event.y - sy
            aspect = self.h / self.w
            span_x = 2 * self.view.scale / aspect
            span_y = 2 * self.view.scale
            dcx = -dx / self.w * span_x
            dcy =  dy / self.h * span_y
            cx0, cy0 = self._pan_start_center
            self.view.cx = cx0 + dcx
            self.view.cy = cy0 + dcy
            self._render()
        elif self.mode.get() == "zoom" and self._rb_start:
            x0, y0 = self._rb_start
            x1, y1 = event.x, event.y
            if self._rb_rect_id:
                self.canvas.coords(self._rb_rect_id, x0, y0, x1, y1)
            else:
                self._rb_rect_id = self.canvas.create_rectangle(
                    x0, y0, x1, y1, outline="#00e0ff", width=2, dash=(4,2)
                )

    def _on_release_left(self, event):
        if self.mode.get() == "pan":
            self._pan_start_px = self._pan_start_center = None
            return
        if not self._rb_start:
            return
        x0, y0 = self._rb_start
        x1, y1 = event.x, event.y
        self._rb_start = None
        if self._rb_rect_id:
            self.canvas.delete(self._rb_rect_id)
            self._rb_rect_id = None

        #If tiny box, treat as simple zoom-in at point
        if abs(x1 - x0) < 6 or abs(y1 - y0) < 6:
            self._zoom_at(0.7, x1, y1)
            return

        #Convert pixel-rect to complex rect
        left, right = sorted([x0, x1])
        bottom, top = sorted([y0, y1])
        #Complex bounds of current view
        aspect = self.h / self.w
        x_min = self.view.cx - self.view.scale / aspect
        x_max = self.view.cx + self.view.scale / aspect
        y_min = self.view.cy - self.view.scale
        y_max = self.view.cy + self.view.scale

        rx_min = x_min + (left / self.w)  * (x_max - x_min)
        rx_max = x_min + (right / self.w) * (x_max - x_min)
        ry_min = y_min + ((self.h - top) / self.h)    * (y_max - y_min)
        ry_max = y_min + ((self.h - bottom) / self.h) * (y_max - y_min)

        #Fit rect to canvas aspect (preserve center)
        rect_w = rx_max - rx_min
        rect_h = ry_max - ry_min
        target_aspect = self.h / self.w
        cx = (rx_min + rx_max) / 2.0
        cy = (ry_min + ry_max) / 2.0

        if rect_h / rect_w > target_aspect:
            #Too tall -> widen
            desired_w = rect_h / target_aspect
            rect_w = desired_w
        else:
            #Too wide -> heighten
            desired_h = rect_w * target_aspect
            rect_h = desired_h

        self.view.cx = cx
        self.view.cy = cy
        self.view.scale = max(MIN_SCALE, rect_h / 2.0)
        self._render()

    # ---------- Wheel zoom
    def _on_wheel(self, event):
        steps = int(event.delta / 120) if event.delta else 0
        if steps > 0:
            factor = 0.85 ** steps
        elif steps < 0:
            factor = (1/0.85) ** (-steps)
        else:
            return
        self._zoom_at(factor, event.x, event.y)

    def _zoom_at(self, factor: float, px: float, py: float):
        aspect = self.h / self.w
        x_min = self.view.cx - self.view.scale / aspect
        x_max = self.view.cx + self.view.scale / aspect
        y_min = self.view.cy - self.view.scale
        y_max = self.view.cy + self.view.scale
        target_x = x_min + (px / self.w) * (x_max - x_min)
        target_y = y_min + ((self.h - py) / self.h) * (y_max - y_min)
        self.view.cx = target_x + (self.view.cx - target_x) * factor
        self.view.cy = target_y + (self.view.cy - target_y) * factor
        self.view.scale = max(MIN_SCALE, self.view.scale * factor)
        self._render()

    def _bump_iter(self, mul: float):
        self.view.max_iter = max(20, int(self.view.max_iter * mul + 1))
        self._render()

    def _reset(self):
        self.view = View()
        self._render()

    # ---------- Render + save
    def _render(self):
        if self.auto_iter.get():
            self.view.max_iter = max(self.view.max_iter, auto_iters(self.view.scale))

        grid = self.view.grid(self.w, self.h)
        vals = mandelbrot_smooth(grid, self.view.max_iter)

        lo, hi = escaped_contrast(vals, self.view.max_iter)
        rgb = map_to_rgb(vals, self.cmap_var.get(), lo, hi, gamma=self.gamma_var.get(), smoothstep=True)

        img = Image.fromarray(rgb, mode="RGB")
        self._imgtk = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self._imgtk, anchor="nw")

        #If in zoom mode and user has started rectangle, recreate it on top
        if self._rb_rect_id:
            self.canvas.tag_raise(self._rb_rect_id)

        self._set_status()

    def _save_dialog(self):
        ts = time.strftime("%Y%m%d_%H%M%S")
        default = f"mandelbrot_{ts}.png"
        path = filedialog.asksaveasfilename(
            title="Save PNG",
            defaultextension=".png",
            filetypes=[("PNG image","*.png")],
            initialfile=default,
        )
        if not path:
            return
        try:
            self._save_png(path)
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def _save_png(self, path: str):
        #2x supersample for crisp output
        W, H = self.w * 2, self.h * 2
        grid = self.view.grid(W, H)
        vals = mandelbrot_smooth(grid, max(self.view.max_iter, int(self.view.max_iter*1.3)))
        lo, hi = escaped_contrast(vals, self.view.max_iter)
        rgb = map_to_rgb(vals, self.cmap_var.get(), lo, hi, gamma=self.gamma_var.get(), smoothstep=True)
        Image.fromarray(rgb, "RGB").save(path, optimize=True)
        print(f"Saved: {os.path.abspath(path)}")

    def run(self): self.root.mainloop()

if __name__ == "__main__":
    TkMandel().run()