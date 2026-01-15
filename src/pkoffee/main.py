import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from pathlib import Path
import seaborn as sns
from abc import ABC, abstractmethod

csv_path = Path("coffee_productivity.csv")
if not csv_path.exists():
    # when running from package location (e.g. with pixi/uv), the working
    # directory may be different; fall back to repo root where the CSV lives
    csv_path = Path(__file__).resolve().parents[2] / "coffee_productivity.csv"
if not csv_path.exists():
    raise FileNotFoundError(f"coffee_productivity.csv not found at {csv_path} or working directory")

data = pd.read_csv(csv_path)
X = data["cups"].values
Y = data["productivity"].values

fig=plt.figure(figsize=(10,6))
ax=plt.gca()
ax.set_xlabel("Cups of Coffee")
ax.set_ylabel("Productivity")
ax.set_title("Productivity vs Coffee")
ax.grid(True,alpha=0.3)

sns.violinplot(data=data,x="cups",y="productivity",hue="cups",ax=ax,inner="quartile",cut=0,density_norm='width',palette="Greens",linewidth=0.8,legend=False)

x_min,x_max=float(np.min(X)),float(np.max(X))
y_min,y_max=float(np.min(Y)),float(np.max(Y))
dy=max(1e-8,y_max-y_min)

class ProductivityModel(ABC):
    def __init__(self, name: str):
        self.name = name
        self.params = None
        self.r2 = None
    
    @abstractmethod
    def get_initial_params(self, x_min, x_max, y_min, y_max, dy):
        pass
    
    @abstractmethod
    def get_bounds(self, x_min, x_max, y_min, y_max, dy):
        pass
    
    @abstractmethod
    def evaluate(self, x, *params):
        pass
    
    def fit(self, X, Y):
        x_min, x_max = float(np.min(X)), float(np.max(X))
        y_min, y_max = float(np.min(Y)), float(np.max(Y))
        dy = max(1e-8, y_max - y_min)
        
        p0 = self.get_initial_params(x_min, x_max, y_min, y_max, dy)
        bounds = self.get_bounds(x_min, x_max, y_min, y_max, dy)
        
        self.params, _ = curve_fit(self.evaluate, X, Y, p0=p0, bounds=bounds, maxfev=20000)
        yhat = self.evaluate(X, *self.params)
        ss_res = float(np.sum((Y - yhat) ** 2))
        ss_tot = float(np.sum((Y - np.mean(Y)) ** 2))
        self.r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return self


class QuadraticModel(ProductivityModel):
    def __init__(self):
        super().__init__("quadratic")
    
    def get_initial_params(self, x_min, x_max, y_min, y_max, dy):
        return [y_min, 0.0, 0.01]
    
    def get_bounds(self, x_min, x_max, y_min, y_max, dy):
        return (-np.inf, np.inf)
    
    def evaluate(self, xx, a0, a1, a2):
        return a0 + a1 * xx + a2 * xx ** 2


class SaturatingModel(ProductivityModel):
    def __init__(self):
        super().__init__("saturating")
    
    def get_initial_params(self, x_min, x_max, y_min, y_max, dy):
        return [dy, max(1.0, 0.2 * (x_min + x_max)), y_min]
    
    def get_bounds(self, x_min, x_max, y_min, y_max, dy):
        return ([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf])
    
    def evaluate(self, xx, Vmax, K, y0):
        return y0 + Vmax * (xx / np.maximum(K + xx, 1e-9))


class LogisticModel(ProductivityModel):
    def __init__(self):
        super().__init__("logistic")
    
    def get_initial_params(self, x_min, x_max, y_min, y_max, dy):
        return [dy, 0.5, 0.5 * (x_min + x_max), y_min]
    
    def get_bounds(self, x_min, x_max, y_min, y_max, dy):
        return ([-np.inf, 0.0, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf])
    
    def evaluate(self, xx, L, k, x0, y0):
        return y0 + L / (1.0 + np.exp(-k * (xx - x0)))


class PeakModel(ProductivityModel):
    def __init__(self):
        super().__init__("peak")
    
    def get_initial_params(self, x_min, x_max, y_min, y_max, dy):
        return [max(y_min, y_max), max(1.0, 0.5 * (x_min + x_max))]
    
    def get_bounds(self, x_min, x_max, y_min, y_max, dy):
        return ([-np.inf, 0.0], [np.inf, np.inf])
    
    def evaluate(self, xx, a, b):
        return a * xx * np.exp(-xx / np.maximum(b, 1e-9))


class Peak2Model(ProductivityModel):
    def __init__(self):
        super().__init__("peak2")
    
    def get_initial_params(self, x_min, x_max, y_min, y_max, dy):
        return [max(1e-6, y_max / max(1.0, x_max ** 2)), max(1.0, 0.5 * (x_min + x_max))]
    
    def get_bounds(self, x_min, x_max, y_min, y_max, dy):
        return ([-np.inf, 0.0], [np.inf, np.inf])
    
    def evaluate(self, xx, a, b):
        return a * (xx ** 2) * np.exp(-xx / np.maximum(b, 1e-9))


# Usage
MODELS = [
    QuadraticModel(),
    SaturatingModel(),
    LogisticModel(),
    PeakModel(),
    Peak2Model(),
]

fits = []
for model in MODELS:
    model.fit(X, Y)
    fits.append({"name": model.name, "func": model.evaluate, "params": model.params, "r2": model.r2})

fits.sort(key=lambda d:(d["r2"]if np.isfinite(d["r2"])else -np.inf),reverse=True)

x_smooth=np.linspace(np.min(X),np.max(X),300)
for idx,res in enumerate(fits):
 y_s=res["func"](x_smooth,*res["params"])
 label=f"{res['name']} (R²={res['r2']:.3f})"
 ax.plot(x_smooth,y_s,lw=2,label=label)


ax.legend()
ax.set_ylim(-0.2,8)
out_path=Path(__file__).with_name("fit_plot.png")
plt.tight_layout()
plt.savefig(out_path,dpi=150)
plt.show()