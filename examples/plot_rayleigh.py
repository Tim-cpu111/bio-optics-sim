"""
Rayleigh 散射交互可视化
----------------------
通过滑块实时调整折射率 n 和最大波长范围。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def rayleigh_cross_section(wavelength, n=1.4):
    lam = wavelength * 1e-9
    term = (n**2 - 1) / (n**2 + 2)
    return (1 / lam**4) * term**2

# 初始参数
n0 = 1.4
wl_min0, wl_max0 = 400, 700

# 创建画布
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121)              # 左边: 截面随波长
ax2 = fig.add_subplot(122, polar=True)  # 右边: 角度分布

# 初始曲线
wavelengths = np.linspace(wl_min0, wl_max0, 300)
sigma = rayleigh_cross_section(wavelengths, n=n0)
line1, = ax1.plot(wavelengths, sigma / sigma.max(), lw=2)

theta = np.linspace(0, np.pi, 360)
intensity = 1 + np.cos(theta)**2
line2, = ax2.plot(theta, intensity / intensity.max(), lw=2)

# 设置标题和标签
ax1.set_title("Rayleigh scattering ∝ 1/λ⁴")
ax1.set_xlabel("Wavelength (nm)")
ax1.set_ylabel("Relative scattering intensity")
ax1.grid(True)

ax2.set_title("Rayleigh Angular Distribution", va="bottom")

plt.subplots_adjust(left=0.1, bottom=0.25)  # 给滑块留空间

# --- 添加滑块 ---
ax_n = plt.axes([0.15, 0.1, 0.65, 0.03])   # [left, bottom, width, height]
ax_wl = plt.axes([0.15, 0.05, 0.65, 0.03])

slider_n = Slider(ax_n, "Refractive index n", 1.0, 2.0, valinit=n0, valstep=0.01)
slider_wl = Slider(ax_wl, "Max λ (nm)", 500, 900, valinit=wl_max0, valstep=10)

# --- 更新函数 ---
def update(val):
    n = slider_n.val
    wl_max = slider_wl.val

    # 更新截面曲线
    wavelengths = np.linspace(wl_min0, wl_max, 300)
    sigma = rayleigh_cross_section(wavelengths, n=n)
    line1.set_xdata(wavelengths)
    line1.set_ydata(sigma / sigma.max())
    ax1.set_xlim(wl_min0, wl_max)
    ax1.set_ylim(0, 1.05)

    # 更新角度分布
    intensity = 1 + np.cos(theta)**2
    line2.set_ydata(intensity / intensity.max())

    fig.canvas.draw_idle()

slider_n.on_changed(update)
slider_wl.on_changed(update)

plt.show()
