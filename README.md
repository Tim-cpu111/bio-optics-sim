# Bio-Optics Simulation

这是一个组织光学仿真模型开发项目，使用 Python 编写，支持光在生物组织中的传播模拟（散射、吸收、反射等）。

## 📁 项目结构

- `src/biooptics/`：主代码模块
- `models/`：组织光学模型（如散射/吸收）
- `simulation/`：Monte Carlo 仿真逻辑
- `utils/`：通用工具函数
- `config.py`：全局配置文件
- `tests/`：单元测试
- `data/`：输入输出数据
- `notebooks/`：实验用 Jupyter 笔记本
- `scripts/`：运行脚本

## 🔧 安装依赖

```bash
pip install -r requirements.txt
pip install -e .
python examples/run_absorption_only.py

进入虚拟环境
source .venv/Scripts/activate


## 2025-08-28
- feat: S1 Step1 吸收-only CPU 内核（simulate_absorption_only）
- test: 能量守恒单测（absorption-only）通过
- demo: examples/run_absorption_only.py 输出 R_d≈0, A≈1
- build: 可编辑安装（pyproject.toml, pip install -e .）
- docs: 统一包名 biooptics，目录结构稳定
## 2025-08-29
- feat: S1 Step2 散射 (HG 相函数) 内核 (simulate_with_scattering)
- feat: Tallies 打点，支持步长/uz/轨迹采集
- test: g=0 各向同性散射验证, 能量守恒验证
- demo: examples/visualize_s1.py, 四合一可视化图
## v0.1.0-S1 (2025-09-01)
- 完成 S1：半无限均匀介质（CPU）
  - 吸收-only、HG 散射、边界 Fresnel、RR
  - Tallies 与四合一可视化
- 能量守恒：R_d + A ≈ 1（统计误差随 N 收敛）
