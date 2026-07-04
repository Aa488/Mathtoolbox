# 📐 MathToolbox — 专业数学分析工具箱

高等数学全覆盖：函数分析 · 微积分 · 无穷级数 · 微分方程 · 多元微积分 · 拉普拉斯变换与向量分析

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动
python math_toolbox.py
```

Windows 用户可直接运行 `dist/MathToolbox/MathToolbox.exe`（已打包，无需 Python 环境）。

## 📦 依赖

| 包 | 用途 |
|---|------|
| `numpy` | 数值计算 |
| `scipy` | 积分 / ODE / 特殊函数 |
| `sympy` | 符号计算 |
| `matplotlib` | 2D / 3D 绑图 |
| `scikit-image` | marching_cubes 三维表面提取 |

## 🧮 功能模块

### 📈 函数分析
- 绘图 + 完整分析：零点、极值、拐点、单调性、凹凸性、渐近线（垂直/水平/斜）、对称性、周期性、值域
- 极限计算（左右极限 / 洛必达 / 等价无穷小 / 夹逼 / 泰勒展开）
- 参数曲线 `(x(t), y(t))` + 弧长
- 极坐标 `r(θ)` + 面积
- 点击图像任意位置标注坐标（自动识别 π 分数）

### ∫ 微积分
- 定积分 / 不定积分 / 反常积分（无穷限 + 瑕点）
- 黎曼和可视化（左/右/中点/梯形）
- 二重积分（支持变量边界 `y = ±√(r² - x²)` + 3D 曲面渲染）
- 三重积分（自动积分排序 + marching_cubes 任意形状 3D 区域渲染）
- 弧长 / 曲率（含密切圆）
- 数值积分对比（梯形 / Simpson / 中点 / Gauss）

### ∑ 无穷级数
- 泰勒展开（任意阶 + 系数逐项展示）
- 傅里叶级数（自动化简 π 系数）
- 级数收敛判定（通项/比值/根值/p-级数/交错级数/比较检验）
- 幂级数收敛半径
- 级数求和

### ⊡ 微分方程
- 方向场可视化
- 初值问题数值解 (RK45)
- 批量解曲线

### ⊿ 多元微积分
- 梯度 / 方向导数（3D 箭头可视化）
- Hessian 矩阵
- 拉格朗日乘数法
- 3D 曲面 `z = f(x, y)`（可拖拽旋转）

### L 拉普拉斯 & 向量分析
- 拉普拉斯正 / 逆变换
- 散度 / 旋度 / 线积分

### 📝 解题过程追踪
- 140+ 定理/公式库
- 每一步标注所用定理，教科书式解题步骤
- 自动检测多种解法（洛必达/等价无穷小/泰勒/夹逼等）
- 结果智能识别 π 表达式（如 `π/4`、`π²/4`）

## ⌨️ 支持的表达式语法

```
三角函数: sin cos tan cot sec csc
反三角:   asin acos atan
双曲:     sinh cosh tanh
指对幂:   exp log ln sqrt ^ **
绝对值:   abs(x) |x|
常数:     pi(π) e(E)
导数:     sin'(x)  (sin(x))'  (sin(x))''  sin''(x)  d/dx sin(x)  d²/dx² sin(x)
方程:     y=sin(x)  x=sin(y)  x²+y²=1
```

## 🏗️ 项目结构

```
├── math_toolbox.py          # 主 GUI 程序 (6 Tab)
├── function_plotter.py      # 函数绑图与分析引擎
├── calculus_engine.py       # 微积分/级数/ODE/极限/多重积分引擎
├── solution_tracer.py       # 解题过程追踪 + 140 定理库
├── function_plotter_gui.py  # 独立轻量 GUI (单窗口版)
├── requirements.txt         # Python 依赖
├── icon.ico / icon.png      # 图标
└── dist/MathToolbox/        # PyInstaller 打包输出
```

## 💬 问题反馈

如遇 Bug 或有功能建议，欢迎联系：

**QQ: 1540893239**

（添加时请备注 "MathToolbox"）
