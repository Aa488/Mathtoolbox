#!/usr/bin/env python3
"""
=============================================================================
  通用函数图像绘制工具 — GUI 版
  基于 tkinter + matplotlib，支持全部初等函数、导数、方程
=============================================================================
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import warnings
import math

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# 导入核心引擎
import function_plotter as fplib
from function_plotter import FunctionPlotter, _preprocess_expr, LANG

# ============================================================================
# GUI 主窗口
# ============================================================================

class FunctionPlotterGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("通用函数图像绘制工具")
        self.root.geometry("1280x900")
        self.root.minsize(900, 650)

        # 配色
        self.bg = "#f5f5f5"
        self.root.configure(bg=self.bg)

        # 核心绘图器
        self.plotter = FunctionPlotter(lang="cn", x_range=(-8, 8), figsize=(10, 5.5), dpi=100)
        self._history: list[str] = []

        self._build_ui()

        # 启动时画一个示例 (等 UI 就绪后)
        self.root.after(500, lambda: self._draw("sin(x)"))

    # ── UI 构建 ────────────────────────────────────────────────
    def _build_ui(self):
        # ---- 顶部工具栏 ----
        toolbar = ttk.Frame(self.root)
        toolbar.pack(fill=tk.X, padx=8, pady=(8, 2))

        ttk.Label(toolbar, text="表达式:", font=("Microsoft YaHei", 11)).pack(side=tk.LEFT, padx=(0, 4))

        self.expr_var = tk.StringVar(value="sin(x)")
        self.entry = ttk.Entry(toolbar, textvariable=self.expr_var, font=("Consolas", 13), width=50)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.entry.bind("<Return>", lambda e: self._draw())
        self.entry.bind("<Up>", self._history_up)
        self.entry.bind("<Down>", self._history_down)

        self.btn_draw = ttk.Button(toolbar, text="✏ 绘制", command=self._draw)
        self.btn_draw.pack(side=tk.LEFT, padx=2)

        self.btn_clear = ttk.Button(toolbar, text="🗑 清除", command=self._clear)
        self.btn_clear.pack(side=tk.LEFT, padx=2)

        self.btn_save = ttk.Button(toolbar, text="💾 保存", command=self._save)
        self.btn_save.pack(side=tk.LEFT, padx=2)

        self.btn_help = ttk.Button(toolbar, text="❓ 帮助", command=self._show_help)
        self.btn_help.pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Label(toolbar, text="标注点 x=", font=("Microsoft YaHei", 10)).pack(side=tk.LEFT, padx=(0, 2))
        self.point_var = tk.StringVar(value="")
        self.point_entry = ttk.Entry(toolbar, textvariable=self.point_var, font=("Consolas", 12), width=8)
        self.point_entry.pack(side=tk.LEFT, padx=(0, 4))
        self.point_entry.bind("<Return>", lambda e: self._annotate_point())
        self.btn_point = ttk.Button(toolbar, text="标注", command=self._annotate_point)
        self.btn_point.pack(side=tk.LEFT, padx=2)

        # ---- 范围控制 ----
        range_frame = ttk.Frame(self.root)
        range_frame.pack(fill=tk.X, padx=8, pady=(2, 4))

        ttk.Label(range_frame, text="X 范围:", font=("Microsoft YaHei", 10)).pack(side=tk.LEFT, padx=(0, 4))
        self.xmin_var = tk.StringVar(value="-8")
        self.xmax_var = tk.StringVar(value="8")
        ttk.Entry(range_frame, textvariable=self.xmin_var, width=6, font=("Consolas", 10)).pack(side=tk.LEFT)
        ttk.Label(range_frame, text=" ~ ").pack(side=tk.LEFT)
        ttk.Entry(range_frame, textvariable=self.xmax_var, width=6, font=("Consolas", 10)).pack(side=tk.LEFT)

        ttk.Label(range_frame, text="  预设:").pack(side=tk.LEFT, padx=(12, 2))
        for label, rng in [("-2π~2π", "-6.28 6.28"), ("0~4π", "0 12.57"), ("-10~10", "-10 10"), ("-4~4", "-4 4")]:
            ttk.Button(range_frame, text=label, width=8,
                       command=lambda r=rng: self._set_range(r)).pack(side=tk.LEFT, padx=1)

        # ---- 主内容区: 绘图 + 分析面板 ----
        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        # 左侧: 绘图区
        plot_frame = ttk.Frame(main)
        main.add(plot_frame, weight=3)

        self.fig = Figure(figsize=(10, 5.5), dpi=100, facecolor="white")
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 点击图上任意位置 → 标注坐标
        self._click_annotations: list = []  # 可清除的标注对象
        self.canvas.mpl_connect("button_press_event", self._on_click)

        # matplotlib 工具栏 (缩放/平移/保存)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self.toolbar.update()

        # 右侧: 分析面板
        right = ttk.Frame(main, width=320)
        main.add(right, weight=1)

        info_label = ttk.Label(right, text="📋 分析结果", font=("Microsoft YaHei", 12, "bold"))
        info_label.pack(pady=(4, 2))

        self.info_text = tk.Text(right, font=("Consolas", 10), wrap=tk.WORD,
                                 bg="#fffff0", relief=tk.SUNKEN, borderwidth=1,
                                 width=36, height=30)
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=(0, 4))

        # 历史列表
        hist_label = ttk.Label(right, text="📜 历史记录 (点击复用)", font=("Microsoft YaHei", 10, "bold"))
        hist_label.pack(pady=(6, 2))
        self.hist_list = tk.Listbox(right, font=("Consolas", 10), height=8,
                                    relief=tk.SUNKEN, borderwidth=1)
        self.hist_list.pack(fill=tk.BOTH, expand=True, padx=2, pady=(0, 4))
        self.hist_list.bind("<Double-Button-1>", self._select_history)

        # ---- 底部状态栏 ----
        self.status_var = tk.StringVar(value="就绪")
        status = ttk.Label(self.root, textvariable=self.status_var,
                           relief=tk.SUNKEN, anchor=tk.W, font=("Microsoft YaHei", 9))
        status.pack(fill=tk.X, side=tk.BOTTOM)

        # 快捷提示
        tips = ttk.Label(self.root, text="💡 导数: sin'(x) | (sin(x))' | d/dx sin(x)  ·  方程: y=... | x=... | f(x,y)=0  ·  多函数用逗号分隔",
                         font=("Microsoft YaHei", 9), foreground="gray")
        tips.pack(fill=tk.X, side=tk.BOTTOM, padx=8, pady=(0, 2))

    # ── 核心操作 ────────────────────────────────────────────────
    def _draw(self, expr: str | None = None):
        """绘制表达式 (主线程中执行)"""
        if expr is None:
            expr = self.expr_var.get().strip()
        if not expr:
            return

        try:
            x0 = float(self.xmin_var.get())
            x1 = float(self.xmax_var.get())
        except ValueError:
            messagebox.showerror("错误", "X 范围必须是数字")
            return

        self.status_var.set(f"正在计算: {expr} ...")
        self.root.update_idletasks()

        try:
            self._click_annotations.clear()
            self.plotter.x_range = (x0, x1)
            self.plotter._analyses.clear()
            self.plotter._color_idx = 0

            # 在 tkinter Figure 上重建坐标轴
            self.fig.clear()
            self.plotter._fig = self.fig
            self.plotter._ax = self.fig.add_subplot(111)

            # 绘制 (与 CLI 完全相同的逻辑)
            if "," in expr:
                exprs = [e.strip() for e in expr.split(",")]
                self.plotter.plot_multi(exprs)
            else:
                self.plotter.plot(expr)

            # 设置四象限坐标轴、π刻度、网格、标题 (与 CLI 的 show() 相同)
            self.plotter._finalize_figure()

            # 渲染
            self.canvas.draw()
            self.toolbar.update()

            # 更新分析面板
            self._update_info()

            # 更新历史
            if expr not in self._history:
                self._history.append(expr)
                self.hist_list.insert(0, expr)

            self.status_var.set(f"完成: {expr}")
        except Exception as exc:
            self.status_var.set(f"错误: {exc}")
            messagebox.showerror("绘图错误", str(exc))

    # ── 点击标注 ────────────────────────────────────────────────
    @staticmethod
    def _format_pi(x: float) -> str:
        """将数值转为 π 分数, 仅当确实接近 π 倍数时使用 (否则返回空字符串)"""
        if abs(x) < 0.001:
            return ""
        best_n, best_d, best_err = 0, 1, 0.005  # 严格阈值
        for d in range(1, 7):  # 分母不超过 6
            n = round(x / math.pi * d)
            err = abs(n * math.pi / d - x)
            if err < best_err:
                best_err = err
                best_n, best_d = n, d
        if best_n == 0 or best_d > 6:
            return ""
        g = math.gcd(abs(best_n), best_d)
        best_n //= g; best_d //= g
        if best_d == 1:
            if best_n == 1: return "π"
            if best_n == -1: return "-π"
            return f"{best_n}π"
        if best_n == 1: return f"π/{best_d}"
        if best_n == -1: return f"-π/{best_d}"
        return f"{best_n}π/{best_d}"

    def _annotate_point(self):
        """输入框输入 x 值 → 标注点"""
        raw = self.point_var.get().strip()
        if not raw:
            return
        # 支持 π 表达式: pi/2, 2*pi, 3pi/2, 3.14 等
        try:
            s = raw.lower().replace("π", "pi").replace(" ", "")
            if s.startswith("pi") and len(s) > 2 and s[2].isdigit():
                s = "pi*" + s[2:]  # "pi2" -> "pi*2"
            import sympy as sp
            x_val = float(sp.N(sp.sympify(s, locals={"pi": sp.pi, "e": sp.E})))
        except Exception:
            self.status_var.set(f"无法解析: {raw}")
            return
        self._do_annotate(x_val)

    def _do_annotate(self, x_val: float):
        """在 x=x_val 处画十字线并标注 (x, f(x))"""
        ax = self.plotter._ax
        if ax is None:
            return

        # 清除上次标注
        for obj in self._click_annotations:
            try: obj.remove()
            except Exception: pass
        self._click_annotations.clear()

        # 用函数求 y (兼容 numpy 2.x)
        fy = None
        if self.plotter._analyses:
            for a in self.plotter._analyses:
                try:
                    result = a.f_lambda(np.array([x_val]))
                    if isinstance(result, np.ndarray):
                        result = result.item()
                    fy = float(result)
                    if np.isfinite(fy):
                        break
                except Exception:
                    continue

        # 十字虚线
        vline = ax.axvline(x=x_val, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        self._click_annotations.append(vline)
        if fy is not None:
            hline = ax.axhline(y=fy, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
            self._click_annotations.append(hline)

        # 坐标标签: x 和 y 各自独立判断 π 格式 → 精度优先
        x_pi = self._format_pi(x_val)
        x_str = f"{x_pi} ≈ {x_val:.6f}" if x_pi else f"{x_val:.6g}"
        if fy is not None:
            y_pi = self._format_pi(fy)
            y_str = f"{y_pi} ≈ {fy:.8f}" if y_pi else f"{fy:.6g}"
        else:
            y_str = "—"
        label = f"({x_str}, {y_str})"

        if fy is not None:
            offset_x = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.03
            offset_y = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05
            ann = ax.annotate(
                label,
                xy=(x_val, fy), xytext=(x_val + offset_x, fy + offset_y),
                fontsize=10, fontweight="bold", color="#cc0000",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cc0000", alpha=0.9),
                zorder=200,
            )
            self._click_annotations.append(ann)
            # 数据点
            dot = ax.plot(x_val, fy, "o", color="#cc0000", markersize=8, zorder=201)
            self._click_annotations.extend(dot)

        # X 轴投影
        dot_x = ax.plot(x_val, 0, "o", color="#cc0000", markersize=6, zorder=201, alpha=0.6)
        self._click_annotations.extend(dot_x)
        x_label = ax.annotate(
            x_str, xy=(x_val, 0), xytext=(0, -18), textcoords="offset points",
            fontsize=8, color="#cc0000", ha="center", zorder=201,
        )
        self._click_annotations.append(x_label)

        self.status_var.set(f"标注: ({x_str}, {y_str})")
        self.canvas.draw_idle()

    def _on_click(self, event):
        """点击图像时标注坐标"""
        if event.inaxes != self.plotter._ax:
            return
        if event.button != 1:
            return
        if self.toolbar.mode:
            return
        x = event.xdata
        if x is None:
            return
        self._do_annotate(x)

    def _clear(self):
        self._click_annotations.clear()
        self.plotter._analyses.clear()
        self.plotter._color_idx = 0
        self.fig.clear()
        self.plotter._ax = self.fig.add_subplot(111)
        self.canvas.draw()
        self.info_text.delete("1.0", tk.END)
        self.status_var.set("已清除")

    def _save(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG 图片", "*.png"), ("SVG 矢量图", "*.svg"), ("PDF", "*.pdf"), ("所有文件", "*.*")]
        )
        if path:
            try:
                self.fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
                self.status_var.set(f"已保存: {path}")
            except Exception as e:
                messagebox.showerror("保存失败", str(e))

    def _set_range(self, rng_str: str):
        parts = rng_str.split()
        self.xmin_var.set(parts[0])
        self.xmax_var.set(parts[1])

    # ── 分析面板 ────────────────────────────────────────────────
    def _update_info(self):
        self.info_text.delete("1.0", tk.END)
        if not self.plotter._analyses:
            self.info_text.insert(tk.END, "无分析数据")
            return

        for i, a in enumerate(self.plotter._analyses):
            if len(self.plotter._analyses) > 1:
                self.info_text.insert(tk.END, f"── f{i+1}(x) ──\n")

            # 表达式
            expr_str = a.expr_str if a.expr_str else str(a.expr).replace("**", "^")
            self.info_text.insert(tk.END, f"表达式: {expr_str}\n")

            if a.domain_info:
                self.info_text.insert(tk.END, f"定义域: {a.domain_info}\n")
            if a.period_value is not None:
                self.info_text.insert(tk.END, f"周期:   T = {a.period_value:.4f}\n")
            if a.y_intercept is not None:
                self.info_text.insert(tk.END, f"y截距:  (0, {a.y_intercept:.4f})\n")

            if a.zeros:
                z_str = ", ".join(f"x={z:.3f}" for z in a.zeros[:8])
                if len(a.zeros) > 8:
                    z_str += f" ...({len(a.zeros)}个)"
                self.info_text.insert(tk.END, f"零点:   {z_str}\n")

            if a.symmetry:
                tag = "偶函数 (关于 y 轴对称)" if a.symmetry == "even" else "奇函数 (关于原点对称)"
                self.info_text.insert(tk.END, f"对称性: {tag}\n")

            if a.extrema:
                maxes = [e for e in a.extrema if e.kind == "max"]
                mins = [e for e in a.extrema if e.kind == "min"]
                for label, lst in [("▲ 极大值", maxes), ("▼ 极小值", mins)]:
                    if lst:
                        pts = ", ".join(f"({e.x:.2f},{e.y:.2f})" for e in lst[:5])
                        self.info_text.insert(tk.END, f"{label}: {pts}\n")

            if a.inflections:
                self.info_text.insert(tk.END, "拐点:   x=" +
                                      ", ".join(f"{v:.4f}" for v in a.inflections[:6]) + "\n")

            is_T = a.period_value is not None and a.period_value > 0.1

            if a.monotonic:
                parts = []
                for iv in a.monotonic:
                    a_s = ("..." if is_T else "—∞") if iv.start is None else f"{iv.start:.3f}"
                    b_s = ("..." if is_T else "+∞") if iv.end is None else f"{iv.end:.3f}"
                    parts.append(f"({a_s},{b_s}){iv.label}")
                self.info_text.insert(tk.END, "单调性: " + " | ".join(parts[:6]) + "\n")

            if a.concavity:
                parts = []
                for iv in a.concavity:
                    a_s = ("..." if is_T else "—∞") if iv.start is None else f"{iv.start:.3f}"
                    b_s = ("..." if is_T else "+∞") if iv.end is None else f"{iv.end:.3f}"
                    parts.append(f"({a_s},{b_s}){iv.label}")
                self.info_text.insert(tk.END, "凹凸性: " + " | ".join(parts[:6]) + "\n")

            if a.range_info and a.range_info != "—":
                self.info_text.insert(tk.END, f"值域:   {a.range_info}\n")

            if a.asymptotes:
                self.info_text.insert(tk.END, "垂直渐近线: x=" +
                                      ", ".join(f"{v:.3f}" for v in a.asymptotes) + "\n")
            if a.h_asymptotes:
                self.info_text.insert(tk.END, "水平渐近线: y=" +
                                      ", ".join(f"{v:.4f}" for v in a.h_asymptotes) + "\n")
            if a.obliques:
                self.info_text.insert(tk.END, "斜渐近线: " +
                      ", ".join(f"y={m:.3f}x+{b:.3f}" for m, b in a.obliques) + "\n")

            self.info_text.insert(tk.END, "\n")

    # ── 历史记录 ────────────────────────────────────────────────
    def _select_history(self, event):
        """双击历史记录复用"""
        sel = self.hist_list.curselection()
        if sel:
            expr = self.hist_list.get(sel[0])
            self.expr_var.set(expr)
            self._draw(expr)

    def _show_help(self):
        """弹出帮助窗口"""
        help_text = """\
═══ 通用函数图像绘制工具 — 使用帮助 ═══

【基本用法】
  在表达式输入框输入函数，回车或点"绘制"即可。
  点击图像任意位置 → 标注该点坐标 (x, f(x))
  在上方"标注点 x="输入 x 值 → 精确标注指定点

【支持的函数】
  三角函数:  sin(x) cos(x) tan(x) cot(x) sec(x) csc(x)
  反三角:    asin(x) acos(x) atan(x)
  双曲:      sinh(x) cosh(x) tanh(x)
  指数对数:  exp(x) e^x log(x) ln(x) sqrt(x)
  绝对值:    abs(x) |x|
  取整:      floor(x) ceiling(x)
  特殊函数:  gamma(x) erf(x)
  常数:      pi(π) E(e) EulerGamma(γ) GoldenRatio(φ)

【导数】三种写法:
  sin'(x)        → 对 sin 求导再代入 x  (即 cos(x))
  (sin(x))'      → 整个表达式对 x 求导  (即 cos(x))
  d/dx sin(x)    → 同上, 显式写法

【方程】(支持 y= / x= / f(x,y)=0 )
  y=sin(x)       →  显函数
  x=sin(y)       →  反函数 / 隐式
  x^2+y^2=1      →  隐式方程 (圆)

【多函数对比】用逗号分隔: sin(x), cos(x), x^2

【操作技巧】
  - 鼠标滚轮缩放, 按住拖动平移 (工具栏)
  - 点击图上任意位置标注坐标 (x 和 y 自动识别 π)
  - 输入框支持 pi 表达式: pi/2, 2*pi, 3pi/2
  - 上下方向键切换历史记录
  - 预设范围按钮快速切换视图

【标注点坐标格式】
  - 普通数值: (2.0000, 4.0000) — 4 位小数
  - 接近 π 倍数: (π/2 ≈ 1.5708, 1.0000) — 含 π 分数
  - 仅当值与 π 分数误差 < 0.005 且分母 ≤ 6 时才显示 π 形式
"""
        top = tk.Toplevel(self.root)
        top.title("使用帮助")
        top.geometry("620x620")
        top.configure(bg="#ffffff")
        txt = tk.Text(top, font=("Consolas", 10), wrap=tk.WORD,
                      bg="#ffffff", relief=tk.FLAT, padx=12, pady=12)
        txt.insert("1.0", help_text)
        txt.configure(state=tk.DISABLED)
        txt.pack(fill=tk.BOTH, expand=True)
        ttk.Button(top, text="关闭", command=top.destroy).pack(pady=(0, 8))
        sel = self.hist_list.curselection()
        if sel:
            expr = self.hist_list.get(sel[0])
            self.expr_var.set(expr)
            self._draw(expr)

    def _history_up(self, event):
        if self._history:
            cur = self.expr_var.get()
            if cur in self._history:
                idx = self._history.index(cur)
                if idx < len(self._history) - 1:
                    self.expr_var.set(self._history[idx + 1])
            else:
                self.expr_var.set(self._history[-1])

    def _history_down(self, event):
        if self._history:
            cur = self.expr_var.get()
            if cur in self._history:
                idx = self._history.index(cur)
                if idx > 0:
                    self.expr_var.set(self._history[idx - 1])


# ============================================================================
# main
# ============================================================================

def main():
    # 全局抑制 numpy 运行时警告
    np.seterr(all='ignore')

    root = tk.Tk()
    app = FunctionPlotterGUI(root)

    # 设置窗口图标 (可选)
    try:
        root.iconbitmap(default="")
    except Exception:
        pass

    root.mainloop()


if __name__ == "__main__":
    main()
