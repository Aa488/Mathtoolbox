#!/usr/bin/env python3
"""
================================================================================
  专业数学分析工具箱 — Math Toolbox
================================================================================
  三模块集成: 函数分析 | 微积分 | 无穷级数
  核心引擎: function_plotter.py (分析) + calculus_engine.py (积分/级数)
================================================================================
"""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import math
import warnings

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import sympy as sp
from function_plotter import FunctionPlotter, _preprocess_expr
from calculus_engine import (CalculusEngine, SeriesAnalysis, AdvancedIntegration,
                               LimitEngine, CurveEngine, OdeEngine,
                               MultivariableEngine, LaplaceEngine, VectorEngine)
from solution_tracer import SolutionTracer, TracedCalculus
import mpl_toolkits.mplot3d  # 注册 '3d' projection
from mpl_toolkits.mplot3d import Axes3D

# ── 全局样式 ──────────────────────────────────────────────────
BG = "#f5f5f5"
FG = "#222222"
ACCENT = "#1a73e8"
FONT = ("Microsoft YaHei UI", 10)
FONT_MONO = ("Consolas", 10)
FONT_TITLE = ("Microsoft YaHei UI", 13, "bold")


def _try_trace(tab, tracer_call):
    """安全的解题追踪: 不影响主计算流程"""
    try:
        _, tr = tracer_call()
        if hasattr(tab, 'trace_cb') and tab.trace_cb:
            tab.trace_cb(tr)
    except Exception as e:
        messagebox.showwarning("追踪失败", str(e))


def _format_value(v: float) -> str:
    """将数值转为最优表达式: 优先 π 形式, 否则小数"""
    if v is None or not np.isfinite(v):
        return "—"
    # 0 特殊处理
    if abs(v) < 1e-12:
        return "0"
    # 整数值: 仅当极精确时才取整 (避免 1.999999 被误判为 2)
    rv = round(v)
    if abs(v - rv) < 1e-13:
        return str(rv)
    # sympy.nsimplify 精确匹配 + 手动 π 分数逼近 (混合策略)
    try:
        import sympy as _sp
        expr = _sp.nsimplify(v, [_sp.pi, _sp.E, _sp.log(2), _sp.sqrt(_sp.pi)],
                             tolerance=1e-8, full=True)
        s = str(expr).replace("**", "^").replace("*", "")
        has_const = any(c in s for c in ['pi', 'PI', 'E', 'log'])
        if not has_const and s.replace('.','').replace('-','').isdigit():
            if abs(float(s) - v) > 1e-12:
                s = None
        if s and s != str(v) and len(s) < 30:
            return f"{s} ≈ {v:.6f}"
    except Exception:
        pass
    # 回落: 手动 π/π² 分数逼近
    best_n, best_d, best_err = 0, 1, 5e-5
    for d in range(1, 13):
        n = round(v / np.pi * d)
        err = abs(n * np.pi / d - v)
        if err < best_err:
            best_err, best_n, best_d = err, n, d
    if best_n != 0 and best_d <= 12:
        g = math.gcd(abs(best_n), best_d)
        best_n //= g; best_d //= g
        if best_d == 1:
            s = "π" if best_n == 1 else ("-π" if best_n == -1 else f"{best_n}π")
        else:
            s = f"π/{best_d}" if best_n == 1 else (f"-π/{best_d}" if best_n == -1 else f"{best_n}π/{best_d}")
        return f"{s} ≈ {v:.6f}"
    return f"{v:.8g}"


def _parse_bound(s: str) -> float:
    """解析积分上下界, 支持 pi, e, sin(pi/2) 等表达式"""
    s = s.strip()
    if not s:
        raise ValueError("空值")
    # 先试纯数字
    try:
        return float(s)
    except ValueError:
        pass
    # 再试 sympy 表达式
    try:
        import sympy as sp
        t = s.lower().replace("π", "pi").replace("^", "**")
        val = sp.N(sp.sympify(t, locals={
            "pi": sp.pi, "e": sp.E, "E": sp.E,
            "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
            "log": sp.log, "ln": sp.log, "exp": sp.exp,
            "sqrt": sp.sqrt, "abs": sp.Abs,
        }))
        return float(val)
    except Exception as e:
        raise ValueError(f"无法解析 '{s}': {e}")


def _make_figure(figsize=(9, 5)) -> tuple[Figure, object]:
    """创建 matplotlib Figure 并返回 (fig, ax)"""
    fig = Figure(figsize=figsize, dpi=100, facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#fafafa")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.axhline(y=0, color="gray", linewidth=0.8, alpha=0.6)
    ax.axvline(x=0, color="gray", linewidth=0.8, alpha=0.6)
    return fig, ax


# ============================================================================
# Tab 1: 函数分析
# ============================================================================

class FunctionAnalysisTab(ttk.Frame):
    """移植 function_plotter_gui 全部功能"""

    trace_cb = None
    def __init__(self, parent, root):
        super().__init__(parent)
        self.root = root
        self.plotter = FunctionPlotter(lang="cn", x_range=(-8, 8), figsize=(9, 5), dpi=100)
        self._history: list[str] = []
        self._anno = None
        self._build()

    def _build(self):
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # ── 左侧控制面板 ──
        left = ttk.Frame(paned, width=300)
        paned.add(left, weight=0)

        # 工具栏
        toolbar = ttk.Frame(left)
        toolbar.pack(fill=tk.X, padx=4, pady=(4, 2))
        self.expr_var = tk.StringVar(value="sin(x)")
        ttk.Label(toolbar, text="f(x)=", font=FONT).pack(side=tk.LEFT)
        self.entry = ttk.Entry(toolbar, textvariable=self.expr_var, font=FONT_MONO, width=18)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self.entry.bind("<Return>", lambda e: self._draw())
        self.entry.bind("<Up>", self._hist_up)
        self.entry.bind("<Down>", self._hist_down)

        ttk.Button(toolbar, text="绘图", command=self._draw).pack(side=tk.LEFT, padx=1)
        ttk.Button(toolbar, text="解题", command=self._show_trace).pack(side=tk.LEFT, padx=1)
        ttk.Button(toolbar, text="清除", command=self._clear).pack(side=tk.LEFT, padx=1)

        # 标注 X=
        frame_x = ttk.Frame(left)
        frame_x.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(frame_x, text="标注点 x=", font=FONT).pack(side=tk.LEFT)
        self.anno_var = tk.StringVar()
        self.anno_entry = ttk.Entry(frame_x, textvariable=self.anno_var, font=FONT_MONO, width=10)
        self.anno_entry.pack(side=tk.LEFT, padx=2)
        self.anno_entry.bind("<Return>", lambda e: self._do_annotate())
        ttk.Button(frame_x, text="标注", command=self._do_annotate).pack(side=tk.LEFT)
        ttk.Button(frame_x, text="✕", width=2, command=self._clear_anno).pack(side=tk.LEFT, padx=1)

        # 预设范围
        frame_r = ttk.Frame(left)
        frame_r.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(frame_r, text="范围:", font=FONT).pack(side=tk.LEFT)
        for label, rng in [("-2π,2π", (-6.28, 6.28)), ("-5,5", (-5, 5)),
                            ("-10,10", (-10, 10)), ("-1,1", (-1, 1))]:
            ttk.Button(frame_r, text=label, width=7,
                       command=lambda r=rng: self._set_range(r)).pack(side=tk.LEFT, padx=1)

        # ── 极限 / 参数 / 极坐标 ──
        ext_lab = ttk.LabelFrame(left, text=" 更多功能 ", padding=4)
        ext_lab.pack(fill=tk.X, padx=4, pady=2)
        frm_ext = ttk.Frame(ext_lab)
        frm_ext.pack(fill=tk.X)
        ttk.Label(frm_ext, text="极限 x→", font=FONT).pack(side=tk.LEFT)
        self.lim_var = tk.StringVar(value="0")
        ttk.Entry(frm_ext, textvariable=self.lim_var, font=FONT_MONO, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Button(frm_ext, text="计算极限", command=self._compute_limit).pack(side=tk.LEFT, padx=2)

        # 坐标模式
        frm_mode = ttk.Frame(ext_lab)
        frm_mode.pack(fill=tk.X, pady=(4, 0))
        self.coord_mode = tk.StringVar(value="cartesian")
        for txt, val in [("直角 y=f(x)", "cartesian"), ("参数 (x(t),y(t))", "parametric"),
                          ("极坐标 r=f(θ)", "polar")]:
            ttk.Radiobutton(frm_mode, text=txt, value=val, variable=self.coord_mode,
                            command=self._switch_coord).pack(anchor=tk.W)

        # 参数/极坐标输入
        self.param_frame = ttk.Frame(ext_lab)
        ttk.Label(self.param_frame, text="x(t)=", font=FONT).pack(side=tk.LEFT)
        self.xt_var = tk.StringVar(value="cos(t)")
        ttk.Entry(self.param_frame, textvariable=self.xt_var, font=FONT_MONO, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(self.param_frame, text="y(t)=", font=FONT).pack(side=tk.LEFT)
        self.yt_var = tk.StringVar(value="sin(t)")
        ttk.Entry(self.param_frame, textvariable=self.yt_var, font=FONT_MONO, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(self.param_frame, text="t∈", font=("Microsoft YaHei UI", 7)).pack(side=tk.LEFT)
        self.t0_var = tk.StringVar(value="0")
        ttk.Entry(self.param_frame, textvariable=self.t0_var, font=FONT_MONO, width=4).pack(side=tk.LEFT)
        self.t1_var = tk.StringVar(value="6.283")
        ttk.Entry(self.param_frame, textvariable=self.t1_var, font=FONT_MONO, width=5).pack(side=tk.LEFT)

        self.polar_frame = ttk.Frame(ext_lab)
        ttk.Label(self.polar_frame, text="r(θ)=", font=FONT).pack(side=tk.LEFT)
        self.r_var = tk.StringVar(value="1+cos(theta)")
        ttk.Entry(self.polar_frame, textvariable=self.r_var, font=FONT_MONO, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Label(self.polar_frame, text="θ∈", font=("Microsoft YaHei UI", 7)).pack(side=tk.LEFT)
        self.th0_var = tk.StringVar(value="0")
        ttk.Entry(self.polar_frame, textvariable=self.th0_var, font=FONT_MONO, width=4).pack(side=tk.LEFT)
        self.th1_var = tk.StringVar(value="6.283")
        ttk.Entry(self.polar_frame, textvariable=self.th1_var, font=FONT_MONO, width=5).pack(side=tk.LEFT)

        # 分析信息
        info_lab = ttk.LabelFrame(left, text=" 分析结果 ", padding=4)
        info_lab.pack(fill=tk.BOTH, expand=True, padx=4, pady=(4, 2))
        self.info_text = tk.Text(info_lab, font=FONT_MONO, wrap=tk.WORD, height=16,
                                 bg="#ffffff", relief=tk.FLAT, padx=4, pady=4)
        self.info_text.pack(fill=tk.BOTH, expand=True)

        # 历史
        hist_lab = ttk.LabelFrame(left, text=" 历史 ", padding=2)
        hist_lab.pack(fill=tk.BOTH, padx=4, pady=(2, 4))
        self.hist_list = tk.Listbox(hist_lab, font=FONT_MONO, height=4, relief=tk.FLAT)
        self.hist_list.pack(fill=tk.BOTH, expand=True)
        self.hist_list.bind("<Double-Button-1>", self._select_history)

        # ── 右侧画布 ──
        right = ttk.Frame(paned)
        paned.add(right, weight=1)
        self.fig, self.ax = _make_figure()
        self.canvas = FigureCanvasTkAgg(self.fig, right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # 把 plotter 绑定到 canvas 的 figure, 否则画在内部 fig 上不可见
        self.plotter._fig = self.fig
        self.plotter._ax = self.ax
        self.fig.tight_layout()
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.draw()

        # 保存按钮
        ttk.Button(right, text="💾 保存图像", command=self._save).pack(anchor=tk.E, padx=4, pady=2)

    # ── 操作 ──
    def _draw(self, expr_str: str | None = None):
        expr = expr_str or self.expr_var.get().strip()
        if not expr:
            return
        try:
            self.plotter.clear()
            self.plotter._fig = self.fig
            self.plotter._ax = self.ax
            self.plotter.plot(expr)
            self._update_info()
            self._update_canvas()
            if expr not in self._history:
                self._history.append(expr)
                self.hist_list.insert(tk.END, expr)
        except Exception as e:
            messagebox.showerror("解析错误", str(e))

    def _update_info(self):
        self.info_text.delete("1.0", tk.END)
        if not self.plotter._analyses:
            self.info_text.insert(tk.END, "无分析数据")
            return
        for i, a in enumerate(self.plotter._analyses):
            if len(self.plotter._analyses) > 1:
                self.info_text.insert(tk.END, f"── f{i+1}(x) ──\n")
            es = a.expr_str if a.expr_str else str(a.expr).replace("**", "^")
            self.info_text.insert(tk.END, f"表达式: {es}\n")
            if a.domain_info:
                self.info_text.insert(tk.END, f"定义域: {a.domain_info}\n")
            if a.period_value:
                self.info_text.insert(tk.END, f"周期:   T={a.period_value:.4f}\n")
            if a.y_intercept is not None:
                self.info_text.insert(tk.END, f"y截距:  (0, {a.y_intercept:.4f})\n")
            if a.symmetry:
                tag = "偶(关于y轴)" if a.symmetry == "even" else "奇(关于原点)"
                self.info_text.insert(tk.END, f"对称性: {tag}\n")
            if a.zeros:
                zs = ", ".join(f"x={z:.3f}" for z in a.zeros[:8])
                self.info_text.insert(tk.END, f"零点:   {zs}\n")
            if a.extrema:
                for label, kind in [("▲ 极大", "max"), ("▼ 极小", "min")]:
                    pts = [e for e in a.extrema if e.kind == kind]
                    if pts:
                        s = ", ".join(f"({e.x:.2f},{e.y:.2f})" for e in pts[:4])
                        self.info_text.insert(tk.END, f"{label}:  {s}\n")
            if a.inflections:
                s = ", ".join(f"x={v:.4f}" for v in a.inflections[:6])
                self.info_text.insert(tk.END, f"拐点:   {s}\n")
            is_T = a.period_value is not None and a.period_value > 0.1
            if a.monotonic:
                parts = []
                for iv in a.monotonic:
                    a_s = ("..." if is_T else "-∞") if iv.start is None else f"{iv.start:.3f}"
                    b_s = ("..." if is_T else "+∞") if iv.end is None else f"{iv.end:.3f}"
                    parts.append(f"({a_s},{b_s}){iv.label}")
                self.info_text.insert(tk.END, f"单调性: {' | '.join(parts[:5])}\n")
            if a.concavity:
                parts = []
                for iv in a.concavity:
                    a_s = ("..." if is_T else "-∞") if iv.start is None else f"{iv.start:.3f}"
                    b_s = ("..." if is_T else "+∞") if iv.end is None else f"{iv.end:.3f}"
                    parts.append(f"({a_s},{b_s}){iv.label}")
                self.info_text.insert(tk.END, f"凹凸性: {' | '.join(parts[:5])}\n")
            if a.range_info and a.range_info != "—":
                self.info_text.insert(tk.END, f"值域:   {a.range_info}\n")
            if a.asymptotes:
                self.info_text.insert(tk.END, "垂直渐近: x=" +
                                      ", ".join(f"{v:.3f}" for v in a.asymptotes) + "\n")
            if a.h_asymptotes:
                self.info_text.insert(tk.END, "水平渐近: y=" +
                                      ", ".join(f"{v:.4f}" for v in a.h_asymptotes) + "\n")
            if a.obliques:
                self.info_text.insert(tk.END, "斜渐近: " +
                      ", ".join(f"y={m:.3f}x+{b:.3f}" for m, b in a.obliques) + "\n")
            self.info_text.insert(tk.END, "\n")

    def _update_canvas(self):
        self.canvas.draw()

    def _show_trace(self):
        """手动弹出解题过程窗口"""
        if not self.plotter._analyses:
            return
        try:
            from solution_tracer import TracedCalculus
            expr = self.expr_var.get().strip()
            _, T = TracedCalculus.function_analysis(expr, self.plotter._analyses[0])
            if hasattr(self, 'trace_cb') and self.trace_cb:
                self.trace_cb(T)
        except Exception as e:
            messagebox.showerror("错误", f"解题追踪失败: {e}")

    def _clear(self):
        self.plotter.clear()
        self.ax.clear()
        self.ax.set_facecolor("#fafafa")
        self.fig.tight_layout()
        self.canvas.draw()
        self.info_text.delete("1.0", tk.END)
        if self._anno:
            try:
                self._anno.remove()
            except Exception:
                pass
            self._anno = None

    def _set_range(self, rng):
        self.plotter.x_range = rng
        self._draw()

    def _save(self):
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("All", "*.*")])
        if path:
            self.fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
            messagebox.showinfo("保存", f"已保存到 {path}")

    # ── 标注 ──
    def _do_annotate(self):
        raw = self.anno_var.get().strip()
        if not raw:
            return
        try:
            import sympy as sp
            s = raw.lower().replace("π", "pi").replace(" ", "")
            if s.startswith("pi") and len(s) > 2 and s[2].isdigit():
                s = "pi*" + s[2:]
            x_val = float(sp.N(sp.sympify(s, locals={"pi": sp.pi, "e": sp.E})))
        except Exception:
            try:
                x_val = float(raw)
            except ValueError:
                messagebox.showerror("错误", f"无法解析 '{raw}'\n支持: 2, pi/2, 3.14")
                return

        self.plotter._ax = self.ax
        self.plotter._fig = self.fig
        if self._anno:
            try:
                self._anno.remove()
            except Exception:
                pass
            self._anno = None

        # 用引擎求 y
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

        # 格式 x
        x_pi = self._format_pi(x_val)
        x_str = f"{x_pi} ≈ {x_val:.4f}" if x_pi else f"{x_val:.4f}"
        # 格式 y
        if fy is not None:
            y_pi = self._format_pi(fy)
            y_str = f"{y_pi} ≈ {fy:.4f}" if y_pi else f"{fy:.4f}"
        else:
            y_str = "—"

        self._anno = self.ax.annotate(
            f"({x_str}, {y_str})",
            xy=(x_val, fy if fy is not None else 0),
            xytext=(x_val + 1, (fy if fy is not None else 0) + 1),
            fontsize=9, color="darkblue", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
        )
        self.fig.tight_layout()
        self.canvas.draw()

    def _clear_anno(self):
        if self._anno:
            try:
                self._anno.remove()
            except Exception:
                pass
            self._anno = None
            self.fig.tight_layout()
            self.canvas.draw()
        self.anno_var.set("")

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.button != 1:
            return
        if event.xdata is None:
            return
        self.anno_var.set(f"{event.xdata:.6g}")
        self._do_annotate()

    @staticmethod
    def _format_pi(x: float) -> str:
        if abs(x) < 0.001:
            return ""
        best_n, best_d, best_err = 0, 1, 0.005
        for d in range(1, 7):
            n = round(x / math.pi * d)
            err = abs(n * math.pi / d - x)
            if err < best_err:
                best_err = err
                best_n, best_d = n, d
        if best_n == 0 or best_d > 6:
            return ""
        g = math.gcd(abs(best_n), best_d)
        best_n //= g
        best_d //= g
        if best_d == 1:
            if best_n == 1:
                return "π"
            if best_n == -1:
                return "-π"
            return f"{best_n}π"
        if best_n == 1:
            return f"π/{best_d}"
        if best_n == -1:
            return f"-π/{best_d}"
        return f"{best_n}π/{best_d}"

    # ── 极限 / 参数 / 极坐标 ──
    def _switch_coord(self):
        mode = self.coord_mode.get()
        self.param_frame.pack_forget()
        self.polar_frame.pack_forget()
        if mode == "parametric":
            self.param_frame.pack(fill=tk.X, pady=2)
        elif mode == "polar":
            self.polar_frame.pack(fill=tk.X, pady=2)

    def _compute_limit(self):
        expr = self.expr_var.get().strip()
        if not expr:
            return
        try:
            a = _parse_bound(self.lim_var.get())
        except ValueError:
            messagebox.showerror("错误", "极限点必须是数字")
            return
        r = LimitEngine.compute(expr, a)
        # 解题过程追踪
        try:
            _, T = TracedCalculus.limit(expr, a)
            if hasattr(self, 'trace_cb') and self.trace_cb:
                self.trace_cb(T)
        except Exception:
            pass
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, f"极限: lim(x→{a}) {expr}\n\n")
        if r["exists"]:
            self.info_text.insert(tk.END, f"极限值 = {_format_value(r['value'])}\n")
        else:
            self.info_text.insert(tk.END, f"极限不存在\n")
            if r["left"] is not None:
                self.info_text.insert(tk.END, f"左极限: {r['left']:.8f}\n")
            if r["right"] is not None:
                self.info_text.insert(tk.END, f"右极限: {r['right']:.8f}\n")
        if r["symbolic"] and r["symbolic"] != str(r.get("value", "")):
            self.info_text.insert(tk.END, f"符号: {r['symbolic']}\n")

    def _draw(self, expr_str: str | None = None):
        mode = self.coord_mode.get()
        if mode == "parametric":
            self._draw_parametric()
            return
        elif mode == "polar":
            self._draw_polar()
            return
        # 正常函数绘制
        expr = expr_str or self.expr_var.get().strip()
        if not expr:
            return
        try:
            self.plotter.clear()
            self.plotter._fig = self.fig
            self.plotter._ax = self.ax
            self.plotter.plot(expr)
            self._update_info()
            self._update_canvas()
            if expr not in self._history:
                self._history.append(expr)
                self.hist_list.insert(tk.END, expr)
        except Exception as e:
            messagebox.showerror("解析错误", str(e))

    def _draw_parametric(self):
        xt = self.xt_var.get().strip(); yt = self.yt_var.get().strip()
        if not xt or not yt:
            return
        try:
            t0 = _parse_bound(self.t0_var.get()); t1 = _parse_bound(self.t1_var.get())
        except ValueError:
            messagebox.showerror("错误", "t 范围必须是数字")
            return
        data = CurveEngine.parametric_curve(xt, yt, (t0, t1))
        _try_trace(self, lambda: TracedCalculus.parametric_curve(xt, yt, (t0, t1)))
        self.fig.clear(); self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#fafafa"); self.ax.grid(True, alpha=0.3, linestyle="--")
        self.ax.axhline(y=0, color="gray", linewidth=0.8, alpha=0.5)
        self.ax.axvline(x=0, color="gray", linewidth=0.8, alpha=0.5)
        self.ax.plot(data["xs"], data["ys"], color="#1f77b4", linewidth=2)
        self.ax.set_xlabel("x"); self.ax.set_ylabel("y")
        self.ax.set_title(f"({xt}, {yt}), t∈[{t0},{t1}]")
        self.fig.tight_layout(); self.canvas.draw()

        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, f"参数曲线: x(t)={xt}, y(t)={yt}\n")
        self.info_text.insert(tk.END, f"t ∈ [{t0}, {t1}]\n")
        if data["arc_length"]:
            self.info_text.insert(tk.END, f"弧长: {data['arc_length']:.6f}\n")

    def _draw_polar(self):
        r_expr = self.r_var.get().strip()
        if not r_expr:
            return
        try:
            th0 = _parse_bound(self.th0_var.get()); th1 = _parse_bound(self.th1_var.get())
        except ValueError:
            messagebox.showerror("错误", "θ 范围必须是数字")
            return
        data = CurveEngine.polar_curve(r_expr, (th0, th1))
        _try_trace(self, lambda: TracedCalculus.polar_curve(r_expr, (th0, th1)))
        self.fig.clear(); self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#fafafa"); self.ax.grid(True, alpha=0.3, linestyle="--")
        self.ax.axhline(y=0, color="gray", linewidth=0.8, alpha=0.5)
        self.ax.axvline(x=0, color="gray", linewidth=0.8, alpha=0.5)
        self.ax.plot(data["xs"], data["ys"], color="#d62728", linewidth=2)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("x"); self.ax.set_ylabel("y")
        self.ax.set_title(f"r = {r_expr}, θ∈[{th0},{th1}]")
        self.fig.tight_layout(); self.canvas.draw()

        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, f"极坐标: r(θ) = {r_expr}\n")
        self.info_text.insert(tk.END, f"θ ∈ [{th0}, {th1}]\n")
        if data["area"]:
            self.info_text.insert(tk.END, f"面积: ½∫r²dθ = {data['area']:.6f}\n")

    # ── 历史 ──
    def _hist_up(self, event):
        if self._history:
            idx = max(0, self._history.index(self.expr_var.get()) - 1) if self.expr_var.get() in self._history else -1
            self.expr_var.set(self._history[idx])

    def _hist_down(self, event):
        if self._history:
            idx = self._history.index(self.expr_var.get()) + 1 if self.expr_var.get() in self._history else len(self._history)
            if idx < len(self._history):
                self.expr_var.set(self._history[idx])

    def _select_history(self, event):
        sel = self.hist_list.curselection()
        if sel:
            self.expr_var.set(self.hist_list.get(sel[0]))
            self._draw()


# ============================================================================
# Tab 2: 微积分 (含子标签)
# ============================================================================

class CalculusTab(ttk.Frame):
    """定积分 / 反常积分 / 多重积分 / 数值方法"""

    trace_cb = None
    def __init__(self, parent, root):
        super().__init__(parent)
        self.root = root
        self._build()

    def _build(self):
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # ── 左侧 ──
        left = ttk.Frame(paned, width=310)
        paned.add(left, weight=0)

        ttk.Label(left, text="∫ 微积分", font=FONT_TITLE).pack(anchor=tk.W, padx=8, pady=(8, 4))
        frm = ttk.Frame(left)
        frm.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(frm, text="f(x)=", font=FONT).pack(side=tk.LEFT)
        self.expr_var = tk.StringVar(value="sin(x)")
        self.entry = ttk.Entry(frm, textvariable=self.expr_var, font=FONT_MONO, width=18)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # 区间
        frm2 = ttk.Frame(left)
        frm2.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(frm2, text="a =", font=FONT).pack(side=tk.LEFT)
        self.a_var = tk.StringVar(value="0")
        ttk.Entry(frm2, textvariable=self.a_var, font=FONT_MONO, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(frm2, text="  b =", font=FONT).pack(side=tk.LEFT)
        self.b_var = tk.StringVar(value="3.14159")
        ttk.Entry(frm2, textvariable=self.b_var, font=FONT_MONO, width=7).pack(side=tk.LEFT, padx=2)
        ttk.Label(frm2, text="  (b可留空表示∞)", font=("Microsoft YaHei UI", 7)).pack(side=tk.LEFT)

        # 子标签
        sub_nb = ttk.Notebook(left)
        sub_nb.pack(fill=tk.BOTH, expand=True, padx=2, pady=4)

        # -- 子1: 定积分 --
        s1 = ttk.Frame(sub_nb)
        sub_nb.add(s1, text="定/不定积分")
        self._build_sub_definite(s1)

        # -- 子2: 反常积分 --
        s2 = ttk.Frame(sub_nb)
        sub_nb.add(s2, text="反常积分")
        self._build_sub_improper(s2)

        # -- 子3: 多重积分 --
        s3 = ttk.Frame(sub_nb)
        sub_nb.add(s3, text="多重积分")
        self._build_sub_multiple(s3)

        # -- 子4: 数值对比 --
        s4 = ttk.Frame(sub_nb)
        sub_nb.add(s4, text="数值对比")
        self._build_sub_numerical(s4)

        # -- 子5: 弧长曲率 --
        s5 = ttk.Frame(sub_nb)
        sub_nb.add(s5, text="弧长 & 曲率")
        self._build_sub_arc(s5)

        # -- 子6: 三重积分 --
        s6 = ttk.Frame(sub_nb)
        sub_nb.add(s6, text="三重积分")
        self._build_sub_triple(s6)

        # 输出
        out_lab = ttk.LabelFrame(left, text=" 计算结果 ", padding=4)
        out_lab.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)
        self.out_text = tk.Text(out_lab, font=FONT_MONO, wrap=tk.WORD, height=8,
                                bg="#ffffff", relief=tk.FLAT, padx=4, pady=4)
        self.out_text.pack(fill=tk.BOTH, expand=True)

        # ── 右侧画布 ──
        right = ttk.Frame(paned)
        paned.add(right, weight=1)
        self.fig, self.ax = _make_figure()
        self.canvas = FigureCanvasTkAgg(self.fig, right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.fig.tight_layout()
        self.canvas.draw()
        ttk.Button(right, text="💾 保存", command=self._save).pack(anchor=tk.E, padx=4, pady=2)

    def _build_sub_definite(self, parent):
        frm = ttk.Frame(parent)
        frm.pack(fill=tk.BOTH, padx=4, pady=4)
        ttk.Button(frm, text="定积分 ∫[a,b] fdx", command=self._definite).pack(fill=tk.X, pady=2)
        ttk.Button(frm, text="不定积分 F(x)", command=self._indefinite).pack(fill=tk.X, pady=2)
        ttk.Button(frm, text="黎曼和可视化", command=self._riemann).pack(fill=tk.X, pady=2)

        # 黎曼参数
        frm2 = ttk.Frame(frm)
        frm2.pack(fill=tk.X, pady=2)
        ttk.Label(frm2, text="n:", font=FONT).pack(side=tk.LEFT)
        self.n_var = tk.IntVar(value=10)
        ttk.Scale(frm2, from_=1, to=100, variable=self.n_var, orient=tk.HORIZONTAL,
                  command=lambda v: self._update_n_label()).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self.n_label = ttk.Label(frm2, text="10", font=FONT, width=3)
        self.n_label.pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value="midpoint")
        ttk.Combobox(frm2, textvariable=self.mode_var, width=10, state="readonly",
                     values=["left", "right", "midpoint", "trapezoid"]).pack(side=tk.LEFT, padx=2)

        # 面积对比
        frm3 = ttk.Frame(frm)
        frm3.pack(fill=tk.X, pady=2)
        ttk.Label(frm3, text="g(x)=", font=FONT).pack(side=tk.LEFT)
        self.g_var = tk.StringVar(value="")
        ttk.Entry(frm3, textvariable=self.g_var, font=FONT_MONO, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(frm3, text="面积对比", command=self._area_between).pack(side=tk.LEFT)

    def _build_sub_improper(self, parent):
        frm = ttk.Frame(parent)
        frm.pack(fill=tk.BOTH, padx=4, pady=4)
        ttk.Label(frm, text="反常积分: b留空=+∞, 或设瑕点",
                  font=("Microsoft YaHei UI", 8), wraplength=260).pack(pady=2)
        ttk.Label(frm, text="瑕点 xₛ =", font=FONT).pack()
        self.sing_var = tk.StringVar(value="")
        ttk.Entry(frm, textvariable=self.sing_var, font=FONT_MONO, width=12).pack(pady=2)
        ttk.Button(frm, text="计算反常积分", command=self._improper).pack(fill=tk.X, pady=4)

    def _build_sub_multiple(self, parent):
        frm = ttk.Frame(parent)
        frm.pack(fill=tk.BOTH, padx=4, pady=4)
        ttk.Label(frm, text="二重积分 ∬ f(x,y) dxdy",
                  font=("Microsoft YaHei UI", 9, "bold")).pack(pady=2)
        ttk.Label(frm, text="被积函数 f(x,y)=", font=FONT).pack()
        self.mf_var = tk.StringVar(value="x*y")
        ttk.Entry(frm, textvariable=self.mf_var, font=FONT_MONO, width=14).pack(pady=2)
        ttk.Label(frm, text="x: 下界= 上界=", font=FONT).pack()
        frm_m1 = ttk.Frame(frm)
        frm_m1.pack()
        self.mx_low = tk.StringVar(value="0")
        self.mx_high = tk.StringVar(value="1")
        ttk.Entry(frm_m1, textvariable=self.mx_low, font=FONT_MONO, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Entry(frm_m1, textvariable=self.mx_high, font=FONT_MONO, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(frm, text="y: 下界= 上界=", font=FONT).pack()
        frm_m2 = ttk.Frame(frm)
        frm_m2.pack()
        self.my_low = tk.StringVar(value="0")
        self.my_high = tk.StringVar(value="1")
        ttk.Entry(frm_m2, textvariable=self.my_low, font=FONT_MONO, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Entry(frm_m2, textvariable=self.my_high, font=FONT_MONO, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(frm, text="(可用 y 的表达式, 如 0, y^2)",
                  font=("Microsoft YaHei UI", 7)).pack()
        ttk.Button(frm, text="计算二重积分", command=self._double_integral).pack(fill=tk.X, pady=4)

    def _build_sub_numerical(self, parent):
        frm = ttk.Frame(parent)
        frm.pack(fill=tk.BOTH, padx=4, pady=4)
        ttk.Label(frm, text="对比多种数值积分方法",
                  font=("Microsoft YaHei UI", 9, "bold")).pack(pady=2)
        ttk.Button(frm, text="对比方法 (梯形/Simpson/Gauss)",
                   command=self._numerical_compare).pack(fill=tk.X, pady=4)
        ttk.Label(frm, text=f"分段数: {self.n_var.get()}", font=FONT).pack()
        self._update_n_label()

    def _build_sub_arc(self, parent):
        frm = ttk.Frame(parent)
        frm.pack(fill=tk.BOTH, padx=4, pady=4)
        ttk.Label(frm, text="弧长 & 曲率",
                  font=("Microsoft YaHei UI", 9, "bold")).pack(pady=2)
        ttk.Label(frm, text="弧长: ∫√(1+f'²)dx", font=("Microsoft YaHei UI", 8)).pack()
        ttk.Button(frm, text="计算弧长", command=self._arc_length).pack(fill=tk.X, pady=2)
        ttk.Label(frm, text="曲率: κ = |f''|/(1+f'²)^(3/2)", font=("Microsoft YaHei UI", 8)).pack()
        frm_c = ttk.Frame(frm)
        frm_c.pack(fill=tk.X, pady=2)
        ttk.Label(frm_c, text="在 x =", font=FONT).pack(side=tk.LEFT)
        self.curv_x_var = tk.StringVar(value="0")
        ttk.Entry(frm_c, textvariable=self.curv_x_var, font=FONT_MONO, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Button(frm_c, text="计算曲率", command=self._curvature).pack(side=tk.LEFT)

    def _build_sub_triple(self, parent):
        frm = ttk.Frame(parent)
        frm.pack(fill=tk.BOTH, padx=4, pady=4)
        ttk.Label(frm, text="三重积分 ∭ f(x,y,z) dV",
                  font=("Microsoft YaHei UI", 9, "bold")).pack(pady=2)
        ttk.Label(frm, text="被积函数 f(x,y,z)=", font=FONT).pack()
        self.triple_var = tk.StringVar(value="x*y*z")
        ttk.Entry(frm, textvariable=self.triple_var, font=FONT_MONO, width=14).pack(pady=2)
        ttk.Label(frm, text="x: 下界= 上界=", font=FONT).pack()
        frm_t1 = ttk.Frame(frm); frm_t1.pack()
        self.tx_low = tk.StringVar(value="0"); self.tx_high = tk.StringVar(value="1")
        ttk.Entry(frm_t1, textvariable=self.tx_low, font=FONT_MONO, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Entry(frm_t1, textvariable=self.tx_high, font=FONT_MONO, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(frm, text="y: 下界= 上界=", font=FONT).pack()
        frm_t2 = ttk.Frame(frm); frm_t2.pack()
        self.ty_low = tk.StringVar(value="0"); self.ty_high = tk.StringVar(value="1")
        ttk.Entry(frm_t2, textvariable=self.ty_low, font=FONT_MONO, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Entry(frm_t2, textvariable=self.ty_high, font=FONT_MONO, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(frm, text="z: 下界= 上界=", font=FONT).pack()
        frm_t3 = ttk.Frame(frm); frm_t3.pack()
        self.tz_low = tk.StringVar(value="0"); self.tz_high = tk.StringVar(value="1")
        ttk.Entry(frm_t3, textvariable=self.tz_low, font=FONT_MONO, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Entry(frm_t3, textvariable=self.tz_high, font=FONT_MONO, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Button(frm, text="计算三重积分", command=self._triple_integral).pack(fill=tk.X, pady=4)

    def _update_n_label(self):
        self.n_label.config(text=str(self.n_var.get()))

    def _get_params(self):
        expr = self.expr_var.get().strip()
        try:
            a = _parse_bound(self.a_var.get())
        except ValueError:
            raise ValueError("a 必须是数字")
        b_str = self.b_var.get().strip()
        b = _parse_bound(b_str) if b_str else None
        if b is not None and a >= b:
            raise ValueError("需要 a < b")
        return expr, a, b

    def _get_exact(self):
        try:
            expr, a, b = self._get_params()
            if b is not None:
                r = CalculusEngine.integrate_definite(expr, a, b)
                return r.get("numeric")
        except Exception:
            pass
        return None

    def _plot_func(self, expr_str: str, a: float, b: float, label: str = "",
                   color: str = "#1f77b4", n_pts: int = 400):
        from function_plotter import _preprocess_expr
        import sympy as sp
        x_sym = sp.Symbol('x')
        s = _preprocess_expr(expr_str)
        loc = {"x": x_sym, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "cot": sp.cot,
               "sec": sp.sec, "csc": sp.csc, "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
               "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
               "log": sp.log, "ln": sp.log, "exp": sp.exp, "sqrt": sp.sqrt, "abs": sp.Abs,
               "pi": sp.pi, "e": sp.E, "E": sp.E}
        expr = sp.sympify(s, locals=loc)
        f = sp.lambdify(x_sym, expr, "numpy")
        margin = (b - a) * 0.2 if b is not None else 2.0
        a_plot = a - margin
        b_plot = (b if b is not None else a + 10) + margin
        xs = np.linspace(a_plot, b_plot, n_pts)
        ys = f(xs)
        ys = np.where(np.isfinite(ys), ys, np.nan)
        self.ax.plot(xs, ys, color=color, linewidth=2, label=label or expr_str)
        self.ax.set_xlim(a_plot, b_plot)
        all_ok = ys[np.isfinite(ys)]
        if len(all_ok) > 0:
            ym = max(abs(float(np.min(all_ok))), abs(float(np.max(all_ok)))) * 0.2 + 0.5
            self.ax.set_ylim(float(np.min(all_ok)) - ym, float(np.max(all_ok)) + ym)

    def _clear_ax(self):
        self.ax.clear()
        self.ax.set_facecolor("#fafafa")
        self.ax.grid(True, alpha=0.3, linestyle="--")
        self.ax.axhline(y=0, color="gray", linewidth=0.8, alpha=0.6)
        self.ax.axvline(x=0, color="gray", linewidth=0.8, alpha=0.6)

    # ── 定积分 ──
    def _definite(self):
        try:
            expr, a, b = self._get_params()
            if b is None:
                self._improper()
                return
        except ValueError as e:
            messagebox.showerror("参数错误", str(e))
            return
        self._clear_ax()
        self._plot_func(expr, a, b)
        from function_plotter import _preprocess_expr
        import sympy as sp
        x_sym = sp.Symbol('x')
        s = _preprocess_expr(expr)
        loc = {"x": x_sym, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
               "log": sp.log, "exp": sp.exp, "sqrt": sp.sqrt, "abs": sp.Abs, "pi": sp.pi, "e": sp.E}
        e = sp.sympify(s, locals=loc)
        f = sp.lambdify(x_sym, e, "numpy")
        xs = np.linspace(a, b, 200)
        ys = np.where(np.isfinite(f(xs)), f(xs), 0)
        self.ax.fill_between(xs, 0, ys, alpha=0.25, color="#1f77b4")
        r = CalculusEngine.integrate_definite(expr, a, b)
        # 解题过程
        try:
            _, T = TracedCalculus.definite_integral(expr, a, b)
            if hasattr(self, 'trace_cb') and self.trace_cb:
                self.trace_cb(T)
        except Exception:
            pass
        self.out_text.delete("1.0", tk.END)
        self.out_text.insert(tk.END, f"定积分 ∫[{a:.4g},{b:.4g}] f(x)dx\n")
        self.out_text.insert(tk.END, f"符号值: {r['symbolic']}\n")
        self.out_text.insert(tk.END, f"数值:   {_format_value(r['numeric'])}\n")
        if r.get("error"):
            self.out_text.insert(tk.END, f"误差:   {r['error']:.2e}\n")
        self.ax.legend(fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw()

    def _indefinite(self):
        expr = self.expr_var.get().strip()
        if not expr:
            return
        F_str, F_fn = CalculusEngine.integrate_indefinite(expr)
        _try_trace(self, lambda: TracedCalculus.indefinite_integral(expr))
        self._clear_ax()
        if F_fn:
            xs = np.linspace(-5, 5, 400)
            try:
                ys = F_fn(xs)
                ys = np.where(np.isfinite(ys), ys, np.nan)
                self.ax.plot(xs, ys, color="#d62728", linewidth=2, label=f"F(x) = {F_str}")
            except Exception:
                pass
        self.ax.set_xlim(-5, 5)
        self.ax.legend(fontsize=8)
        self.out_text.delete("1.0", tk.END)
        self.out_text.insert(tk.END, f"不定积分 ∫ f(x)dx\n")
        self.out_text.insert(tk.END, f"原函数: F(x) = {F_str}\n")
        self.fig.tight_layout()
        self.canvas.draw()

    def _riemann(self):
        try:
            expr, a, b = self._get_params()
            if b is None:
                raise ValueError("黎曼和需要有限区间")
        except ValueError as e:
            messagebox.showerror("参数错误", str(e))
            return
        n = self.n_var.get()
        mode = self.mode_var.get()
        data = CalculusEngine.riemann_data(expr, a, b, n, mode)
        _try_trace(self, lambda: TracedCalculus.riemann_sum(expr, a, b, n, mode))
        self._clear_ax()
        xs = np.array(data["xs"])
        ys = np.array(data["ys"])
        self.ax.plot(xs, ys, color="#1f77b4", linewidth=2, label=expr)
        for rect in data["rects"]:
            if rect["type"] == "rectangle":
                self.ax.add_patch(plt.Rectangle(
                    (rect["x_left"], rect["y_bottom"]),
                    rect["width"], rect["y_top"] - rect["y_bottom"],
                    facecolor="orange", edgecolor="darkorange", alpha=0.5, linewidth=0.5))
            elif rect["type"] == "trapezoid":
                self.ax.fill_between(
                    [rect["x_left"], rect["x_right"]],
                    [0, 0], [rect["y_left"], rect["y_right"]],
                    alpha=0.4, color="green")
        self.ax.set_xlim(a - (b - a) * 0.1, b + (b - a) * 0.1)
        self.ax.legend(fontsize=8)
        mode_cn = {"left": "左端点", "right": "右端点", "midpoint": "中点", "trapezoid": "梯形"}
        self.out_text.delete("1.0", tk.END)
        self.out_text.insert(tk.END, f"黎曼和 ({mode_cn.get(mode, mode)}), n={n}\n")
        self.out_text.insert(tk.END, f"近似值: {_format_value(data['value'])}\n")
        exact = self._get_exact()
        if exact:
            self.out_text.insert(tk.END, f"精确值: {_format_value(exact)}\n")
            self.out_text.insert(tk.END, f"误差:   {abs(data['value']-exact):.2e}\n")
        self.fig.tight_layout()
        self.canvas.draw()

    def _area_between(self):
        expr1 = self.expr_var.get().strip()
        expr2 = self.g_var.get().strip()
        if not expr1 or not expr2:
            messagebox.showerror("参数错误", "需要 f(x) 和 g(x)")
            return
        try:
            a = _parse_bound(self.a_var.get())
            b = _parse_bound(self.b_var.get())
        except ValueError:
            messagebox.showerror("参数错误", "a, b 必须是数字")
            return
        self._clear_ax()
        self._plot_func(expr1, a, b, label=f"f={expr1}", color="#1f77b4")
        self._plot_func(expr2, a, b, label=f"g={expr2}", color="#d62728")
        from function_plotter import _preprocess_expr
        import sympy as sp
        x_sym = sp.Symbol('x')
        loc = {"x": x_sym, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
               "log": sp.log, "exp": sp.exp, "sqrt": sp.sqrt, "abs": sp.Abs, "pi": sp.pi, "e": sp.E}
        s1 = _preprocess_expr(expr1); s2 = _preprocess_expr(expr2)
        e1 = sp.sympify(s1, locals=loc); e2 = sp.sympify(s2, locals=loc)
        f1 = sp.lambdify(x_sym, e1, "numpy"); f2 = sp.lambdify(x_sym, e2, "numpy")
        xs = np.linspace(a, b, 300)
        y1 = np.where(np.isfinite(f1(xs)), f1(xs), 0)
        y2 = np.where(np.isfinite(f2(xs)), f2(xs), 0)
        self.ax.fill_between(xs, y1, y2, alpha=0.2, color="purple")
        r = CalculusEngine.area_between(expr1, expr2, a, b)
        self.ax.legend(fontsize=8)
        self.out_text.delete("1.0", tk.END)
        self.out_text.insert(tk.END, f"面积对比: |f-g| 在 [{a:.4g},{b:.4g}]\n")
        self.out_text.insert(tk.END, f"面积 ≈ {_format_value(r['numeric'])}\n")
        self.fig.tight_layout()
        self.canvas.draw()

    # ── 反常积分 ──
    def _improper(self):
        expr = self.expr_var.get().strip()
        try:
            a = _parse_bound(self.a_var.get())
        except ValueError:
            messagebox.showerror("参数错误", "a 必须是数字")
            return
        b_str = self.b_var.get().strip()
        b = _parse_bound(b_str) if b_str else None
        sing_str = self.sing_var.get().strip()
        singularity = _parse_bound(sing_str) if sing_str else None

        self._clear_ax()
        if b is not None:
            self._plot_func(expr, a, b)
            xs_fill = np.linspace(a, b, 200)
        else:
            self._plot_func(expr, a, a + 10)
            xs_fill = np.linspace(a, a + 10, 300)
        from function_plotter import _preprocess_expr
        import sympy as sp
        x_sym = sp.Symbol('x')
        loc = {"x": x_sym, "sin": sp.sin, "cos": sp.cos,
               "log": sp.log, "exp": sp.exp, "sqrt": sp.sqrt, "abs": sp.Abs, "pi": sp.pi}
        s = _preprocess_expr(expr); e = sp.sympify(s, locals=loc)
        f = sp.lambdify(x_sym, e, "numpy")
        ys = np.where(np.isfinite(f(xs_fill)), f(xs_fill), 0)
        self.ax.fill_between(xs_fill, 0, ys, alpha=0.15, color="#9467bd")

        r = AdvancedIntegration.improper_integral(expr, a, b, singularity)
        _try_trace(self, lambda: TracedCalculus.improper_integral(expr, a, b, singularity))
        self.out_text.delete("1.0", tk.END)
        desc = f"反常积分 ∫[{a:.4g}," + (f"{b:.4g}" if b else "∞") + "] f(x)dx"
        self.out_text.insert(tk.END, f"{desc}\n")
        self.out_text.insert(tk.END, f"收敛性: {'收敛' if r['converges'] else ('发散' if r['converges'] is False else '待定')}\n")
        self.out_text.insert(tk.END, f"符号值: {r['symbolic']}\n")
        if r["numeric"]:
            self.out_text.insert(tk.END, f"数值:   {_format_value(r['numeric'])}\n")
            if r["error"]:
                self.out_text.insert(tk.END, f"误差:   {r['error']:.2e}\n")
        self.fig.tight_layout()
        self.canvas.draw()

    # ── 多重积分 ──
    def _double_integral(self):
        expr = self.mf_var.get().strip()
        if not expr:
            return
        import sympy as sp
        x_s, y_s = sp.symbols('x y')
        loc2 = {"x": x_s, "y": y_s, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
                "log": sp.log, "exp": sp.exp, "sqrt": sp.sqrt, "abs": sp.Abs, "pi": sp.pi, "e": sp.E}
        def _parse_dbl(s):
            s = s.strip()
            try: return float(s)
            except ValueError: pass
            try: return _parse_bound(s)
            except ValueError: pass
            try: return sp.sympify(_preprocess_expr(s), locals=loc2)
            except Exception: raise ValueError(f"无法解析 '{s}'")
        try:
            xl = _parse_dbl(self.mx_low.get()); xh = _parse_dbl(self.mx_high.get())
            yl = _parse_dbl(self.my_low.get()); yh = _parse_dbl(self.my_high.get())
        except ValueError as e:
            messagebox.showerror("积分限格式错误", f"{e}")
            return

        r = AdvancedIntegration.double_integral(expr, (xl, xh), (yl, yh))
        _try_trace(self, lambda: TracedCalculus.double_integral(expr, (xl, xh), (yl, yh)))

        # ── 3D 图: 任意区域曲面 + 边界 ──
        def _f2d(b):
            if isinstance(b, (int, float)): return lambda *a: float(b)
            fs = list(b.free_symbols) if hasattr(b, 'free_symbols') and b.free_symbols else []
            if not fs: return lambda *a: float(b.evalf())
            return sp.lambdify(tuple(fs), b, 'numpy')
        xlf, xhf = _f2d(xl), _f2d(xh)
        ylf, yhf = _f2d(yl), _f2d(yh)
        # 判断常量还是变量边界
        all_const_2d = isinstance(xl,(int,float)) and isinstance(xh,(int,float)) and isinstance(yl,(int,float)) and isinstance(yh,(int,float))
        if all_const_2d:
            xlv, xhv = float(xl), float(xh); ylv, yhv = float(yl), float(yh)
        else:
            xlv = float(xlf(0)); xhv = float(xhf(0))
            ylv = float(ylf(0)); yhv = float(yhf(0))

        self.fig.clear()
        self._ax3d = self.fig.add_subplot(111, projection='3d')
        # 在区域内采样 f(x,y)
        import sympy as sp
        x_s, y_s = sp.symbols('x y')
        s = _preprocess_expr(expr)
        loc_f = {"x": x_s, "y": y_s, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
                 "log": sp.log, "exp": sp.exp, "sqrt": sp.sqrt, "abs": sp.Abs, "pi": sp.pi, "e": sp.E}
        f_fn = sp.lambdify((x_s, y_s), sp.sympify(s, locals=loc_f), "numpy")
        px, py, pz = [], [], []
        n = 60
        for yi in np.linspace(ylv, yhv, n):
            xli = float(xlf(yi)); xhi = float(xhf(yi))
            for xi in np.linspace(max(xli, xlv-1), min(xhi, xhv+1), n):
                zi = float(f_fn(xi, yi))
                if np.isfinite(zi):
                    px.append(xi); py.append(yi); pz.append(zi)
        if px:
            from matplotlib.tri import Triangulation
            tri = Triangulation(px, py)
            self._ax3d.plot_trisurf(px, py, tri.triangles, pz, cmap='viridis', alpha=0.8, edgecolor='none')
        # 区域边界 (在 z=z_min 平面)
        bx, by_lo, by_hi = [], [], []
        for xi in np.linspace(xlv, xhv, 200):
            yli = float(ylf(xi)); yhi = float(yhf(xi))
            if np.isfinite(yli): bx.append(xi); by_lo.append(yli)
            if np.isfinite(yhi): by_hi.append(yhi)
        z_min = np.nanmin(pz)-1 if pz else -1
        if bx:
            self._ax3d.plot(bx, by_lo, [z_min]*len(bx), color='red', lw=2)
            self._ax3d.plot(bx, by_hi, [z_min]*len(bx), color='red', lw=2)
        self._ax3d.set_xlabel("x"); self._ax3d.set_ylabel("y"); self._ax3d.set_zlabel("z")
        self._ax3d.set_title(f"f(x,y) = {expr}", fontsize=9)
        self.fig.tight_layout(); self.canvas.draw()

        self.out_text.delete("1.0", tk.END)
        self.out_text.insert(tk.END, f"二重积分 ∬ f(x,y) dxdy\n")
        self.out_text.insert(tk.END, f"被积函数: f(x,y) = {expr}\n")
        self.out_text.insert(tk.END, f"区域: x∈[{xl},{xh}], y∈[{yl},{yh}]\n")
        self.out_text.insert(tk.END, f"内层积分: {r.get('inner','')}\n")
        self.out_text.insert(tk.END, f"符号值: {r['symbolic']}\n")
        if r["numeric"]:
            self.out_text.insert(tk.END, f"数值:   {_format_value(r['numeric'])}\n")
            if r.get("error"):
                self.out_text.insert(tk.END, f"误差:   {r['error']:.2e}\n")

    # ── 数值对比 ──
    def _numerical_compare(self):
        try:
            expr, a, b = self._get_params()
            if b is None:
                raise ValueError("数值对比需要有限区间")
        except ValueError as e:
            messagebox.showerror("参数错误", str(e))
            return
        r = AdvancedIntegration.numerical_compare(expr, a, b)
        _try_trace(self, lambda: TracedCalculus.numerical_compare(expr, a, b))
        self._clear_ax()
        self._plot_func(expr, a, b)
        self.ax.legend(fontsize=8)
        self.out_text.delete("1.0", tk.END)
        self.out_text.insert(tk.END, f"数值积分方法对比 | {expr} 在 [{a:.4g},{b:.4g}]\n\n")
        if r["exact"]:
            self.out_text.insert(tk.END, f"精确值 (符号): {r['exact']:.10f}\n\n")
        for method, val in r["methods"].items():
            err_str = ""
            if r["exact"]:
                err_str = f"  误差={abs(val-r['exact']):.2e}"
            self.out_text.insert(tk.END, f"{method:12s}: {_format_value(val)}{err_str}\n")

        self.fig.tight_layout()
        self.canvas.draw()

    # ── 弧长 ──
    def _arc_length(self):
        try:
            expr, a, b = self._get_params()
            if b is None:
                raise ValueError("弧长需要有限区间")
        except ValueError as e:
            messagebox.showerror("参数错误", str(e))
            return
        r = CurveEngine.arc_length(expr, a, b)
        _try_trace(self, lambda: TracedCalculus.arc_length(expr, a, b))
        self._clear_ax()
        self._plot_func(expr, a, b)
        self.ax.set_title(f"弧长 L = {_format_value(r['numeric'])}" if r['numeric'] else "弧长")
        self.ax.legend(fontsize=8)
        self.fig.tight_layout(); self.canvas.draw()
        self.out_text.delete("1.0", tk.END)
        self.out_text.insert(tk.END, f"弧长: y={expr} 在 [{a:.4g},{b:.4g}]\n")
        self.out_text.insert(tk.END, f"公式: ∫√(1+f'²)dx = ∫ {r['integrand']} dx\n")
        self.out_text.insert(tk.END, f"符号值: {r['symbolic']}\n")
        self.out_text.insert(tk.END, f"数值:   {_format_value(r['numeric'])}\n" if r['numeric'] else "—\n")

    # ── 曲率 ──
    def _curvature(self):
        expr = self.expr_var.get().strip()
        if not expr:
            return
        try:
            xv = _parse_bound(self.curv_x_var.get())
        except ValueError:
            messagebox.showerror("参数错误", "x 必须是数字")
            return
        r = CurveEngine.curvature(expr, xv)
        _try_trace(self, lambda: TracedCalculus.curvature(expr, xv))
        self._clear_ax()
        self._plot_func(expr, xv - 3, xv + 3)
        # 画密切圆
        if r["radius"] and r["radius"] < 100:
            import sympy as sp
            from function_plotter import _preprocess_expr
            x_sym = sp.Symbol('x')
            s = _preprocess_expr(expr)
            loc = {"x": x_sym, "sin": sp.sin, "cos": sp.cos, "log": sp.log,
                   "exp": sp.exp, "sqrt": sp.sqrt, "abs": sp.Abs, "pi": sp.pi, "e": sp.E}
            e = sp.sympify(s, locals=loc)
            f_fn = sp.lambdify(x_sym, e, "numpy")
            fp = sp.diff(e, x_sym)
            fp_fn = sp.lambdify(x_sym, fp, "numpy")
            yv = float(f_fn(np.array([xv])).item())
            slope = float(fp_fn(np.array([xv])).item())
            # 法向方向 (指向曲率中心)
            norm = np.sqrt(1 + slope**2)
            nx = -slope / norm; ny = 1.0 / norm
            cx = xv + nx * r["radius"]
            cy = yv + ny * r["radius"]
            circle = plt.Circle((cx, cy), r["radius"], fill=False, color="green",
                                linestyle=":", linewidth=1.2, alpha=0.7)
            self.ax.add_patch(circle)
            self.ax.plot(xv, yv, "go", markersize=6)
        self.ax.legend(fontsize=8)
        self.ax.set_title(f"x={xv}: κ={_format_value(r['curvature'])}, R={r['radius']:.4f}" if r['curvature'] else "曲率")
        self.fig.tight_layout(); self.canvas.draw()
        self.out_text.delete("1.0", tk.END)
        self.out_text.insert(tk.END, f"曲率: y={expr} 在 x={xv}\n")
        self.out_text.insert(tk.END, f"κ = {r['kappa_expr']}\n")
        if r['curvature'] is not None:
            self.out_text.insert(tk.END, f"κ = {_format_value(r['curvature'])}\n")
        if r['radius'] is not None:
            self.out_text.insert(tk.END, f"曲率半径 R = {r['radius']:.6f}\n")

    # ── 三重积分 ──
    def _triple_integral(self):
        expr = self.triple_var.get().strip()
        if not expr:
            return
        import sympy as sp
        x, y, z = sp.symbols('x y z')
        loc_mv = {"x": x, "y": y, "z": z,
                  "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
                  "log": sp.log, "exp": sp.exp, "sqrt": sp.sqrt, "abs": sp.Abs,
                  "pi": sp.pi, "e": sp.E}
        # 解析积分限: 支持常量 + 含 x,y,z 的表达式
        def _parse_mv_bound(raw, vars_ok=None):
            """解析多元积分限: 数字→float, 表达式→sympy Expr"""
            s_v = raw.strip()
            if not s_v:
                raise ValueError("空值")
            try:
                return float(s_v)
            except ValueError:
                pass
            try:
                return _parse_bound(s_v)
            except ValueError:
                pass
            try:
                pp = _preprocess_expr(s_v)
                return sp.sympify(pp, locals=loc_mv)
            except Exception:
                raise ValueError(f"无法解析 '{s_v}'")
        try:
            xl = _parse_mv_bound(self.tx_low.get())
            xh = _parse_mv_bound(self.tx_high.get())
            yl = _parse_mv_bound(self.ty_low.get())
            yh = _parse_mv_bound(self.ty_high.get())
            zl = _parse_mv_bound(self.tz_low.get())
            zh = _parse_mv_bound(self.tz_high.get())
        except ValueError as e:
            messagebox.showerror("积分限格式错误",
                f"{e}\n\n支持的格式:\n"
                "  常量: 0, 3.14, pi/2, 2*pi, sqrt(2)\n"
                "  含变量的表达式: sqrt(pi^2 - y^2), 1-x, sin(y)\n"
                "  (x/y/z 分别对应重积分的外层变量)")
            return
        s = _preprocess_expr(expr)
        f_expr = sp.sympify(s, locals=loc_mv)
        # 自动确定积分顺序: 被依赖的变量必须外层的(后积分)
        # y限含x → x是外层, y是内层 → 先积y再积x
        bounds = [(x, xl, xh), (y, yl, yh), (z, zl, zh)]
        deps = {v: set() for v, _, _ in bounds}
        for v, lo, hi in bounds:
            for sym in [x, y, z]:
                if hasattr(lo, 'free_symbols') and sym in lo.free_symbols: deps[v].add(sym)
                if hasattr(hi, 'free_symbols') and sym in hi.free_symbols: deps[v].add(sym)
        # 拓扑排序: 依赖多的先积分 (内层), 被依赖的后积分 (外层)
        order = sorted(bounds, key=lambda b: -len(deps[b[0]]))
        try:
            expr = f_expr
            for v, lo, hi in order:
                expr = sp.integrate(expr, (v, lo, hi))
                expr = sp.simplify(expr)
            symbolic = str(expr).replace("**", "^")
        except Exception:
            symbolic = "—"
        # 数值积分: 按依赖顺序重排
        try:
            f_fn = sp.lambdify((x, y, z), f_expr, "numpy")
            def _mk_fn(bound_expr, vars_sym):
                if isinstance(bound_expr, (int, float)):
                    return lambda *a: float(bound_expr)
                return sp.lambdify(vars_sym, bound_expr, "numpy")
            from scipy import integrate as scipy_integrate
            # 按依赖顺序构建嵌套调用
            vmap = {x: 'x', y: 'y', z: 'z'}
            o = order  # [(var, lo, hi), ...] sorted by deps
            # 构建最内层积分
            inner_v, inner_lo, inner_hi = o[0]
            mid_v, mid_lo, mid_hi = o[1]
            outer_v, outer_lo, outer_hi = o[2]
            # 按 tplquad(z, y, x) 适配
            numeric = None
            if all(isinstance(b, (int, float)) or (hasattr(b,'is_number') and b.is_number())
                   for _, lo, hi in order for b in [lo, hi]):
                # 全常量: 直接用 tplquad
                val, err = scipy_integrate.tplquad(
                    f_fn, float(outer_lo), float(outer_hi),
                    lambda m: float(mid_lo), lambda m: float(mid_hi),
                    lambda m, i: float(inner_lo), lambda m, i: float(inner_hi),
                    epsabs=1e-6, limit=100)
                numeric = round(val, 10); error = round(float(err), 10)
            else:
                # 含变量: 用 nquad 处理任意顺序
                ranges = []
                for v, lo, hi in reversed(order):
                    lo_fn = _mk_fn(lo, ()) if isinstance(lo, (int,float)) else _mk_fn(lo, tuple(deps[v]))
                    hi_fn = _mk_fn(hi, ()) if isinstance(hi, (int,float)) else _mk_fn(hi, tuple(deps[v]))
                    ranges.append((lo_fn, hi_fn))
                val, err = scipy_integrate.nquad(lambda *args: f_fn(*args), ranges)
                numeric = round(val, 10); error = round(float(err), 10)
        except Exception:
            numeric = None; error = None

        _try_trace(self, lambda: TracedCalculus.triple_integral(expr, (xl, xh), (yl, yh), (zl, zh)))

        # ── 3D 区域图 (marching_cubes 提取任意形状表面) ──
        from skimage.measure import marching_cubes
        self.fig.clear()
        self._ax3d = self.fig.add_subplot(111, projection='3d')
        # 边界→可调用函数
        def _f(b):
            if isinstance(b, (int, float)): return lambda *a: float(b) if not a else np.full_like(np.atleast_1d(a[0]), float(b))
            fs = list(b.free_symbols) if hasattr(b, 'free_symbols') and b.free_symbols else []
            if not fs:
                v = float(b.evalf() if hasattr(b, 'evalf') else b)
                return lambda *a: v if not a else np.full_like(np.atleast_1d(a[0]), v)
            return sp.lambdify(tuple(fs), b, 'numpy')
        xlf, xhf = _f(xl), _f(xh)
        ylf, yhf = _f(yl), _f(yh)
        zlf, zhf = _f(zl), _f(zh)
        # 确定包围盒 (根据各边界函数的 free_symbols 传正确参数)
        xlv = float(xlf(0,0).item()) if hasattr(xlf(0,0), 'item') else float(xlf(0,0))
        xhv = float(xhf(0,0).item()) if hasattr(xhf(0,0), 'item') else float(xhf(0,0))
        # y 边界可能依赖 x 或 z, 先试 x=0
        ylv = float(ylf(0).item()) if hasattr(ylf(0), 'item') else float(ylf(0))
        yhv = float(yhf(0).item()) if hasattr(yhf(0), 'item') else float(yhf(0))
        zlv = float(zlf(0,0).item()) if hasattr(zlf(0,0), 'item') else float(zlf(0,0))
        zhv = float(zhf(0,0).item()) if hasattr(zhf(0,0), 'item') else float(zhf(0,0))
        # 生成体素网格
        n = 60
        margin = 0.2
        xs = np.linspace(xlv-margin, xhv+margin, n)
        ys = np.linspace(ylv-margin, yhv+margin, n)
        zs = np.linspace(zlv-margin, zhv+margin, n)
        Xg, Yg, Zg = np.meshgrid(xs, ys, zs, indexing='ij')
        # 隐式函数: 根据各边界的自由变量选择正确的网格维度
        def _eval(bound_fn, bound_expr, default_grid):
            """调用 bound_fn, 传入选自 {Xg,Yg,Zg} 中对应 free_symbols 的网格"""
            fs = list(bound_expr.free_symbols) if hasattr(bound_expr, 'free_symbols') and bound_expr.free_symbols else []
            args = []
            for s in fs:
                if s == x: args.append(Xg)
                elif s == y: args.append(Yg)
                elif s == z: args.append(Zg)
            if not args:
                args.append(default_grid)  # 常量函数, 传任意网格
            return bound_fn(*args)

        F = np.maximum.reduce([
            _eval(xlf, xl, Zg) - Xg,
            Xg - _eval(xhf, xh, Zg),
            _eval(ylf, yl, Zg) - Yg,
            Yg - _eval(yhf, yh, Zg),
            _eval(zlf, zl, Xg) - Zg,
            Zg - _eval(zhf, zh, Xg),
        ])
        F = np.nan_to_num(F, nan=1e9)  # NaN→大数, 确保 marching_cubes 不崩溃
        # marching_cubes 提取 F=0 等值面
        try:
            verts, faces, _, _ = marching_cubes(F, level=0, spacing=(xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0]))
            verts[:,0] += xs[0]; verts[:,1] += ys[0]; verts[:,2] += zs[0]
            self._ax3d.plot_trisurf(verts[:,0], verts[:,1], faces, verts[:,2],
                                    color='steelblue', alpha=0.6, edgecolor='none')
        except Exception:
            self._ax3d.text(0, 0, 0, "该区域无法生成表面", fontsize=10, color='red')
        self._ax3d.set_xlim(xlv-0.5, xhv+0.5)
        self._ax3d.set_ylim(ylv-0.5, yhv+0.5)
        self._ax3d.set_zlim(zlv-0.5, zhv+0.5)
        self._ax3d.set_xlabel("x"); self._ax3d.set_ylabel("y"); self._ax3d.set_zlabel("z")
        self._ax3d.set_title("积分区域", fontsize=9)
        self.fig.tight_layout(); self.canvas.draw()
        self.fig.tight_layout(); self.canvas.draw()

        self.out_text.delete("1.0", tk.END)
        self.out_text.insert(tk.END, f"三重积分 ∭ {expr} dxdydz\n")
        self.out_text.insert(tk.END, f"区域: x∈[{xl},{xh}], y∈[{yl},{yh}], z∈[{zl},{zh}]\n")
        self.out_text.insert(tk.END, f"符号值: {symbolic}\n")
        if numeric:
            self.out_text.insert(tk.END, f"数值:   {_format_value(numeric)}\n")
            self.out_text.insert(tk.END, f"误差:   {error:.2e}\n")

    def _save(self):
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("All", "*.*")])
        if path:
            self.fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")


# ============================================================================
# Tab 3: 无穷级数 (含子标签)
# ============================================================================

class SeriesTab(ttk.Frame):
    """泰勒 / 傅里叶 / 收敛判定 / 幂级数 / 求和"""

    trace_cb = None
    def __init__(self, parent, root):
        super().__init__(parent)
        self.root = root
        self._build()

    def _build(self):
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # ── 左侧 ──
        left = ttk.Frame(paned, width=310)
        paned.add(left, weight=0)

        ttk.Label(left, text="∑ 无穷级数", font=FONT_TITLE).pack(anchor=tk.W, padx=8, pady=(8, 4))
        frm = ttk.Frame(left)
        frm.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(frm, text="f(x) / aₙ =", font=FONT).pack(side=tk.LEFT)
        self.expr_var = tk.StringVar(value="sin(x)")
        self.entry = ttk.Entry(frm, textvariable=self.expr_var, font=FONT_MONO, width=18)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # 子标签
        sub_nb = ttk.Notebook(left)
        sub_nb.pack(fill=tk.BOTH, expand=True, padx=2, pady=4)

        # -- 子1: 泰勒 --
        s1 = ttk.Frame(sub_nb)
        sub_nb.add(s1, text="泰勒展开")
        self._build_sub_taylor(s1)

        # -- 子2: 傅里叶 --
        s2 = ttk.Frame(sub_nb)
        sub_nb.add(s2, text="傅里叶级数")
        self._build_sub_fourier(s2)

        # -- 子3: 收敛判定 --
        s3 = ttk.Frame(sub_nb)
        sub_nb.add(s3, text="收敛判定")
        self._build_sub_convergence(s3)

        # -- 子4: 幂级数 --
        s4 = ttk.Frame(sub_nb)
        sub_nb.add(s4, text="幂级数半径")
        self._build_sub_power(s4)

        # -- 子5: 级数求和 --
        s5 = ttk.Frame(sub_nb)
        sub_nb.add(s5, text="级数求和")
        self._build_sub_sum(s5)

        # 输出
        out_lab = ttk.LabelFrame(left, text=" 计算结果 ", padding=4)
        out_lab.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)
        self.out_text = tk.Text(out_lab, font=FONT_MONO, wrap=tk.WORD, height=6,
                                bg="#ffffff", relief=tk.FLAT, padx=4, pady=4)
        self.out_text.pack(fill=tk.BOTH, expand=True)

        # ── 右侧画布 ──
        right = ttk.Frame(paned)
        paned.add(right, weight=1)
        self.fig, self.ax = _make_figure()
        self.canvas = FigureCanvasTkAgg(self.fig, right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.fig.tight_layout()
        self.canvas.draw()
        ttk.Button(right, text="💾 保存", command=self._save).pack(anchor=tk.E, padx=4, pady=2)

    def _build_sub_taylor(self, parent):
        frm = ttk.Frame(parent)
        frm.pack(fill=tk.BOTH, padx=4, pady=4)
        ttk.Label(frm, text="中心 x₀ =", font=FONT).pack()
        self.x0_var = tk.StringVar(value="0")
        ttk.Entry(frm, textvariable=self.x0_var, font=FONT_MONO, width=8).pack(pady=2)
        ttk.Label(frm, text="阶数 n =", font=FONT).pack()
        frm_t = ttk.Frame(frm)
        frm_t.pack(fill=tk.X, pady=2)
        self.n_taylor_var = tk.IntVar(value=5)
        ttk.Scale(frm_t, from_=1, to=20, variable=self.n_taylor_var, orient=tk.HORIZONTAL,
                  command=lambda v: self._update_tn_label()).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.tn_label = ttk.Label(frm_t, text="5", font=FONT, width=3)
        self.tn_label.pack(side=tk.LEFT)
        ttk.Button(frm, text="泰勒展开", command=self._taylor).pack(fill=tk.X, pady=(4, 0))

    def _build_sub_fourier(self, parent):
        frm = ttk.Frame(parent)
        frm.pack(fill=tk.BOTH, padx=4, pady=4)
        ttk.Label(frm, text="周期 T =", font=FONT).pack()
        self.period_var = tk.StringVar(value="6.283185")
        ttk.Entry(frm, textvariable=self.period_var, font=FONT_MONO, width=10).pack(pady=2)
        ttk.Label(frm, text="谐波 n =", font=FONT).pack()
        frm_f = ttk.Frame(frm)
        frm_f.pack(fill=tk.X, pady=2)
        self.n_fourier_var = tk.IntVar(value=3)
        ttk.Scale(frm_f, from_=1, to=15, variable=self.n_fourier_var, orient=tk.HORIZONTAL,
                  command=lambda v: self._update_fn_label()).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.fn_label = ttk.Label(frm_f, text="3", font=FONT, width=3)
        self.fn_label.pack(side=tk.LEFT)
        ttk.Button(frm, text="傅里叶展开", command=self._fourier).pack(fill=tk.X, pady=(4, 0))

    def _build_sub_convergence(self, parent):
        frm = ttk.Frame(parent)
        frm.pack(fill=tk.BOTH, padx=4, pady=4)
        ttk.Label(frm, text="级数收敛判定",
                  font=("Microsoft YaHei UI", 9, "bold")).pack(pady=2)
        ttk.Label(frm, text="通项 aₙ = (含 n 的表达式)",
                  font=("Microsoft YaHei UI", 8)).pack()
        frm_s = ttk.Frame(frm)
        frm_s.pack(fill=tk.X, pady=2)
        ttk.Label(frm_s, text="aₙ =", font=FONT).pack(side=tk.LEFT)
        self.series_var = tk.StringVar(value="1/n^2")
        ttk.Entry(frm_s, textvariable=self.series_var, font=FONT_MONO, width=14).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(frm, text="判定收敛性", command=self._series_test).pack(fill=tk.X, pady=4)
        ttk.Label(frm, text="支持: 1/n^2, 1/2^n, (-1)^n/n, ...",
                  font=("Microsoft YaHei UI", 7)).pack()

    def _build_sub_power(self, parent):
        frm = ttk.Frame(parent)
        frm.pack(fill=tk.BOTH, padx=4, pady=4)
        ttk.Label(frm, text="幂级数收敛半径",
                  font=("Microsoft YaHei UI", 9, "bold")).pack(pady=2)
        ttk.Label(frm, text="系数 aₙ = (含 n 的表达式)",
                  font=("Microsoft YaHei UI", 8)).pack()
        frm_p = ttk.Frame(frm)
        frm_p.pack(fill=tk.X, pady=2)
        ttk.Label(frm_p, text="aₙ =", font=FONT).pack(side=tk.LEFT)
        self.power_var = tk.StringVar(value="1/2^n")
        ttk.Entry(frm_p, textvariable=self.power_var, font=FONT_MONO, width=14).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(frm, text="求收敛半径", command=self._power_radius).pack(fill=tk.X, pady=4)
        ttk.Label(frm, text="∑ aₙ·(x-x₀)ⁿ 的收敛半径 R",
                  font=("Microsoft YaHei UI", 7)).pack()

    def _build_sub_sum(self, parent):
        frm = ttk.Frame(parent)
        frm.pack(fill=tk.BOTH, padx=4, pady=4)
        ttk.Label(frm, text="级数求和",
                  font=("Microsoft YaHei UI", 9, "bold")).pack(pady=2)
        frm_s1 = ttk.Frame(frm)
        frm_s1.pack(fill=tk.X, pady=2)
        ttk.Label(frm_s1, text="aₙ =", font=FONT).pack(side=tk.LEFT)
        self.sum_var = tk.StringVar(value="1/n^2")
        ttk.Entry(frm_s1, textvariable=self.sum_var, font=FONT_MONO, width=14).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        frm_s2 = ttk.Frame(frm)
        frm_s2.pack(fill=tk.X, pady=2)
        ttk.Label(frm_s2, text="从 n =", font=FONT).pack(side=tk.LEFT)
        self.sum_from_var = tk.StringVar(value="1")
        ttk.Entry(frm_s2, textvariable=self.sum_from_var, font=FONT_MONO, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Label(frm_s2, text="到 (留空=∞)", font=("Microsoft YaHei UI", 8)).pack(side=tk.LEFT)
        self.sum_to_var = tk.StringVar(value="")
        ttk.Entry(frm_s2, textvariable=self.sum_to_var, font=FONT_MONO, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Button(frm, text="计算级数和", command=self._series_sum).pack(fill=tk.X, pady=4)
        ttk.Label(frm, text="例: ∑1/n² = π²/6",
                  font=("Microsoft YaHei UI", 7)).pack()

    def _update_tn_label(self):
        self.tn_label.config(text=str(self.n_taylor_var.get()))

    def _update_fn_label(self):
        self.fn_label.config(text=str(self.n_fourier_var.get()))

    def _clear_ax(self):
        self.ax.clear()
        self.ax.set_facecolor("#fafafa")
        self.ax.grid(True, alpha=0.3, linestyle="--")
        self.ax.axhline(y=0, color="gray", linewidth=0.8, alpha=0.6)
        self.ax.axvline(x=0, color="gray", linewidth=0.8, alpha=0.6)

    def _plot_ref_func(self, expr_str: str, label: str = "f(x)", color: str = "#1f77b4",
                       x_range=(-6, 6), n_pts=500):
        from function_plotter import _preprocess_expr
        import sympy as sp
        x_sym = sp.Symbol('x')
        s = _preprocess_expr(expr_str)
        loc = {"x": x_sym, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
               "log": sp.log, "exp": sp.exp, "sqrt": sp.sqrt, "abs": sp.Abs, "pi": sp.pi, "e": sp.E}
        expr = sp.sympify(s, locals=loc)
        f = sp.lambdify(x_sym, expr, "numpy")
        xs = np.linspace(x_range[0], x_range[1], n_pts)
        ys = f(xs)
        ys = np.where(np.isfinite(ys), ys, np.nan)
        self.ax.plot(xs, ys, color=color, linewidth=2.5, label=label, zorder=3)
        self.ax.set_xlim(x_range[0], x_range[1])
        return xs

    # ── 泰勒 ──
    def _taylor(self):
        expr = self.expr_var.get().strip()
        if not expr:
            return
        try:
            x0 = _parse_bound(self.x0_var.get())
        except ValueError:
            messagebox.showerror("参数错误", "x₀ 必须是数字")
            return
        n = self.n_taylor_var.get()
        data = CalculusEngine.taylor(expr, x0, n)
        try:
            _, T = TracedCalculus.taylor(expr, x0, n)
            if hasattr(self, 'trace_cb') and self.trace_cb:
                self.trace_cb(T)
        except Exception:
            pass
        self._clear_ax()
        margin = max(2.0, abs(x0) + 3)
        xr = (x0 - margin, x0 + margin)
        self._plot_ref_func(expr, label=f"f(x)={expr}", x_range=xr)
        poly_fn = data["poly_fn"]
        xs = np.linspace(xr[0], xr[1], 500)
        try:
            ys = poly_fn(xs)
            self.ax.plot(xs, np.where(np.isfinite(ys), ys, np.nan),
                         "--", color="#ff7f0e", linewidth=2, label=f"T{n}(x)", zorder=4)
        except Exception:
            pass
        self.ax.legend(fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw()
        self.out_text.delete("1.0", tk.END)
        self.out_text.insert(tk.END, f"泰勒展开: f(x)={expr}, x₀={x0}, n={n}\n\n")
        self.out_text.insert(tk.END, f"T{n}(x) = {data['poly_str']}\n\n")
        if data["coefficients"]:
            self.out_text.insert(tk.END, "系数:\n")
            for k, cv in data["coefficients"]:
                self.out_text.insert(tk.END, f"  f^({k})({x0})/{k}! = {cv:.6f}\n")

    # ── 傅里叶 ──
    def _fourier(self):
        expr = self.expr_var.get().strip()
        if not expr:
            return
        try:
            period = _parse_bound(self.period_var.get())
        except ValueError:
            messagebox.showerror("参数错误", "周期必须是数字")
            return
        n = self.n_fourier_var.get()
        data = CalculusEngine.fourier(expr, period=period, n_terms=n)
        try:
            _, T = TracedCalculus.fourier(expr, period, n)
            if hasattr(self, 'trace_cb') and self.trace_cb:
                self.trace_cb(T)
        except Exception:
            pass
        self._clear_ax()
        L = data["L"]
        xr = (-L * 1.5, L * 1.5)
        self._plot_ref_func(expr, label=f"f(x)={expr}", x_range=xr)
        try:
            xs = np.linspace(xr[0], xr[1], 600)
            ys = data["approx_fn"](xs)
            self.ax.plot(xs, np.where(np.isfinite(ys), ys, np.nan),
                         "--", color="#2ca02c", linewidth=2, label=f"F{n}(x)", zorder=4)
        except Exception:
            pass
        self.ax.legend(fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw()
        self.out_text.delete("1.0", tk.END)
        self.out_text.insert(tk.END, f"傅里叶级数: f(x)={expr}, T={period:.4f}, n={n}\n\n")
        self.out_text.insert(tk.END, f"a₀/2 = {_format_value(data['a0'])}\n")
        for n_i, av in data["an"]:
            self.out_text.insert(tk.END, f"  a{n_i} = {av:.6f}\n")
        for n_i, bv in data["bn"]:
            self.out_text.insert(tk.END, f"  b{n_i} = {bv:.6f}\n")
        self.out_text.insert(tk.END, f"\n逼近: F{n}(x) ≈ {data['approx_str']}\n")

    # ── 收敛判定 ──
    def _series_test(self):
        expr = self.series_var.get().strip()
        if not expr:
            return
        r = SeriesAnalysis.convergence_test(expr)
        try:
            _, T = TracedCalculus.series_convergence(expr)
            if hasattr(self, 'trace_cb') and self.trace_cb:
                self.trace_cb(T)
        except Exception:
            pass
        self.out_text.delete("1.0", tk.END)
        self.out_text.insert(tk.END, f"级数收敛判定: Σ aₙ,  aₙ = {expr}\n\n")
        for t in r["tests"]:
            self.out_text.insert(tk.END, f"  [{t['result']}] {t['name']}\n")
            self.out_text.insert(tk.END, f"    {t['detail']}\n\n")
        self.out_text.insert(tk.END, f"结论: {r['conclusion']}\n")

    # ── 幂级数半径 ──
    def _power_radius(self):
        expr = self.power_var.get().strip()
        if not expr:
            return
        r = SeriesAnalysis.power_series_radius(expr)
        _try_trace(self, lambda: TracedCalculus.power_radius(expr))
        self.out_text.delete("1.0", tk.END)
        self.out_text.insert(tk.END, f"幂级数收敛半径: 系数 aₙ = {expr}\n\n")
        self.out_text.insert(tk.END, f"方法: {r['method']}\n")
        if r.get("L") is not None:
            self.out_text.insert(tk.END, f"L = lim |aₙ₊₁/aₙ| = {r['L']:.6f}\n")
        self.out_text.insert(tk.END, f"收敛半径 R = {r['radius_str']}\n")
        if r["radius"] is not None and r["radius"] != float('inf'):
            self.out_text.insert(tk.END, f"收敛区间: |x| < {r['radius_str']}\n")

    # ── 级数求和 ──
    def _series_sum(self):
        expr = self.sum_var.get().strip()
        if not expr:
            return
        try:
            from_n = int(self.sum_to_var.get() or "1")
            to_str = self.sum_to_var.get().strip()
            to_n = int(to_str) if to_str else None
        except ValueError:
            messagebox.showerror("参数错误", "n 必须是整数")
            return
        r = SeriesAnalysis.series_sum(expr, from_n, to_n)
        _try_trace(self, lambda: TracedCalculus.series_sum(expr, from_n, to_n))
        self.out_text.delete("1.0", tk.END)
        desc = f"级数求和: Σ({expr}), n={from_n}" + (f"..{to_n}" if to_n else "..∞")
        self.out_text.insert(tk.END, f"{desc}\n\n")
        self.out_text.insert(tk.END, f"结果: {r['symbolic']}\n")
        if r.get("numeric"):
            self.out_text.insert(tk.END, f"数值: {_format_value(r['numeric'])}\n")
        self.out_text.insert(tk.END, f"{'解析形式' if r['closed_form'] else '未找到闭式'}\n")

    def _save(self):
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("All", "*.*")])
        if path:
            self.fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")


# ============================================================================
# Tab 4: 微分方程
# ============================================================================

class OdeTab(ttk.Frame):
    """ODE 方向场 + 初值问题数值解"""

    trace_cb = None
    def __init__(self, parent, root):
        super().__init__(parent)
        self.root = root
        self._build()

    def _build(self):
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(paned, width=300)
        paned.add(left, weight=0)

        ttk.Label(left, text="⊡ 微分方程", font=FONT_TITLE).pack(anchor=tk.W, padx=8, pady=(8, 4))

        # y' = f(x, y)
        frm = ttk.Frame(left)
        frm.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(frm, text="y' =", font=FONT).pack(side=tk.LEFT)
        self.ode_var = tk.StringVar(value="-y + sin(x)")
        ttk.Entry(frm, textvariable=self.ode_var, font=FONT_MONO, width=20).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # 显示范围
        frm_r = ttk.Frame(left)
        frm_r.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(frm_r, text="x∈", font=("Microsoft YaHei UI", 7)).pack(side=tk.LEFT)
        self.ode_x0 = tk.StringVar(value="-3"); self.ode_x1 = tk.StringVar(value="3")
        ttk.Entry(frm_r, textvariable=self.ode_x0, font=FONT_MONO, width=4).pack(side=tk.LEFT)
        ttk.Entry(frm_r, textvariable=self.ode_x1, font=FONT_MONO, width=4).pack(side=tk.LEFT)
        ttk.Label(frm_r, text=" y∈", font=("Microsoft YaHei UI", 7)).pack(side=tk.LEFT)
        self.ode_y0 = tk.StringVar(value="-3"); self.ode_y1 = tk.StringVar(value="3")
        ttk.Entry(frm_r, textvariable=self.ode_y0, font=FONT_MONO, width=4).pack(side=tk.LEFT)
        ttk.Entry(frm_r, textvariable=self.ode_y1, font=FONT_MONO, width=4).pack(side=tk.LEFT)

        ttk.Button(left, text="画方向场", command=self._plot_field).pack(fill=tk.X, padx=4, pady=2)

        # 初值问题
        ivp_lab = ttk.LabelFrame(left, text=" 初值问题 y(x₀)=y₀ ", padding=4)
        ivp_lab.pack(fill=tk.X, padx=4, pady=4)
        frm_i = ttk.Frame(ivp_lab)
        frm_i.pack(fill=tk.X)
        ttk.Label(frm_i, text="x₀=", font=FONT).pack(side=tk.LEFT)
        self.ivp_x0 = tk.StringVar(value="0")
        ttk.Entry(frm_i, textvariable=self.ivp_x0, font=FONT_MONO, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Label(frm_i, text="y₀=", font=FONT).pack(side=tk.LEFT)
        self.ivp_y0 = tk.StringVar(value="1")
        ttk.Entry(frm_i, textvariable=self.ivp_y0, font=FONT_MONO, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Label(frm_i, text="到 x=", font=FONT).pack(side=tk.LEFT)
        self.ivp_xend = tk.StringVar(value="5")
        ttk.Entry(frm_i, textvariable=self.ivp_xend, font=FONT_MONO, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Button(ivp_lab, text="求解并叠加", command=self._solve_ivp).pack(fill=tk.X, pady=(4, 0))

        # 多条解曲线
        mlab = ttk.LabelFrame(left, text=" 批量初值 ", padding=4)
        mlab.pack(fill=tk.X, padx=4, pady=4)
        frm_m = ttk.Frame(mlab)
        frm_m.pack(fill=tk.X)
        ttk.Label(frm_m, text="y₀ 列表:", font=("Microsoft YaHei UI", 8)).pack(side=tk.LEFT)
        self.multi_y0 = tk.StringVar(value="-2,-1,0,1,2")
        ttk.Entry(frm_m, textvariable=self.multi_y0, font=FONT_MONO, width=14).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Label(mlab, text=f"x₀={self.ivp_x0.get()}, 到 x={self.ivp_xend.get()}",
                  font=("Microsoft YaHei UI", 7)).pack()
        ttk.Button(mlab, text="批量解曲线", command=self._multi_ivp).pack(fill=tk.X, pady=(2, 0))

        # 输出
        out_lab = ttk.LabelFrame(left, text=" 结果 ", padding=4)
        out_lab.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)
        self.out_text = tk.Text(out_lab, font=FONT_MONO, wrap=tk.WORD, height=6,
                                bg="#ffffff", relief=tk.FLAT, padx=4, pady=4)
        self.out_text.pack(fill=tk.BOTH, expand=True)

        # 画布
        right = ttk.Frame(paned)
        paned.add(right, weight=1)
        self.fig, self.ax = _make_figure()
        self.canvas = FigureCanvasTkAgg(self.fig, right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.fig.tight_layout(); self.canvas.draw()
        ttk.Button(right, text="💾 保存", command=self._save).pack(anchor=tk.E, padx=4, pady=2)

        self._field_data = None

    def _plot_field(self):
        expr = self.ode_var.get().strip()
        if not expr:
            return
        try:
            x0 = _parse_bound(self.ode_x0.get()); x1 = _parse_bound(self.ode_x1.get())
            y0 = _parse_bound(self.ode_y0.get()); y1 = _parse_bound(self.ode_y1.get())
        except ValueError:
            messagebox.showerror("参数错误", "范围必须是数字")
            return
        data = OdeEngine.direction_field(expr, (x0, x1), (y0, y1), nx=28, ny=28)
        self._field_data = data
        self.ax.clear()
        self.ax.set_facecolor("#fafafa")
        X = np.array(data["X"]); Y = np.array(data["Y"]); U = np.array(data["U"]); V = np.array(data["V"])
        self.ax.quiver(X, Y, U, V, color="gray", alpha=0.5, scale=40, width=0.003)
        self.ax.set_xlim(x0, x1); self.ax.set_ylim(y0, y1)
        self.ax.set_xlabel("x"); self.ax.set_ylabel("y")
        self.ax.set_title(f"y' = {expr}")
        self.fig.tight_layout(); self.canvas.draw()
        self.out_text.delete("1.0", tk.END)
        self.out_text.insert(tk.END, f"方向场: y' = {expr}\n")
        self.out_text.insert(tk.END, f"网格: 28×28, 范围 x∈[{x0},{x1}], y∈[{y0},{y1}]\n")

    def _solve_ivp(self):
        expr = self.ode_var.get().strip()
        if not expr:
            return
        try:
            x0 = _parse_bound(self.ivp_x0.get()); y0 = _parse_bound(self.ivp_y0.get())
            xend = _parse_bound(self.ivp_xend.get())
        except ValueError:
            messagebox.showerror("参数错误", "初值必须是数字")
            return
        sol = OdeEngine.solve_ivp(expr, x0, y0, xend)
        try:
            _, T = TracedCalculus.ode_solve(expr, x0, y0, xend)
            if hasattr(self, 'trace_cb') and self.trace_cb:
                self.trace_cb(T)
        except Exception:
            pass
        self.ax.plot(sol["xs"], sol["ys"], color="#d62728", linewidth=2.5,
                     label=f"y({x0})={y0}", zorder=5)
        self.ax.legend(fontsize=8)
        self.fig.tight_layout(); self.canvas.draw()
        self.out_text.insert(tk.END, f"\n解曲线: y({x0})={y0}, x∈[{x0},{xend}]\n")
        self.out_text.insert(tk.END, f"终点: y({xend}) ≈ {_format_value(sol['ys'][-1])}\n")

    def _multi_ivp(self):
        expr = self.ode_var.get().strip()
        if not expr:
            return
        try:
            x0 = _parse_bound(self.ivp_x0.get()); xend = _parse_bound(self.ivp_xend.get())
        except ValueError:
            messagebox.showerror("参数错误", "初值必须是数字")
            return
        # 先确保有方向场
        if self._field_data is None:
            self._plot_field()
        y0s = [float(v.strip()) for v in self.multi_y0.get().split(",") if v.strip()]
        colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd",
                  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        for i, y0 in enumerate(y0s):
            sol = OdeEngine.solve_ivp(expr, x0, y0, xend)
            color = colors[i % len(colors)]
            self.ax.plot(sol["xs"], sol["ys"], color=color, linewidth=1.5,
                         label=f"y₀={y0}", zorder=4, alpha=0.8)
        self.ax.legend(fontsize=7, ncol=2)
        self.fig.tight_layout(); self.canvas.draw()

    def _save(self):
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("All", "*.*")])
        if path:
            self.fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")


# ============================================================================
# Tab 5: 多元微积分 + 3D 曲面
# ============================================================================

class MultivariableTab(ttk.Frame):
    """偏导数 / 梯度 / 方向导数 / Hessian / 拉格朗日 + 3D 曲面"""

    trace_cb = None
    def __init__(self, parent, root):
        super().__init__(parent)
        self.root = root
        self._build()

    def _build(self):
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(paned, width=320)
        paned.add(left, weight=0)

        ttk.Label(left, text="⊿ 多元微积分", font=FONT_TITLE).pack(anchor=tk.W, padx=8, pady=(8, 4))
        frm = ttk.Frame(left)
        frm.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(frm, text="f(x,y)=", font=FONT).pack(side=tk.LEFT)
        self.expr_var = tk.StringVar(value="x^2 + y^2")
        ttk.Entry(frm, textvariable=self.expr_var, font=FONT_MONO, width=18).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # 子标签
        sub_nb = ttk.Notebook(left)
        sub_nb.pack(fill=tk.BOTH, expand=True, padx=2, pady=4)

        # -- 偏导/梯度 --
        s1 = ttk.Frame(sub_nb); sub_nb.add(s1, text="梯度&方向导数")
        self._build_sub_gradient(s1)

        # -- Hessian & Lagrange --
        s2 = ttk.Frame(sub_nb); sub_nb.add(s2, text="Hessian&乘数")
        self._build_sub_hess_lag(s2)

        # -- 3D 曲面 --
        s3 = ttk.Frame(sub_nb); sub_nb.add(s3, text="3D 曲面")
        self._build_sub_3d(s3)

        # 输出
        out_lab = ttk.LabelFrame(left, text=" 结果 ", padding=4)
        out_lab.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)
        self.out_text = tk.Text(out_lab, font=FONT_MONO, wrap=tk.WORD, height=6,
                                bg="#ffffff", relief=tk.FLAT, padx=4, pady=4)
        self.out_text.pack(fill=tk.BOTH, expand=True)

        # 画布
        right = ttk.Frame(paned)
        paned.add(right, weight=1)
        self.fig = Figure(figsize=(8, 6), dpi=100, facecolor="white")
        self.canvas = FigureCanvasTkAgg(self.fig, right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.fig.tight_layout(); self.canvas.draw()
        ttk.Button(right, text="💾 保存", command=self._save).pack(anchor=tk.E, padx=4, pady=2)

    def _build_sub_gradient(self, parent):
        frm = ttk.Frame(parent); frm.pack(fill=tk.BOTH, padx=4, pady=4)
        ttk.Label(frm, text="在点 (x₀, y₀) =", font=FONT).pack()
        frm_p = ttk.Frame(frm); frm_p.pack(pady=2)
        self.gx_var = tk.StringVar(value="1"); self.gy_var = tk.StringVar(value="2")
        ttk.Entry(frm_p, textvariable=self.gx_var, font=FONT_MONO, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Entry(frm_p, textvariable=self.gy_var, font=FONT_MONO, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Button(frm, text="梯度 ∇f", command=self._gradient).pack(fill=tk.X, pady=2)
        ttk.Label(frm, text="方向 u = (u₁, u₂)", font=FONT).pack()
        frm_d = ttk.Frame(frm); frm_d.pack(pady=2)
        self.ux_var = tk.StringVar(value="1"); self.uy_var = tk.StringVar(value="0")
        ttk.Entry(frm_d, textvariable=self.ux_var, font=FONT_MONO, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Entry(frm_d, textvariable=self.uy_var, font=FONT_MONO, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Button(frm, text="方向导数 D_u f", command=self._directional).pack(fill=tk.X, pady=2)

    def _build_sub_hess_lag(self, parent):
        frm = ttk.Frame(parent); frm.pack(fill=tk.BOTH, padx=4, pady=4)
        ttk.Button(frm, text="Hessian 矩阵", command=self._hessian).pack(fill=tk.X, pady=3)
        ttk.Separator(frm, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        ttk.Label(frm, text="拉格朗日乘数法", font=("Microsoft YaHei UI", 9, "bold")).pack(pady=2)
        ttk.Label(frm, text="约束 g(x,y) = 0", font=FONT).pack()
        self.lag_g = tk.StringVar(value="x^2 + y^2 - 1")
        ttk.Entry(frm, textvariable=self.lag_g, font=FONT_MONO, width=18).pack(pady=2)
        ttk.Button(frm, text="求条件极值", command=self._lagrange).pack(fill=tk.X, pady=2)

    def _build_sub_3d(self, parent):
        frm = ttk.Frame(parent); frm.pack(fill=tk.BOTH, padx=4, pady=4)
        ttk.Label(frm, text="z = f(x, y)", font=("Microsoft YaHei UI", 9, "bold")).pack(pady=2)
        frm_r = ttk.Frame(frm); frm_r.pack(pady=2)
        ttk.Label(frm_r, text="x,y∈", font=("Microsoft YaHei UI", 7)).pack(side=tk.LEFT)
        self.s3d_lo = tk.StringVar(value="-3"); self.s3d_hi = tk.StringVar(value="3")
        ttk.Entry(frm_r, textvariable=self.s3d_lo, font=FONT_MONO, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Entry(frm_r, textvariable=self.s3d_hi, font=FONT_MONO, width=4).pack(side=tk.LEFT)
        ttk.Button(frm, text="绘制 3D 曲面", command=self._plot_3d).pack(fill=tk.X, pady=3)
        ttk.Label(frm, text="(可用鼠标拖拽旋转/缩放)", font=("Microsoft YaHei UI", 7)).pack()

    def _gradient(self):
        expr = self.expr_var.get().strip()
        try:
            px = _parse_bound(self.gx_var.get()); py = _parse_bound(self.gy_var.get())
        except ValueError:
            messagebox.showerror("错误", "坐标必须是数字"); return
        r = MultivariableEngine.gradient(expr, (px, py))
        try:
            _, T = TracedCalculus.gradient(expr, (px, py))
            if hasattr(self, 'trace_cb') and self.trace_cb:
                self.trace_cb(T)
        except Exception:
            pass
        self.out_text.delete("1.0", tk.END)
        self.out_text.insert(tk.END, f"f(x,y) = {expr}\n")
        self.out_text.insert(tk.END, f"∇f = [{', '.join(r['gradient'])}]\n")
        if "values" in r:
            gx, gy = r['values'][0], r['values'][1] if len(r['values'])>1 else 0
            self.out_text.insert(tk.END, f"\n在 ({px},{py}):\n")
            self.out_text.insert(tk.END, f"∇f = [{gx:.6f}, {gy:.6f}]\n")
            self.out_text.insert(tk.END, f"|∇f| = {_format_value(r['magnitude'])}\n")
            # ── 3D 曲面 + 梯度箭头 ──
            data = MultivariableEngine.surface_sample(expr, (px-2, px+2), (py-2, py+2), 60)
            self.fig.clear()
            self._ax3d = self.fig.add_subplot(111, projection='3d')
            X, Y, Z = np.array(data["X"]), np.array(data["Y"]), np.array(data["Z"])
            self._ax3d.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8, edgecolor="none")
            # 梯度箭头
            import sympy as sp
            x_s, y_s = sp.symbols('x y')
            loc = {"x": x_s, "y": y_s, "sin": sp.sin, "cos": sp.cos, "log": sp.log,
                   "exp": sp.exp, "sqrt": sp.sqrt, "abs": sp.Abs, "pi": sp.pi, "e": sp.E}
            f_sp = sp.sympify(_preprocess_expr(expr), locals=loc)
            f_fn = sp.lambdify((x_s, y_s), f_sp, "numpy")
            pz = float(f_fn(np.array([px]), np.array([py])).item())
            scale = 1.0 / (r['magnitude'] + 0.5) * 1.5  # 箭头长度
            self._ax3d.quiver(px, py, pz, gx*scale, gy*scale, 0,
                              color="red", linewidth=2, arrow_length_ratio=0.2)
            self._ax3d.scatter([px], [py], [pz], color="red", s=50)
            self._ax3d.set_xlabel("x"); self._ax3d.set_ylabel("y"); self._ax3d.set_zlabel("z")
            self._ax3d.set_title(f"∇f({px},{py}) = [{gx:.3f}, {gy:.3f}]   |∇f| = {r['magnitude']:.4f}")
            self.fig.tight_layout(); self.canvas.draw()

    def _directional(self):
        expr = self.expr_var.get().strip()
        try:
            px = _parse_bound(self.gx_var.get()); py = _parse_bound(self.gy_var.get())
            ux = _parse_bound(self.ux_var.get()); uy = _parse_bound(self.uy_var.get())
        except ValueError:
            messagebox.showerror("错误", "坐标/方向必须是数字"); return
        if ux == 0 and uy == 0:
            messagebox.showerror("错误", "方向向量不能为零向量 (0,0)"); return
        r = MultivariableEngine.directional_derivative(expr, (px, py), (ux, uy))
        _try_trace(self, lambda: TracedCalculus.directional_derivative(expr, (px, py), (ux, uy)))
        self.out_text.delete("1.0", tk.END)
        self.out_text.insert(tk.END, f"f(x,y) = {expr}\n")
        self.out_text.insert(tk.END, f"点: ({px},{py}), 方向: ({ux},{uy})\n")
        self.out_text.insert(tk.END, f"单位方向: [{', '.join(f'{v:.4f}' for v in r['unit_direction'])}]\n")
        if "directional_derivative" in r:
            self.out_text.insert(tk.END, f"D_u f = {r['directional_derivative']:.6f}\n")
            # ── 3D 曲面 + 方向箭头 ──
            data = MultivariableEngine.surface_sample(expr, (px-2, px+2), (py-2, py+2), 60)
            self.fig.clear()
            self._ax3d = self.fig.add_subplot(111, projection='3d')
            X, Y, Z = np.array(data["X"]), np.array(data["Y"]), np.array(data["Z"])
            self._ax3d.plot_surface(X, Y, Z, cmap="plasma", alpha=0.8, edgecolor="none")
            import sympy as sp
            x_s, y_s = sp.symbols('x y')
            loc = {"x": x_s, "y": y_s, "sin": sp.sin, "cos": sp.cos, "log": sp.log,
                   "exp": sp.exp, "sqrt": sp.sqrt, "abs": sp.Abs, "pi": sp.pi, "e": sp.E}
            f_sp = sp.sympify(_preprocess_expr(expr), locals=loc)
            f_fn = sp.lambdify((x_s, y_s), f_sp, "numpy")
            pz = float(f_fn(np.array([px]), np.array([py])).item())
            norm = np.sqrt(ux**2 + uy**2)
            udx, udy = ux/norm, uy/norm
            self._ax3d.quiver(px, py, pz, udx, udy, 0,
                              color="cyan", linewidth=2.5, arrow_length_ratio=0.25)
            self._ax3d.scatter([px], [py], [pz], color="cyan", s=50)
            self._ax3d.set_xlabel("x"); self._ax3d.set_ylabel("y"); self._ax3d.set_zlabel("z")
            self._ax3d.set_title(f"D_u f({px},{py}) = {r['directional_derivative']:.4f}  dir=({udx:.2f},{udy:.2f})")
            self.fig.tight_layout(); self.canvas.draw()

    def _hessian(self):
        expr = self.expr_var.get().strip()
        r = MultivariableEngine.hessian(expr)
        _try_trace(self, lambda: TracedCalculus.hessian(expr))
        self.out_text.delete("1.0", tk.END)
        self.out_text.insert(tk.END, f"Hessian of f(x,y) = {expr}\n")
        for row in r["hessian"]:
            self.out_text.insert(tk.END, f"  [{', '.join(row)}]\n")

    def _lagrange(self):
        f_str = self.expr_var.get().strip()
        g_str = self.lag_g.get().strip()
        if not g_str:
            return
        r = MultivariableEngine.lagrange_multipliers(f_str, g_str)
        _try_trace(self, lambda: TracedCalculus.lagrange_multipliers(f_str, g_str))
        self.out_text.delete("1.0", tk.END)
        self.out_text.insert(tk.END, f"拉格朗日乘数法\n")
        self.out_text.insert(tk.END, f"f = {f_str}, 约束 g = {g_str} = 0\n")
        self.out_text.insert(tk.END, f"L = {r.get('lagrangian','')}\n\n")
        if r.get("critical_points"):
            for cp in r["critical_points"]:
                self.out_text.insert(tk.END, f"点: ({cp['x']:.4f}, {cp['y']:.4f}), λ={cp['lambda']:.4f}, f={cp['f']:.6f}\n")

    def _plot_3d(self):
        expr = self.expr_var.get().strip()
        try:
            lo = _parse_bound(self.s3d_lo.get()); hi = _parse_bound(self.s3d_hi.get())
        except ValueError:
            messagebox.showerror("错误", "范围必须是数字"); return
        data = MultivariableEngine.surface_sample(expr, (lo, hi), (lo, hi), 80)
        self.fig.clear()
        self._ax3d = self.fig.add_subplot(111, projection='3d')
        X = np.array(data["X"]); Y = np.array(data["Y"]); Z = np.array(data["Z"])
        self._ax3d.plot_surface(X, Y, Z, cmap="viridis", alpha=0.88, edgecolor="none")
        self._ax3d.contour(X, Y, Z, zdir='z', offset=np.nanmin(Z) - 1,
                           cmap="viridis", alpha=0.5)
        self._ax3d.set_xlabel("x"); self._ax3d.set_ylabel("y"); self._ax3d.set_zlabel("z")
        self._ax3d.set_title(f"z = {expr}")
        self.fig.tight_layout(); self.canvas.draw()

    def _save(self):
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("All", "*.*")])
        if path:
            self.fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")


# ============================================================================
# Tab 6: 拉普拉斯 & 向量分析
# ============================================================================

class LaplaceVectorTab(ttk.Frame):
    """拉普拉斯变换 / 逆变换 + 散度 / 旋度 / 线积分"""

    trace_cb = None
    def __init__(self, parent, root):
        super().__init__(parent)
        self.root = root
        self._build()

    def _build(self):
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(paned, width=320)
        paned.add(left, weight=0)

        ttk.Label(left, text="L 拉普拉斯 & 向量分析", font=FONT_TITLE).pack(anchor=tk.W, padx=8, pady=(8, 4))

        sub_nb = ttk.Notebook(left)
        sub_nb.pack(fill=tk.BOTH, expand=True, padx=2, pady=4)

        # -- 拉普拉斯 --
        s1 = ttk.Frame(sub_nb); sub_nb.add(s1, text="拉普拉斯变换")
        self._build_sub_laplace(s1)

        # -- 向量分析 --
        s2 = ttk.Frame(sub_nb); sub_nb.add(s2, text="向量分析")
        self._build_sub_vector(s2)

        # 输出
        out_lab = ttk.LabelFrame(left, text=" 结果 ", padding=4)
        out_lab.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)
        self.out_text = tk.Text(out_lab, font=FONT_MONO, wrap=tk.WORD, height=8,
                                bg="#ffffff", relief=tk.FLAT, padx=4, pady=4)
        self.out_text.pack(fill=tk.BOTH, expand=True)

        right = ttk.Frame(paned)
        paned.add(right, weight=1)
        self.fig, self.ax = _make_figure()
        self.canvas = FigureCanvasTkAgg(self.fig, right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.fig.tight_layout(); self.canvas.draw()
        ttk.Button(right, text="💾 保存", command=self._save).pack(anchor=tk.E, padx=4, pady=2)

    def _build_sub_laplace(self, parent):
        frm = ttk.Frame(parent); frm.pack(fill=tk.BOTH, padx=4, pady=4)
        ttk.Label(frm, text="f(t) (时域函数)", font=FONT).pack()
        self.lt_var = tk.StringVar(value="sin(t)")
        ttk.Entry(frm, textvariable=self.lt_var, font=FONT_MONO, width=18).pack(pady=2)
        ttk.Button(frm, text="L{f(t)} 正变换", command=self._laplace).pack(fill=tk.X, pady=2)
        ttk.Separator(frm, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        ttk.Label(frm, text="F(s) (频域函数)", font=FONT).pack()
        self.ls_var = tk.StringVar(value="1/(s^2+1)")
        ttk.Entry(frm, textvariable=self.ls_var, font=FONT_MONO, width=18).pack(pady=2)
        ttk.Button(frm, text="L⁻¹{F(s)} 逆变换", command=self._inv_laplace).pack(fill=tk.X, pady=2)

    def _build_sub_vector(self, parent):
        frm = ttk.Frame(parent); frm.pack(fill=tk.BOTH, padx=4, pady=4)
        ttk.Label(frm, text="向量场 F = [Fx, Fy, Fz]",
                  font=("Microsoft YaHei UI", 9, "bold")).pack(pady=2)
        self.vf_var = tk.StringVar(value="x*y, y*z, z*x")
        ttk.Entry(frm, textvariable=self.vf_var, font=FONT_MONO, width=18).pack(pady=2)
        ttk.Label(frm, text="在点 (x,y,z)", font=FONT).pack()
        frm_p = ttk.Frame(frm); frm_p.pack(pady=2)
        self.vx_var = tk.StringVar(value="1"); self.vy_var = tk.StringVar(value="1"); self.vz_var = tk.StringVar(value="1")
        ttk.Entry(frm_p, textvariable=self.vx_var, font=FONT_MONO, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Entry(frm_p, textvariable=self.vy_var, font=FONT_MONO, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Entry(frm_p, textvariable=self.vz_var, font=FONT_MONO, width=4).pack(side=tk.LEFT, padx=2)
        frm_b = ttk.Frame(frm); frm_b.pack(pady=4)
        ttk.Button(frm_b, text="散度 div", command=self._divergence).pack(side=tk.LEFT, padx=2)
        ttk.Button(frm_b, text="旋度 curl", command=self._curl).pack(side=tk.LEFT, padx=2)
        ttk.Separator(frm, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        ttk.Label(frm, text="线积分 ∫_C F·dr", font=("Microsoft YaHei UI", 9, "bold")).pack(pady=2)
        ttk.Label(frm, text="曲线: x(t)=, y(t)=", font=FONT).pack()
        frm_c = ttk.Frame(frm); frm_c.pack(pady=2)
        self.li_xt = tk.StringVar(value="cos(t)"); self.li_yt = tk.StringVar(value="sin(t)")
        ttk.Entry(frm_c, textvariable=self.li_xt, font=FONT_MONO, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Entry(frm_c, textvariable=self.li_yt, font=FONT_MONO, width=8).pack(side=tk.LEFT, padx=2)
        frm_c2 = ttk.Frame(frm); frm_c2.pack(pady=2)
        ttk.Label(frm_c2, text="t∈", font=("Microsoft YaHei UI", 7)).pack(side=tk.LEFT)
        self.li_t0 = tk.StringVar(value="0"); self.li_t1 = tk.StringVar(value="6.283")
        ttk.Entry(frm_c2, textvariable=self.li_t0, font=FONT_MONO, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Entry(frm_c2, textvariable=self.li_t1, font=FONT_MONO, width=5).pack(side=tk.LEFT)
        ttk.Button(frm, text="计算线积分", command=self._line_integral).pack(fill=tk.X, pady=3)

    def _laplace(self):
        expr = self.lt_var.get().strip()
        r = LaplaceEngine.transform(expr)
        try:
            _, T = TracedCalculus.laplace(expr)
            if hasattr(self, 'trace_cb') and self.trace_cb:
                self.trace_cb(T)
        except Exception:
            pass
        self.out_text.delete("1.0", tk.END)
        self.out_text.insert(tk.END, f"拉普拉斯正变换\n")
        self.out_text.insert(tk.END, f"f(t) = {expr}\n")
        self.out_text.insert(tk.END, f"F(s) = L{{f(t)}} = {r['result']}\n")

    def _inv_laplace(self):
        expr = self.ls_var.get().strip()
        r = LaplaceEngine.inverse_transform(expr)
        self.out_text.delete("1.0", tk.END)
        self.out_text.insert(tk.END, f"拉普拉斯逆变换\n")
        self.out_text.insert(tk.END, f"F(s) = {expr}\n")
        self.out_text.insert(tk.END, f"f(t) = L⁻¹{{F(s)}} = {r['result']}\n")

    def _divergence(self):
        expr = self.vf_var.get().strip()
        try:
            px = _parse_bound(self.vx_var.get()); py = _parse_bound(self.vy_var.get()); pz = _parse_bound(self.vz_var.get())
        except ValueError:
            messagebox.showerror("错误", "坐标必须是数字"); return
        r = VectorEngine.divergence(expr, (px, py, pz))
        _try_trace(self, lambda: TracedCalculus.divergence(expr, (px, py, pz)))
        self.out_text.delete("1.0", tk.END)
        self.out_text.insert(tk.END, f"散度 div F\n")
        self.out_text.insert(tk.END, f"F = {expr}\n")
        self.out_text.insert(tk.END, f"div F = {r['divergence']}\n")
        if "value" in r:
            self.out_text.insert(tk.END, f"\n在 ({px},{py},{pz}): div F = {_format_value(r['value'])}\n")

    def _curl(self):
        expr = self.vf_var.get().strip()
        try:
            px = _parse_bound(self.vx_var.get()); py = _parse_bound(self.vy_var.get()); pz = _parse_bound(self.vz_var.get())
        except ValueError:
            messagebox.showerror("错误", "坐标必须是数字"); return
        r = VectorEngine.curl(expr, (px, py, pz))
        _try_trace(self, lambda: TracedCalculus.curl(expr, (px, py, pz)))
        self.out_text.delete("1.0", tk.END)
        self.out_text.insert(tk.END, f"旋度 curl F = ∇ × F\n")
        self.out_text.insert(tk.END, f"F = {expr}\n")
        self.out_text.insert(tk.END, f"curl F = [{', '.join(r['curl'])}]\n")
        if "value" in r:
            self.out_text.insert(tk.END, f"\n在 ({px},{py},{pz}): curl F = [{', '.join(f'{v:.6f}' for v in r['value'])}]\n")

    def _line_integral(self):
        Fstr = self.vf_var.get().strip()
        xt = self.li_xt.get().strip(); yt = self.li_yt.get().strip()
        try:
            t0 = _parse_bound(self.li_t0.get()); t1 = _parse_bound(self.li_t1.get())
        except ValueError:
            messagebox.showerror("错误", "t范围必须是数字"); return
        r = VectorEngine.line_integral(Fstr, xt, yt, (t0, t1))
        _try_trace(self, lambda: TracedCalculus.line_integral(Fstr, xt, yt, (t0, t1)))
        self._clear_ax()
        # 画曲线
        import sympy as sp
        t = sp.Symbol('t')
        loc_t = {"t": t, "sin": sp.sin, "cos": sp.cos, "pi": sp.pi}
        xt_fn = sp.lambdify(t, sp.sympify(_preprocess_expr(xt), locals=loc_t), "numpy")
        yt_fn = sp.lambdify(t, sp.sympify(_preprocess_expr(yt), locals=loc_t), "numpy")
        ts = np.linspace(t0, t1, 300)
        self.ax.plot(xt_fn(ts), yt_fn(ts), color="#1f77b4", linewidth=2, label=f"C: ({xt},{yt})")
        self.ax.set_aspect("equal")
        self.ax.legend(fontsize=8)
        self.fig.tight_layout(); self.canvas.draw()
        self.out_text.delete("1.0", tk.END)
        self.out_text.insert(tk.END, f"线积分 ∫_C F·dr\n")
        self.out_text.insert(tk.END, f"F = {Fstr}, C: ({xt},{yt}), t∈[{t0},{t1}]\n")
        self.out_text.insert(tk.END, f"被积函数: {r['integrand']}\n")
        self.out_text.insert(tk.END, f"符号值: {r['symbolic']}\n")
        if r["numeric"]:
            self.out_text.insert(tk.END, f"数值:   {_format_value(r['numeric'])}\n")

    def _clear_ax(self):
        self.ax.clear(); self.ax.set_facecolor("#fafafa")
        self.ax.grid(True, alpha=0.3, linestyle="--")
        self.ax.axhline(y=0, color="gray", linewidth=0.8, alpha=0.5)
        self.ax.axvline(x=0, color="gray", linewidth=0.8, alpha=0.5)

    def _save(self):
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("All", "*.*")])
        if path:
            self.fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")


# ============================================================================
# 主窗口
# ============================================================================

class MathToolbox:
    """四模块 tab 主应用"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("专业数学分析工具箱 — Math Toolbox")
        self.root.geometry("1280x860")
        self.root.minsize(1000, 680)
        self.root.configure(bg=BG)

        # 样式
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=[18, 6], font=FONT)
        style.map("TNotebook.Tab", background=[("selected", "#ffffff")],
                  foreground=[("selected", ACCENT)])
        style.configure("TFrame", background=BG)
        style.configure("TLabelframe", background=BG)
        style.configure("TLabelframe.Label", background=BG, font=FONT)

        # 标题
        title_bar = tk.Frame(self.root, bg=ACCENT, height=42)
        title_bar.pack(fill=tk.X)
        tk.Label(title_bar, text="📐 专业数学分析工具箱",
                 font=("Microsoft YaHei UI", 15, "bold"),
                 bg=ACCENT, fg="white").pack(side=tk.LEFT, padx=14, pady=6)
        tk.Label(title_bar, text="高等数学全覆盖 · 函数分析 · 微积分 · 级数 · 微分方程  |  Powered by SymPy + SciPy",
                 font=("Microsoft YaHei UI", 8), bg=ACCENT, fg="#c8daf5").pack(side=tk.RIGHT, padx=14)

        # Notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Tab 1: 函数分析
        self.tab1 = FunctionAnalysisTab(self.notebook, self.root)
        self.tab1.trace_cb = self.show_solution
        self.notebook.add(self.tab1, text="  📈 函数分析  ")

        # Tab 2: 微积分
        self.tab2 = CalculusTab(self.notebook, self.root)
        self.tab2.trace_cb = self.show_solution
        self.notebook.add(self.tab2, text="  ∫ 微积分  ")

        # Tab 3: 级数
        self.tab3 = SeriesTab(self.notebook, self.root)
        self.tab3.trace_cb = self.show_solution
        self.notebook.add(self.tab3, text="  ∑ 无穷级数  ")

        # Tab 4: ODE
        self.tab4 = OdeTab(self.notebook, self.root)
        self.tab4.trace_cb = self.show_solution
        self.notebook.add(self.tab4, text="  ⊡ 微分方程  ")

        # Tab 5: 多元微积分
        self.tab5 = MultivariableTab(self.notebook, self.root)
        self.tab5.trace_cb = self.show_solution
        self.notebook.add(self.tab5, text="  ⊿ 多元微积分  ")

        # Tab 6: 拉普拉斯 & 向量
        self.tab6 = LaplaceVectorTab(self.notebook, self.root)
        self.tab6.trace_cb = self.show_solution
        self.notebook.add(self.tab6, text="  L 变换&向量  ")

        # 帮助按钮 (全局)
        help_btn = tk.Button(title_bar, text="?", font=("Microsoft YaHei UI", 12, "bold"),
                             bg="#ffffff", fg=ACCENT, relief=tk.FLAT, width=3,
                             command=self._show_help, cursor="hand2")
        help_btn.pack(side=tk.RIGHT, padx=(0, 8), pady=6)

        # ── 解题过程面板 (底部, 可折叠) ──
        self.solution_visible = tk.BooleanVar(value=False)
        sol_bar = tk.Frame(self.root, bg="#e8e8e8", height=28)
        sol_bar.pack(fill=tk.X, side=tk.BOTTOM, before=self.notebook)
        tk.Checkbutton(sol_bar, text="📝 显示解题过程", variable=self.solution_visible,
                       font=("Microsoft YaHei UI", 9), bg="#e8e8e8",
                       command=lambda: None).pack(side=tk.LEFT, padx=8, pady=2)

        self.solution_panel = tk.Text(self.root, font=("Consolas", 9), wrap=tk.WORD,
                                       height=10, bg="#fffef5", relief=tk.FLAT,
                                       padx=10, pady=8, state=tk.DISABLED)
        # 初始隐藏

        # 状态栏
        status = tk.Frame(self.root, bg="#e0e0e0", height=22)
        status.pack(fill=tk.X, side=tk.BOTTOM)
        tk.Label(status, text="就绪 | 高等数学全覆盖 · 函数分析 · 微积分 · 级数 · ODE · 多元微积分 · 拉普拉斯 · 向量分析",
                 font=("Microsoft YaHei UI", 8), bg="#e0e0e0", fg="#666666").pack(side=tk.LEFT, padx=8)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _show_help(self):
        help_text = """\
═══  专业数学分析工具箱 — 使用帮助  ═══

【支持函数】
  三角: sin cos tan cot sec csc
  反三角: asin acos atan acot asec acsc
  双曲: sinh cosh tanh coth sech csch
  反双曲: asinh acosh atanh acoth asech acsch
  指数对数: exp e^x log ln sqrt
  绝对值: abs(x) |x|
  取整: floor(x) ceiling(x)
  特殊: gamma(x) erf(x)
  常数: pi(π) E(e) EulerGamma(γ) GoldenRatio(φ)

【导数写法】
  sin'(x)      → 对 sin 求导再代入 (得 cos(x))
  (sin(x))'    → 对整体求导 (链式法则)
  (sin(x))''   → 二阶导
  sin''(x)     → 二阶导简写
  d/dx sin(x)  → Leibniz 记号
  d²/dx² sin(x)→ 二阶 Leibniz

【方程】逗号分隔多函数: sin(x), cos(x), x²

【操作技巧】
  - 鼠标滚轮缩放, 按住拖动平移
  - 点击图上任意位置标注坐标
  - 标注输入框支持 pi 表达式: pi/2, 2*pi
  - 上下方向键切换历史记录
  - 预设范围按钮快速切换视图

【Tab 1 · 函数分析】
  输入 f(x) → 回车绘图 → 自动分析:
  零点/极值/拐点/单调/凹凸/渐近线/周期/对称/值域
  额外: 极限计算 / 参数曲线 (x(t),y(t)) / 极坐标 r(θ)

【Tab 2 · 微积分】
  定/不定积分: 输入上下界 a,b (支持 pi, e, pi/2 等)
  反常积分: b 留空表示 ∞, 可设瑕点
  黎曼和: 滑块调 n, 选左/右/中点/梯形
  二重/三重积分: 3D 区域可视化
  弧长曲率: 含密切圆绘制
  数值对比: 梯形/Simpson/Gauss 四种方法

【Tab 3 · 无穷级数】
  泰勒展开: 输入中心 x₀ + 阶数 n
  傅里叶级数: 输入周期 T (默认 2π) + 谐波数
  收敛判定: 通项 aₙ = 1/n², (-1)^n/n 等形式
  幂级数半径: 系数 aₙ → 收敛半径 R
  级数求和: ∑₁ᵢⁿᶠ aₙ 求解析和

【Tab 4 · 微分方程】
  y' = f(x,y): 输入后画方向场
  初值问题: 设 x₀,y₀, 到 x_end → 数值解
  批量解: y₀ 逗号列表, 一次性画多条曲线

【Tab 5 · 多元微积分】
  梯度 ∇f / 方向导数 D_uf / Hessian 矩阵
  拉格朗日乘数法: 求 g(x,y)=0 约束下 f 极值
  3D 曲面: z=f(x,y), 鼠标拖拽旋转

【Tab 6 · 拉普拉斯 & 向量分析】
  拉普拉斯: L{f(t)} 正变换 / L⁻¹{F(s)} 逆变换
  向量场: 散度 div / 旋度 curl
  线积分: ∫_C F·dr, 自定义曲线+向量场
"""
        top = tk.Toplevel(self.root)
        top.title("使用帮助 — Math Toolbox")
        top.geometry("660x740")
        top.configure(bg="#ffffff")
        txt = tk.Text(top, font=("Consolas", 10), wrap=tk.WORD,
                      bg="#ffffff", relief=tk.FLAT, padx=14, pady=14)
        txt.insert("1.0", help_text)
        txt.configure(state=tk.DISABLED)
        txt.pack(fill=tk.BOTH, expand=True)
        ttk.Button(top, text="关闭", command=top.destroy).pack(pady=(0, 10))

    def show_solution(self, tracer: SolutionTracer):
        """固定弹窗 — 首次创建, 之后仅更新文本; 受底部复选框控制"""
        if not self.solution_visible.get():
            return  # 复选框未勾选, 不弹窗
        text = tracer.render()
        if hasattr(self, '_sol_win') and self._sol_win and self._sol_win.winfo_exists():
            self._sol_txt.configure(state=tk.NORMAL)
            self._sol_txt.delete("1.0", tk.END)
            self._sol_txt.insert("1.0", text)
            self._sol_txt.configure(state=tk.DISABLED)
            self._sol_win.lift()
            return
        self._sol_win = tk.Toplevel(self.root)
        self._sol_win.title("解题过程")
        self._sol_win.geometry("680x720")
        self._sol_win.minsize(480, 380)
        self._sol_win.configure(bg="#ffffff")
        self._sol_win.protocol("WM_DELETE_WINDOW", self._on_sol_close)
        tk.Label(self._sol_win, text="📝 解题过程", font=("Microsoft YaHei UI", 13, "bold"),
                 bg="#ffffff", fg=ACCENT).pack(pady=(10, 4))
        frame = ttk.Frame(self._sol_win)
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        sb = ttk.Scrollbar(frame)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._sol_txt = tk.Text(frame, font=("Consolas", 11), wrap=tk.WORD,
                                bg="#fffef5", relief=tk.FLAT, padx=14, pady=12,
                                yscrollcommand=sb.set)
        self._sol_txt.insert("1.0", text)
        self._sol_txt.configure(state=tk.DISABLED)
        self._sol_txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.config(command=self._sol_txt.yview)
        btn_frame = ttk.Frame(self._sol_win)
        btn_frame.pack(pady=(0, 10))
        ttk.Button(btn_frame, text="复制", command=lambda: (
            self._sol_win.clipboard_clear(),
            self._sol_win.clipboard_append(self._sol_txt.get("1.0", tk.END))
        )).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="关闭", command=self._on_sol_close).pack(side=tk.LEFT, padx=4)

    def _on_sol_close(self):
        if hasattr(self, '_sol_win') and self._sol_win:
            self._sol_win.destroy()
        self._sol_win = None

    def _on_close(self):
        plt.close("all")
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ============================================================================
# 入口
# ============================================================================
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    app = MathToolbox()
    app.run()
