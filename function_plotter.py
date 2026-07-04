#!/usr/bin/env python3
"""
================================================================================
  通用函数图像绘制工具  ──  Function Plotter with Full Analysis
================================================================================
  像数学软件一样输入表达式，自动绘制图像并标注：
    · 极大值 / 极小值（自动求解一阶导=0，二阶导判正负）
    · 零点（函数与 x 轴交点）
    · y 轴截距
    · 周期（自动检测周期函数）
    · 垂直渐近线（分母零点）
    · 定义域 / 值域分析

  支持符号输入：sin cos tan log exp sqrt abs |x| ^ π e 等
================================================================================
  用法:
    python function_plotter.py                          # 交互模式
    python function_plotter.py "sin(x)"                 # 单个函数
    python function_plotter.py "sin(x), cos(x)"         # 多个函数
    python function_plotter.py "x^3 - 3x" -r -4 4       # 指定范围
    python function_plotter.py "sin(x)" -s output.png   # 保存图像
================================================================================
"""

from __future__ import annotations
import re
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import Locator, Formatter

import sympy as sp
from sympy import (
    # ── 导数 ──
    Derivative, Subs,
    # ── 6 三角函数 ──
    sin, cos, tan, cot, sec, csc,
    # ── 6 反三角函数 ──
    asin, acos, atan, acot, asec, acsc,
    # ── 6 双曲函数 ──
    sinh, cosh, tanh, coth, sech, csch,
    # ── 6 反双曲函数 ──
    asinh, acosh, atanh, acoth, asech, acsch,
    # ── 指数 & 对数 ──
    log, exp, sqrt,
    # ── 其它常用 ──
    Abs, Piecewise, Heaviside,
    floor, ceiling, sign,
    factorial,
    Max, Min, Mod,
    gamma, erf, erfc,
    Symbol, Expr, S, oo,
    pi, E,
    diff, solveset, denom, periodicity,
    Interval, FiniteSet, Union, ImageSet,
)
from sympy.calculus.util import continuous_domain
from sympy.utilities.lambdify import lambdify

# ============================================================================
# 双语字符串
# ============================================================================
LANG: str = "cn"

T_CN = {
    "app_title":       "通用函数图像绘制工具",
    "app_subtitle":    "输入数学表达式即可绘图 | 覆盖全部初等函数 + 自然常数",
    "prompt":          "请输入表达式",
    "title":           "函数图像: ",
    "xlabel":          "x 轴",
    "ylabel":          "y 轴",
    "max_point":       "极大值",
    "min_point":       "极小值",
    "zero_point":      "零点",
    "y_intercept":     "y截距",
    "period":          "周期",
    "func_expr":       "表达式",
    "domain":          "定义域",
    "range":           "值域",
    "no_extrema":      "未找到极值点",
    "not_periodic":    "非周期函数",
    "vertical_asymptote": "渐近线",
    "piecewise_warn":  "注意: 分段函数部分分析可能不完整",
    "legend_title":    "图例",
    "info_title":      "函数分析结果",
    "quit_help":       "输入 q/quit/exit 退出, h/help 帮助, c/clear 清除",
    "range_set":       "x 范围已设为",
    "saved_to":        "图像已保存到",
    "error_parse":     "无法解析表达式",
    "goodbye":         "再见!",
    "help_text": """
╔══════════════════════════════════════════════════════════════════╗
║                    📖  帮助 ─ 使用说明                           ║
╠══════════════════════════════════════════════════════════════════╣
║  【基本运算】+  -  *  /  ** (幂)  ^ (幂)  ² ³ ⁴ (上标)  √ (根号) ║
║                                                                  ║
║  【6 三角函数】sin(x)  cos(x)  tan(x)  cot(x)  sec(x)  csc(x)   ║
║  【6 反三角】  asin(x)  acos(x)  atan(x)  acot/sec/csc(x)      ║
║               也可写 arcsin, arccos, arctan 等                   ║
║  【6 双曲】   sinh(x)  cosh(x)  tanh(x)  coth/sech/csch(x)     ║
║  【6 反双曲】 asinh(x)  acosh(x)  atanh(x)  acoth/sech/csch    ║
║                                                                  ║
║  【指数】exp(x)  e^x  a^x (任意底)                               ║
║  【对数】log(x)  ln(x)  lg(x)  lb(x)  log(x, 底)               ║
║  【幂】  sqrt(x)  cbrt(x)  x^(任意)                              ║
║                                                                  ║
║  【绝对值/分段】abs(x)  |x|  Heaviside(x)  Piecewise            ║
║  【取整】      floor(x)  ceiling(x)                             ║
║  【符号】      sign(x)  Max(a,b)  Min(a,b)  Mod(x,m)           ║
║  【阶乘/Gamma】factorial(x)  gamma(x)                           ║
║  【误差函数】  erf(x)  erfc(x)                                   ║
║                                                                  ║
║  【自然常数】pi(π)  E(e)  EulerGamma(γ)  GoldenRatio(φ)        ║
║                                                                  ║
║  【隐式乘法】2x→2*x  x(x+1)→x*(x+1)  3sin(x)→3*sin(x)          ║
╠══════════════════════════════════════════════════════════════════╣
║  命令: range a b   save 路径   clear / c   q / quit   h / help  ║
╠══════════════════════════════════════════════════════════════════╣
║  示例: sin(x)  |  x^2-4x+3  |  exp(-0.3x)*sin(5x)  |  tan(x)   ║
║        sin(x), cos(x), x^2  |  e^x  |  log(x)  |  1/(x^2-1)   ║
║        gamma(x)  |  erf(x)  |  sinh(x)  |  acosh(x)           ║
╠══════════════════════════════════════════════════════════════════╣
║  【方程】y=sin(x)  x=sin(y)  x^2+y^2=1  y^2=x  x=y^3-3y      ║
╚══════════════════════════════════════════════════════════════════╝
""",
}

T_EN = {
    "app_title":       "Function Plotter",
    "app_subtitle":    "Enter math expressions | Symbols: sin cos tan log exp sqrt abs |x| ^ pi e",
    "prompt":          "Enter expression",
    "title":           "Function Plot: ",
    "xlabel":          "x-axis",
    "ylabel":          "y-axis",
    "max_point":       "Local Max",
    "min_point":       "Local Min",
    "zero_point":      "Root / Zero",
    "y_intercept":     "y-intercept",
    "period":          "Period",
    "func_expr":       "Expression",
    "domain":          "Domain",
    "range":           "Range",
    "no_extrema":      "No extrema found",
    "not_periodic":    "Not periodic",
    "vertical_asymptote": "Asymptote",
    "piecewise_warn":  "Note: piecewise analysis may be incomplete",
    "legend_title":    "Legend",
    "info_title":      "Analysis Result",
    "quit_help":       "Type q/quit/exit to quit, h/help for help, c/clear to clear",
    "range_set":       "x range set to",
    "saved_to":        "Image saved to",
    "error_parse":     "Cannot parse expression",
    "goodbye":         "Goodbye!",
    "help_text": """
╔══════════════════════════════════════════════════════════════╗
║                    HELP ─ Usage Guide                        ║
╠══════════════════════════════════════════════════════════════╣
║  Arithmetic    +  -  *  /  ** (power)  ^ (power)            ║
║  Trig          sin(x)  cos(x)  tan(x)  cot(x)               ║
║  Inverse       asin(x)  acos(x)  atan(x)                    ║
║  Hyperbolic    sinh(x)  cosh(x)  tanh(x)                    ║
║  Exp/Log       exp(x)  log(x)  log(x,10)  sqrt(x)           ║
║  Absolute      abs(x)  or |x|                                ║
║  Constants     pi  π  E  e                                   ║
║  Floor/Ceil    floor(x)  ceiling(x)                          ║
║  Sign/Step     sign(x)  Heaviside(x)                         ║
║  Factorial     factorial(x)                                  ║
╠══════════════════════════════════════════════════════════════╣
║  Commands                                                     ║
║    range a b    Set x-axis range (default -8, 8)             ║
║    save path    Save figure to file                          ║
║    clear / c    Clear current figure                         ║
║    q / quit     Exit                                         ║
║    h / help     Show this help                               ║
╚══════════════════════════════════════════════════════════════╝
""",
}


def _t(key: str) -> str:
    tbl = T_CN if LANG == "cn" else T_EN
    return tbl.get(key, key)


# ============================================================================
# 表达式预处理  —  让用户像在纸上一样写公式
# ============================================================================

def _preprocess_expr(expr_str: str) -> str:
    """预处理输入字符串, 兼容自然数学写法 → sympy 语法"""
    s = expr_str.strip()
    # 清除不可见字符 (零宽空格、软连字符等, 从网页/PDF 复制时常带入)
    s = re.sub(r'[​‌‍‎‏­﻿⁠]', '', s)

    # ── 上标 / Unicode 符号 ──
    s = s.replace("²", "**2")
    s = s.replace("³", "**3")
    s = s.replace("⁴", "**4")
    s = s.replace("⁵", "**5")
    s = s.replace("⁶", "**6")
    s = s.replace("⁷", "**7")
    s = s.replace("⁸", "**8")
    s = s.replace("⁹", "**9")
    s = s.replace("⁰", "**0")
    s = s.replace("√", "sqrt")
    s = s.replace("π", "pi")
    s = s.replace("θ", "theta")
    s = s.replace("×", "*")
    s = s.replace("÷", "/")
    s = s.replace("∞", "oo")
    s = s.replace("≤", "<=")
    s = s.replace("≥", ">=")
    s = s.replace("≠", "!=")

    # ── 三角函数别名 ──
    s = re.sub(r'\barcsin\b', 'asin', s)
    s = re.sub(r'\barccos\b', 'acos', s)
    s = re.sub(r'\barctan\b', 'atan', s)
    s = re.sub(r'\barccot\b', 'acot', s)
    s = re.sub(r'\barcsec\b', 'asec', s)
    s = re.sub(r'\barccsc\b', 'acsc', s)
    s = re.sub(r'\barsinh?\b', 'asinh', s)
    s = re.sub(r'\barcosh?\b', 'acosh', s)
    s = re.sub(r'\bartanh?\b', 'atanh', s)
    s = re.sub(r'\btg\b', 'tan', s)
    s = re.sub(r'\bctg\b', 'cot', s)
    s = re.sub(r'\bsh\b', 'sinh', s)
    s = re.sub(r'\bch\b', 'cosh', s)
    s = re.sub(r'\bth\b', 'tanh', s)
    s = re.sub(r'\blg\b', 'log10', s)
    s = re.sub(r'\bln\b', 'log', s)
    s = re.sub(r'\blb\b', 'log2', s)
    s = re.sub(r'\bsinc\b', 'sinc', s)
    # ── 常数别名 (注意: 只替换独立出现的, 不影响函数调用) ──
    s = re.sub(r'\bphi\b', 'GoldenRatio', s)      # phi → 黄金比例
    s = re.sub(r'\bvarphi\b', 'GoldenRatio', s)
    s = re.sub(r'φ\b', 'GoldenRatio', s)           # φ → 黄金比例

    # ── e^... → exp(...)   [在 ^ → ** 之前处理] ──
    s = _convert_e_power(s)

    # ── 导数记号: sin'(x) / (expr)' / d/dx / d^2/dx^2  [在 ^→** 之前] ──
    s = _fix_derivative(s)

    # ── 幂运算: ^ → ** ──
    s = s.replace("^", "**")

    # ── 绝对值 |...| → Abs(...) ──
    s = _fix_abs(s)

    # ── 隐式乘法: 2x→2*x, x(x+1)→x*(x+1) ──
    s = _fix_implicit_mul(s)

    # ── 统一 log 写法 ──
    s = re.sub(r'\blog10\b', 'log(x,10)', s)
    s = re.sub(r'\blog2\b', 'log(x,2)', s)

    return s


def _convert_e_power(s: str) -> str:
    """将 e^{...}, e^(...), e^x 等转换为 exp(...) 形式"""
    # e^{...} → exp(...)
    s = re.sub(r'\be\s*\^\s*\{([^}]*)\}', r'exp(\1)', s)
    # e^(...) → exp(...)  — 匹配括号内不含嵌套括号的内容
    s = re.sub(r'\be\s*\^\s*\(([^()]*)\)', r'exp(\1)', s)
    # e^number 或 e^var → exp(number)/exp(var)
    s = re.sub(r'\be\s*\^\s*([a-zA-Z0-9.]+)', r'exp(\1)', s)
    return s


def _fix_derivative(s: str) -> str:
    """将导数记号转为 sympy 可计算的形式 (会在 analyze 中 .doit())

    语义:
      f'(g(x))    = 先对 f 求导, 再代入 g(x)     → Subs(Derivative(f(t),t), t, g(x))
      f''(g(x))   = 二阶导, 再代入              → Subs(Derivative(f(t),(t,2)), t, g(x))
      (f(x))'     = 对整体求导 (链式法则)        → Derivative(f(x), x)
      (f(x))''    = 二阶导                      → Derivative(f(x), (x, 2))
      d/dx f(x)   = 对整体求导                   → Derivative(f(x), x)
      d²/dx² f(x) = 二阶导                      → Derivative(f(x), (x, 2))
    """
    # 1. func'(args) / func''(args) / func'''(args)
    #    先对 func 求导 (可多阶), 再代入 args
    result = []
    i = 0
    while i < len(s):
        m = re.match(r"(\w+)('+)\((\w)", s[i:])
        if m:
            func = m.group(1)
            primes = m.group(2)
            order = len(primes)
            paren_pos = i + len(func) + order
            depth = 1
            j = paren_pos + 1
            while j < len(s) and depth > 0:
                if s[j] == '(':
                    depth += 1
                elif s[j] == ')':
                    depth -= 1
                j += 1
            arg = s[paren_pos + 1:j - 1]
            if order == 1:
                result.append(f"Subs(Derivative({func}(t), t), t, {arg})")
            else:
                result.append(f"Subs(Derivative({func}(t), (t, {order})), t, {arg})")
            i = j
        else:
            result.append(s[i])
            i += 1
    s = ''.join(result)

    # 2. (expr)' / func(expr)' / (expr)'' → Derivative(expr, x [, order])
    #    也处理 sin(x)'' (撇号在 ) 后面, 前面有函数名)
    result2 = []
    i = 0
    while i < len(s):
        if s[i] == '(':
            depth = 1
            j = i + 1
            while j < len(s) and depth > 0:
                if s[j] == '(':
                    depth += 1
                elif s[j] == ')':
                    depth -= 1
                j += 1
            # j 在匹配的 ) 之后, 检查后面跟了几个 '
            k = j
            while k < len(s) and s[k] == "'":
                k += 1
            order = k - j
            if order >= 1:
                inner = s[i + 1:j - 1]
                # 回溯: 如果 ( 前面有函数名, 包进来; 并从 result2 中移除已追加的字符
                func_start = i - 1
                while func_start >= 0 and (s[func_start].isalpha() or s[func_start].isdigit()):
                    func_start -= 1
                func_start += 1
                if func_start < i:
                    # 移除 result2 中已追加的函数名字符
                    remove_len = i - func_start
                    while remove_len > 0 and result2:
                        remove_len -= 1
                        result2.pop()
                    inner = s[func_start:i] + "(" + inner + ")"
                else:
                    inner = "(" + inner + ")"
                if order == 1:
                    result2.append(f"Derivative({inner}, x)")
                else:
                    result2.append(f"Derivative({inner}, (x, {order}))")
                i = k
                continue
        result2.append(s[i])
        i += 1
    s = ''.join(result2)

    # 3. d/dx f(x) / d^2/dx^2 f(x)
    s = re.sub(r"d\^?(\d*)/dx\^?\1\s+(\w+\([^)]+\))",
               lambda m: f"Derivative({m.group(2)}, (x, {m.group(1) or 1}))", s)
    s = re.sub(r"d\^?(\d*)/dx\^?\1\s+\(([^()]+)\)",
               lambda m: f"Derivative({m.group(2)}, (x, {m.group(1) or 1}))", s)
    # fallback: d/dx
    s = re.sub(r"d/dx\s+(\w+\([^)]+\))", r"Derivative(\1, x)", s)
    s = re.sub(r"d/dx\s+\(([^()]+)\)", r"Derivative(\1, x)", s)

    return s


def _fix_abs(s: str) -> str:
    """将 |...| 转为 Abs(...)"""
    while "|" in s:
        start = s.find("|")
        end = s.find("|", start + 1)
        if end == -1:
            break
        inner = s[start + 1:end]
        s = s[:start] + f"Abs({inner})" + s[end + 1:]
    return s


def _fix_implicit_mul(s: str) -> str:
    """隐式乘法补全: 2x→2*x, 3sin(x)→3*sin(x), x(x+1)→x*(x+1)"""
    # 数字后跟字母
    s = re.sub(r'(\d)([a-zA-Zα-ωπ])', r'\1*\2', s)
    # 数字后跟左括号: 2( → 2*(
    s = re.sub(r'(\d)\s*\(', r'\1*(', s)
    # 右括号后跟字母: )x → )*x
    s = re.sub(r'\)\s*([a-zA-Z])', r')*\1', s)
    # 右括号后跟左括号: )( → )*(
    s = re.sub(r'\)\s*\(', r')*(', s)
    # 字母后跟左括号: x( → x*( (但要保护函数名)
    # 只对非函数名做
    s = re.sub(r'(?<![a-zA-Z])x\s*\(', r'x*(', s)
    return s


# ============================================================================
# π 刻度 — 让三角函数图的 X 轴显示 π 的分数
# ============================================================================

class PiTickLocator(Locator):
    """在 π/2 的整数倍处放置刻度"""
    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        ticks = []
        step = np.pi / 2
        start = np.ceil(vmin / step) * step
        v = start
        while v <= vmax + 1e-10:
            ticks.append(float(v))
            v += step
        return ticks


class PiTickFormatter(Formatter):
    """将数值格式化为 π 分数: 0, π/2, π, 3π/2, 2π, ..."""
    def __call__(self, x, pos=None):
        # 在分母 1~6 范围内寻找最简 π 分数
        best = None
        best_err = 0.03
        for d in range(1, 7):
            n = round(x / np.pi * d)
            val = n * np.pi / d
            err = abs(val - x)
            if err < best_err:
                best_err = err
                best = (n, d)
        if best is None:
            return f"{x:.2f}"
        n, d = best
        # 简化分数
        from math import gcd
        g = gcd(abs(n), d)
        n //= g; d //= g

        if n == 0:
            return "0"
        if d == 1:
            if n == 1: return "π"
            if n == -1: return "-π"
            return f"{n}π"
        if n == 1: return f"π/{d}"
        if n == -1: return f"-π/{d}"
        return f"{n}π/{d}"


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class Extremum:
    x: float
    y: float
    kind: str  # "max" | "min"


@dataclass
class Interval:
    """单调 / 凹凸区间, None 表示 ±∞"""
    start: float | None
    end: float | None
    label: str  # ↗ ↗↗ concave_up concave_down


@dataclass
class AnalysisResult:
    expr: Expr
    x: Symbol
    f_lambda: Callable
    expr_str: str = ""
    extrema: list[Extremum] = field(default_factory=list)
    zeros: list[float] = field(default_factory=list)
    y_intercept: float | None = None
    period_value: float | None = None
    domain_info: str = ""
    range_info: str = ""
    asymptotes: list[float] = field(default_factory=list)
    h_asymptotes: list[float] = field(default_factory=list)  # 水平渐近线
    obliques: list[tuple[float, float]] = field(default_factory=list)  # 斜渐近线 [(m,b)]
    inflections: list[float] = field(default_factory=list)  # 拐点 x 坐标
    monotonic: list[Interval] = field(default_factory=list)  # 单调区间
    concavity: list[Interval] = field(default_factory=list)  # 凹凸区间
    symmetry: str = ""  # "even" | "odd" | ""
    is_piecewise: bool = False


# ============================================================================
# 核心: FunctionPlotter
# ============================================================================

class FunctionPlotter:
    """通用函数图像绘制器 ── 输入表达式, 自动分析并绘制"""

    _DEFAULT_COLORS = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf",
    ]

    def __init__(
        self,
        x_range: tuple[float, float] = (-8, 8),
        num_points: int = 3000,
        figsize: tuple[float, float] = (15, 9),
        dpi: int = 130,
        lang: str = "cn",
        style: str = "default",
    ):
        global LANG
        LANG = lang if lang in ("cn", "en") else "cn"
        self.x_range = x_range
        self.num_points = num_points
        self.figsize = figsize
        self.dpi = dpi

        self._analyses: list[AnalysisResult] = []
        self._color_idx = 0
        self._fig: Any = None
        self._ax: Any = None

        # 全局抑制 numpy 的 sqrt/divide 警告 (域外求值很正常)
        np.seterr(all='ignore')

        # 样式必须在字体之前设置 (plt.style.use 会重置 rcParams)
        if style == "dark":
            plt.style.use("dark_background")
        else:
            plt.style.use("default")

        self._setup_chinese()

    # ── 字体 ──────────────────────────────────────────────────
    @staticmethod
    def _setup_chinese() -> None:
        global LANG
        try:
            import matplotlib.font_manager as fm
            preferred = [
                "Microsoft YaHei", "SimHei", "PingFang SC",
                "Noto Sans CJK SC", "WenQuanYi Micro Hei",
                "Arial Unicode MS", "DejaVu Sans",
            ]
            # 找一个支持中文的 monospace 字体
            mono_candidates = [
                "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                "FangSong", "KaiTi", "DejaVu Sans Mono",
            ]
            available = {f.name for f in fm.fontManager.ttflist}
            for font in preferred:
                if font in available:
                    matplotlib.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
                    matplotlib.rcParams["axes.unicode_minus"] = False
                    break
            # 同时设置 monospace 后备
            for mf in mono_candidates:
                if mf in available:
                    matplotlib.rcParams["font.monospace"] = [mf, "DejaVu Sans Mono"]
                    break
        except Exception:
            pass
        # 如果没有任何中文字体, 切到英文
        if "Microsoft YaHei" not in str(matplotlib.rcParams.get("font.sans-serif", "")) \
           and "SimHei" not in str(matplotlib.rcParams.get("font.sans-serif", "")):
            LANG = "en"

    # ── 完整分析 ──────────────────────────────────────────────
    def analyze(self, raw_expr: str) -> AnalysisResult:
        """对表达式进行完整分析：零点、极值、周期、渐近线、定义域、值域"""
        x = Symbol("x", real=True)
        preprocessed = _preprocess_expr(raw_expr)

        # 解析为 sympy 表达式
        local_dict = {
            "x": x, "t": Symbol("t"),
            # ── 6 三角 ──
            "sin": sin, "cos": cos, "tan": tan,
            "cot": cot, "sec": sec, "csc": csc,
            # ── 6 反三角 ──
            "asin": asin, "acos": acos, "atan": atan,
            "acot": acot, "asec": asec, "acsc": acsc,
            # ── 6 双曲 ──
            "sinh": sinh, "cosh": cosh, "tanh": tanh,
            "coth": coth, "sech": sech, "csch": csch,
            # ── 6 反双曲 ──
            "asinh": asinh, "acosh": acosh, "atanh": atanh,
            "acoth": acoth, "asech": asech, "acsch": acsch,
            # ── 指对幂 ──
            "exp": exp, "log": log, "sqrt": sqrt,
            # ── 绝对值 / 分段 / 取整 ──
            "Abs": Abs, "abs": Abs,
            "Heaviside": Heaviside,
            "floor": floor, "ceiling": ceiling,
            "sign": sign, "factorial": factorial,
            "Max": Max, "Min": Min, "Mod": Mod,
            "Piecewise": Piecewise,
            # ── 特殊函数 ──
            "gamma": gamma, "erf": erf, "erfc": erfc,
            # ── 常数 ──
            "Derivative": Derivative, "Subs": Subs,
            "pi": pi, "E": E, "e": E,
            "EulerGamma": sp.EulerGamma, "eulergamma": sp.EulerGamma,
            "GoldenRatio": sp.GoldenRatio, "goldenratio": sp.GoldenRatio,
        }
        try:
            expr = sp.sympify(preprocessed, locals=local_dict)
        except Exception as e:
            raise ValueError(f"{_t('error_parse')} '{raw_expr}': {e}")

        # 计算导数标记 (sin'(x) → cos(x), cot'(sqrt(x)) → -1/sin²(sqrt(x)))
        if expr.has(Derivative) or expr.has(Subs):
            expr = expr.doit()

        is_pw = expr.has(Piecewise, Heaviside, floor, ceiling, sign, Abs)

        # Lambdify — 数值化
        f_lambda = self._make_lambda(expr, x)

        result = AnalysisResult(
            expr=expr, x=x, f_lambda=f_lambda,
            expr_str=raw_expr, is_piecewise=is_pw,
        )

        # 定义域
        if self._count_ops(expr) < 40:
            try:
                dom = continuous_domain(expr, x, S.Reals)
                result.domain_info = self._format_domain(dom, x)
            except Exception:
                result.domain_info = "R"
        else:
            result.domain_info = "R"

        # 对称性
        result.symmetry = self._detect_symmetry(expr, x)

        # 周期 (提前计算, 零点/拐点展开要用)
        result.period_value = self._find_period(expr, x)

        # 渐近线 (垂直 + 水平 + 斜)
        result.asymptotes = self._find_asymptotes(expr, x)
        result.h_asymptotes = self._find_horizontal_asymptotes(expr, x)
        result.obliques = self._find_oblique_asymptotes(expr, x)

        # 零点
        result.zeros = self._find_zeros(expr, x, exclude=result.asymptotes,
                                         period=result.period_value)

        # y 截距
        result.y_intercept = self._find_y_intercept(expr, x)

        # 极值 (先符号后数值)
        result.extrema = self._find_extrema(expr, x)

        # 拐点
        result.inflections = self._find_inflections(expr, x, period=result.period_value)

        # 单调性
        result.monotonic = self._find_monotonic(expr, x, result.extrema, result.asymptotes,
                                                  period=result.period_value)

        # 凹凸性
        result.concavity = self._find_concavity(expr, x, result.inflections, result.asymptotes,
                                                  period=result.period_value)

        # 值域 (数值估算)
        result.range_info = self._find_range(expr, x, result)

        return result

    def _make_lambda(self, expr: Expr, x: Symbol) -> Callable:
        import scipy.special as scisp  # optional
        mods: Any = ["numpy", {
            # ── 三角 (numpy 原生) ──
            "sin": np.sin, "cos": np.cos, "tan": np.tan,
            # ── 三角 (手动实现, numpy 无) ──
            "cot": lambda v: 1.0 / np.tan(v),
            "sec": lambda v: 1.0 / np.cos(v),
            "csc": lambda v: 1.0 / np.sin(v),
            # ── 反三角 (numpy 原生) ──
            "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
            # ── 反三角 (手动) ──
            "acot": lambda v: np.pi / 2 - np.arctan(v),
            "asec": lambda v: np.arccos(1.0 / v),
            "acsc": lambda v: np.arcsin(1.0 / v),
            # ── 双曲 (numpy 原生) ──
            "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
            # ── 双曲 (手动) ──
            "coth": lambda v: 1.0 / np.tanh(v),
            "sech": lambda v: 1.0 / np.cosh(v),
            "csch": lambda v: 1.0 / np.sinh(v),
            # ── 反双曲 (numpy 原生) ──
            "asinh": np.arcsinh, "acosh": np.arccosh, "atanh": np.arctanh,
            # ── 反双曲 (手动) ──
            "acoth": lambda v: np.arctanh(1.0 / v),
            "asech": lambda v: np.arccosh(1.0 / v),
            "acsch": lambda v: np.arcsinh(1.0 / v),
            # ── 指对幂 ──
            "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
            # ── 取整 / 符号 ──
            "Abs": np.abs, "abs": np.abs,
            "floor": np.floor, "ceiling": np.ceil, "sign": np.sign,
            "Heaviside": lambda v: np.heaviside(v, 0.5),
            "factorial": lambda v: scisp.gamma(v + 1),
            "Max": np.maximum, "Min": np.minimum,
            "Mod": np.fmod,
            # ── 特殊函数 (scipy) ──
            "gamma": scisp.gamma,
            "erf": scisp.erf, "erfc": scisp.erfc,
            # ── 常数 ──
            "pi": np.pi, "E": np.e,
            "EulerGamma": np.float64(0.5772156649015329),
            "GoldenRatio": np.float64(1.618033988749895),
        }]
        try:
            f = lambdify(x, expr, modules=mods)
            f(np.array([0.5], dtype=float))
            return f
        except Exception:
            return lambdify(x, expr, modules=["numpy"])

    @staticmethod
    def _format_domain(dom, x) -> str:
        """将 sympy 定义域转为可读字符串"""
        s = str(dom)
        if s == "Reals":
            return "R"
        # 太复杂的域显示简化版
        if len(s) > 120:
            # 提取主要区间
            if "Complement((0, oo)" in s or "Complement(Interval.open(0, oo)" in s:
                return "x > 0 (排除部分点)"
            if "Complement(Reals," in s:
                return "R (排除部分点)"
            return "定义域复杂, 见图"
        # Complement(Reals, Union(ImageSet(...), ...)) → "R \ {kπ | k∈Z}"
        if "Complement(Reals," in s:
            # 尝试提取 ImageSet 的模式
            import re as _re
            imagesets = _re.findall(r'ImageSet\(Lambda\(_n, ([^)]+)\), Integers\)', s)
            if imagesets:
                parts = []
                for formula in imagesets:
                    formula = formula.replace('**', '^').replace(' ', '')
                    formula = formula.replace('_n', 'k')
                    formula = formula.replace('*', '')
                    # 去掉前导的 +
                    formula = _re.sub(r'^\+', '', formula)
                    parts.append(formula)
                # 简化常见模式
                patterns = [
                    (["2kpi", "2kpi+pi"], "kpi"),
                    (["2kpi+pi/2", "2kpi+3pi/2"], "kpi+pi/2"),
                    (["2kpi+pi", "2kpi"], "kpi"),  # 顺序可能颠倒
                    (["2kpi+3pi/2", "2kpi+pi/2"], "kpi+pi/2"),
                ]
                parts_set = set(parts)
                for pat, simplified in patterns:
                    if parts_set == set(pat):
                        parts = [simplified]
                        break
                exclude_str = ", ".join(parts)
                return f"R \\ {{{exclude_str} | k∈Z}}"
            return "R \\ {{...}}"  # 有排除点但无法解析
        # Interval 类型
        if "Interval.open" in s:
            return s.replace("Interval.open", "").replace("oo", "∞").replace(",", ", ")
        if "Interval" in s:
            return s.replace("Interval", "").replace("oo", "∞").replace(",", ", ")
        if "Union" in s and "ImageSet" not in s:
            return s.replace("Union", "").replace("Interval.open", "").replace("oo", "∞").replace(",", ", ").replace("(", "[").replace(")", "]")
        return s

    @staticmethod
    def _count_ops(expr: Expr) -> int:
        """估算表达式复杂度：操作节点数量（快速版，截断）"""
        try:
            n = 0
            for _ in sp.preorder_traversal(expr):
                n += 1
                if n > 500:  # 截断，避免遍历超大表达式
                    return 1000
            return n
        except Exception:
            return 1000

    @staticmethod
    def _to_float(val) -> float:
        """安全地将 numpy 标量/数组或普通数值转换为 Python float"""
        if isinstance(val, np.ndarray):
            return float(val.item())
        return float(val)

    # ── 零点 ──────────────────────────────────────────────────
    def _find_zeros(self, expr: Expr, x: Symbol, exclude: list[float] | None = None,
                     period: float | None = None) -> list[float]:
        """先符号求解, 周期展开, 去重"""
        ops = self._count_ops(expr)
        has_trig = expr.has(sp.sin, sp.cos, sp.tan, sp.cot)
        has_pw = expr.has(sp.floor, sp.ceiling, sp.sign, sp.Heaviside, sp.Piecewise)
        if ops < 20 and not (ops > 5 and has_trig) and not has_pw:
            try:
                sols = solveset(expr, x, domain=S.Reals)
                z = self._flatten_set(sols)
                if z:
                    z = self._expand_periodic(z, period)
                    return self._dedup_zeros(z, exclude)
            except Exception:
                pass
        z = self._numeric_scan(expr, x)
        z = self._expand_periodic(z, period)
        return self._dedup_zeros(z, exclude)

    def _expand_periodic(self, candidates: list[float], period: float | None) -> list[float]:
        """对周期函数展开候选点: 在 x_range 内复制 ±n*T"""
        if not period or period < 0.1:
            return candidates
        x0, x1 = self.x_range
        margin = period * 0.55  # 多展开半个周期, 保证覆盖边缘
        expanded = set()
        for c in candidates:
            for n in range(-5, 6):
                p = round(c + n * period, 8)
                if x0 - margin <= p <= x1 + margin:
                    expanded.add(p)
        return sorted(expanded)

    @staticmethod
    def _dedup_zeros(zeros: list[float], exclude: list[float] | None) -> list[float]:
        """去重: 合并相近值 + 规范 -0.000→0.000 + 过滤渐近线伪根"""
        if not zeros:
            return []
        merged = []
        for z in sorted(zeros):
            if merged and abs(z - merged[-1]) < 0.001:
                continue
            v = round(z, 8)
            if abs(v) < 0.0005:
                v = 0.0  # 规范化 -0.000 → 0.000
            merged.append(v)
        if not exclude:
            return merged
        return [z for z in merged
                if not any(abs(z - asym) < 0.05 for asym in exclude)]

    @staticmethod
    def _filter_pseudo_roots(zeros: list[float], exclude: list[float] | None) -> list[float]:
        """过滤掉靠近渐近线的伪零点"""
        if not exclude:
            return zeros
        result = []
        for z in zeros:
            is_pseudo = any(abs(z - asym) < 0.05 for asym in exclude)
            if not is_pseudo:
                result.append(z)
        return result

    # ── 极值 ──────────────────────────────────────────────────
    def _find_extrema(self, expr: Expr, x: Symbol) -> list[Extremum]:
        """一阶导=0 + 二阶导判定 → 极大/极小值"""
        extrema: list[Extremum] = []
        x0, x1 = self.x_range
        try:
            df = diff(expr, x)
            d2f = diff(df, x)

            # 符号求解临界点 — 仅对简单非三角/非分段表达式
            cp_vals: list[float] = []
            ops = self._count_ops(df)
            has_trig = df.has(sp.sin, sp.cos, sp.tan, sp.cot)
            has_pw = df.has(sp.floor, sp.ceiling, sp.sign, sp.Heaviside, sp.Piecewise)
            if ops < 20 and not (ops > 5 and has_trig) and not has_pw:
                try:
                    cp_sols = solveset(df, x, domain=S.Reals)
                    cp_vals = self._flatten_set(cp_sols)
                except Exception:
                    pass

            f_l = lambdify(x, expr, modules="numpy")
            d2f_l = lambdify(x, d2f, modules="numpy")

            for cp in cp_vals:
                if not (x0 - 0.1 <= cp <= x1 + 0.1):
                    continue
                try:
                    fy = self._to_float(f_l(np.array([cp])))
                    if not math.isfinite(fy) or abs(fy) > 1e100:
                        continue
                    dd = self._to_float(d2f_l(np.array([cp])))
                    if dd < -1e-8:
                        extrema.append(Extremum(cp, fy, "max"))
                    elif dd > 1e-8:
                        extrema.append(Extremum(cp, fy, "min"))
                    else:
                        eps = 1e-4
                        left = self._to_float(f_l(np.array([cp - eps])))
                        right = self._to_float(f_l(np.array([cp + eps])))
                        if math.isfinite(left) and math.isfinite(right):
                            if fy > left and fy > right:
                                extrema.append(Extremum(cp, fy, "max"))
                            elif fy < left and fy < right:
                                extrema.append(Extremum(cp, fy, "min"))
                except Exception:
                    continue

            # 数值补充搜索
            df_l = lambdify(x, df, modules="numpy")
            numeric_cps = self._numeric_scan(df, x)
            for cp in numeric_cps:
                if any(abs(cp - e.x) < 1e-5 for e in extrema):
                    continue
                try:
                    fy = self._to_float(f_l(np.array([cp])))
                    if not math.isfinite(fy) or abs(fy) > 1e100:
                        continue
                    dd = self._to_float(d2f_l(np.array([cp])))
                    if dd < -1e-8:
                        extrema.append(Extremum(cp, fy, "max"))
                    elif dd > 1e-8:
                        extrema.append(Extremum(cp, fy, "min"))
                except Exception:
                    continue

        except Exception:
            pass

        extrema.sort(key=lambda e: e.x)
        x0, x1 = self.x_range
        merged: list[Extremum] = []
        for e in extrema:
            if merged and abs(e.x - merged[-1].x) < 1e-4:
                continue
            if x0 - 0.5 <= e.x <= x1 + 0.5:
                merged.append(e)
        return merged

    # ── y 截距 ────────────────────────────────────────────────
    def _find_y_intercept(self, expr: Expr, x: Symbol) -> float | None:
        try:
            val = float(expr.subs(x, 0).evalf())
            if math.isfinite(val):
                return val
        except Exception:
            pass
        return None

    # ── 周期 ──────────────────────────────────────────────────
    def _find_period(self, expr: Expr, x: Symbol) -> float | None:
        if self._count_ops(expr) > 30:
            return None
        try:
            p = periodicity(expr, x)
            if p is not None and p != 0 and p != oo:
                val = float(p.evalf())
                if 1e-6 < val < 1000:
                    # 偶次幂修正: sin^2n 和 cos^2n 的周期是 π 而非 2π
                    val = self._fix_trig_power_period(expr, x, val)
                    return val
        except Exception:
            pass
        return None

    @staticmethod
    def _fix_trig_power_period(expr: Expr, x: Symbol, base_period: float) -> float:
        """如果所有 sin/cos 都只以偶次幂出现, 周期减半"""
        if abs(base_period - 2 * np.pi) > 0.01:
            return base_period  # 非 2π 周期的不处理
        # 检查是否所有 sin(x), cos(x) 都出现在偶次幂中
        has_odd = False
        for sub in sp.preorder_traversal(expr):
            if sub.func in (sp.sin, sp.cos):
                arg = sub.args[0]
                if arg == x or (arg.is_Add and x in arg.atoms()):
                    # 查找这个 sin/cos 的指数
                    parent = None
                    # 简单检查: expr 中是否有 sin(x)^odd 或 cos(x)^odd
                    pass
        # 简化方案: 检查 expr 是否等于 f(x+π)
        try:
            shifted = expr.subs(x, x + sp.pi)
            diff_expr = sp.simplify(expr - shifted)
            if diff_expr == 0:
                return base_period / 2
        except Exception:
            pass
        return base_period

    # ── 渐近线 (垂直) ─────────────────────────────────────────
    def _find_asymptotes(self, expr: Expr, x: Symbol) -> list[float]:
        asymptotes: list[float] = []
        x0, x1 = self.x_range

        def _find_denom_zeros(denom_expr):
            """符号+数值混合求解分母零点"""
            vals = []
            # 符号求解
            try:
                sols = solveset(denom_expr, x, domain=S.Reals)
                vals = self._flatten_set(sols)
            except Exception:
                pass
            # 符号失败时用数值扫描
            if not vals:
                vals = self._numeric_scan(denom_expr, x)
            return vals

        # 方法1: 通分后检查分母
        try:
            combined = sp.together(expr)
            _, d = sp.fraction(combined)
            if d != 1 and d.has(x):
                for val in _find_denom_zeros(d):
                    if x0 - 0.5 <= val <= x1 + 0.5:
                        asymptotes.append(val)
        except Exception:
            pass

        # 方法2: tan/cot/sec/csc 有内在渐近线
        trig_asym_funcs = {tan: cos, cot: sin, sec: cos, csc: sin}
        for trig_func, denom_func in trig_asym_funcs.items():
            if expr.has(trig_func):
                for sub_expr in sp.preorder_traversal(expr):
                    if sub_expr.func == trig_func:
                        arg = sub_expr.args[0]
                        if arg.has(x):
                            inner = denom_func(arg)
                            for val in _find_denom_zeros(inner):
                                if x0 - 0.5 <= val <= x1 + 0.5:
                                    asymptotes.append(val)
        return sorted(set(round(v, 10) for v in asymptotes))

    # ── 水平渐近线 ─────────────────────────────────────────
    def _find_horizontal_asymptotes(self, expr: Expr, x: Symbol) -> list[float]:
        """检测水平渐近线: lim_{x→±∞} f(x) = L (有限值)"""
        result: list[float] = []
        for direction, sgn in [("+∞", oo), ("-∞", -oo)]:
            try:
                lim = sp.limit(expr, x, sgn)
                if lim.is_real and lim.is_finite:
                    val = float(lim.evalf())
                    if abs(val) < 1e8:
                        result.append(val)
            except Exception:
                pass
        return sorted(set(round(v, 10) for v in result))

    # ── 斜渐近线 ─────────────────────────────────────────
    def _find_oblique_asymptotes(self, expr: Expr, x: Symbol) -> list[tuple[float, float]]:
        """检测斜渐近线 y=mx+b: m=lim f(x)/x, b=lim (f(x)-mx)"""
        result: list[tuple[float, float]] = []
        for direction, sgn in [("+∞", oo), ("-∞", -oo)]:
            try:
                m = sp.limit(expr / x, x, sgn)
                if not (m.is_real and m.is_finite):
                    continue
                m_val = float(m.evalf())
                if abs(m_val) < 1e-10:  # 退化→水平渐近线, 跳过
                    continue
                b = sp.limit(expr - m_val * x, x, sgn)
                if b.is_real and b.is_finite:
                    b_val = float(b.evalf())
                    result.append((round(m_val, 10), round(b_val, 10)))
            except Exception:
                pass
        # 去重 (同一条渐近线两个方向会算出相同结果)
        seen = set()
        unique = []
        for item in result:
            if item not in seen:
                seen.add(item)
                unique.append(item)
        return unique

    # ── 对称性 ─────────────────────────────────────────
    def _detect_symmetry(self, expr: Expr, x: Symbol) -> str:
        """检测偶函数 f(-x)=f(x) / 奇函数 f(-x)=-f(x)"""
        try:
            expr_neg = sp.simplify(expr.subs(x, -x))
            # 偶函数: f(-x) - f(x) = 0
            if sp.simplify(expr_neg - expr) == 0:
                return "even"
            # 奇函数: f(-x) + f(x) = 0
            if sp.simplify(expr_neg + expr) == 0:
                return "odd"
        except Exception:
            pass
        return ""

    # ── 拐点 ─────────────────────────────────────────
    def _find_inflections(self, expr: Expr, x: Symbol,
                          period: float | None = None) -> list[float]:
        """求拐点: f''(x)=0 且符号改变"""
        try:
            fpp = sp.diff(expr, x, 2)
            fpp_simp = sp.simplify(fpp)
            cand = sp.solve(fpp_simp, x, domain=S.Reals)
            cand_set = self._flatten_set(cand)
            # 周期展开
            cand_set = self._expand_periodic(cand_set, period)
            # 筛选在定义域内且符号改变
            x0, x1 = self.x_range
            valid = []
            for c in cand_set:
                try:
                    if not (x0 - 1 <= c <= x1 + 1):
                        continue
                    # 检查左右 f'' 符号是否相反
                    eps = 0.01
                    left = float(fpp_simp.subs(x, c - eps).evalf())
                    right = float(fpp_simp.subs(x, c + eps).evalf())
                    if left * right < 0:  # 符号改变 → 真正拐点
                        valid.append(round(c, 8))
                except Exception:
                    continue
            return sorted(set(valid))
        except Exception:
            return []

    # ── 单调性 ─────────────────────────────────────────
    def _find_monotonic(self, expr: Expr, x: Symbol,
                        extrema: list, asymptotes: list[float],
                        period: float | None = None) -> list[Interval]:
        """根据 f'(x) 符号划分递增/递减区间"""
        is_periodic = period is not None and period > 0.1
        try:
            fp = sp.simplify(sp.diff(expr, x))
            # 取临界点: 极值点 + 渐近线
            crit = sorted(set(
                e.x for e in extrema
            ) | set(asymptotes))
            if not crit:
                # 无临界点时抽样判断
                mid = sum(self.x_range) / 2
                try:
                    val = float(fp.subs(x, mid).evalf())
                    label = "↗" if val > 0 else "↘"
                except Exception:
                    label = "?"
                return [Interval(None, None, label)]

            intervals = []
            pts = [None] + crit + [None]
            for i in range(len(pts) - 1):
                a, b = pts[i], pts[i + 1]
                if a is None and b is None:
                    mid = 0
                elif a is None:
                    mid = b - 1
                elif b is None:
                    mid = a + 1
                else:
                    mid = (a + b) / 2
                try:
                    val = float(fp.subs(x, mid).evalf())
                    label = "↗" if val > 0 else ("↘" if val < 0 else "→")
                except Exception:
                    label = "?"
                intervals.append(Interval(a, b, label))
            return intervals
        except Exception:
            return []

    # ── 凹凸性 ─────────────────────────────────────────
    def _find_concavity(self, expr: Expr, x: Symbol,
                        inflections: list[float], asymptotes: list[float],
                        period: float | None = None) -> list[Interval]:
        """根据 f''(x) 符号划分凹凸区间"""
        _ = period  # 接口统一, reserved
        try:
            fpp = sp.simplify(sp.diff(expr, x, 2))
            crit = sorted(set(inflections) | set(asymptotes))
            if not crit:
                mid = sum(self.x_range) / 2
                try:
                    val = float(fpp.subs(x, mid).evalf())
                    label = "∪" if val > 0 else "∩"
                except Exception:
                    label = "?"
                return [Interval(None, None, label)]

            intervals = []
            pts = [None] + crit + [None]
            for i in range(len(pts) - 1):
                a, b = pts[i], pts[i + 1]
                if a is None and b is None:
                    mid = 0
                elif a is None:
                    mid = b - 1
                elif b is None:
                    mid = a + 1
                else:
                    mid = (a + b) / 2
                try:
                    val = float(fpp.subs(x, mid).evalf())
                    label = "∪" if val > 0 else ("∩" if val < 0 else "—")
                except Exception:
                    label = "?"
                intervals.append(Interval(a, b, label))
            return intervals
        except Exception:
            return []

    # ── 值域 ─────────────────────────────────────────
    def _find_range(self, expr: Expr, x: Symbol, result: AnalysisResult) -> str:
        """数值估算值域: 结合极值、渐近线、端点"""
        try:
            vals = []
            # 极值点取值
            for e in result.extrema:
                vals.append(e.y)
            # 两端极限
            x0, x1 = self.x_range
            for end, sign in [(x0, -oo), (x1, oo)]:
                try:
                    lim = sp.limit(expr, x, sign)
                    if lim.is_real:
                        v = float(lim.evalf())
                        if abs(v) < 1e10:
                            vals.append(v)
                except Exception:
                    pass
            if not vals:
                return "R"

            lo, hi = min(vals), max(vals)
            # 检查有无水平渐近线扩展
            for hy in result.h_asymptotes:
                lo = min(lo, hy); hi = max(hi, hy)

            # 用采样点扩充
            try:
                f = sp.lambdify(x, expr, "numpy")
                xs = np.linspace(x0, x1, 200)
                ys = f(xs)
                ys = ys[np.isfinite(ys)]
                if len(ys) > 0:
                    lo = min(lo, float(np.min(ys)))
                    hi = max(hi, float(np.max(ys)))
            except Exception:
                pass

            lo_r = round(lo, 4)
            hi_r = round(hi, 4)
            return f"[{lo_r}, {hi_r}]"
        except Exception:
            return "R"

    # ── 集合扁平化 (核心) ─────────────────────────────────────
    def _flatten_set(self, solutions) -> list[float]:
        """将 sympy 的各种返回类型统一转为 float 列表"""
        x0, x1 = self.x_range
        vals: list[float] = []

        def _walk(s):
            if isinstance(s, (list, tuple)):
                for item in s:
                    _walk(item)
            elif isinstance(s, FiniteSet):
                for item in s:
                    try:
                        v = complex(item.evalf())
                        if abs(v.imag) < 1e-10:
                            vals.append(float(v.real))
                    except Exception:
                        pass
            elif isinstance(s, Interval):
                try:
                    if s.left_open and s.start != -oo:
                        vals.append(float(s.start.evalf()) + 1e-6)
                    elif s.start != -oo:
                        vals.append(float(s.start.evalf()))
                    if s.right_open and s.end != oo:
                        vals.append(float(s.end.evalf()) - 1e-6)
                    elif s.end != oo:
                        vals.append(float(s.end.evalf()))
                except Exception:
                    pass
            elif isinstance(s, Union):
                for sub in s.args:
                    _walk(sub)
            elif isinstance(s, ImageSet):
                _walk_image_set(s)
            elif isinstance(s, sp.ConditionSet):
                pass  # 跳过条件集
            elif isinstance(s, sp.Complement):
                _walk(s.args[0])  # 近似处理
            elif isinstance(s, type(sp.EmptySet)):
                pass
            elif hasattr(s, 'evalf'):
                # 裸 sympy 数 / 表达式: 直接求值
                try:
                    v = complex(s.evalf())
                    if abs(v.imag) < 1e-10:
                        vals.append(float(v.real))
                except Exception:
                    pass

        def _walk_image_set(img: ImageSet):
            """ImageSet(Lambda(_n, formula), base_set) — 在范围内采样"""
            try:
                # 只处理 base_set 是 Integers/Naturals 的简单 ImageSet
                # 嵌套 ImageSet (如 {x² | x∈{2nπ}}) 太复杂, 跳过让数值扫描处理
                base = img.base_set
                if not base.is_iterable:
                    return
                # 检查 base_set 是否是简单的整数集
                is_simple = isinstance(base, (sp.Integers, sp.Naturals, sp.Naturals0))
                is_interval = isinstance(base, Interval)
                if not (is_simple or is_interval):
                    return  # 嵌套集合, 太复杂

                lam = img.lamda
                formula = lam.expr
                raw_sym = lam.args[0]
                if isinstance(raw_sym, (tuple, sp.Tuple)):
                    n_sym = raw_sym[0]
                else:
                    n_sym = raw_sym

                try:
                    step_expr = abs((formula.subs(n_sym, 1) - formula.subs(n_sym, 0)).evalf())
                    step = float(step_expr)
                except Exception:
                    step = 1.0
                if step < 1e-8:
                    step = 1.0

                margin = max(step, 2.0)
                n_min = int((x0 - margin) / step) - 2
                n_max = int((x1 + margin) / step) + 2
                n_min = max(n_min, -2000)
                n_max = min(n_max, 2000)

                for n_val in range(n_min, n_max + 1):
                    try:
                        v = complex(formula.subs(n_sym, n_val).evalf())
                        if abs(v.imag) < 1e-10:
                            rv = float(v.real)
                            if x0 - 0.5 <= rv <= x1 + 0.5:
                                vals.append(rv)
                    except Exception:
                        pass
            except Exception:
                pass

        _walk(solutions)
        filtered = [v for v in vals if x0 - 0.5 <= v <= x1 + 0.5]
        return sorted(set(round(v, 10) for v in filtered))

    # ── 数值扫描 (降级方案) ────────────────────────────────────
    def _numeric_scan(
        self, expr: Expr, x: Symbol, *, find: str = "zeros"
    ) -> list[float]:
        """在区间内用高密度采样 + 二分法搜索零点或临界点"""
        results: list[float] = []
        x0, x1 = self.x_range
        f = lambdify(x, expr, modules="numpy")

        # 在域边界和关键点增加采样 (0, 整数, π 倍数等)
        extra = [0.0]
        for k in range(-10, 11):
            extra.extend([float(k), k * np.pi / 2, k * np.pi])
        extra = sorted(set(v for v in extra if x0 <= v <= x1))

        xs = np.linspace(x0, x1, 6000)
        if extra:
            xs = np.sort(np.unique(np.concatenate([xs, np.array(extra)])))

        try:
            ys = f(xs)
            ys = np.asarray(ys, dtype=float)

            for i in range(len(xs) - 1):
                yi, yi1 = ys[i], ys[i + 1]
                if np.isnan(yi) or np.isnan(yi1):
                    continue
                if not (np.isfinite(yi) and np.isfinite(yi1)):
                    continue

                # 域边界: NaN → finite 过渡, 检查边界点是否为 0
                if i > 0 and np.isnan(ys[i - 1]) and not np.isnan(yi):
                    if abs(yi) < 0.01:
                        results.append(xs[i])
                        continue

                # 两端都接近0 (floor 类函数在平区间)
                if abs(yi) < 1e-10 and abs(yi1) < 1e-10:
                    results.append((xs[i] + xs[i + 1]) / 2)
                    continue

                if yi * yi1 <= 0:
                    a, b = xs[i], xs[i + 1]
                    for _ in range(35):
                        m = (a + b) / 2
                        try:
                            fm = self._to_float(f(np.array([m])))
                        except Exception:
                            break
                        if not np.isfinite(fm):
                            break
                        if abs(fm) < 1e-12:
                            a = b = m
                            break
                        try:
                            fa = self._to_float(f(np.array([a])))
                        except Exception:
                            break
                        if not np.isfinite(fa):
                            break
                        if fa * fm <= 0:
                            b = m
                        else:
                            a = m
                    root = (a + b) / 2
                    try:
                        froot_val = self._to_float(f(np.array([root])))
                        if np.isfinite(froot_val) and abs(froot_val) < 0.01:
                            results.append(root)
                    except Exception:
                        pass
        except Exception:
            pass
        return results

    # ── 绘图入口 ──────────────────────────────────────────────
    def plot(
        self,
        expr_str: str,
        *,
        label: str | None = None,
        color: str | None = None,
        linewidth: float = 2.0,
        linestyle: str = "-",
        show_analysis: bool = True,
    ):
        """绘制单个函数或方程 (自动检测 y=f(x) / x=f(y) / f(x,y)=0)"""
        if self._ax is None:
            self._fig, self._ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        if color is None:
            color = self._DEFAULT_COLORS[self._color_idx % len(self._DEFAULT_COLORS)]
            self._color_idx += 1

        # ── 检测方程类型 ──
        eq_type, eq_data = self._parse_equation(expr_str)
        lbl = label or expr_str

        if eq_type == "y_explicit":
            return self._plot_y_explicit(eq_data, lbl, color, linewidth, linestyle, show_analysis)
        elif eq_type == "x_explicit":
            self._plot_x_equals_fy(eq_data, lbl, color, linewidth, linestyle)
            return self._dummy_result(expr_str)
        elif eq_type == "implicit":
            self._plot_implicit_eq(eq_data, lbl, color, linewidth)
            return self._dummy_result(expr_str)
        else:
            return self._plot_y_explicit(expr_str, lbl, color, linewidth, linestyle, show_analysis)

    @staticmethod
    def _dummy_result(expr_str: str) -> AnalysisResult:
        """为隐式曲线创建占位分析结果"""
        return AnalysisResult(
            expr=sp.sympify(0), x=Symbol("x"),
            f_lambda=lambda v: np.zeros_like(v), expr_str=expr_str,
        )

    # ── 方程解析 ────────────────────────────────────────────────
    @staticmethod
    def _parse_equation(s: str) -> tuple[str, str]:
        """
        解析输入字符串, 返回 (类型, 数据)
        类型: "y_explicit" | "x_explicit" | "implicit" | "plain"
        """
        s = s.strip()
        if "=" not in s:
            return ("plain", s)

        # 拆分为左右两边
        parts = s.split("=", 1)
        lhs = parts[0].strip()
        rhs = parts[1].strip()

        if not lhs or not rhs:
            return ("plain", s)

        # y = f(x)  →  y_explicit
        if lhs == "y" or lhs == "Y":
            return ("y_explicit", rhs)

        # x = f(y)  →  x_explicit
        if lhs == "x" or lhs == "X":
            return ("x_explicit", rhs)

        # f(y) = x  →  x_explicit (反向)
        if rhs in ("x", "X") and "y" in lhs:
            return ("x_explicit", lhs)

        # f(x) = y  →  y_explicit (反向)
        if rhs in ("y", "Y") and "x" in lhs:
            return ("y_explicit", lhs)

        # 其他含 = 的 → implicit: f(x,y) = 0
        return ("implicit", s)

    def _plot_y_explicit(self, expr_str, lbl, color, lw, ls, show_analysis):
        """y = f(x) 或普通表达式 — 标准函数绘图"""
        analysis = self.analyze(expr_str)
        self._analyses.append(analysis)

        x0, x1 = self.x_range
        xs = np.linspace(x0, x1, self.num_points)

        # 在渐近线附近增加采样点 (使曲线更锐利)
        if analysis.asymptotes:
            extra = []
            for asym in analysis.asymptotes:
                if x0 <= asym <= x1:
                    for eps in [1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.03, 0.06, 0.1,
                                -1e-4, -5e-4, -1e-3, -5e-3, -0.01, -0.03, -0.06, -0.1]:
                        pt = asym + eps
                        if x0 <= pt <= x1:
                            extra.append(pt)
            if extra:
                xs = np.sort(np.unique(np.concatenate([xs, np.array(extra)])))

        ys = self._safe_eval(analysis.f_lambda, xs)

        # 渐近线断开
        if analysis.asymptotes:
            for asym in analysis.asymptotes:
                idx = np.searchsorted(xs, asym)
                if 0 < idx < len(xs) - 1:
                    ys[idx - 1:idx + 2] = np.nan

        # 间断函数 (floor, ceil, sign, Heaviside) 在跳跃处断开
        if analysis.is_piecewise:
            self._break_at_jumps(xs, ys)

        self._ax.plot(xs, ys, label=lbl, color=color,
                      linewidth=lw, linestyle=ls, alpha=0.9)

        if show_analysis:
            self._annotate_analysis(analysis, color)

        return analysis

    @staticmethod
    def _break_at_jumps(xs: np.ndarray, ys: np.ndarray):
        """在跳跃间断处插入 NaN, 避免陡峭斜线连接"""
        if len(ys) < 2:
            return
        # 计算相邻点斜率, 斜率突变处即为跳跃
        dy = np.abs(np.diff(ys))
        dx = np.abs(np.diff(xs))
        with np.errstate(divide='ignore', invalid='ignore'):
            slope = np.where(dx > 0, dy / dx, 0)
        # 斜率阈值: 超过相邻段中位数的 200 倍即视为跳跃
        if len(slope) > 3:
            median_slope = np.median(slope[np.isfinite(slope)])
            if median_slope > 0:
                threshold = max(median_slope * 200, 50)
                for i in range(1, len(ys)):
                    if slope[i - 1] > threshold:
                        ys[i - 1] = np.nan
                        ys[i] = np.nan

    def _plot_x_equals_fy(self, expr_str, lbl, color, lw, ls):
        """x = f(y) — 以 y 为自变量绘制隐式曲线"""
        x0, x1 = self.x_range
        y_span = max(abs(x0), abs(x1), 10) * 1.5

        # 需要两个变量 x, y, 方程: x - f(y) = 0
        x_sym = Symbol("x", real=True)
        y_sym = Symbol("y", real=True)
        preprocessed = _preprocess_expr(expr_str)
        local_dict = {
            "y": y_sym,
            "sin": sin, "cos": cos, "tan": tan, "cot": cot, "sec": sec, "csc": csc,
            "asin": asin, "acos": acos, "atan": atan,
            "sinh": sinh, "cosh": cosh, "tanh": tanh,
            "exp": exp, "log": log, "sqrt": sqrt, "Abs": Abs, "abs": Abs,
            "pi": pi, "E": E,
        }
        try:
            f_expr = sp.sympify(preprocessed, locals=local_dict)
        except Exception:
            return

        F_expr = x_sym - f_expr  # x - f(y) = 0

        self._draw_implicit(F_expr, x_sym, y_sym, x0, x1, -y_span, y_span,
                            lbl, color, lw)

    def _plot_implicit_eq(self, expr_str, lbl, color, lw):
        """f(x,y) = 0 或 f(x,y) = g(x,y) — 通用隐式方程"""
        x0, x1 = self.x_range
        y_span = max(abs(x0), abs(x1), 10) * 1.5

        x_sym = Symbol("x", real=True)
        y_sym = Symbol("y", real=True)

        # 拆分 =
        if "=" in expr_str:
            parts = expr_str.split("=", 1)
            lhs_str = parts[0].strip()
            rhs_str = parts[1].strip()
            eq_str = f"({lhs_str}) - ({rhs_str})"
        else:
            eq_str = expr_str

        preprocessed = _preprocess_expr(eq_str)
        local_dict = {
            "x": x_sym, "y": y_sym,
            "sin": sin, "cos": cos, "tan": tan, "cot": cot, "sec": sec, "csc": csc,
            "asin": asin, "acos": acos, "atan": atan,
            "sinh": sinh, "cosh": cosh, "tanh": tanh,
            "exp": exp, "log": log, "sqrt": sqrt, "Abs": Abs, "abs": Abs,
            "pi": pi, "E": E,
        }
        try:
            F_expr = sp.sympify(preprocessed, locals=local_dict)
        except Exception:
            return

        self._draw_implicit(F_expr, x_sym, y_sym, x0, x1, -y_span, y_span,
                            lbl, color, lw)

    def _draw_implicit(self, F_expr, x_sym, y_sym,
                       x0, x1, y0, y1, label, color, lw):
        """用 contour 绘制隐式曲线 F(x,y) = 0"""
        import scipy.special as scisp
        num = 600
        X, Y = np.meshgrid(np.linspace(x0, x1, num),
                           np.linspace(y0, y1, num))

        mods: Any = ["numpy", {
            "sin": np.sin, "cos": np.cos, "tan": np.tan,
            "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
            "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
            "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
            "Abs": np.abs, "abs": np.abs, "sign": np.sign,
            "pi": np.pi, "E": np.e,
        }]
        F = lambdify((x_sym, y_sym), F_expr, modules=mods)
        try:
            Z = F(X, Y)
        except Exception:
            return
        Z = np.asarray(Z, dtype=float)
        Z = np.clip(Z, -1e10, 1e10)

        self._ax.contour(X, Y, Z, levels=[0], colors=[color],
                         linewidths=lw, alpha=0.9)
        # 给 contour 加图例
        self._ax.plot([], [], color=color, linewidth=lw, label=label)

    def plot_multi(
        self,
        expr_list: Sequence[str],
        *,
        labels: Sequence[str] | None = None,
        colors: Sequence[str] | None = None,
        show_analysis: bool = True,
    ):
        """在一张图上绘制多个函数/方程"""
        if labels is None:
            labels = [""] * len(expr_list)
        if colors is None:
            colors = [self._DEFAULT_COLORS[i % len(self._DEFAULT_COLORS)]
                      for i in range(len(expr_list))]

        for i, e in enumerate(expr_list):
            self.plot(e, label=labels[i] or e, color=colors[i],
                      show_analysis=show_analysis)

    def _safe_eval(self, f: Callable, xs: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(all='ignore'):
                try:
                    ys = f(xs)
                    if np.isscalar(ys):
                        ys = np.full_like(xs, ys, dtype=float)
                    ys = np.asarray(ys, dtype=float)
                except Exception:
                    ys = np.full_like(xs, np.nan)
        # 只去掉 inf/nan, 不截断精度
        ys[~np.isfinite(ys)] = np.nan
        return ys

    # ── 标注 ──────────────────────────────────────────────────
    def _annotate_analysis(self, a: AnalysisResult, color: str):
        ax = self._ax

        # 极大值 ▲
        for e in a.extrema:
            if e.kind == "max":
                ax.plot(e.x, e.y, "r^", markersize=11, markeredgecolor="black",
                        markeredgewidth=0.8, zorder=8)
                ax.annotate(
                    f'{_t("max_point")}\n({e.x:.3f}, {e.y:.3f})',
                    xy=(e.x, e.y), xytext=(0, 18), textcoords="offset points",
                    fontsize=7.5, color="red", fontweight="bold", ha="center",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="lightyellow",
                              edgecolor="red", alpha=0.9),
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
                    zorder=9,
                )

        # 极小值 ▼
        for e in a.extrema:
            if e.kind == "min":
                ax.plot(e.x, e.y, "bv", markersize=11, markeredgecolor="black",
                        markeredgewidth=0.8, zorder=8)
                ax.annotate(
                    f'{_t("min_point")}\n({e.x:.3f}, {e.y:.3f})',
                    xy=(e.x, e.y), xytext=(0, -22), textcoords="offset points",
                    fontsize=7.5, color="blue", fontweight="bold", ha="center",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="lightyellow",
                              edgecolor="blue", alpha=0.9),
                    arrowprops=dict(arrowstyle="->", color="blue", lw=1.2),
                    zorder=9,
                )

        # 零点 ●
        for z in a.zeros:
            ax.plot(z, 0, "go", markersize=9, markeredgecolor="black",
                    markeredgewidth=0.8, zorder=8)
            ax.annotate(
                f"({z:.3f}, 0)",
                xy=(z, 0), xytext=(0, -15), textcoords="offset points",
                fontsize=7, color="green", fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow",
                          edgecolor="green", alpha=0.85),
                arrowprops=dict(arrowstyle="->", color="green", lw=1.0),
                zorder=9,
            )

        # y 截距 ◆
        if a.y_intercept is not None:
            ax.plot(0, a.y_intercept, "mD", markersize=10, markeredgecolor="black",
                    markeredgewidth=0.8, zorder=8)
            ax.annotate(
                f'{_t("y_intercept")}: ({0}, {a.y_intercept:.3f})',
                xy=(0, a.y_intercept), xytext=(50, 12),
                textcoords="offset points",
                fontsize=7.5, color="purple", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="lightyellow",
                          edgecolor="purple", alpha=0.9),
                arrowprops=dict(arrowstyle="->", color="purple", lw=1.2),
                zorder=9,
            )

        # 垂直渐近线 ┄
        y_min, y_max = ax.get_ylim()
        for asym in a.asymptotes:
            ax.axvline(x=asym, color="orange", linestyle="--", linewidth=1.2, alpha=0.75)
            ax.annotate(
                f'{_t("vertical_asymptote")}\nx={asym:.3f}',
                xy=(asym, y_max * 0.85), fontsize=7, color="orange",
                rotation=90, ha="right", va="top",
            )

        # 水平渐近线 ─
        x_min, x_max = ax.get_xlim()
        for hy in a.h_asymptotes:
            ax.axhline(y=hy, color="brown", linestyle="--", linewidth=1.2, alpha=0.6)
            ax.annotate(
                f'y={hy:.3f}',
                xy=(x_max * 0.92, hy), fontsize=7, color="brown",
                ha="right", va="bottom",
            )

        # 拐点 ● (青色圆点)
        for infl in a.inflections:
            try:
                fy = float(a.f_lambda(np.array([infl])).item())
                if np.isfinite(fy):
                    ax.plot(infl, fy, "o", color="cyan", markersize=7, zorder=10)
            except Exception:
                pass

        # 斜渐近线 ─
        for m_val, b_val in a.obliques:
            xs_draw = np.linspace(x_min, x_max, 100)
            ys_draw = m_val * xs_draw + b_val
            ax.plot(xs_draw, ys_draw, linestyle=":", color="gray",
                    linewidth=1.0, alpha=0.7)
            ax.annotate(
                f'y={m_val:.3f}x+{b_val:.3f}',
                xy=(x_max * 0.7, m_val * x_max * 0.7 + b_val),
                fontsize=7, color="gray", ha="right", va="bottom",
            )

    # ── 整图修饰 ──────────────────────────────────────────────
    def _finalize_figure(self):
        ax = self._ax
        if ax is None:
            return

        # 坐标轴移动
        ax.spines["left"].set_position("zero")
        ax.spines["bottom"].set_position("zero")
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")
        ax.spines["left"].set_linewidth(2.0)
        ax.spines["bottom"].set_linewidth(2.0)

        # 箭头
        ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False, markersize=9)
        ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False, markersize=9)

        # 网格: 主 + 次
        ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.35, color="gray")
        ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.2, color="gray")

        # X 轴刻度: 有三角函数用 π 分数, 否则用十进制
        trig_names = ('sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'tg', 'ctg')
        trig_funcs = (sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc)
        has_trig = (
            any(a.expr.has(*trig_funcs) for a in self._analyses) or
            any(any(t in a.expr_str.lower() for t in trig_names) for a in self._analyses)
        )
        if has_trig:
            ax.xaxis.set_major_locator(PiTickLocator())
            ax.xaxis.set_major_formatter(PiTickFormatter())
        else:
            # 非三角函数: 用整洁刻度, 避免 0.2/0.4/0.6 这种无意义小数
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10]))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10]))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

        # 刻度
        ax.tick_params(axis="both", which="major", labelsize=10, width=1.2, length=6)
        ax.tick_params(axis="both", which="minor", width=0.8, length=3)

        # 范围
        x0, x1 = self.x_range
        ax.set_xlim(x0, x1)
        self._auto_ylim()

        # 轴标签
        ax.set_xlabel(_t("xlabel"), fontsize=13, fontweight="bold")
        ax.set_ylabel(_t("ylabel"), fontsize=13, fontweight="bold")
        ax.xaxis.set_label_coords(0.98, 0.52)
        ax.yaxis.set_label_coords(0.52, 0.98)

        # 标题
        ax.set_title(self._build_title(), fontsize=15, fontweight="bold", pad=18)

        # 图例
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="upper right", fontsize=10,
                      framealpha=0.92, title=_t("legend_title"),
                      title_fontsize=10, edgecolor="gray")

        # 信息面板
        self._draw_info_panel(ax)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                if self._fig is not None:
                    self._fig.tight_layout(pad=1.5)
                else:
                    plt.tight_layout(pad=1.5)
            except Exception:
                pass

    def _auto_ylim(self):
        ax = self._ax
        all_y = []
        for line in ax.get_lines():
            ydata = np.asarray(line.get_ydata(), dtype=float).ravel()
            fin = ydata[np.isfinite(ydata)]
            if len(fin) > 0:
                all_y.extend(fin.tolist())
        if all_y:
            arr = np.array(all_y)
            lo, hi = np.percentile(arr, [5, 95])
            if hi - lo > 500:
                lo, hi = np.percentile(arr, [10, 90])
            margin = max((hi - lo) * 0.2, 2.0)
            # 确保始终显示四个象限 (y 轴至少覆盖到 ±margin)
            lo = min(lo, -margin * 0.5)
            hi = max(hi, margin * 0.5)
            ax.set_ylim(lo - margin, hi + margin)

    def _build_title(self) -> str:
        parts = []
        for i, a in enumerate(self._analyses):
            s = a.expr_str if a.expr_str else str(a.expr).replace("**", "^")
            if len(s) > 50:
                s = s[:47] + "..."
            parts.append(f"f{i+1}(x) = {s}")
        return _t("title") + "   |   ".join(parts)

    def _draw_info_panel(self, ax):
        """在左下角绘制分析结果信息面板"""
        if not self._analyses:
            return

        lines = [f"══ {_t('info_title')} ══"]
        for i, a in enumerate(self._analyses):
            if len(self._analyses) > 1:
                lines.append(f"── f{i+1}(x) ──")
            lines.append(f"  {_t('func_expr')}: {a.expr_str[:40] if a.expr_str else str(a.expr).replace('**', '^')[:40]}")
            if a.domain_info:
                lines.append(f"  {_t('domain')}: {a.domain_info}")
            if a.range_info and a.range_info != "—":
                lines.append(f"  {_t('range')}: {a.range_info}")
            if a.period_value is not None:
                lines.append(f"  {_t('period')}: T = {a.period_value:.4f}")
            if a.y_intercept is not None:
                lines.append(f"  {_t('y_intercept')}: (0, {a.y_intercept:.4f})")
            if a.zeros:
                z_str = ", ".join(f"x={z:.3f}" for z in a.zeros[:6])
                if len(a.zeros) > 6:
                    z_str += f" ...({len(a.zeros)} total)"
                lines.append(f"  {_t('zero_point')}: {z_str}")
            if a.extrema:
                maxes = [e for e in a.extrema if e.kind == "max"]
                mins = [e for e in a.extrema if e.kind == "min"]
                if maxes:
                    pts = ", ".join(f"({e.x:.2f},{e.y:.2f})" for e in maxes[:4])
                    lines.append(f"  ▲ {_t('max_point')}: {pts}")
                if mins:
                    pts = ", ".join(f"({e.x:.2f},{e.y:.2f})" for e in mins[:4])
                    lines.append(f"  ▼ {_t('min_point')}: {pts}")
            if a.symmetry:
                tag = "偶函数 (关于 y 轴对称)" if a.symmetry == "even" else "奇函数 (关于原点对称)"
                lines.append(f"  ⚭ {tag}")
            if a.extrema:
                maxes = [e for e in a.extrema if e.kind == "max"]
                mins = [e for e in a.extrema if e.kind == "min"]
                if maxes:
                    pts = ", ".join(f"({e.x:.2f},{e.y:.2f})" for e in maxes[:4])
                    lines.append(f"  ▲ {_t('max_point')}: {pts}")
                if mins:
                    pts = ", ".join(f"({e.x:.2f},{e.y:.2f})" for e in mins[:4])
                    lines.append(f"  ▼ {_t('min_point')}: {pts}")
            if a.inflections:
                lines.append(f"  ∿ 拐点: x=" + ", ".join(f"{v:.4f}" for v in a.inflections[:6]))
            is_T = a.period_value is not None and a.period_value > 0.1
            if a.monotonic:
                parts = []
                for iv in a.monotonic:
                    a_s = ("..." if is_T else "—∞") if iv.start is None else f"{iv.start:.3f}"
                    b_s = ("..." if is_T else "+∞") if iv.end is None else f"{iv.end:.3f}"
                    parts.append(f"({a_s},{b_s}){iv.label}")
                lines.append(f"  ↕ 单调: " + " | ".join(parts[:6]))
            if a.concavity:
                parts = []
                for iv in a.concavity:
                    a_s = ("..." if is_T else "—∞") if iv.start is None else f"{iv.start:.3f}"
                    b_s = ("..." if is_T else "+∞") if iv.end is None else f"{iv.end:.3f}"
                    parts.append(f"({a_s},{b_s}){iv.label}")
                lines.append(f"  ⌣ 凹凸: " + " | ".join(parts[:6]))
            if a.range_info and a.range_info != "—":
                lines.append(f"  值域: {a.range_info}")
            if a.asymptotes:
                lines.append(f"  {_t('vertical_asymptote')}: " +
                             ", ".join(f"x={v:.3f}" for v in a.asymptotes))
            if a.h_asymptotes:
                lines.append(f"  水平渐近线: " +
                             ", ".join(f"y={v:.4f}" for v in a.h_asymptotes))
            if a.obliques:
                lines.append(f"  斜渐近线: " +
                             ", ".join(f"y={m:.3f}x+{b:.3f}" for m, b in a.obliques))
            if a.is_piecewise:
                lines.append(f"  ⚠ {_t('piecewise_warn')}")

        text = "\n".join(lines)
        self._ax.text(
            0.02, 0.98, text, transform=self._ax.transAxes,
            fontsize=7.5, verticalalignment="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="wheat",
                      edgecolor="gray", alpha=0.92),
            zorder=99,
        )

    # ── 显示 / 保存 / 清除 ────────────────────────────────────
    def show(self):
        self._finalize_figure()
        plt.show()

    def save(self, filepath: str):
        self._finalize_figure()
        self._fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight",
                          facecolor="white", edgecolor="none")
        print(f"{_t('saved_to')}: {filepath}")

    def clear(self):
        plt.close("all")
        self._fig, self._ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        self._analyses.clear()
        self._color_idx = 0


# ============================================================================
# 交互式控制台
# ============================================================================

def _print_banner():
    print()
    print("  ╔" + "═" * 58 + "╗")
    print(f"  ║  {_t('app_title'):^54}  ║")
    print(f"  ║  {_t('app_subtitle'):^52}  ║")
    print("  ╠" + "═" * 58 + "╣")
    print(f"  ║  {_t('quit_help'):^54}  ║")
    print("  ╚" + "═" * 58 + "╝")
    print()


def _interactive_mode():
    """交互式命令行模式 — 像软件一样使用"""
    _print_banner()
    plotter = FunctionPlotter(x_range=(-8, 8))

    while True:
        try:
            cmd = input(f"\n  {_t('prompt')} >>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {_t('goodbye')}")
            break

        if not cmd:
            continue

        low = cmd.lower()
        if low in ("q", "quit", "exit"):
            print(f"  {_t('goodbye')}")
            break
        if low in ("h", "help", "?"):
            print(_t("help_text"))
            continue
        if low in ("c", "clear", "clr"):
            plotter.clear()
            print("  图像已清除。")
            continue
        if low.startswith("range"):
            parts = cmd.split()
            if len(parts) == 3:
                try:
                    x0, x1 = float(parts[1]), float(parts[2])
                    plotter.x_range = (x0, x1)
                    print(f"  {_t('range_set')}: [{x0}, {x1}]")
                except ValueError:
                    print("  用法: range <min> <max>  例: range -10 10")
            else:
                print("  用法: range <min> <max>  例: range -10 10")
            continue
        if low.startswith("save"):
            parts = cmd.split(maxsplit=1)
            if len(parts) == 2:
                plotter.save(parts[1])
            else:
                print("  用法: save <文件路径>  例: save myplot.png")
            continue

        # ── 绘制 ──
        plotter.clear()
        try:
            if "," in cmd:
                exprs = [e.strip() for e in cmd.split(",")]
                plotter.plot_multi(exprs)
            else:
                plotter.plot(cmd)
            plotter.show()
        except ValueError as e:
            print(f"  ❌ {e}")
        except Exception as e:
            print(f"  ❌ 错误: {e}")


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="通用函数图像绘制工具 — 极值/零点/周期/渐近线全面分析",
    )
    parser.add_argument(
        "expr", nargs="*",
        help="函数表达式 (空格分隔多个, 逗号也可); 留空进入交互模式",
    )
    parser.add_argument("-r", "--range", nargs=2, type=float, default=[-8, 8],
                        metavar=("MIN", "MAX"), help="x 轴范围 (默认 -8 8)")
    parser.add_argument("-s", "--save", type=str, default=None,
                        help="保存图像到文件")
    parser.add_argument("--no-analysis", action="store_true",
                        help="不标注极值/零点/周期")
    parser.add_argument("--dark", action="store_true", help="暗色主题")
    parser.add_argument("--lang", choices=["cn", "en"], default="cn",
                        help="界面语言")
    parser.add_argument("--dpi", type=int, default=130, help="图像分辨率")

    args = parser.parse_args()

    plotter = FunctionPlotter(
        x_range=tuple(args.range),  # type: ignore[arg-type]
        lang=args.lang,
        style="dark" if args.dark else "default",
        dpi=args.dpi,
    )

    if not args.expr:
        _interactive_mode()
        return

    combined = " ".join(args.expr)
    if "," in combined:
        exprs = [e.strip() for e in combined.split(",")]
    else:
        exprs = [combined]

    for e in exprs:
        try:
            plotter.plot(e, show_analysis=not args.no_analysis)
        except ValueError as exc:
            print(f"错误: {exc}")
            return
        except Exception as exc:
            print(f"未预期的错误: {exc}")
            return

    if args.save:
        plotter.save(args.save)
    else:
        plotter.show()


if __name__ == "__main__":
    main()
