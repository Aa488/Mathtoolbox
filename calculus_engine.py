#!/usr/bin/env python3
"""
================================================================================
  微积分 & 级数计算引擎 — Calculus & Series Engine
================================================================================
  纯计算层, 不依赖 tkinter/matplotlib。
  封装 sympy / scipy 的符号与数值计算, 供 math_toolbox GUI 调用。
================================================================================
"""

from __future__ import annotations
import numpy as np
import sympy as sp
from sympy import (
    sin, cos, tan, cot, sec, csc,
    asin, acos, atan, acot, asec, acsc,
    sinh, cosh, tanh, coth, sech, csch,
    log, exp, sqrt, Abs,
    Symbol, Expr, S, oo, pi, E,
    integrate, diff, series, fourier_series,
    lambdify, simplify, expand,
)
from typing import Callable

# ── 与 function_plotter 相同的预处理 ──
from function_plotter import _preprocess_expr


def _parse(expr_str: str) -> Expr:
    """将字符串转为 sympy 表达式"""
    x = Symbol('x')
    s = _preprocess_expr(expr_str)
    return sp.sympify(s, locals={
        "x": x, "pi": sp.pi, "e": sp.E, "E": sp.E,
        "sin": sin, "cos": cos, "tan": tan, "cot": cot, "sec": sec, "csc": csc,
        "asin": asin, "acos": acos, "atan": atan, "acot": acot, "asec": asec, "acsc": acsc,
        "sinh": sinh, "cosh": cosh, "tanh": tanh, "coth": coth, "sech": sech, "csch": csch,
        "log": log, "ln": log, "exp": exp, "sqrt": sqrt, "abs": Abs,
        "gamma": sp.gamma, "erf": sp.erf,
    })


# ============================================================================
# 微积分引擎
# ============================================================================

class CalculusEngine:
    """微积分 & 级数计算"""

    x = Symbol('x')

    # ── 不定积分 ─────────────────────────────────────────
    @classmethod
    def integrate_indefinite(cls, expr_str: str) -> tuple[str, Callable | None]:
        """
        返回: (原函数字符串, lambdified 函数)
        例: "sin(x)" → ("-cos(x)", <callable>)
        """
        expr = _parse(expr_str)
        F = integrate(expr, cls.x)
        F_simplified = simplify(F)
        F_str = str(F_simplified).replace("**", "^")
        try:
            F_fn = lambdify(cls.x, F_simplified, "numpy")
        except Exception:
            F_fn = None
        return F_str, F_fn

    # ── 定积分 ─────────────────────────────────────────
    @classmethod
    def integrate_definite(cls, expr_str: str, a: float, b: float) -> dict:
        """
        返回: {"symbolic": str, "numeric": float, "error": float|None}
        """
        expr = _parse(expr_str)
        result = {"symbolic": "", "numeric": 0.0, "error": None}

        # 符号积分
        try:
            sym_val = integrate(expr, (cls.x, a, b))
            result["symbolic"] = str(sym_val.simplify()).replace("**", "^")
        except Exception:
            result["symbolic"] = "—"

        # 数值积分 (scipy quad)
        try:
            from scipy import integrate as scipy_integrate
            f = lambdify(cls.x, expr, "numpy")
            val, err = scipy_integrate.quad(f, a, b, limit=200)
            result["numeric"] = round(val, 10)
            result["error"] = round(float(err), 10)
        except Exception:
            # 退路: numpy 简单采样
            try:
                f = lambdify(cls.x, expr, "numpy")
                xs = np.linspace(a, b, 2000)
                ys = f(xs)
                ys = ys[np.isfinite(ys)]
                if len(ys) > 1:
                    result["numeric"] = round(float(np.trapz(ys, np.linspace(a, b, len(ys)))), 6)
            except Exception:
                result["numeric"] = float('nan')

        return result

    # ── 黎曼和 ─────────────────────────────────────────
    @classmethod
    def riemann_data(cls, expr_str: str, a: float, b: float, n: int = 10,
                     mode: str = "midpoint") -> dict:
        """
        mode: "left" | "right" | "midpoint" | "trapezoid"
        返回: {"rects": [(x_left, y_top, width), ...], "value": float,
               "xs": [], "ys": []}  # 函数曲线采样点
        """
        expr = _parse(expr_str)
        f = lambdify(cls.x, expr, "numpy")

        # 函数采样 (高密度, 用于画曲线)
        xs_dense = np.linspace(a, b, 500)
        ys_dense = f(xs_dense)
        ys_dense = np.where(np.isfinite(ys_dense), ys_dense, np.nan)

        # 矩形
        dx = (b - a) / n
        rects = []
        total = 0.0

        for i in range(n):
            x_left = a + i * dx
            x_right = x_left + dx
            if mode == "left":
                x_eval = x_left
            elif mode == "right":
                x_eval = x_right
            elif mode == "midpoint":
                x_eval = (x_left + x_right) / 2
            elif mode == "trapezoid":
                x_eval = None  # 梯形特殊处理
            else:
                x_eval = (x_left + x_right) / 2

            if mode == "trapezoid":
                y_l = float(f(np.array([x_left])).item())
                y_r = float(f(np.array([x_right])).item())
                if np.isfinite(y_l) and np.isfinite(y_r):
                    area = (y_l + y_r) / 2 * dx
                    rects.append({
                        "type": "trapezoid",
                        "x_left": x_left, "x_right": x_right,
                        "y_left": y_l, "y_right": y_r,
                    })
                    total += area
            else:
                y_val = float(f(np.array([x_eval])).item())
                if np.isfinite(y_val):
                    # 矩形底部到 y=0 (或到曲线, 看符号)
                    y_top = y_val
                    y_bottom = 0.0
                    rects.append({
                        "type": "rectangle",
                        "x_left": x_left,
                        "y_bottom": min(y_bottom, y_top),
                        "y_top": max(y_bottom, y_top),
                        "width": dx,
                        "y_val": y_val,
                    })
                    total += y_val * dx

        return {
            "rects": rects,
            "value": round(total, 8),
            "xs": xs_dense.tolist(),
            "ys": ys_dense.tolist(),
            "dx": dx,
            "n": n,
            "mode": mode,
        }

    # ── 两函数间面积 ────────────────────────────────────
    @classmethod
    def area_between(cls, expr1: str, expr2: str, a: float, b: float) -> dict:
        """返回两曲线间的区域面积"""
        try:
            from scipy import integrate as scipy_integrate
            e1 = _parse(expr1)
            e2 = _parse(expr2)
            diff_expr = Abs(e1 - e2)
            f = lambdify(cls.x, diff_expr, "numpy")
            val, err = scipy_integrate.quad(f, a, b, limit=200)
            return {"numeric": round(val, 8), "error": round(float(err), 10)}
        except Exception:
            return {"numeric": float('nan'), "error": None}

    # ── 泰勒级数 ─────────────────────────────────────────
    @classmethod
    def taylor(cls, expr_str: str, x0: float = 0.0, n: int = 5) -> dict:
        """
        返回:
          "polynomial": sympy 多项式
          "poly_str":    格式化字符串 (e.g. "x - x^3/6 + x^5/120")
          "poly_fn":      lambdified 函数
          "coefficients": [(order, coeff), ...]
        """
        expr = _parse(expr_str)
        s = series(expr, cls.x, x0, n + 1).removeO()
        poly = simplify(s)
        poly_str = str(poly).replace("**", "^")
        poly_fn = lambdify(cls.x, poly, "numpy")

        # 提取系数
        coeffs = []
        for k in range(n + 1):
            c = diff(expr, cls.x, k).subs(cls.x, x0) / sp.factorial(k)
            cv = float(c.evalf()) if c.is_real else None
            if cv is not None and abs(cv) > 1e-14:
                coeffs.append((k, cv))

        return {
            "polynomial": poly,
            "poly_str": poly_str,
            "poly_fn": poly_fn,
            "coefficients": coeffs,
            "x0": x0,
            "n": n,
        }

    # ── 傅里叶级数 ─────────────────────────────────────────
    @classmethod
    def fourier(cls, expr_str: str, L: float | None = None,
                n_terms: int = 3, period: float | None = None) -> dict:
        """
        在 [-L, L] 上展开傅里叶级数, 截断到 n_terms 项。
        若 period 指定, 则 L = period/2。
        返回:
          "a0": 常数项
          "an": [(n, value), ...]  余弦系数
          "bn": [(n, value), ...]  正弦系数
          "approx_str": 逼近表达式字符串
          "approx_fn":  lambdified 函数
        """
        expr = _parse(expr_str)
        pi_val = float(sp.pi.evalf())

        # ── 智能周期→符号 L (消除 π/L 浮点误差) ──
        raw = period if period else (L * 2 if L else 2 * pi_val)
        L_sym = cls._to_symbolic_L(raw, pi_val)
        L_val = float(L_sym.evalf()) if hasattr(L_sym, 'evalf') else float(L_sym)

        # ── 符号积分求系数 ──
        a0_val = float((integrate(expr, (cls.x, -L_sym, L_sym)) / L_sym).evalf())
        an_list = []; bn_list = []
        for n in range(1, n_terms + 1):
            cn = cos(n * sp.pi * cls.x / L_sym)
            sn = sin(n * sp.pi * cls.x / L_sym)
            an_val = float((integrate(expr * cn, (cls.x, -L_sym, L_sym)) / L_sym).evalf())
            bn_val = float((integrate(expr * sn, (cls.x, -L_sym, L_sym)) / L_sym).evalf())
            if abs(an_val) > 1e-12:
                an_list.append((n, cls._snap_coeff(an_val)))
            if abs(bn_val) > 1e-12:
                bn_list.append((n, cls._snap_coeff(bn_val)))

        # ── 构建逼近 ──
        approx = a0_val / 2
        for n, av in an_list:
            approx = approx + av * cos(n * sp.pi * cls.x / L_sym)
        for n, bv in bn_list:
            approx = approx + bv * sin(n * sp.pi * cls.x / L_sym)

        approx_simplified = simplify(approx)
        approx_str = str(approx_simplified).replace("**", "^")
        approx_fn = lambdify(cls.x, approx_simplified, "numpy")

        return {
            "a0": cls._snap_coeff(a0_val / 2),
            "an": an_list, "bn": bn_list,
            "approx_str": approx_str, "approx_fn": approx_fn,
            "L": L_val, "n_terms": n_terms,
        }

    @staticmethod
    def _to_symbolic_L(raw_period: float, pi_val: float):
        """将数值周期映射到符号 L (消除 0.3183*pi*x 这种丑陋表达)"""
        # 匹配已知 π 倍数
        for mult, sym_L in [(2, sp.pi), (1, sp.pi / 2), (4, 2 * sp.pi),
                             (1/2, sp.pi / 4), (3, 3 * sp.pi / 2)]:
            if abs(raw_period - mult * pi_val) < 0.005:
                return sym_L
        # 整数周期
        r = round(raw_period)
        if abs(raw_period - r) < 0.01:
            return sp.Rational(r) / 2
        # 回落
        return raw_period / 2

    @staticmethod
    def _snap_coeff(v: float) -> float:
        """浮点系数修到精确值 (1.00000002→1, 0.66666667→2/3)"""
        for target in [0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5]:
            if abs(v - target) < 1e-6:
                return target
        for num, den in [(1, 3), (2, 3), (1, 2), (3, 2), (1, 4), (3, 4)]:
            if abs(v - num / den) < 2e-7:
                return round(num / den, 10)
        return round(v, 8)


# ============================================================================
# 级数收敛分析引擎
# ============================================================================

class SeriesAnalysis:
    """级数收敛判定 / 幂级数 / 求和"""

    @staticmethod
    def _make_sequence(expr_str: str):
        """将含 n 的表达式转为 sympy 序列项 a_n"""
        n = sp.Symbol('n', integer=True, positive=True)
        s = _preprocess_expr(expr_str)
        a_n = sp.sympify(s, locals={
            "n": n, "sin": sin, "cos": cos, "tan": tan,
            "log": log, "ln": log, "exp": exp, "sqrt": sqrt, "abs": Abs,
            "pi": sp.pi, "e": sp.E, "factorial": sp.factorial,
        })
        return a_n, n

    @classmethod
    def convergence_test(cls, expr_str: str) -> dict:
        """
        对级数 ∑ a_n 进行多种收敛判定。
        expr_str 是通项 a_n, 例如 "1/n^2", "(-1)^n/n", "1/(2^n)"

        返回: {"converges": bool|null, "tests": [{"name":..., "result":..., "detail":...}]}
        """
        a_n, n = cls._make_sequence(expr_str)
        tests = []

        # 1. 必要条件: lim a_n = 0 ?
        try:
            lim = sp.limit(a_n, n, sp.oo)
            lim_val = float(lim.evalf()) if lim.is_real else None
            if lim_val is not None:
                if abs(lim_val) > 0.001:
                    tests.append({"name": "通项极限检验",
                                  "result": "发散",
                                  "detail": f"lim a_n = {lim:.4f} ≠ 0, 级数发散",
                                  "converges": False})
                    return {"converges": False, "conclusion": "发散 (通项不趋于0)", "tests": tests}
                else:
                    tests.append({"name": "通项极限检验",
                                  "result": "通过",
                                  "detail": f"lim a_n = {lim:.4f} = 0, 必要条件满足",
                                  "converges": None})
        except Exception:
            pass

        # 2. 比值检验 (Ratio Test)
        try:
            ratio = sp.simplify(sp.Abs(a_n.subs(n, n + 1) / a_n))
            lim_ratio = sp.limit(ratio, n, sp.oo)
            rv = float(lim_ratio.evalf()) if lim_ratio.is_real else None
            if rv is not None:
                if rv < 0.999:
                    tests.append({"name": "比值检验 (Ratio Test)",
                                  "result": "收敛",
                                  "detail": f"lim |a_{{n+1}}/a_n| = {rv:.4f} < 1",
                                  "converges": True})
                elif rv > 1.001:
                    tests.append({"name": "比值检验 (Ratio Test)",
                                  "result": "发散",
                                  "detail": f"lim |a_{{n+1}}/a_n| = {rv:.4f} > 1",
                                  "converges": False})
                else:
                    tests.append({"name": "比值检验 (Ratio Test)",
                                  "result": "不确定",
                                  "detail": f"lim |a_{{n+1}}/a_n| = {rv:.4f} ≈ 1, 无法判定",
                                  "converges": None})
        except Exception:
            pass

        # 3. 根值检验 (Root Test)
        try:
            root = sp.simplify(sp.Abs(a_n) ** (1 / n))
            lim_root = sp.limit(root, n, sp.oo)
            rrv = float(lim_root.evalf()) if lim_root.is_real else None
            if rrv is not None and not any(t["name"] == "根值检验 (Root Test)" for t in tests):
                if rrv < 0.999:
                    tests.append({"name": "根值检验 (Root Test)",
                                  "result": "收敛",
                                  "detail": f"lim |a_n|^(1/n) = {rrv:.4f} < 1",
                                  "converges": True})
                elif rrv > 1.001:
                    tests.append({"name": "根值检验 (Root Test)",
                                  "result": "发散",
                                  "detail": f"lim |a_n|^(1/n) = {rrv:.4f} > 1",
                                  "converges": False})
                else:
                    tests.append({"name": "根值检验 (Root Test)",
                                  "result": "不确定",
                                  "detail": f"lim |a_n|^(1/n) = {rrv:.4f} ≈ 1, 无法判定",
                                  "converges": None})
        except Exception:
            pass

        # 4. p-级数比较
        try:
            # 尝试识别 1/n^p 形式
            degree = sp.degree(sp.denom(sp.together(a_n)), n) if a_n.has(n) else 0
            if degree > 1.01:
                tests.append({"name": "p-级数比较",
                              "result": "收敛",
                              "detail": f"分母最高次 ≈ {degree:.1f} > 1, 类似 Σ1/n^{degree:.1f}",
                              "converges": True})
            elif 0 < degree <= 1.01:
                tests.append({"name": "p-级数比较",
                              "result": "发散",
                              "detail": f"分母最高次 ≈ {degree:.1f} ≤ 1, 类似 Σ1/n^{degree:.1f}",
                              "converges": False})
        except Exception:
            pass

        # 5. 交错级数检验
        try:
            # 检测 (-1)^n 因子 (更鲁棒的检测)
            alt_factor = a_n / ((-1) ** n)
            alt_factor_simplified = sp.simplify(alt_factor)
            # 或者检测 a_n * (-1)^n 是否恒正
            alt_check = sp.simplify(a_n * ((-1) ** n))
            is_alternating = False
            try:
                # 检查是否 a_n = (-1)^n * b_n 且 b_n > 0 (符号交替)
                if alt_check.subs(n, 10).evalf() > 0 and alt_check.subs(n, 11).evalf() > 0:
                    is_alternating = True
            except Exception:
                pass
            # 回退: a_{{n+1}}/a_n ≈ -1
            if not is_alternating:
                try:
                    alt_ratio = sp.simplify(a_n.subs(n, n + 1) / a_n)
                    if alt_ratio == -1 or str(alt_ratio) == "-1":
                        is_alternating = True
                except Exception:
                    pass

            if is_alternating:
                abs_a = sp.Abs(a_n)
                lim_abs = sp.limit(abs_a, n, sp.oo)
                try:
                    lim_v = float(lim_abs.evalf())
                except Exception:
                    lim_v = None
                if lim_v is not None and lim_v < 0.001:
                    # 检查递减
                    diff = sp.simplify(abs_a.subs(n, n + 1) - abs_a)
                    try:
                        is_decreasing = float(diff.subs(n, 10).evalf()) < 0
                    except Exception:
                        is_decreasing = True  # 大概率递减
                    if is_decreasing:
                        tests.append({"name": "交错级数检验 (Leibniz)",
                                      "result": "条件收敛",
                                      "detail": "|a_n| 单调递减且趋于 0",
                                      "converges": True})
        except Exception:
            pass

        # 总结
        has_conditional = any(t["name"].startswith("交错级数") for t in tests)
        any_conv = any(t.get("converges") is True for t in tests)
        any_div = any(t.get("converges") is False for t in tests)
        if any_div and not any_conv:
            conclusion = "发散"
        elif any_conv and not any_div:
            if has_conditional:
                conclusion = "条件收敛 (交错级数, 绝对值级数发散)"
            else:
                conclusion = "收敛"
        elif any_conv and any_div:
            if has_conditional:
                conclusion = "条件收敛 (交错级数, 绝对值级数发散)"
            else:
                conclusion = "判定结果不一致, 需要进一步分析"
        else:
            conclusion = "无法确定 (比值/根值检验失效, 可尝试比较检验)"

        return {"converges": True if any_conv and not any_div else (False if any_div and not any_conv else None),
                "conclusion": conclusion, "tests": tests}

    @classmethod
    def power_series_radius(cls, expr_str: str) -> dict:
        """
        对幂级数 ∑ a_n * (x - x0)^n, 求收敛半径。
        expr_str 是系数 a_n 的通项, 例如 "1/n!", "1/2^n", "1/n"
        """
        a_n, n = cls._make_sequence(expr_str)
        try:
            ratio = sp.simplify(sp.Abs(a_n.subs(n, n + 1) / a_n))
            L = sp.limit(ratio, n, sp.oo)
            L_val = float(L.evalf()) if L.is_real else None
            if L_val is not None and L_val > 0:
                R = 1.0 / L_val
            elif L_val == 0:
                R = float('inf')
            else:
                R = 0.0
            return {"radius": R if R != float('inf') else None,
                    "radius_str": f"{R:.6f}" if R != float('inf') else "∞",
                    "L": L_val, "method": "比值法"}
        except Exception:
            return {"radius": None, "radius_str": "无法计算", "L": None, "method": "失败"}

    @classmethod
    def series_sum(cls, expr_str: str, from_n: int = 1, to_n=None) -> dict:
        """计算级数和 ∑_{n=from_n}^{to_n or ∞} a_n"""
        a_n, n = cls._make_sequence(expr_str)
        try:
            if to_n is None:
                s = sp.summation(a_n, (n, from_n, sp.oo))
            else:
                s = sp.summation(a_n, (n, from_n, to_n))
            val = s.evalf() if s.has(sp.Sum) else s
            return {"symbolic": str(s).replace("**", "^"),
                    "numeric": float(val.evalf()) if hasattr(val, 'evalf') else None,
                    "closed_form": not s.has(sp.Sum)}
        except Exception as e:
            return {"symbolic": str(e), "numeric": None, "closed_form": False}


# ============================================================================
# 高级积分引擎
# ============================================================================

class AdvancedIntegration:
    """反常积分 / 多重积分 / 数值方法对比"""

    @classmethod
    def improper_integral(cls, expr_str: str, a: float, b=None,
                          singularity: float | None = None) -> dict:
        """
        反常积分: b=None 表示 ∫[a,∞); singularity 表示瑕点
        返回收敛性 + 数值
        """
        x = sp.Symbol('x')
        expr = _parse(expr_str)
        try:
            if b is None:
                # ∫[a, ∞)
                sym_val = sp.integrate(expr, (x, a, sp.oo))
                converges = sym_val.is_finite
            elif singularity is not None:
                sym_val = sp.integrate(expr, (x, a, b))
                converges = sym_val.is_finite
            else:
                sym_val = sp.integrate(expr, (x, a, b))
                converges = sym_val.is_finite

            # 数值积分
            try:
                from scipy import integrate as scipy_integrate
                f = sp.lambdify(x, expr, "numpy")
                if b is None:
                    val, err = scipy_integrate.quad(f, a, np.inf, limit=200)
                elif singularity is not None:
                    val, err = scipy_integrate.quad(f, a, singularity - 1e-8, limit=100)
                    v2, e2 = scipy_integrate.quad(f, singularity + 1e-8, b, limit=100)
                    val += v2
                    err = max(err, e2)
                else:
                    val, err = scipy_integrate.quad(f, a, b, limit=200)
                numeric = round(val, 10)
                error = round(float(err), 10)
            except Exception:
                numeric = None
                error = None

            return {
                "converges": bool(converges),
                "symbolic": str(sym_val).replace("**", "^") if converges else "发散",
                "numeric": numeric,
                "error": error,
            }
        except Exception as e:
            return {"converges": None, "symbolic": str(e), "numeric": None, "error": None}

    @classmethod
    def double_integral(cls, expr_str: str,
                        x_range: tuple, y_range: tuple) -> dict:
        """
        二重积分 ∬ f(x,y) dxdy
        x_range: (x_low, x_high) 可以是数字或 y 的表达式
        y_range: (y_low, y_high) 数字
        """
        x, y = sp.symbols('x y')
        s = _preprocess_expr(expr_str)
        f = sp.sympify(s, locals={
            "x": x, "y": y, "sin": sin, "cos": cos, "tan": tan,
            "log": log, "exp": exp, "sqrt": sqrt, "abs": Abs,
            "pi": sp.pi, "e": sp.E,
        })

        try:
            # 先对 x 积分
            inner = sp.integrate(f, (x, x_range[0], x_range[1]))
            inner_simplified = sp.simplify(inner)
            # 再对 y 积分
            outer = sp.integrate(inner_simplified, (y, y_range[0], y_range[1]))
            outer_simplified = sp.simplify(outer)
            numeric = float(outer_simplified.evalf()) if outer_simplified.is_real else None
            return {
                "symbolic": str(outer_simplified).replace("**", "^"),
                "numeric": round(numeric, 10) if numeric else None,
                "inner": str(inner_simplified).replace("**", "^"),
            }
        except Exception as e:
            # 回退纯数值
            try:
                from scipy import integrate as scipy_integrate
                f_fn = sp.lambdify((x, y), f, "numpy")
                val, err = scipy_integrate.dblquad(
                    f_fn, y_range[0], y_range[1],
                    lambda y: x_range[0], lambda y: x_range[1],
                    epsabs=1e-6, limit=100
                )
                return {"symbolic": "—", "numeric": round(val, 10),
                        "inner": "—", "error": round(float(err), 10)}
            except Exception as e2:
                return {"symbolic": str(e), "numeric": None, "inner": str(e2)}

    @classmethod
    def numerical_compare(cls, expr_str: str, a: float, b: float) -> dict:
        """对比多种数值积分方法"""
        x = sp.Symbol('x')
        expr = _parse(expr_str)
        f = sp.lambdify(x, expr, "numpy")
        try:
            exact_r = cls.improper_integral(expr_str, a, b)
            exact = exact_r.get("numeric")
        except Exception:
            exact = None

        results = {}
        n_seg = 20
        xs_uniform = np.linspace(a, b, n_seg + 1)
        ys_uniform = np.array([float(f(np.array([xv])).item()) for xv in xs_uniform])

        # Trapezoidal
        trap = float(np.trapezoid(ys_uniform, xs_uniform)
                     if hasattr(np, 'trapezoid') else np.trapz(ys_uniform, xs_uniform))
        results["梯形法"] = round(trap, 8)

        # Simpson (scipy)
        try:
            from scipy.integrate import simpson
            simp = simpson(y=ys_uniform, x=xs_uniform)
            results["Simpson"] = round(float(simp), 8)
        except Exception:
            pass

        # Midpoint
        mids = (xs_uniform[:-1] + xs_uniform[1:]) / 2
        ys_mid = np.array([float(f(np.array([mv])).item()) for mv in mids])
        mid_val = np.sum(ys_mid) * (b - a) / n_seg
        results["中点法"] = round(float(mid_val), 8)

        # Gaussian quadrature (scipy)
        try:
            from scipy.integrate import fixed_quad
            gauss, _ = fixed_quad(f, a, b, n=10)
            results["Gauss-10"] = round(float(gauss), 8)
        except Exception:
            pass

        return {
            "exact": exact,
            "methods": results,
            "n_segments": n_seg,
        }


# ============================================================================
# 自测
# ============================================================================
if __name__ == "__main__":
    print("=== 不定积分 ===")
    F_str, F_fn = CalculusEngine.integrate_indefinite("sin(x)")
    print(f"  ∫sin(x)dx = {F_str}")

    print("\n=== 定积分 ===")
    r = CalculusEngine.integrate_definite("sin(x)", 0, np.pi)
    print(f"  ∫[0,π] sin(x)dx = {r}")

    print("\n=== 黎曼和 ===")
    ri = CalculusEngine.riemann_data("sin(x)", 0, np.pi, n=6, mode="midpoint")
    print(f"  midpoint(6) ≈ {ri['value']}  (exact: 2)")

    print("\n=== 泰勒级数 ===")
    t = CalculusEngine.taylor("sin(x)", x0=0, n=5)
    print(f"  sin(x) ≈ {t['poly_str']}")
    print(f"  coeffs: {t['coefficients']}")

    print("\n=== 傅里叶级数 ===")
    fo = CalculusEngine.fourier("x", period=2 * np.pi, n_terms=3)
    print(f"  f(x)=x on [-π,π], 3 terms:")
    print(f"  a0/2 = {fo['a0']}")
    print(f"  an: {fo['an']}")
    print(f"  bn: {fo['bn']}")
    print(f"  approx: {fo['approx_str']}")


# ============================================================================
# 极限 & 弧长曲率 & 三重积分
# ============================================================================

class LimitEngine:
    """极限计算: lim_{x→a} f(x) / 左右极限"""

    @classmethod
    def compute(cls, expr_str: str, a: float, direction: str = "both") -> dict:
        """
        direction: "both" | "left" | "right"
        返回: {"value": float|null, "left": float|null, "right": float|null,
               "exists": bool, "symbolic": str}
        """
        x = sp.Symbol('x')
        expr = _parse(expr_str)
        result = {"symbolic": "", "value": None, "left": None, "right": None, "exists": False}

        try:
            lim_two_sided = sp.limit(expr, x, a)
            if lim_two_sided.is_real and lim_two_sided.is_finite:
                v = float(lim_two_sided.evalf())
                result["value"] = round(v, 10)
                result["exists"] = True
                result["symbolic"] = str(lim_two_sided).replace("**", "^")
        except Exception:
            pass

        try:
            lim_left = sp.limit(expr, x, a, dir="-")
            if lim_left.is_real:
                result["left"] = round(float(lim_left.evalf()), 10)
        except Exception:
            pass

        try:
            lim_right = sp.limit(expr, x, a, dir="+")
            if lim_right.is_real:
                result["right"] = round(float(lim_right.evalf()), 10)
        except Exception:
            pass

        if result["left"] is not None and result["right"] is not None:
            result["exists"] = abs(result["left"] - result["right"]) < 1e-8
            if result["exists"]:
                result["value"] = result["left"]
        elif result["value"] is not None:
            result["exists"] = True

        return result


class CurveEngine:
    """弧长 / 曲率 / 参数曲线 / 极坐标"""

    @classmethod
    def arc_length(cls, expr_str: str, a: float, b: float) -> dict:
        """曲线 y=f(x) 在 [a,b] 上的弧长 L = ∫√(1+f'²)dx"""
        x = sp.Symbol('x')
        expr = _parse(expr_str)
        fp = sp.diff(expr, x)
        integrand = sp.sqrt(1 + fp ** 2)
        try:
            sym = sp.integrate(integrand, (x, a, b))
            sym_str = str(sp.simplify(sym)).replace("**", "^") if sym.is_real else "—"
        except Exception:
            sym_str = "—"
        try:
            f_fn = sp.lambdify(x, integrand, "numpy")
            from scipy import integrate as scipy_integrate
            val, err = scipy_integrate.quad(f_fn, a, b, limit=200)
            numeric = round(val, 8)
            error = round(float(err), 10)
        except Exception:
            numeric = None; error = None
        return {"symbolic": sym_str, "numeric": numeric, "error": error,
                "integrand": str(integrand).replace("**", "^")}

    @classmethod
    def curvature(cls, expr_str: str, x_val: float) -> dict:
        """曲率 κ = |f''| / (1 + f'²)^(3/2)  曲率半径 R = 1/κ"""
        x = sp.Symbol('x')
        expr = _parse(expr_str)
        fp = sp.diff(expr, x)
        fpp = sp.diff(fp, x)
        kappa_expr = sp.Abs(fpp) / (1 + fp ** 2) ** (sp.Rational(3, 2))
        try:
            k_val = float(kappa_expr.subs(x, x_val).evalf())
        except Exception:
            k_val = None
        R = 1.0 / k_val if k_val and k_val > 1e-12 else None
        return {"curvature": round(k_val, 8) if k_val else None,
                "radius": round(R, 6) if R else None,
                "kappa_expr": str(kappa_expr).replace("**", "^"),
                "at_x": x_val}

    @classmethod
    def parametric_curve(cls, xt_str: str, yt_str: str, t_range: tuple, n_pts=500) -> dict:
        """参数曲线 (x(t), y(t)), t∈[t0,t1]"""
        t = sp.Symbol('t')
        s1 = _preprocess_expr(xt_str); s2 = _preprocess_expr(yt_str)
        loc = {"t": t, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
               "log": sp.log, "exp": sp.exp, "sqrt": sp.sqrt, "abs": sp.Abs, "pi": sp.pi, "e": sp.E}
        xt = sp.sympify(s1, locals=loc); yt = sp.sympify(s2, locals=loc)
        xt_fn = sp.lambdify(t, xt, "numpy"); yt_fn = sp.lambdify(t, yt, "numpy")
        ts = np.linspace(t_range[0], t_range[1], n_pts)
        xs = xt_fn(ts); ys = yt_fn(ts)
        # 弧长
        try:
            dxdt = sp.diff(xt, t); dydt = sp.diff(yt, t)
            ds = sp.sqrt(dxdt**2 + dydt**2)
            ds_fn = sp.lambdify(t, ds, "numpy")
            from scipy import integrate as scipy_integrate
            L, err = scipy_integrate.quad(ds_fn, t_range[0], t_range[1], limit=200)
            arc_len = round(L, 8)
        except Exception:
            arc_len = None
        return {"xs": xs.tolist(), "ys": ys.tolist(), "arc_length": arc_len,
                "t_range": t_range}

    @classmethod
    def polar_curve(cls, r_str: str, theta_range=(0, 2*np.pi), n_pts=800) -> dict:
        """极坐标 r = f(θ)"""
        th = sp.Symbol('theta')
        s = _preprocess_expr(r_str)
        loc = {"theta": th, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
               "log": sp.log, "exp": sp.exp, "sqrt": sp.sqrt, "abs": sp.Abs, "pi": sp.pi, "e": sp.E}
        r_expr = sp.sympify(s, locals=loc)
        r_fn = sp.lambdify(th, r_expr, "numpy")
        ths = np.linspace(theta_range[0], theta_range[1], n_pts)
        rs = r_fn(ths)
        rs = np.where(np.isfinite(rs), rs, np.nan)
        xs = rs * np.cos(ths); ys = rs * np.sin(ths)
        # 极坐标面积 A = 1/2 ∫ r² dθ
        try:
            r2_fn = sp.lambdify(th, r_expr**2, "numpy")
            from scipy import integrate as scipy_integrate
            A, err = scipy_integrate.quad(r2_fn, theta_range[0], theta_range[1], limit=200)
            area = round(0.5 * A, 8)
        except Exception:
            area = None
        return {"xs": xs.tolist(), "ys": ys.tolist(), "rs": rs.tolist(), "ths": ths.tolist(),
                "area": area, "r_str": r_str}


class OdeEngine:
    """一阶 / 二阶 ODE 方向场 + 数值解"""

    @classmethod
    def direction_field(cls, expr_str: str, x_range=(-3, 3), y_range=(-3, 3),
                        nx=25, ny=25) -> dict:
        """
        y' = f(x, y) 方向场。
        expr_str 是 f(x,y), 含 x 和 y。
        """
        x_s, y_s = sp.symbols('x y')
        s = _preprocess_expr(expr_str)
        loc = {"x": x_s, "y": y_s, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
               "log": sp.log, "exp": sp.exp, "sqrt": sp.sqrt, "abs": sp.Abs,
               "pi": sp.pi, "e": sp.E}
        f_expr = sp.sympify(s, locals=loc)
        f_fn = sp.lambdify((x_s, y_s), f_expr, "numpy")
        xs = np.linspace(x_range[0], x_range[1], nx)
        ys = np.linspace(y_range[0], y_range[1], ny)
        X, Y = np.meshgrid(xs, ys)
        try:
            U = np.ones_like(X)
            V = f_fn(X, Y)
            # 归一化
            norms = np.sqrt(U**2 + V**2)
            norms[norms < 1e-10] = 1
            U = U / norms; V = V / norms
        except Exception:
            U = np.ones_like(X); V = np.zeros_like(X)
        return {"X": X.tolist(), "Y": Y.tolist(), "U": U.tolist(), "V": V.tolist(),
                "x_range": x_range, "y_range": y_range, "expr": expr_str}

    @classmethod
    def solve_ivp(cls, expr_str: str, x0: float, y0: float,
                  x_end: float, n_steps: int = 200) -> dict:
        """数值求解 y'=f(x,y), y(x0)=y0, 用 RK45"""
        x_s, y_s = sp.symbols('x y')
        s = _preprocess_expr(expr_str)
        loc = {"x": x_s, "y": y_s, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
               "log": sp.log, "exp": sp.exp, "sqrt": sp.sqrt, "abs": sp.Abs,
               "pi": sp.pi, "e": sp.E}
        f_expr = sp.sympify(s, locals=loc)
        f_fn = sp.lambdify((x_s, y_s), f_expr, "numpy")

        try:
            from scipy.integrate import solve_ivp as solve_ivp_scipy
            sol = solve_ivp_scipy(
                lambda x, y: f_fn(x, y[0]),
                (x0, x_end), [y0],
                t_eval=np.linspace(x0, x_end, n_steps),
                method="RK45", rtol=1e-8, atol=1e-10,
            )
            return {"xs": sol.t.tolist(), "ys": sol.y[0].tolist(),
                    "success": sol.success, "message": str(sol.message)}
        except Exception as e:
            # 回退 Euler
            xs = [x0]; ys = [y0]
            h = (x_end - x0) / n_steps
            for i in range(n_steps):
                try:
                    k1 = float(f_fn(np.array([xs[-1]]), np.array([ys[-1]])).item())
                    xs.append(xs[-1] + h)
                    ys.append(ys[-1] + h * k1)
                except Exception:
                    break
            return {"xs": xs, "ys": ys, "success": False, "message": f"Euler fallback: {e}"}


# ============================================================================
# 多元微积分引擎
# ============================================================================

class MultivariableEngine:
    """偏导数 / 梯度 / 方向导数 / Hessian / 拉格朗日乘数 / 3D 采样"""

    @classmethod
    def gradient(cls, expr_str: str, at_point: tuple = None) -> dict:
        """∇f = [∂f/∂x, ∂f/∂y, ∂f/∂z]"""
        x, y, z = sp.symbols('x y z')
        expr = cls._parse_mv(expr_str, (x, y, z))
        vars_used = list(expr.free_symbols & {x, y, z})
        grad = [sp.diff(expr, v) for v in vars_used]
        grad_simplified = [sp.simplify(g) for g in grad]
        result = {"gradient": [str(g).replace("**", "^") for g in grad_simplified],
                  "variables": [str(v) for v in vars_used]}
        if at_point:
            subs_dict = {v: at_point[i] for i, v in enumerate(vars_used) if i < len(at_point)}
            vals = [float(g.subs(subs_dict).evalf()) for g in grad_simplified]
            result["at_point"] = at_point
            result["values"] = [round(v, 8) for v in vals]
            result["magnitude"] = round(np.sqrt(sum(v**2 for v in vals)), 8)
        return result

    @classmethod
    def directional_derivative(cls, expr_str: str, point: tuple, direction: tuple) -> dict:
        """D_u f = ∇f · u (单位方向)"""
        g = cls.gradient(expr_str, point)
        if "values" not in g:
            return {"error": "无法计算梯度"}
        grad_vals = g["values"]
        norm = np.sqrt(sum(d**2 for d in direction))
        u = [d / norm for d in direction]
        dd = sum(gv * ui for gv, ui in zip(grad_vals, u[:len(grad_vals)]))
        return {"directional_derivative": round(dd, 8),
                "gradient": grad_vals, "unit_direction": [round(ui, 6) for ui in u]}

    @classmethod
    def hessian(cls, expr_str: str) -> dict:
        """Hessian 矩阵 H_ij = ∂²f/∂x_i∂x_j"""
        x, y = sp.symbols('x y')
        expr = cls._parse_mv(expr_str, (x, y))
        vars_used = list(expr.free_symbols & {x, y})
        n = len(vars_used)
        H = [[sp.simplify(sp.diff(expr, v1, v2)) for v2 in vars_used] for v1 in vars_used]
        return {"hessian": [[str(h).replace("**", "^") for h in row] for row in H],
                "variables": [str(v) for v in vars_used]}

    @classmethod
    def lagrange_multipliers(cls, f_str: str, g_str: str) -> dict:
        """拉格朗日乘数法: 约束 g(x,y)=0 下求 f(x,y) 极值"""
        x, y, lam = sp.symbols('x y lambda')
        f = cls._parse_mv(f_str, (x, y))
        g = cls._parse_mv(g_str, (x, y))
        L = f - lam * g
        eqs = [sp.diff(L, x), sp.diff(L, y), g]
        try:
            sols = sp.solve(eqs, (x, y, lam), dict=True)
            critical = []
            for sol in sols:
                xv = float(sol[x].evalf()) if sol[x].is_real else None
                yv = float(sol[y].evalf()) if sol[y].is_real else None
                lv = float(sol[lam].evalf()) if sol[lam].is_real else None
                if xv is not None and yv is not None:
                    fv = float(f.subs({x: xv, y: yv}).evalf())
                    critical.append({"x": round(xv, 6), "y": round(yv, 6),
                                     "lambda": round(lv, 6) if lv else None, "f": round(fv, 8)})
            return {"critical_points": critical, "lagrangian": str(L).replace("**", "^")}
        except Exception as e:
            return {"critical_points": [], "error": str(e)}

    @classmethod
    def surface_sample(cls, expr_str: str, x_range=(-3, 3), y_range=(-3, 3),
                       n_pts=80) -> dict:
        """采样 z=f(x,y) 曲面, 返回网格数据给 3D 绘图"""
        x, y = sp.symbols('x y')
        expr = cls._parse_mv(expr_str, (x, y))
        f = sp.lambdify((x, y), expr, "numpy")
        xs = np.linspace(x_range[0], x_range[1], n_pts)
        ys = np.linspace(y_range[0], y_range[1], n_pts)
        X, Y = np.meshgrid(xs, ys)
        Z = f(X, Y)
        Z = np.where(np.isfinite(Z), Z, np.nan)
        return {"X": X.tolist(), "Y": Y.tolist(), "Z": Z.tolist(),
                "x_range": x_range, "y_range": y_range}

    @staticmethod
    def _parse_mv(expr_str: str, symbols):
        s = _preprocess_expr(expr_str)
        loc = {str(v): v for v in symbols}
        loc.update({"sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
                     "log": sp.log, "exp": sp.exp, "sqrt": sp.sqrt, "abs": sp.Abs,
                     "pi": sp.pi, "e": sp.E})
        return sp.sympify(s, locals=loc)


# ============================================================================
# 拉普拉斯变换引擎
# ============================================================================

class LaplaceEngine:
    """拉普拉斯变换 L{f(t)} = F(s) / 逆变换"""

    @classmethod
    def transform(cls, expr_str: str) -> dict:
        """L{f(t)} = ∫₀^∞ f(t) e^{-st} dt"""
        t, s = sp.symbols('t s')
        expr = cls._parse_lt(expr_str, t)
        try:
            F = sp.laplace_transform(expr, t, s, noconds=True)
            F_str = str(sp.simplify(F)).replace("**", "^")
            return {"result": F_str, "converges": True}
        except Exception as e:
            return {"result": str(e), "converges": False}

    @classmethod
    def inverse_transform(cls, expr_str: str) -> dict:
        """L^{-1}{F(s)} = f(t)"""
        t, s = sp.symbols('t s')
        expr = cls._parse_lt(expr_str, s)
        try:
            f = sp.inverse_laplace_transform(expr, s, t)
            f_str = str(sp.simplify(f)).replace("**", "^")
            return {"result": f_str}
        except Exception as e:
            return {"result": str(e)}

    @staticmethod
    def _parse_lt(expr_str: str, var):
        s = _preprocess_expr(expr_str)
        loc = {str(var): var, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
               "log": sp.log, "exp": sp.exp, "sqrt": sp.sqrt, "abs": sp.Abs,
               "pi": sp.pi, "e": sp.E, "Heaviside": sp.Heaviside, "DiracDelta": sp.DiracDelta}
        return sp.sympify(s, locals=loc)


# ============================================================================
# 向量分析引擎
# ============================================================================

class VectorEngine:
    """散度 / 旋度 / 线积分"""

    @classmethod
    def divergence(cls, F_str: str, at_point: tuple = None) -> dict:
        """div F = ∂Fx/∂x + ∂Fy/∂y + ∂Fz/∂z"""
        x, y, z = sp.symbols('x y z')
        Fx, Fy, Fz = cls._parse_vector(F_str, (x, y, z))
        div = sp.diff(Fx, x) + sp.diff(Fy, y) + sp.diff(Fz, z)
        div_simp = sp.simplify(div)
        result = {"divergence": str(div_simp).replace("**", "^")}
        if at_point:
            subs_d = {x: at_point[0], y: at_point[1]}
            if len(at_point) > 2:
                subs_d[z] = at_point[2]
            result["at_point"] = at_point
            result["value"] = round(float(div_simp.subs(subs_d).evalf()), 8)
        return result

    @classmethod
    def curl(cls, F_str: str, at_point: tuple = None) -> dict:
        """curl F = ∇ × F (3D)"""
        x, y, z = sp.symbols('x y z')
        Fx, Fy, Fz = cls._parse_vector(F_str, (x, y, z))
        Cx = sp.diff(Fz, y) - sp.diff(Fy, z)
        Cy = sp.diff(Fx, z) - sp.diff(Fz, x)
        Cz = sp.diff(Fy, x) - sp.diff(Fx, y)
        curl_vec = [sp.simplify(Cx), sp.simplify(Cy), sp.simplify(Cz)]
        result = {"curl": [str(c).replace("**", "^") for c in curl_vec]}
        if at_point:
            subs_d = {x: at_point[0], y: at_point[1], z: at_point[2]}
            result["at_point"] = at_point
            result["value"] = [round(float(c.subs(subs_d).evalf()), 8) for c in curl_vec]
        return result

    @classmethod
    def line_integral(cls, F_str: str, curve_xt: str, curve_yt: str,
                      t_range: tuple) -> dict:
        """线积分 ∫_C F·dr, C: (x(t), y(t)), t∈[t0,t1]"""
        t, x_s, y_s = sp.symbols('t x y')
        Fx_s, Fy_s, _ = cls._parse_vector(F_str, (x_s, y_s))
        loc_t = {"t": t, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
                 "log": sp.log, "exp": sp.exp, "sqrt": sp.sqrt, "abs": sp.Abs, "pi": sp.pi}
        xt = sp.sympify(_preprocess_expr(curve_xt), locals=loc_t)
        yt = sp.sympify(_preprocess_expr(curve_yt), locals=loc_t)
        # 参数化 F
        Fx_t = Fx_s.subs({x_s: xt, y_s: yt})
        Fy_t = Fy_s.subs({x_s: xt, y_s: yt})
        dxdt = sp.diff(xt, t); dydt = sp.diff(yt, t)
        integrand = Fx_t * dxdt + Fy_t * dydt
        integrand_s = sp.simplify(integrand)
        try:
            sym_val = sp.integrate(integrand_s, (t, t_range[0], t_range[1]))
            sym_str = str(sp.simplify(sym_val)).replace("**", "^")
        except Exception:
            sym_str = "—"
        try:
            f_int = sp.lambdify(t, integrand_s, "numpy")
            from scipy import integrate as scipy_integrate
            val, err = scipy_integrate.quad(f_int, t_range[0], t_range[1], limit=200)
            numeric = round(val, 8)
        except Exception:
            numeric = None
        return {"symbolic": sym_str, "numeric": numeric,
                "integrand": str(integrand_s).replace("**", "^")}

    @staticmethod
    def _parse_vector(F_str: str, symbols):
        """解析向量场 [Fx, Fy, Fz]"""
        loc = {str(v): v for v in symbols}
        loc.update({
               "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
               "log": sp.log, "exp": sp.exp, "sqrt": sp.sqrt, "abs": sp.Abs,
               "pi": sp.pi, "e": sp.E})
        # F_str 格式: "[Fx, Fy, Fz]" 或 "Fx, Fy, Fz"
        s = F_str.strip().strip("[]()")
        parts = [p.strip() for p in s.split(",")]
        Fx = sp.sympify(_preprocess_expr(parts[0]), locals=loc) if len(parts) > 0 else 0
        Fy = sp.sympify(_preprocess_expr(parts[1]), locals=loc) if len(parts) > 1 else 0
        Fz = sp.sympify(_preprocess_expr(parts[2]), locals=loc) if len(parts) > 2 else 0
        return Fx, Fy, Fz
