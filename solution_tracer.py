#!/usr/bin/env python3
"""
================================================================================
  解题过程追踪器 — Solution Tracer
================================================================================
  记录每一步计算、所用定理/公式，输出教科书式解题过程。
  覆盖: 极限 · 积分 · 泰勒 · 傅里叶 · ODE · 函数分析 · 级数 · 向量
================================================================================
"""

from __future__ import annotations
from dataclasses import dataclass, field
import re
from typing import Callable


@dataclass
class Step:
    """一个解题步骤"""
    num: int
    description: str
    formula: str = ""
    theorem: str = ""
    result: str = ""
    indent: int = 0


class SolutionTracer:
    """解题过程追踪器 — 链式调用, 教科书风格"""

    def __init__(self, title: str = "", category: str = ""):
        self.title = title
        self.category = category  # limit/integral/series/fourier/ode/multivariable/vector/laplace/derivative
        self.steps: list[Step] = []
        self._num = 0
        self._indent = 0

    def step(self, description: str, formula: str = "",
             theorem: str = "", result: str = "") -> "SolutionTracer":
        self._num += 1
        self.steps.append(Step(num=self._num, description=description,
                                formula=formula, theorem=theorem, result=result,
                                indent=self._indent))
        return self

    def indent(self) -> "SolutionTracer":
        self._indent += 1; return self

    def dedent(self) -> "SolutionTracer":
        if self._indent > 0: self._indent -= 1
        return self

    def sub(self, description: str, formula: str = "",
            theorem: str = "", result: str = "") -> "SolutionTracer":
        self.indent()
        self.step(description, formula, theorem, result)
        self.dedent()
        return self

    def method(self, name: str) -> "SolutionTracer":
        """标记一个新解法路径"""
        self._num += 1
        self.steps.append(Step(num=0, description=f"◆ 解法: {name}",
                                formula="", theorem="", result="", indent=0))
        return self

    def fallback(self, description: str, detail: str = "",
                 method: str = "") -> "SolutionTracer":
        """智能回退: 当预设分支未命中时, 至少解释 sympy 做了什么"""
        if method:
            self.method(method)
        self.step(description, detail if detail else "",
                   "SymPy 符号引擎 (Risch算法 / 模式匹配 / 查表法)")
        return self

    @staticmethod
    def _natural_desc(s: Step) -> str:
        """将结构化步骤转为人类解题书写风格"""
        d = s.description
        thm = f"  [{s.theorem}]" if s.theorem else ""
        patterns = [
            (r"^直接代入$", f"将 x 值直接代入函数表达式{thm}"),
            (r"^求原函数", f"求被积函数的原函数{thm}"),
            (r"^应用(牛顿|莱布|N-L)", f"由牛顿-莱布尼茨公式{thm}"),
            (r"^代入上下限", f"将上下限代入原函数{thm}"),
            (r"^数值验证", f"数值方法交叉验证{thm}"),
            (r"^求零点", f"解方程 f(x)=0 求函数零点{thm}"),
            (r"^求一阶导数", f"对 f(x) 求一阶导数{thm}"),
            (r"^求二阶导数", f"对 f'(x) 再求导得二阶导数{thm}"),
            (r"^求临界点", f"令 f'(x)=0 解临界点{thm}"),
            (r"^求拐点", f"令 f''(x)=0 且检查凹凸性变化, 求拐点{thm}"),
            (r"^单调性分析", f"由 f'(x) 符号判断单调区间{thm}"),
            (r"^凹凸性", f"由 f''(x) 符号判断凹凸区间{thm}"),
            (r"^渐近线分析", f"用极限方法求渐近线{thm}"),
            (r"^对称性分析", f"比较 f(-x) 与 ±f(x) 判断奇偶性{thm}"),
            (r"^求 (极值|极值点)", f"由一阶导=0 和二阶导符号求极值点{thm}"),
            (r"^审题", f"分析题目条件, 确定解题策略{thm}"),
            (r"^问题分类", f"判断方程类型{thm}"),
            (r"^识别", f"观察被积函数特征{thm}"),
            (r"^泰勒公式", f"写出泰勒展开通式{thm}"),
            (r"^泰勒定理", f"由泰勒定理, 函数可展开为幂级数{thm}"),
            (r"^泰勒多项式", f"整理得泰勒多项式{thm}"),
            (r"^系数", f"逐项计算展开系数{thm}"),
            (r"^傅里叶级数公式", f"写出傅里叶级数通式{thm}"),
            (r"^求偏导数", f"分别对 x, y 求偏导数{thm}"),
            (r"^代入点坐标", f"将点坐标代入偏导表达式{thm}"),
            (r"^散度定义", f"由散度定义 div F = ∇·F{thm}"),
            (r"^旋度定义", f"由旋度定义 curl F = ∇×F{thm}"),
            (r"^拉普拉斯变换定义", f"由拉普拉斯变换定义{thm}"),
            (r"^查表速算", f"利用拉普拉斯变换表, 快速匹配{thm}"),
            (r"^变换结果", f"查表/积分计算得{thm}"),
            (r"^定义域分析", f"考虑分母≠0, 根号内≥0, 对数真数>0, 确定定义域{thm}"),
            (r"^微分学分析", f"进入微分学分析阶段, 求各阶导数{thm}"),
            (r"^分析完成", f"综合以上分析, 函数性态全面刻画完毕{thm}"),
            (r"^积分策略分析", f"分析被积函数结构, 选择积分策略{thm}"),
            (r"^降幂法", f"── 降幂法 (Power Reduction) ──"),
            (r"^识别高次三角函数", f"检测到 sin/cos 的高次幂, 需先降幂再积分{thm}"),
            (r"^积分结果", f"积分结果{thm}"),
            (r"^数值求解", f"采用 RK45 自适应步长法数值求解{thm}"),
            (r"^求解结果", f"数值积分结果{thm}"),
        ]
        for pat, repl in patterns:
            if re.search(pat, d):
                return repl
        if s.indent > 0:
            return f"  {d}{thm}"
        return f"▸ {d}{thm}"

    @staticmethod
    def _fmt(f: str) -> str:
        if not f: return ""
        return (f.replace("**", "^").replace("*", "·")
                 .replace("pi", "π").replace("inf", "∞")
                 .replace("oo", "∞").replace("->", "→"))

    @staticmethod
    def _fmt_val(v) -> str:
        """智能识别 π/e 等常数: 先用 sympy.nsimplify 精确匹配, 再用分数逼近"""
        import numpy as _np, math
        if v is None:
            return "—"
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return str(v)
        if not _np.isfinite(fv):
            return str(v)
        if abs(fv) < 1e-12:
            return "0"
        rv = round(fv)
        if abs(fv - rv) < 1e-13:
            return str(rv)
        # 第1步: sympy.nsimplify 精确匹配 (仅接受简短表达式)
        try:
            import sympy as _sp
            expr = _sp.nsimplify(fv, [_sp.pi, _sp.E, _sp.log(2), _sp.sqrt(_sp.pi)],
                                 tolerance=1e-8, full=True)
            s = str(expr).replace("**", "^").replace("*", "")
            has_const = any(c in s for c in ['pi', 'PI', 'E', 'log'])
            # 纯数字结果仅当极精确才采纳 (nsimplify 可能把 2.000003→2)
            if not has_const and s.replace('.','').replace('-','').isdigit():
                try:
                    if abs(float(s) - fv) > 1e-12:
                        s = None  # 拒绝不精确的整数简化
                except Exception:
                    pass
            if s and s != str(fv) and len(s) < 30:
                return f"{s} (≈ {fv:.6f})"
        except Exception:
            pass
        # 第2步: 手动 π 分数逼近 (分母≤12, 容差 5e-5)
        best_n, best_d, best_err = 0, 1, 5e-5
        for d in range(1, 13):
            n = round(fv / math.pi * d)
            err = abs(n * math.pi / d - fv)
            if err < best_err:
                best_err, best_n, best_d = err, n, d
        if best_n != 0 and best_d <= 12:
            g = math.gcd(abs(best_n), best_d)
            best_n //= g; best_d //= g
            if best_d == 1:
                s = "π" if best_n == 1 else ("-π" if best_n == -1 else f"{best_n}π")
            else:
                s = f"π/{best_d}" if best_n == 1 else (f"-π/{best_d}" if best_n == -1 else f"{best_n}π/{best_d}")
            return f"{s} (≈ {fv:.6f})"
        return f"{fv:.8g}"

    def render(self, related_theorems: list[str] = None) -> str:
        lines = []
        if self.title:
            lines.append(f"═══ {self.title} ═══")
            lines.append("")
        for s in self.steps:
            prefix = "  " * s.indent
            if s.num == 0:
                # 解法标记 ─ 分隔线
                lines.append("")
                lines.append(f"  {s.description}")
                lines.append("  " + "─" * 50)
                continue
            # 自然语言风格渲染
            desc = self._natural_desc(s)
            lines.append(f"{prefix}{desc}")
            if s.formula:
                lines.append(f"{prefix}    {self._fmt(s.formula)}")
            if s.result:
                formatted = self._fmt(s.result)
                import re as _re
                nums = _re.findall(r'[\d]+\.[\d]+', formatted)
                for num in nums:
                    try:
                        fv = float(num)
                        pi_str = self._fmt_val(fv)
                        if 'π' in pi_str or 'π' in pi_str:
                            formatted = formatted.replace(num, pi_str)
                    except Exception:
                        pass
                lines.append(f"{prefix}    ∴ {formatted}")
            lines.append("")
        # 自动附加相关定理
        if related_theorems is None and self.category:
            related_theorems = suggest_theorems(self.category)
        if related_theorems:
            lines.append("── 相关定理与公式参考 ──")
            for thm in related_theorems[:25]:
                desc = THEOREMS.get(thm, thm)
                lines.append(f"  · {thm}: {desc[:130]}")
            lines.append("")
        return "\n".join(lines)


# ============================================================================
# 全能定理 & 公式库 — 高等数学全覆盖
# ============================================================================

THEOREMS = {
    # ═══ 极限与连续 ═══
    "ε-δ定义": "极限 ε-δ 定义: ∀ε>0, ∃δ>0, 0<|x-a|<δ ⇒ |f(x)-L|<ε",
    "极限四则": "极限四则运算法则: lim(f±g)=lim f±lim g, lim(fg)=lim f·lim g, lim(f/g)=lim f/lim g (g≠0)",
    "代入法": "代入法 (直接代入): 若 f 在 a 处连续, 则 lim_{x→a} f(x) = f(a)",
    "洛必达": "洛必达法则 (L'Hopital): 0/0 或 ∞/∞ 不定型时, lim f/g = lim f'/g' (可反复使用)",
    "夹逼": "夹逼定理 (Squeeze Theorem): g(x)≤f(x)≤h(x) 且 lim g=lim h=L ⇒ lim f=L",
    "单调有界": "单调有界定理: 单调递增有上界数列必收敛; 单调递减有下界数列必收敛",
    "柯西收敛": "柯西收敛准则: 数列{a_n}收敛 ⇔ ∀ε>0, ∃N, m,n>N ⇒ |a_m-a_n|<ε",
    "两个重要极限": "两个重要极限: ① lim_{x→0} sinx/x = 1  ② lim_{x→∞} (1+1/x)^x = e = lim_{x→0} (1+x)^{1/x}",
    "等价无穷小": "等价无穷小代换 (x→0): sinx~x, tanx~x, arcsinx~x, arctanx~x, e^x-1~x, ln(1+x)~x, 1-cosx~x²/2, (1+x)^α-1~αx",
    "无穷小阶": "无穷小阶的比较: 高阶/低阶/同阶/等价无穷小, k阶无穷小: f(x)/(x-a)^k → 非零常数",
    "连续性": "连续函数定义: lim_{x→a} f(x) = f(a); 间断点分类: 可去/跳跃/无穷/振荡",
    "闭区间连续": "闭区间上连续函数的性质: 有界性定理, 最值定理, 介值定理 (IVT), 零点定理, 一致连续性 (Cantor定理)",
    "零点定理": "零点定理 (Bolzano): f∈C[a,b], f(a)f(b)<0 ⇒ ∃ξ∈[a,b], f(ξ)=0",

    # ═══ 微分学 ═══
    "导数定义": "导数定义: f'(x₀) = lim_{h→0} [f(x₀+h)-f(x₀)]/h = lim_{x→x₀} [f(x)-f(x₀)]/(x-x₀)",
    "求导法则": "求导四则法则: (u±v)'=u'±v', (uv)'=u'v+uv', (u/v)'=(u'v-uv')/v²",
    "反函数求导": "反函数求导法则: [f^{-1}(y)]' = 1/f'(x), 其中 y=f(x)",
    "链式法则": "链式法则 (Chain Rule): dy/dx = dy/du · du/dx, [f(g(x))]' = f'(g(x))·g'(x)",
    "隐函数求导": "隐函数求导: 对方程 F(x,y)=0 两边对 x 求导, 解得 dy/dx = -F_x/F_y",
    "参数求导": "参数方程求导: x=φ(t), y=ψ(t) ⇒ dy/dx = ψ'(t)/φ'(t), d²y/dx² = [ψ''φ'-ψ'φ'']/(φ')³",
    "高阶导数": "高阶导数: f^(n)(x) = d^n f/dx^n; 莱布尼茨公式: (uv)^(n) = Σ C(n,k) u^(n-k) v^(k)",
    "费马": "费马定理 (Fermat): 可导极值点处 f'(x₀)=0",
    "罗尔": "罗尔定理 (Rolle): f∈C[a,b]∩D(a,b), f(a)=f(b) ⇒ ∃ξ∈(a,b), f'(ξ)=0",
    "拉格朗日中值": "拉格朗日中值定理 (LMVT): f∈C[a,b]∩D(a,b) ⇒ ∃ξ∈(a,b), f(b)-f(a)=f'(ξ)(b-a)",
    "柯西中值": "柯西中值定理 (CMVT): f,g∈C[a,b]∩D(a,b) ⇒ ∃ξ, [f(b)-f(a)]g'(ξ)=[g(b)-g(a)]f'(ξ)",
    "单调性定理": "单调性判定: f'(x)>0→严格递增, f'(x)<0→严格递减, f'(x)=0→常数",
    "一阶导检验": "一阶导数极值检验: f' 由正变负→极大; f' 由负变正→极小; f' 不变号→无极值",
    "二阶导检验": "二阶导数极值检验: f'(x₀)=0, f''(x₀)<0→极大; f''(x₀)>0→极小; f''(x₀)=0→待定(用高阶)",
    "凹凸性": "凹凸性: f''>0→下凸(∪/Convex), f''<0→上凸(∩/Concave), 拐点: f'' 变号处",
    "拐点判定": "拐点判定: f''(x₀)=0 或不存在, 且 f'' 在 x₀ 两侧异号; 或 f'''(x₀)≠0 当 f''(x₀)=0",
    "渐近线": "渐近线: 垂直(分母→0,|f|→∞); 水平(lim_{±∞}f=L); 斜(lim f/x=m≠0, lim(f-mx)=b)",
    "弧微分": "弧微分公式: ds = √(1+y'²)dx = √(x'²+y'²)dt (参数); ds = √(r²+r'²)dθ (极坐标)",
    "曲率": "曲率: κ = |y''|/(1+y'²)^{3/2}; 曲率半径: R = 1/κ; 曲率圆(密切圆)中心: (x-y'(1+y'²)/y'', y+(1+y'²)/y'')",
    "泰勒": "泰勒定理: f(x)=Σ_{k=0}^n f^(k)(x₀)/k! · (x-x₀)^k + R_n(x); R_n 可为拉格朗日余项或佩亚诺余项",
    "拉格朗日余项": "拉格朗日余项: R_n(x) = f^(n+1)(ξ)/(n+1)! · (x-x₀)^{n+1}, ξ介于x₀与x之间",
    "佩亚诺余项": "佩亚诺余项: R_n(x) = o((x-x₀)^n), 即 R_n/(x-x₀)^n → 0 (x→x₀)",
    "麦克劳林": "麦克劳林展开 (x₀=0的特例): e^x, sinx, cosx, ln(1+x), (1+x)^α, 1/(1±x), arctanx 等",

    # ═══ 积分学 ═══
    "不定积分定义": "不定积分: ∫f(x)dx = F(x) + C, 其中 F'(x)=f(x); 是求导的逆运算",
    "基本积分表": "基本积分表: ∫xⁿ=x^{n+1}/(n+1); ∫1/x=ln|x|; ∫eˣ=eˣ; ∫sin=-cos; ∫cos=sin; ∫sec²=tan; ∫csc²=-cot; ∫1/(1+x²)=arctan; ∫1/√(1-x²)=arcsin",
    "第一类换元": "第一类换元法(凑微分): ∫f(g(x))g'(x)dx = ∫f(u)du, 令 u=g(x)",
    "第二类换元": "第二类换元法: ∫f(x)dx {x=φ(t)} = ∫f(φ(t))φ'(t)dt, 常用: 三角代换/倒代换/根式代换",
    "三角代换": "三角代换: √(a²-x²)→x=a·sint; √(a²+x²)→x=a·tant; √(x²-a²)→x=a·sect",
    "分部积分": "分部积分法: ∫u dv = uv - ∫v du; 口诀: 反对幂指三(U优先次序)",
    "有理函数积分": "有理函数积分: 分解为部分分式之和, 结合待定系数法",
    "定积分定义": "定积分(黎曼和): ∫[a,b] f = lim_{n→∞} Σ_{i=1}^n f(ξ_i)Δx_i, 与分割和取法无关",
    "定积分性质": "定积分性质: 线性性/区间可加性/保序性/估值定理/积分中值定理",
    "积分中值": "积分中值定理: f∈C[a,b] ⇒ ∃ξ∈[a,b], ∫[a,b] f = f(ξ)(b-a); 推广: ∫fg = f(c)∫g",
    "变上限积分": "变上限积分: Φ(x) = ∫[a,x] f(t)dt ⇒ Φ'(x) = f(x); d/dx ∫[a,g(x)] f = f(g(x))g'(x)",
    "微积分基本定理": "微积分基本定理 (FTC): (1) 变上限积分求导; (2) ∫[a,b] f = F(b)-F(a) (牛顿-莱布尼茨)",
    "牛顿莱布尼茨": "牛顿-莱布尼茨公式: ∫[a,b] f(x)dx = F(b)-F(a), F'=f",
    "反常积分": "反常积分: (1)无穷限: lim_{R→∞}∫[a,R]; (2)瑕积分: lim_{ε→0}∫[a,b-ε]; 柯西主值 PV",
    "反常积分审敛": "反常积分审敛法: 比较判别法/极限比较法; p-积分: ∫[1,∞]1/x^p dx, p>1收敛; ∫[0,1]1/x^p dx, p<1收敛",
    "Γ函数": "Γ函数: Γ(s)=∫₀^∞ t^{s-1}e^{-t}dt; Γ(n+1)=n!; Γ(1/2)=√π",
    "B函数": "B函数: B(p,q)=∫₀¹ t^{p-1}(1-t)^{q-1}dt = Γ(p)Γ(q)/Γ(p+q)",
    "定积分应用": "定积分应用: 面积/体积(旋转体/截面)/弧长/旋转曲面面积/质心/转动惯量/变力作功/液体压力",
    "旋转体体积": "旋转体体积: 绕x轴 V=π∫y²dx; 绕y轴 V=2π∫xydx (柱壳法)",
    "弧长公式": "弧长: y=f(x)→L=∫√(1+y'²)dx; 参数→L=∫√(x'²+y'²)dt; 极坐标→L=∫√(r²+r'²)dθ",

    # ═══ 多元函数微分学 ═══
    "偏导数": "偏导数: ∂f/∂x = lim_{Δx→0} [f(x+Δx,y)-f(x,y)]/Δx; 几何意义: 曲面与坐标平面交线的切线斜率",
    "高阶偏导": "高阶偏导数: ∂²f/∂x², ∂²f/∂x∂y, ∂²f/∂y²; 混合偏导连续则相等 (Clairaut定理)",
    "全微分": "全微分: dz = (∂f/∂x)dx + (∂f/∂y)dy; 可微 ⇒ 连续+偏导存在; 偏导连续 ⇒ 可微",
    "多元链式": "多元链式法则: z=f(u,v), u=u(x,y), v=v(x,y) ⇒ ∂z/∂x = ∂z/∂u·∂u/∂x + ∂z/∂v·∂v/∂x",
    "隐函数定理": "隐函数定理: F(x,y)=0且F_y≠0 ⇒ y=y(x)可导, dy/dx=-F_x/F_y",
    "雅可比": "雅可比行列式: J = ∂(u,v)/∂(x,y); 隐函数组求导; 重积分换元 dxdy = |J| dudv",
    "方向导数": "方向导数: D_u f = ∇f · u, 其中 u 为单位方向向量",
    "梯度定义": "梯度: ∇f = [∂f/∂x₁, ..., ∂f/∂x_n]; 方向导数最大值 = |∇f|; 梯度⊥等值线(面)",
    "多元极值": "多元函数极值: 驻点(∇f=0)→Hessian判定: 正定→极小; 负定→极大; 不定→鞍点; 半定→待定",
    "拉格朗日乘数": "拉格朗日乘数法: 约束 g=0 下求 f 极值, ∇f = λ∇g; 解方程组得驻点",
    "条件极值": "条件极值(Karush-Kuhn-Tucker): 多个约束 g_i=0 时, ∇f = Σ λ_i ∇g_i",
    "二元泰勒": "二元泰勒公式: f≈f(P₀)+∇f·Δr + ½Δrᵀ·H·Δr + ...",

    # ═══ 多元函数积分学 ═══
    "二重积分": "二重积分 ∬_D f(x,y)dσ: 化二次积分; 先x后y或先y后x; 选择积分次序",
    "极坐标二重": "二重积分极坐标换元: x=rcosθ, y=rsinθ, dσ = r dr dθ; ∬ f = ∬ f(rcosθ,rsinθ)·r drdθ",
    "三重积分": "三重积分 ∭_Ω f dV: 化三次积分; 投影法/截面法",
    "柱坐标": "柱坐标: x=rcosθ, y=rsinθ, z=z, dV = r dr dθ dz (适用于圆柱/旋转体)",
    "球坐标": "球坐标: x=ρsinφcosθ, y=ρsinφsinθ, z=ρcosφ, dV = ρ²sinφ dρdφdθ (适用于球/锥)",
    "曲线积分1": "第一类曲线积分(对弧长): ∫_L f(x,y)ds = ∫ f(x(t),y(t))·√(x'²+y'²) dt",
    "曲线积分2": "第二类曲线积分(对坐标): ∫_L Pdx+Qdy = ∫ [Px'(t)+Qy'(t)] dt",
    "曲面积分1": "第一类曲面积分(对面积): ∬_Σ f dS = ∬ f(x,y,z(x,y))·√(1+z_x²+z_y²) dxdy",
    "曲面积分2": "第二类曲面积分(对坐标): ∬_Σ Pdydz+Qdzdx+Rdxdy = ∬ (Pcosα+Qcosβ+Rcosγ) dS",
    "格林": "格林公式: ∮_C Pdx+Qdy = ∬_D (∂Q/∂x - ∂P/∂y) dxdy; C为D的正向边界",
    "曲线积分与路径": "曲线积分与路径无关 ⇔ ∂P/∂y = ∂Q/∂x (单连通) ⇔ 存在势函数 u, du=Pdx+Qdy",
    "高斯": "高斯公式/散度定理: ∬_Σ F·dS = ∭_Ω div(F) dV; 闭曲面通量 = 散度的体积积分",
    "斯托克斯": "斯托克斯公式: ∮_C F·dr = ∬_Σ (curl F)·dS; 环量 = 旋度的面积分",
    "散度定义": "散度: div F = ∇·F = ∂P/∂x + ∂Q/∂y + ∂R/∂z; >0源, <0汇, =0无源",
    "旋度定义": "旋度: curl F = ∇×F = (∂R/∂y-∂Q/∂z, ∂P/∂z-∂R/∂x, ∂Q/∂x-∂P/∂y); 无旋场⇔保守场",

    # ═══ 无穷级数 ═══
    "级数定义": "级数: Σa_n 收敛 ⇔ 部分和数列 S_n = Σ_{k=1}^n a_k 收敛; 必要条件: lim a_n = 0 (反之不真!)",
    "几何级数": "几何级数: Σ ar^{n-1}, |r|<1收敛和为 a/(1-r); |r|≥1发散 (a≠0)",
    "p级数": "p-级数: Σ 1/n^p, p>1收敛, p≤1发散; 调和级数 Σ1/n 发散",
    "比较检验": "比较检验: 0≤a_n≤b_n, Σb_n⇒Σa_n; Σa_n发散⇒Σb_n发散; 极限比较: lim a_n/b_n = c∈(0,∞) 同敛散",
    "比值检验": "比值检验 (d'Alembert): lim|a_{n+1}/a_n| = ρ; ρ<1收敛, ρ>1发散, ρ=1失效",
    "根值检验": "根值检验 (Cauchy): lim|a_n|^{1/n} = ρ; ρ<1收敛, ρ>1发散, ρ=1失效",
    "积分检验": "积分检验 (Cauchy): f(x)≥0递减, Σf(n) 与 ∫₁^∞ f(x)dx 同敛散",
    "交错级数": "莱布尼茨交错级数: a_n≥0递减且→0 ⇒ Σ(-1)^n a_n 收敛; 余项 |R_n| ≤ a_{n+1}",
    "绝对收敛": "绝对收敛: Σ|a_n|收敛 ⇒ Σa_n收敛; 条件收敛=收敛但不绝对收敛",
    "幂级数": "幂级数 Σ a_n(x-x₀)^n: 收敛半径 R=1/lim|a_{n+1}/a_n|=1/lim|a_n|^{1/n}; 阿贝尔定理",
    "阿贝尔": "阿贝尔定理: 幂级数在|x-x₀|<R内收敛; >R发散; =R需单独判定; 在收敛区间内可逐项积分/求导",
    "泰勒级数": "泰勒级数: f(x) = Σ f^(n)(x₀)/n! (x-x₀)^n; 收敛于f(x)的充要条件: R_n→0",
    "常见泰勒": "常见麦克劳林级数: e^x, sinx, cosx, ln(1+x), (1+x)^α, 1/(1-x)=Σx^n, arctanx 等及其收敛域",
    "傅里叶级数": "傅里叶级数 (周期2L): f(x) = a₀/2 + Σ [a_n·cos(nπx/L) + b_n·sin(nπx/L)]",
    "欧拉傅里叶": "欧拉-傅里叶系数公式: a_n = 1/L ∫_{-L}^L f(x)cos(nπx/L)dx; b_n = 1/L ∫_{-L}^L f(x)sin(nπx/L)dx",
    "狄利克雷": "狄利克雷收敛定理: f分段光滑 ⇒ 傅里叶级数收敛于 [f(x+0)+f(x-0)]/2; 间断点有Gibbs现象",
    "奇偶延拓": "奇延拓(正弦级数): 奇函数→a_n=0; 偶延拓(余弦级数): 偶函数→b_n=0",

    # ═══ 微分方程 ═══
    "ODE定义": "常微分方程 (ODE): 含一元未知函数及其导数的方程; 阶=最高阶导数; 解=满足方程的函数",
    "通解特解": "通解: 含任意常数个数=阶数的解; 特解: 满足初始条件的解; 奇解: 不在通解族中的解",
    "分离变量": "分离变量法: dy/dx = f(x)g(y) ⇒ ∫ dy/g(y) = ∫ f(x)dx (g(y)≠0)",
    "齐次方程": "齐次方程: dy/dx = φ(y/x); 令 u=y/x ⇒ du/dx = (φ(u)-u)/x 化为可分离变量",
    "一阶线性": "一阶线性 ODE: y' + P(x)y = Q(x); 通解 y = e^{-∫Pdx} [∫ Q e^{∫Pdx} dx + C]",
    "积分因子": "积分因子法: μ(x) = exp(∫ P(x)dx); μy' + μPy = μQ ⇒ (μy)' = μQ",
    "伯努利": "伯努利方程: y' + P(x)y = Q(x)y^n (n≠0,1); 令 z = y^{1-n} 化为一阶线性",
    "全微分方程": "全微分(恰当)方程: Pdx+Qdy=0, ∂P/∂y=∂Q/∂x; 解法: 求势函数 u, du=Pdx+Qdy=0 ⇒ u=C",
    "可降阶": "可降阶高阶ODE: (1) y^(n)=f(x)直接积分; (2) 缺y型: 令p=y'; (3) 缺x型: 令p=y', y''=p·dp/dy",
    "二阶常系数齐次": "二阶常系数线性齐次: y''+py'+qy=0; 特征方程 r²+pr+q=0; Δ>0→y=C₁e^{r₁x}+C₂e^{r₂x}; Δ=0→y=(C₁+C₂x)e^{rx}; Δ<0→y=e^{αx}(C₁cosβx+C₂sinβx)",
    "二阶常系数非齐次": "二阶常系数非齐次 y''+py'+qy=f(x): 通解 = 齐次通解 + 特解; 特解用待定系数法或常数变易法",
    "待定系数法": "待定系数法: 根据 f(x) 形式设特解: P_n(x)→Q_n(x); e^{kx}→Ae^{kx}; sin/cos→Asin+Bcos; 乘积组合",
    "欧拉方程": "欧拉(Euler)方程: x²y''+pxy'+qy=f(x); 令 x=e^t 化为常系数线性ODE",
    "常数变易": "常数变易法: 已知齐次解 y_h=C₁y₁+C₂y₂, 设特解 y_p=u₁y₁+u₂y₂, 解 u₁',u₂' 的线性方程组",
    "拉普拉斯变换法": "拉普拉斯变换解ODE: L{f(t)}=F(s); 将ODE转换为代数方程; 解出F(s)后求逆变换得f(t)",
    "RK45": "Runge-Kutta-Fehlberg 4(5): 变步长RK方法, 4阶推进+5阶误差估计; 局部截断误差O(h⁵)",
    "Picard": "Picard-Lindelof定理: y'=f(x,y), y(x₀)=y₀, 若f对y满足Lipschitz条件, 则存在唯一解",

    # ═══ 向量与空间解析几何 ═══
    "向量代数": "向量运算: 加减(平行四边形); 数乘; 点积 a·b=|a||b|cosθ; 叉积 |a×b|=|a||b|sinθ, 方向右手定则",
    "混合积": "混合积: [a b c] = (a×b)·c = 平行六面体体积; 共面 ⇔ 混合积=0",
    "平面方程": "平面方程: 点法式 A(x-x₀)+B(y-y₀)+C(z-z₀)=0; 一般式 Ax+By+Cz+D=0; n=(A,B,C)",
    "直线方程": "空间直线: 点向式 (x-x₀)/m = (y-y₀)/n = (z-z₀)/p = t; 方向向量 s=(m,n,p)",
    "曲面方程": "常见曲面: 球面/椭球面/柱面/锥面/旋转曲面; 二次曲面: 椭圆抛物面/双曲抛物面/单叶双曲面等",
    "空间曲线": "空间曲线: 两曲面交线; 参数式 (x(t),y(t),z(t)); 切向量 s=(x',y',z')",
    "切平面": "曲面的切平面与法线: F(x,y,z)=0在P₀处: F_x(P₀)(x-x₀)+F_y(P₀)(y-y₀)+F_z(P₀)(z-z₀)=0",
    "方向余弦": "方向余弦: cosα=a/|s|, cosβ=b/|s|, cosγ=c/|s|; cos²α+cos²β+cos²γ=1",

    # ═══ 三角学公式 ═══
    "三角恒等式": "基本恒等式: sin²+cos²=1; 1+tan²=sec²; 1+cot²=csc²; tan=sin/cos; cot=cos/sin; sec=1/cos; csc=1/sin",
    "诱导公式": "诱导公式: sin(π±x)=∓sinx; cos(π±x)=-cosx; tan(π±x)=±tanx; sin(π/2±x)=cosx; cos(π/2±x)=∓sinx",
    "和角公式": "和角公式: sin(A±B)=sinAcosB±cosAsinB; cos(A±B)=cosAcosB∓sinAsinB; tan(A±B)=(tanA±tanB)/(1∓tanAtanB)",
    "倍角公式": "倍角: sin2x=2sinxcosx; cos2x=cos²x-sin²x=2cos²x-1=1-2sin²x; tan2x=2tanx/(1-tan²x); cot2x=(cot²x-1)/(2cotx)",
    "三倍角": "三倍角: sin3x=3sinx-4sin³x; cos3x=4cos³x-3cosx",
    "半角公式": "半角: sin²(x/2)=(1-cosx)/2; cos²(x/2)=(1+cosx)/2; tan(x/2)=(1-cosx)/sinx=sinx/(1+cosx)",
    "和差化积": "和差化积: sinA±sinB=2sin((A±B)/2)cos((A∓B)/2); cosA+cosB=2cos((A+B)/2)cos((A-B)/2); cosA-cosB=-2sin((A+B)/2)sin((A-B)/2)",
    "积化和差": "积化和差: sinAcosB=½[sin(A+B)+sin(A-B)]; cosAsinB=½[sin(A+B)-sin(A-B)]; cosAcosB=½[cos(A+B)+cos(A-B)]; sinAsinB=-½[cos(A+B)-cos(A-B)]",
    "万能公式": "万能代换: 令 t=tan(x/2), 则 sinx=2t/(1+t²), cosx=(1-t²)/(1+t²), tanx=2t/(1-t²), dx=2dt/(1+t²)",

    # ═══ 积分变换 ═══
    "拉普拉斯定义": "拉普拉斯变换: L{f(t)} = F(s) = ∫₀^∞ f(t)e^{-st}dt",
    "拉普拉斯表": "拉普拉斯变换表: 1→1/s; t→1/s²; tⁿ→n!/s^{n+1}; e^{at}→1/(s-a); sin(at)→a/(s²+a²); cos(at)→s/(s²+a²); sinh→a/(s²-a²); cosh→s/(s²-a²); tⁿe^{at}→n!/(s-a)^{n+1}; u(t-a)→e^{-as}/s; δ(t)→1",
    "拉普拉斯性质": "拉普拉斯性质: 线性; 时移 L{f(t-a)u(t-a)}=e^{-as}F(s); 频移 L{e^{at}f(t)}=F(s-a); 导数 L{f'}=sF-f(0); 积分 L{∫f}=F/s; 卷积 L{f*g}=F·G",
    "卷积定理": "卷积定理: L{f(t)*g(t)} = F(s)·G(s), 其中 f*g = ∫₀^t f(τ)g(t-τ)dτ",
    "傅里叶变换": "傅里叶变换: F{f(t)} = F(ω) = ∫_{-∞}^∞ f(t)e^{-iωt}dt; 逆变换 f(t) = 1/2π ∫ F(ω)e^{iωt}dω",

    # ═══ 特殊函数 ═══
    "Γ函数": "Gamma函数: Γ(z)=∫₀^∞ t^{z-1}e^{-t}dt; Γ(n+1)=n!; Γ(1/2)=√π; Γ(z+1)=zΓ(z)",
    "B函数": "Beta函数: B(p,q)=∫₀¹ t^{p-1}(1-t)^{q-1}dt = Γ(p)Γ(q)/Γ(p+q)",
    "误差函数": "误差函数: erf(x) = 2/√π ∫₀^x e^{-t²}dt; erfc(x)=1-erf(x); 高斯积分 ∫₀^∞ e^{-x²}dx=√π/2",
}

# ── 根据计算类型推荐相关定理 ──
def suggest_theorems(category: str) -> list[str]:
    by_cat = {
        "limit": ["ε-δ定义", "代入法", "洛必达", "夹逼", "两个重要极限", "等价无穷小", "单调有界", "柯西收敛", "连续性", "罗尔", "拉格朗日中值"],
        "derivative": ["导数定义", "求导法则", "反函数求导", "链式法则", "隐函数求导", "参数求导", "高阶导数",
                       "费马", "罗尔", "拉格朗日中值", "柯西中值", "一阶导检验", "二阶导检验",
                       "凹凸性", "拐点判定", "渐近线", "弧微分", "曲率", "泰勒", "麦克劳林"],
        "integral": ["不定积分定义", "基本积分表", "第一类换元", "第二类换元", "三角代换", "分部积分",
                     "有理函数积分", "定积分定义", "定积分性质", "积分中值", "变上限积分",
                     "微积分基本定理", "牛顿莱布尼茨", "反常积分", "反常积分审敛",
                     "旋转体体积", "弧长公式", "Γ函数", "B函数"],
        "series": ["级数定义", "几何级数", "p级数", "比较检验", "比值检验", "根值检验", "积分检验",
                   "交错级数", "绝对收敛", "幂级数", "阿贝尔", "泰勒级数", "常见泰勒"],
        "fourier": ["欧拉傅里叶", "狄利克雷", "奇偶延拓", "三角恒等式", "和差化积", "积化和差", "倍角公式"],
        "ode": ["ODE定义", "通解特解", "分离变量", "齐次方程", "一阶线性", "积分因子", "伯努利",
                "全微分方程", "可降阶", "二阶常系数齐次", "二阶常系数非齐次",
                "待定系数法", "欧拉方程", "常数变易", "拉普拉斯变换法", "RK45", "Picard"],
        "multivariable": ["偏导数", "高阶偏导", "全微分", "多元链式", "隐函数定理", "雅可比",
                          "方向导数", "梯度定义", "多元极值", "拉格朗日乘数", "条件极值", "二元泰勒"],
        "vector": ["格林", "高斯", "斯托克斯", "散度定义", "旋度定义", "曲线积分1", "曲线积分2",
                   "曲面积分1", "曲面积分2", "曲线积分与路径", "向量代数", "切平面"],
        "laplace": ["拉普拉斯定义", "拉普拉斯表", "拉普拉斯性质", "卷积定理", "傅里叶变换"],
        "trig": ["三角恒等式", "诱导公式", "和角公式", "倍角公式", "三倍角", "半角公式", "和差化积", "积化和差", "万能公式"],
        "polar": ["极坐标二重", "柱坐标", "球坐标", "弧长公式"],
    }
    return by_cat.get(category, [])

# ── 根据计算类型推荐相关定理 ──
def suggest_theorems(category: str) -> list[str]:
    """根据计算类型返回可能相关的定理名称列表"""
    by_cat = {
        "limit": ["代入法", "洛必达", "夹逼", "两个重要极限", "等价无穷小", "连续性", "罗尔", "拉格朗日中值"],
        "derivative": ["求导法则", "链式法则", "费马", "二阶导检验", "凹凸性", "罗尔", "拉格朗日中值", "柯西中值"],
        "integral": ["基本积分表", "分部积分", "换元积分", "牛顿莱布尼茨", "微积分基本定理"],
        "series": ["比值检验", "根值检验", "积分检验", "交错级数", "p级数", "比较检验", "泰勒", "麦克劳林"],
        "fourier": ["欧拉傅里叶", "狄利克雷", "三角恒等式", "和差化积", "积化和差"],
        "ode": ["分离变量", "积分因子", "RK45"],
        "multivariable": ["梯度定义", "方向导数", "拉格朗日乘数", "链式法则"],
        "vector": ["格林", "斯托克斯", "散度定理"],
        "laplace": ["拉普拉斯表", "拉普拉斯定义", "分部积分"],
        "trig": ["三角恒等式", "倍角公式", "半角公式", "和差化积", "积化和差"],
    }
    return by_cat.get(category, [])


# ============================================================================
# 带追踪的引擎包装
# ============================================================================

class TracedCalculus:
    """带解题追踪的微积分计算"""

    @staticmethod
    def limit(expr: str, a) -> tuple[dict, SolutionTracer]:
        from calculus_engine import LimitEngine
        import sympy as sp
        T = SolutionTracer(f"极限 lim(x→{a}) {expr}", "limit")

        x = sp.Symbol('x')
        from function_plotter import _preprocess_expr
        s = _preprocess_expr(expr)
        e = sp.sympify(s, locals={"x": x, "sin": sp.sin, "cos": sp.cos,
                                   "tan": sp.tan, "log": sp.log, "exp": sp.exp,
                                   "sqrt": sp.sqrt, "pi": sp.pi, "e": sp.E})

        # Step 1: 检查连续性→直接代入
        try:
            direct = e.subs(x, a)
            has_nan = direct.has(sp.nan) if hasattr(direct, 'has') else False
            is_inf = direct.is_infinite if hasattr(direct, 'is_infinite') else False
            if not has_nan and not is_inf:
                T.step("直接代入", f"f({a}) = {sp.simplify(direct)}",
                       THEOREMS["连续性"])
            else:
                raise ValueError("not continuous")
        except Exception:
            T.step("判别不定型", f"x={a} 代入得到不定型 (0/0 或 ∞/∞ 或 0·∞)",
                   "不定型: 0/0, ∞/∞, 0·∞, ∞-∞, 1^∞, 0^0, ∞^0")

            # 三角函数检查
            if e.has(sp.sin) or e.has(sp.cos) or e.has(sp.tan):
                # sin(ax)/x 型 → 重要极限
                try:
                    if e.has(sp.sin) and not e.has(sp.cos) and e.has(x):
                        T.step("识别重要极限形式",
                               f"含三角函数, 可尝试 lim sin(Ax)/Ax = 1",
                               THEOREMS["两个重要极限"])
                        T.sub("等价无穷小代换", "x→0时: sinx~x, tanx~x, 1-cosx~x²/2",
                              THEOREMS["等价无穷小"])
                except Exception:
                    pass

            # L'Hopital attempt
            try:
                num = sp.numer(sp.together(e)) if not e.is_Pow else e
                den = sp.denom(sp.together(e))
                n0 = sp.limit(num, x, a); d0 = sp.limit(den, x, a)
                if (n0 == 0 and d0 == 0) or (n0.is_infinite and d0.is_infinite):
                    T.step("应用洛必达法则",
                           f"分子分母同趋于{n0}, 对分子分母分别求导",
                           THEOREMS["洛必达"])
                    fp = sp.diff(num, x); gp = sp.diff(den, x)
                    T.sub("分子求导", f"({num})' = {sp.simplify(fp)}",
                          THEOREMS["求导法则"])
                    T.sub("分母求导", f"({den})' = {sp.simplify(gp)}",
                          THEOREMS["求导法则"])
                    lim_ratio = sp.limit(fp/gp, x, a)
                    T.sub("再求极限", f"lim f'/g' = {sp.simplify(lim_ratio)}")
            except Exception:
                pass

        # Final
        r = LimitEngine.compute(expr, a)
        if r["exists"]:
            T.step("极限结果",
                   result=f"lim(x→{a}) {expr} = {r['value']:.8f}")
        else:
            T.step("极限不存在",
                   f"左极限: {r.get('left','?')}, 右极限: {r.get('right','?')}")
            T.sub("分析", "左右极限不相等 → 极限不存在",
                  "极限存在的充要条件: 左右极限相等")
        return r, T

    @staticmethod
    def definite_integral(expr: str, a, b) -> tuple[dict, SolutionTracer]:
        from calculus_engine import CalculusEngine
        import sympy as sp
        x = sp.Symbol('x')
        from function_plotter import _preprocess_expr
        s = _preprocess_expr(expr)
        e = sp.sympify(s, locals={"x": x, "sin": sp.sin, "cos": sp.cos,
                                   "tan": sp.tan, "log": sp.log, "exp": sp.exp,
                                   "sqrt": sp.sqrt, "pi": sp.pi, "e": sp.E})

        T = SolutionTracer(f"定积分 ∫[{a},{b}] {expr} dx", "integral")
        T.step("积分策略分析", f"被积函数: f(x) = {expr}",
               THEOREMS["基本积分表"])

        # ── 智能三角策略检测 ──
        has_sin = e.has(sp.sin); has_cos = e.has(sp.cos); has_tan = e.has(sp.tan)
        trig_str = str(e)
        # 检测 sin^n 或 cos^n 高次幂 → 降幂法
        has_sin_pow = has_sin and ('sin' in trig_str and ('**' in trig_str or '^' in trig_str or 'sin(x)^' in trig_str or 'sin(x)**' in trig_str))
        has_cos_pow = has_cos and ('cos' in trig_str and ('**' in trig_str or '^' in trig_str or 'cos(x)^' in trig_str or 'cos(x)**' in trig_str))
        if has_sin_pow or has_cos_pow or (has_sin and 'sin(x)^' in trig_str.replace(' ','')) or (has_cos and 'cos(x)^' in trig_str.replace(' ','')):
            T.method("降幂法 (Power Reduction)")
            T.step("识别高次三角函数",
                   "sin^n(x) 或 cos^n(x) (n≥2) → 用倍角公式降幂",
                   THEOREMS["倍角公式"])
            T.sub("sin²x 降幂", "sin²x = (1 - cos(2x))/2", THEOREMS["半角公式"])
            T.sub("cos²x 降幂", "cos²x = (1 + cos(2x))/2", THEOREMS["半角公式"])
            if has_sin and ('**4' in trig_str or '^4' in trig_str):
                T.sub("sin⁴x 降幂", "sin⁴x = (sin²x)² = [(1-cos2x)/2]² = 3/8 - cos2x/2 + cos4x/8")
            if has_cos and ('**4' in trig_str or '^4' in trig_str):
                T.sub("cos⁴x 降幂", "cos⁴x = (cos²x)² = [(1+cos2x)/2]² = 3/8 + cos2x/2 + cos4x/8")

        # 检测 sin·cos 乘积 → 积化和差 (和差形式不触发)
        if has_sin and has_cos:
            # 仅乘积 sin(x)*cos(x) 才提示积化和差; sin(x)+cos(x) 直接用线性拆分
            has_product = '*' in trig_str or 'sin(x)cos' in trig_str.replace(' ','') or 'cos(x)sin' in trig_str.replace(' ','')
            if has_product:
                T.sub("三角恒等式: 积化和差",
                      "sinA·cosB = ½[sin(A+B)+sin(A-B)]",
                      THEOREMS["积化和差"])
            else:
                T.sub("线性拆分", "sin(x) + cos(x) 可逐项积分, 无需三角恒等式",
                      THEOREMS["基本积分表"])

        # 检测 tan/cot → 三角恒等变换
        if has_tan:
            T.sub("正切积分策略", "tanx = sinx/cosx, 可用换元 u=cosx 或恒等式 tan²x=sec²x-1",
                  THEOREMS["三角恒等式"])

        F = sp.integrate(e, x)
        F_simp = sp.simplify(F)
        # 智能回退: 如果原函数含有特殊函数, 标注 sympy 使用了什么
        has_special = any(fn in str(F_simp) for fn in
                          ["erf", "gamma", "Si", "Ci", "Ei", "li", "polylog", "hyper", "meijerg"])
        theorem_tag = (THEOREMS["基本积分表"] + " + " + THEOREMS["第一类换元"])
        if has_special:
            theorem_tag = "SymPy Risch算法 + 特殊函数库 (erf/gamma/Si等)"
        T.step("求原函数 (不定积分)",
               f"∫ {expr} dx = {F_simp}".replace("**", "^"),
               theorem_tag)
        if has_special:
            T.sub("说明", "此积分不含初等原函数, SymPy 调用特殊函数库给出表达式",
                  "Liouville定理: 某些初等函数的积分不是初等函数")

        T.step("应用牛顿-莱布尼茨公式",
               f"∫[{a},{b}] f = F({b}) - F({a})",
               THEOREMS["牛顿莱布尼茨"])
        T.sub("理论基础", "微积分基本定理: 积分与微分互为逆运算",
              THEOREMS["微积分基本定理"])

        import numpy as _np
        try:
            Fa_val = float(sp.N(F_simp.subs(x, a)))
            Fb_val = float(sp.N(F_simp.subs(x, b)))
        except (TypeError, ValueError, AttributeError):
            Fa_val = Fb_val = 0  # 复杂表达式, 跳过符号计算
        if Fa_val is None or Fb_val is None:
            result_v = None
        else:
            result_v = Fb_val - Fa_val
        result_v = Fb_val - Fa_val
        # 分别处理: 整数或 π 边界用 sympy 化简, 普通浮点保留符号形式
        a_tag = str(a)
        b_tag = str(b)
        Fs = str(F_simp).replace("**", "^")
        def _is_exact(v):
            """判断是否精确值 (整数或接近 π 的符号常量)"""
            if v == int(v):
                return sp.Integer(int(v))
            if abs(v - _np.pi) < 1e-12:
                return sp.pi
            if abs(v - _np.pi/2) < 1e-12:
                return sp.pi/2
            return None
        def _sub(v, tag):
            exact = _is_exact(v)
            if exact is not None:
                return str(sp.simplify(F_simp.subs(x, exact))).replace("**", "^")
            return Fs.replace("x", tag)
        Fa_sym = _sub(a, a_tag)
        Fb_sym = _sub(b, b_tag)
        sym_expr = f"{Fb_sym} - ({Fa_sym})"
        sym_expr = sym_expr.replace("--", "+")
        sym_expr = sym_expr.replace("- (-1)", "+ 1").replace("- (- 1)", "+ 1")
        sym_expr = sym_expr.replace("- (0)", "").replace("+ (0)", "")
        sym_expr = sym_expr.replace("  ", " ").strip()
        # 纯数值表达式进一步化简: 1 + 1 → 2 等 (仅当原表达式无 trig/特殊函数时才简化)
        has_fn = any(c in sym_expr for c in ['cos', 'sin', 'tan', 'log', 'exp', 'sqrt'])
        if not has_fn:
            try:
                sym_expr = str(sp.simplify(sp.sympify(sym_expr))).replace("**", "^")
            except Exception:
                pass
        T.step("代入上下限并相减",
               f"F({b_tag}) - F({a_tag}) = {sym_expr}",
               result=f"≈ {result_v:.15g}" if result_v is not None else "—")

        r = CalculusEngine.integrate_definite(expr,
            float(a) if isinstance(a, (int, float)) else float(sp.N(a)),
            float(b) if isinstance(b, (int, float)) else float(sp.N(b)))
        if r.get("numeric") and result_v != "?":
            if abs(float(result_v) - r["numeric"]) < 0.001:
                T.step("数值验证通过",
                       f"数值积分 = {r['numeric']:.8f} (与符号值一致)", "scipy quad 自适应积分")
        return r, T

    @staticmethod
    def taylor(expr: str, x0: float, n: int) -> tuple[dict, SolutionTracer]:
        from calculus_engine import CalculusEngine
        import sympy as sp
        T = SolutionTracer(f"{expr} 在 x₀={x0} 的泰勒展开 (n={n})", "series")

        T.step("泰勒定理",
               "f(x) = Sum[k=0..n] f^(k)(x0)/k! * (x-x0)^k + R_n(x)",
               THEOREMS["泰勒"])
        if x0 == 0:
            T.sub("麦克劳林展开特例", "x₀=0, 即麦克劳林级数",
                  THEOREMS["麦克劳林"])

        x = sp.Symbol('x')
        from function_plotter import _preprocess_expr
        s = _preprocess_expr(expr)
        e = sp.sympify(s, locals={"x": x, "sin": sp.sin, "cos": sp.cos,
                                   "tan": sp.tan, "log": sp.log, "exp": sp.exp,
                                   "sqrt": sp.sqrt, "pi": sp.pi, "e": sp.E})

        for k in range(n + 1):
            fk = sp.diff(e, x, k)
            fk_at_x0 = sp.simplify(fk.subs(x, x0))
            coeff = float(fk_at_x0.evalf()) / float(sp.factorial(k))
            if abs(coeff) > 1e-14:
                T.sub(f"k={k}: 系数 c{k} = f^({k})({x0})/{k}!",
                      f"f^({k})({x0}) = {fk_at_x0}",
                      result=f"c{k} = {coeff:.6f}")

        r = CalculusEngine.taylor(expr, x0, n)
        T.step("泰勒多项式",
               f"T{n}(x) = {r['poly_str']}",
               result=f"余项: R_{n}(x) = f^({n+1})(ξ)/(n+1)! · (x-{x0})^{n+1} (拉格朗日余项)")
        T.sub("误差分析", f"当 |x-{x0}| 较小时, |R_{n}(x)| 很小 (O((x-{x0})^{n+1}))")
        return r, T

    @staticmethod
    def series_convergence(expr: str) -> tuple[dict, SolutionTracer]:
        from calculus_engine import SeriesAnalysis
        T = SolutionTracer(f"级数收敛判定: Σ a_n, a_n = {expr}", "series")

        T.step("审题", f"判定级数 Σ {expr} 的敛散性",
               "级数收敛定义: 部分和数列收敛")
        T.sub("必要条件", "若 Σa_n 收敛, 则 lim a_n = 0",
              "级数收敛的必要条件 (Term Test)")

        r = SeriesAnalysis.convergence_test(expr)
        for test in r.get("tests", []):
            icon = {"收敛": "✓", "发散": "✗", "条件收敛": "≈", "通过": "✓",
                    "不确定": "?"}.get(test["result"], "-")
            thm_name = test["name"].split("(")[0].strip()
            T.step(f"{icon} {test['name']}", test["detail"],
                   theorem=THEOREMS.get(thm_name, thm_name))

        T.step("结论", result=r["conclusion"])
        # 推荐相关定理
        sug = suggest_theorems("series")
        if sug:
            T.sub("相关定理参考", " · ".join(sug[:4]))
        return r, T

    @staticmethod
    def fourier(expr: str, period: float, n_terms: int) -> tuple[dict, SolutionTracer]:
        from calculus_engine import CalculusEngine
        import numpy as np
        T = SolutionTracer(f"{expr} 的傅里叶级数 (T={period:.4g}, n={n_terms})", "fourier")

        L = period / 2
        T.step("傅里叶级数公式",
               "f(x) = a₀/2 + Σ (a_n·cos(nπx/L) + b_n·sin(nπx/L))",
               THEOREMS["欧拉傅里叶"])
        T.sub("半周期", f"L = T/2 = {L:.4f}")

        # 三角相关提示
        T.sub("三角恒等式预备", "计算系数可能需要: 积化和差, 分部积分",
              THEOREMS["三角恒等式"])
        T.sub("收敛条件", "f(x) 分段连续+分段单调 → 傅里叶级数收敛",
              THEOREMS["狄利克雷"])

        r = CalculusEngine.fourier(expr, period=period, n_terms=n_terms)
        if r["a0"] != 0:
            T.step("常数项 a₀/2", result=f"= {r['a0']:.6f}")
        for n_i, v in r["an"]:
            T.step(f"余弦系数 a{n_i}", f"a{n_i} = 1/L ∫f·cos({n_i}πx/L)dx",
                   result=f"= {v:.6f}")
        for n_i, v in r["bn"]:
            T.step(f"正弦系数 b{n_i}", f"b{n_i} = 1/L ∫f·sin({n_i}πx/L)dx",
                   result=f"= {v:.6f}")

        T.step("截断逼近", f"F{n_terms}(x) = {r['approx_str']}",
               f"取前{n_terms}个谐波, 舍去高频项 (Gibbs现象可能出现在跳变点)")
        return r, T

    @staticmethod
    def ode_solve(expr: str, x0: float, y0: float, x_end: float) -> tuple[dict, SolutionTracer]:
        from calculus_engine import OdeEngine
        T = SolutionTracer(f"ODE 初值问题: y' = {expr}, y({x0}) = {y0}", "ode")

        T.step("问题分类", f"一阶常微分方程: dy/dx = {expr}",
               "一阶 ODE 初值问题 (IVP)")
        T.sub("初始条件", f"y({x0}) = {y0}",
              "初始条件唯一确定解曲线 (Picard-Lindelof定理)")

        # 判断类型
        if 'y' in expr and expr.replace('y', '').strip().replace('=', ''):
            T.sub("方程类型分析", "可尝试: 分离变量法 / 积分因子法 / 换元法",
                  THEOREMS["分离变量"] + " / " + THEOREMS["积分因子"])

        T.step("数值求解", f"从 x={x0} 到 x={x_end}",
               THEOREMS["RK45"])
        T.sub("方法细节", "RK45 = 4阶 Runge-Kutta + 5阶误差估计, 自适应步长")

        r = OdeEngine.solve_ivp(expr, x0, y0, x_end)
        T.step("求解结果",
               result=f"y({x_end}) ≈ {r['ys'][-1]:.6f}")
        T.sub("局部误差控制", "每步截断误差 O(h^5), 累积误差 O(h^4)",
              "RK45 误差分析")
        return r, T

    @staticmethod
    def gradient(expr: str, point: tuple) -> tuple[dict, SolutionTracer]:
        from calculus_engine import MultivariableEngine
        px, py = point[0], point[1]
        T = SolutionTracer(f"梯度计算: f(x,y) = {expr} 在 ({px},{py})", "multivariable")

        T.step("梯度定义",
               "∇f = [∂f/∂x, ∂f/∂y] (函数增长最快方向)",
               THEOREMS["梯度定义"])
        T.sub("几何意义", "梯度方向 = 等值线法线方向, |∇f| = 最大方向导数",
              THEOREMS["方向导数"])

        r = MultivariableEngine.gradient(expr, point)
        gx, gy = r["gradient"][0], r.get("gradient", ["", ""])[1]
        T.step("求偏导数",
               f"∂f/∂x = {gx}\n∂f/∂y = {gy}",
               THEOREMS["链式法则"] if "(" in expr else "偏导数定义")

        if "values" in r:
            T.step("代入点坐标",
                   f"∇f({px},{py}) = [{r['values'][0]:.4f}, {r['values'][1]:.4f}]",
                   result=f"|∇f| = {r['magnitude']:.4f}")

        return r, T

    @staticmethod
    def laplace(expr: str) -> tuple[dict, SolutionTracer]:
        from calculus_engine import LaplaceEngine
        T = SolutionTracer(f"拉普拉斯变换: L{{{expr}}}", "laplace")

        T.step("变换定义",
               "L{f(t)} = ∫₀^∞ f(t)·e^(-st) dt",
               THEOREMS["拉普拉斯定义"])
        T.sub("查表速算",
              "sin(at)→a/(s²+a²), cos(at)→s/(s²+a²), t^n→n!/s^{n+1}, e^{at}→1/(s-a)",
              THEOREMS["拉普拉斯表"])

        r = LaplaceEngine.transform(expr)
        T.step("变换结果", f"L{{{expr}}} = {r['result']}",
               result=f"收敛域: Re(s) > 0 (对因果信号)")

        return r, T

    @staticmethod
    def function_analysis(expr: str, result) -> tuple[dict, SolutionTracer]:
        """完整函数分析追踪"""
        import sympy as sp
        x = sp.Symbol('x')
        from function_plotter import _preprocess_expr
        s = _preprocess_expr(expr)
        e = sp.sympify(s, locals={"x": x, "sin": sp.sin, "cos": sp.cos,
                                   "tan": sp.tan, "log": sp.log, "exp": sp.exp,
                                   "sqrt": sp.sqrt, "pi": sp.pi, "e": sp.E})

        T = SolutionTracer(f"函数完整分析: f(x) = {expr}", "derivative")

        T.step("定义域分析", f"确定 f(x)={expr} 的自然定义域",
               "分母≠0, 偶次根号内≥0, 对数真数>0")
        if result.domain_info:
            T.sub("定义域", result=result.domain_info)

        if result.symmetry:
            tag = "偶函数 f(-x)=f(x), 关于y轴对称" if result.symmetry == "even" else "奇函数 f(-x)=-f(x), 关于原点对称"
            T.step("对称性", tag, "奇偶性判定: 比较 f(-x) 与 ±f(x)")

        if result.y_intercept is not None:
            T.step("y轴截距", f"令 x=0: f(0) = {result.y_intercept:.4f}",
                   "截距定义")

        if result.period_value:
            T.step("周期性", f"T = {result.period_value:.4f}",
                   "周期函数: f(x+T)=f(x)")

        T.step("微分学分析开始", "求 f'(x) 和 f''(x)",
               THEOREMS["求导法则"])
        fp = sp.diff(e, x)
        fpp = sp.diff(fp, x)
        T.sub("一阶导数", f"f'(x) = {sp.simplify(fp)}".replace("**", "^"))
        T.sub("二阶导数", f"f''(x) = {sp.simplify(fpp)}".replace("**", "^"))

        if result.zeros:
            T.step("求零点", f"解方程 f(x) = 0",
                   "牛顿法 / 二分法 + 符号求解")
            zs = sorted(set(round(abs(z), 6) for z in result.zeros))
            T.sub("零点", f"x = {', '.join(f'{z:.4f}' for z in zs[:6])}")

        if result.extrema:
            T.step("求极值点", "令 f'(x) = 0, 解临界点",
                   THEOREMS["费马"])
            for e_pt in result.extrema:
                tag = "极大" if e_pt.kind == "max" else "极小"
                T.sub(f"{tag}值", f"x≈{e_pt.x:.4f}, f={e_pt.y:.4f}",
                      THEOREMS["二阶导检验"] if e_pt.kind else "")

        if result.inflections:
            T.step("求拐点", "解 f''(x) = 0 且 f'' 变号",
                   THEOREMS["凹凸性"])
            T.sub("拐点", f"x = {', '.join(f'{v:.4f}' for v in result.inflections[:5])}")

        if result.monotonic:
            parts = []
            for iv in result.monotonic:
                s_e = ("-∞" if iv.start is None else f"{iv.start:.3f}")
                e_e = ("+∞" if iv.end is None else f"{iv.end:.3f}")
                parts.append(f"({s_e},{e_e}){iv.label}")
            T.step("单调性", " | ".join(parts[:5]),
                   THEOREMS["拉格朗日中值"] + ": f'(x)>0↗, f'(x)<0↘")

        if result.concavity:
            parts = []
            for iv in result.concavity:
                s_e = ("-∞" if iv.start is None else f"{iv.start:.3f}")
                e_e = ("+∞" if iv.end is None else f"{iv.end:.3f}")
                parts.append(f"({s_e},{e_e}){iv.label}")
            T.step("凹凸性", " | ".join(parts[:5]),
                   THEOREMS["凹凸性"])

        if result.asymptotes:
            T.step("垂直渐近线", f"x = {', '.join(f'{v:.3f}' for v in result.asymptotes)}",
                   "分母→0 或 |f|→∞")
        if result.h_asymptotes:
            T.step("水平渐近线", f"y = {', '.join(f'{v:.4f}' for v in result.h_asymptotes)}",
                   "lim_{x→±∞} f(x) = 有限值")
        if result.obliques:
            T.step("斜渐近线",
                   ", ".join(f"y={m:.3f}x+{b:.3f}" for m, b in result.obliques),
                   "lim f(x)/x=m, lim(f(x)-mx)=b")

        if result.range_info and result.range_info != "—":
            T.step("值域", result.range_info,
                   "结合极值/渐近线/端点综合确定")

        T.step("分析完成", f"综合运用: 微分学 + 零点定理 + 极限理论 + 凹凸性",
               "")  # no theorem tag for summary
        T.sub("涉及定理", " · ".join(suggest_theorems("derivative")[:5]))
        T.sub("涉及定理(积分)", " · ".join(suggest_theorems("integral")[:3]))
        T.sub("涉及定理(三角)", " · ".join(suggest_theorems("trig")[:3])
              if e.has(sp.sin) or e.has(sp.cos) or e.has(sp.tan) else "")

        return {}, T

    # ── 以下为补充 tracer: 覆盖所有计算类型 ──

    @staticmethod
    def indefinite_integral(expr: str) -> tuple[dict, SolutionTracer]:
        T = SolutionTracer(f"不定积分: ∫ {expr} dx", "integral")
        T.method("基本积分法")
        T.step("识别被积函数形式", f"f(x) = {expr}", THEOREMS["基本积分表"])
        from calculus_engine import CalculusEngine
        F_str, _ = CalculusEngine.integrate_indefinite(expr)
        T.step("积分结果", f"∫ {expr} dx = {F_str}".replace("**", "^"),
               THEOREMS["不定积分定义"])
        T.sub("验证", f"对结果求导应还原为 {expr}", "微分与积分互为逆运算")
        return {"result": F_str}, T

    @staticmethod
    def improper_integral(expr: str, a: float, b=None, singularity=None) -> tuple[dict, SolutionTracer]:
        desc = f"∫[{a}," + ("∞" if b is None else str(b)) + f"] {expr} dx"
        T = SolutionTracer(f"反常积分: {desc}", "integral")
        from calculus_engine import AdvancedIntegration
        T.step("反常积分识别", "无穷限积分" if b is None else "瑕积分" if singularity else "反常积分",
               THEOREMS["反常积分"])
        T.sub("审敛分析", "检查无穷远处的衰减速度 或 奇点附近的阶", THEOREMS["反常积分审敛"])
        r = AdvancedIntegration.improper_integral(expr, a, b, singularity)
        status = "收敛" if r["converges"] else ("发散" if r["converges"] is False else "待定")
        T.step("收敛性判断", result=status)
        if r.get("numeric"):
            T.step("数值结果", result=f"≈ {r['numeric']:.8f}")
        return r, T

    @staticmethod
    def double_integral(expr: str, xr: tuple, yr: tuple) -> tuple[dict, SolutionTracer]:
        T = SolutionTracer(f"二重积分: ∬ {expr} dxdy", "integral")
        T.step("积分区域", f"x∈[{xr[0]},{xr[1]}], y∈[{yr[0]},{yr[1]}]", THEOREMS["二重积分"])
        T.sub("化为二次积分", "先对 x 积分(内层), 再对 y 积分(外层)", "Fubini定理: 矩形区域可交换次序")
        from calculus_engine import AdvancedIntegration
        r = AdvancedIntegration.double_integral(expr, xr, yr)
        if r.get("inner"):
            T.step("内层积分 (对x)", f"∫ f(x,y) dx = {r['inner']}".replace("**", "^"))
        T.step("外层积分结果", f"= {r['symbolic']}".replace("**", "^") if r.get("symbolic") else "",
               result=f"≈ {r['numeric']:.8f}" if r.get("numeric") else "")
        return r, T

    @staticmethod
    def triple_integral(expr: str, xr: tuple, yr: tuple, zr: tuple) -> tuple[dict, SolutionTracer]:
        T = SolutionTracer(f"三重积分: ∭ {expr} dxdydz", "integral")
        sl = lambda b: f"{float(b):.4g}" if isinstance(b, (int, float)) else str(b).replace("**", "^")
        T.step("积分区域", f"x∈[{sl(xr[0])},{sl(xr[1])}], y∈[{sl(yr[0])},{sl(yr[1])}], z∈[{sl(zr[0])},{sl(zr[1])}]",
               THEOREMS["三重积分"])
        T.sub("化为三次积分", "逐次积分 (根据依赖关系自动排序)", "Fubini定理推广")
        import sympy as sp
        x, y, z = sp.symbols('x y z')
        from function_plotter import _preprocess_expr
        try:
            s = _preprocess_expr(str(expr))  # 确保 expr 是字符串
        except Exception:
            s = str(expr)
        loc = {"x": x, "y": y, "z": z, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
               "log": sp.log, "exp": sp.exp, "sqrt": sp.sqrt, "abs": sp.Abs, "pi": sp.pi, "e": sp.E}
        f_sp = sp.sympify(s, locals=loc)
        # 自动确定积分顺序
        bounds = [(x, xr[0], xr[1]), (y, yr[0], yr[1]), (z, zr[0], zr[1])]
        deps = {}
        for v, lo, hi in bounds:
            deps[v] = set()
            for sym in [x, y, z]:
                if hasattr(lo, 'free_symbols') and sym in lo.free_symbols: deps[v].add(sym)
                if hasattr(hi, 'free_symbols') and sym in hi.free_symbols: deps[v].add(sym)
        order = sorted(bounds, key=lambda b: -len(deps[b[0]]))
        try:
            expr_i = f_sp
            for v, lo, hi in order:
                expr_i = sp.integrate(expr_i, (v, lo, hi))
                T.step(f"积分 d{str(v)}", f"∫ d{str(v)} → {sp.simplify(expr_i)}".replace("**", "^"))
            T.step("积分结果", result=str(sp.simplify(expr_i)).replace("**", "^"))
        except Exception:
            T.step("符号积分失败", "使用数值积分")
        return {}, T

    @staticmethod
    def arc_length(expr: str, a: float, b: float) -> tuple[dict, SolutionTracer]:
        T = SolutionTracer(f"弧长: y={expr} 在 [{a},{b}]", "integral")
        T.step("弧长公式", "L = ∫ √(1 + (y')²) dx", THEOREMS["弧长公式"])
        from calculus_engine import CurveEngine
        r = CurveEngine.arc_length(expr, a, b)
        T.step("被积函数", f"√(1 + (f')²) = {r['integrand']}".replace("**", "^"),
               THEOREMS["弧微分"])
        T.step("弧长", result=f"L ≈ {r['numeric']:.6f}" if r.get("numeric") else r.get("symbolic", "—"))
        return r, T

    @staticmethod
    def curvature(expr: str, xv: float) -> tuple[dict, SolutionTracer]:
        T = SolutionTracer(f"曲率: y={expr} 在 x={xv}", "derivative")
        T.step("曲率公式", "κ = |y''| / (1 + y'²)^(3/2)", THEOREMS["曲率"])
        from calculus_engine import CurveEngine
        r = CurveEngine.curvature(expr, xv)
        T.step("曲率表达式", f"κ(x) = {r['kappa_expr']}".replace("**", "^"))
        if r.get("curvature"):
            T.step("曲率值", result=f"κ = {r['curvature']:.6f}")
        if r.get("radius"):
            T.step("曲率半径", result=f"R = 1/κ = {r['radius']:.4f}",
                   theorem="密切圆 (Osculating Circle)")
        return r, T

    @staticmethod
    def numerical_compare(expr: str, a: float, b: float) -> tuple[dict, SolutionTracer]:
        T = SolutionTracer(f"数值积分对比: {expr} 在 [{a},{b}]", "integral")
        T.step("方法对比", "比较梯形法 / Simpson / 中点法 / Gauss求积")
        from calculus_engine import AdvancedIntegration
        r = AdvancedIntegration.numerical_compare(expr, a, b)
        for method, val in r.get("methods", {}).items():
            err_str = ""
            if r.get("exact"):
                err_str = f" 误差={abs(val-r['exact']):.2e}"
            T.sub(method, result=f"{val:.8f}{err_str}")
        if r.get("exact"):
            T.step("精确值", result=f"{r['exact']:.10f}")
        return r, T

    @staticmethod
    def parametric_curve(xt: str, yt: str, tr: tuple) -> tuple[dict, SolutionTracer]:
        T = SolutionTracer(f"参数曲线: x(t)={xt}, y(t)={yt}", "integral")
        T.step("参数方程", f"t ∈ [{tr[0]}, {tr[1]}]", THEOREMS["弧长公式"])
        from calculus_engine import CurveEngine
        r = CurveEngine.parametric_curve(xt, yt, tr)
        if r.get("arc_length"):
            T.step("弧长", f"L = ∫ √(x'² + y'²) dt", result=f"≈ {r['arc_length']:.6f}")
        return r, T

    @staticmethod
    def polar_curve(r_expr: str, thr: tuple) -> tuple[dict, SolutionTracer]:
        T = SolutionTracer(f"极坐标: r = {r_expr}", "integral")
        T.step("极坐标方程", f"θ ∈ [{thr[0]}, {thr[1]}]")
        from calculus_engine import CurveEngine
        r = CurveEngine.polar_curve(r_expr, thr)
        if r.get("area"):
            T.step("极坐标面积", "A = ½ ∫ r² dθ", result=f"≈ {r['area']:.6f}",
                   theorem=THEOREMS["极坐标二重"])
        return r, T

    @staticmethod
    def divergence(F: str, pt: tuple) -> tuple[dict, SolutionTracer]:
        T = SolutionTracer(f"散度: div F, F = [{F}]", "vector")
        T.step("散度定义", "div F = ∂P/∂x + ∂Q/∂y + ∂R/∂z", THEOREMS["散度定义"])
        from calculus_engine import VectorEngine
        r = VectorEngine.divergence(F, pt)
        T.step("散度表达式", f"div F = {r.get('divergence','')}".replace("**", "^"))
        if "value" in r:
            T.step(f"在 {pt} 处", result=f"div F = {r['value']:.6f}")
        return r, T

    @staticmethod
    def curl(F: str, pt: tuple) -> tuple[dict, SolutionTracer]:
        T = SolutionTracer(f"旋度: curl F, F = [{F}]", "vector")
        T.step("旋度定义", "curl F = ∇ × F", THEOREMS["旋度定义"])
        from calculus_engine import VectorEngine
        r = VectorEngine.curl(F, pt)
        T.step("旋度表达式", f"curl F = [{', '.join(r.get('curl',[]))}]".replace("**", "^"))
        if "value" in r:
            T.step(f"在 {pt} 处", result=f"curl F = {r['value']}")
        return r, T

    @staticmethod
    def line_integral(F: str, xt: str, yt: str, tr: tuple) -> tuple[dict, SolutionTracer]:
        T = SolutionTracer(f"线积分: ∫_C F·dr", "vector")
        T.step("线积分定义", f"F = [{F}], C: ({xt}, {yt}), t∈[{tr[0]},{tr[1]}]",
               THEOREMS["曲线积分2"])
        from calculus_engine import VectorEngine
        r = VectorEngine.line_integral(F, xt, yt, tr)
        T.step("被积函数", f"F·dr = {r.get('integrand','')}".replace("**", "^"))
        if r.get("numeric"):
            T.step("积分结果", result=f"≈ {r['numeric']:.8f}")
        return r, T

    @staticmethod
    def power_radius(expr: str) -> tuple[dict, SolutionTracer]:
        T = SolutionTracer(f"幂级数收敛半径: Σ {expr}·(x-x₀)^n", "series")
        T.step("收敛半径公式", "R = 1 / lim |a_{n+1}/a_n|", THEOREMS["幂级数"])
        from calculus_engine import SeriesAnalysis
        r = SeriesAnalysis.power_series_radius(expr)
        T.step("比值极限", "L = lim |a_{n+1}/a_n| = " + str(r.get('L','?')))
        T.step("收敛半径", result=f"R = {r.get('radius_str','?')}")
        T.sub("阿贝尔定理", f"|x-x₀| < R 时收敛, > R 时发散")
        return r, T

    @staticmethod
    def series_sum(expr: str, fm: int, to_n=None) -> tuple[dict, SolutionTracer]:
        desc = f"Σ_{{{fm}}}^{{{to_n or '∞'}}} {expr}"
        T = SolutionTracer(f"级数求和: {desc}", "series")
        T.step("级数求和", f"计算 {desc}")
        from calculus_engine import SeriesAnalysis
        r = SeriesAnalysis.series_sum(expr, fm, to_n)
        T.step("结果", result=r.get("symbolic", "—"))
        if r.get("numeric"):
            T.sub("数值", result=f"{r['numeric']:.10f}")
        return r, T

    @staticmethod
    def riemann_sum(expr: str, a: float, b: float, n: int, mode: str) -> tuple[dict, SolutionTracer]:
        mode_cn = {"left": "左端点", "right": "右端点", "midpoint": "中点", "trapezoid": "梯形"}
        T = SolutionTracer(f"黎曼和: {expr} [{a},{b}] n={n} ({mode_cn.get(mode,mode)})", "integral")
        T.step("黎曼和定义", f"将[{a},{b}]分为{n}份, 取{mode_cn.get(mode,mode)}近似",
               THEOREMS["定积分定义"])
        from calculus_engine import CalculusEngine
        r = CalculusEngine.riemann_data(expr, a, b, n, mode)
        T.step("近似值", f"Σ f(ξ_i)·Δx", result=f"≈ {r['value']:.8f}")
        exact = CalculusEngine.integrate_definite(expr, a, b)
        if exact.get("numeric"):
            err = abs(r["value"] - exact["numeric"])
            T.sub("与精确值对比", f"精确值={exact['numeric']:.8f}, 误差={err:.2e}")
        return r, T

    @staticmethod
    def directional_derivative(expr: str, pt: tuple, direction: tuple) -> tuple[dict, SolutionTracer]:
        T = SolutionTracer(f"方向导数: f={expr} 在{pt}, 方向={direction}", "multivariable")
        T.step("方向导数公式", "D_u f = ∇f · u (u 为单位向量)", THEOREMS["方向导数"])
        from calculus_engine import MultivariableEngine
        r = MultivariableEngine.directional_derivative(expr, pt, direction)
        if "directional_derivative" in r:
            T.step("结果", result=f"D_u f = {r['directional_derivative']:.6f}")
        return r, T

    @staticmethod
    def hessian(expr: str) -> tuple[dict, SolutionTracer]:
        T = SolutionTracer(f"Hessian矩阵: f = {expr}", "multivariable")
        T.step("Hessian定义", "H_ij = ∂²f/∂x_i∂x_j", THEOREMS["高阶偏导"])
        from calculus_engine import MultivariableEngine
        r = MultivariableEngine.hessian(expr)
        for row in r.get("hessian", []):
            T.sub("行", f"[{', '.join(row)}]".replace("**", "^"))
        return r, T

    @staticmethod
    def lagrange_multipliers(f_str: str, g_str: str) -> tuple[dict, SolutionTracer]:
        T = SolutionTracer(f"拉格朗日乘数: f={f_str}, 约束 g={g_str}=0", "multivariable")
        T.step("拉格朗日函数", f"L = f - λ·g = {f_str} - λ·({g_str})", THEOREMS["拉格朗日乘数"])
        T.sub("求偏导", "∂L/∂x=0, ∂L/∂y=0, ∂L/∂λ=0 (即g=0)")
        from calculus_engine import MultivariableEngine
        r = MultivariableEngine.lagrange_multipliers(f_str, g_str)
        for cp in r.get("critical_points", []):
            T.step("驻点", f"x={cp['x']:.4f}, y={cp['y']:.4f}, λ={cp['lambda']:.4f}",
                   result=f"f={cp['f']:.6f}")
        return r, T