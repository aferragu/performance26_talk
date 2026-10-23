"""
Extra figure for the Performance 2026 talk, using the parameters of Section 4 of the paper
(p2=150 kW, p1=60 kW, p0=p0*=36 kW, 1/mu=20 kWh, lambda in {2,3,4} cars/h).

The sample paths, stationary CDF and slowdown figures are taken from the paper repo.
This script only produces figuras/blocking_vs_B.tex: blocking probability (Eq. PB of the
paper) as a function of B, with exact event-driven simulation of the PDMP as a check.

Run:  python3 make_figures.py
"""
import os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figuras")
rng = np.random.default_rng(2026)

p2, p1, p0 = 150.0, 60.0, 36.0
c = p2 - p1
mu = 1 / 20.0
LAMBDAS = [2.0, 3.0, 4.0]
COLORS = {2.0: "blue!70!black", 3.0: "red!70!black", 4.0: "green!50!black"}
LABELS = {2.0: r"$\lambda=2$", 3.0: r"$\lambda=3$", 4.0: r"$\lambda=4$"}


def blocking(lam, B):
    th = mu * p2 / c - lam / p0
    E = B if abs(th) < 1e-12 else np.expm1(th * B) / th
    K = (p0 + c) / c * E + p0 / (mu * p1) + p0 / lam * np.exp(th * B)
    return p0 / K * (E / c + 1 / (mu * p1))


def simulate_PB(lam, B, ncars):
    x, t, busy = B, 0.0, 0.0
    for _ in range(ncars):
        A = rng.exponential(1 / lam)
        x = min(B, x + p0 * A)
        t += A
        y = rng.exponential(1 / mu)
        if y <= p2 * x / c:
            T = y / p2
            x -= c * T
        else:
            T = x / c + (y - p2 * x / c) / p1
            x = 0.0
        busy += T
        t += T
    return busy / t


def table(xs, ys):
    rows = "\n".join(f"            {a:.6g}  {b:.6g}  \\\\" for a, b in zip(xs, ys))
    return "        table[row sep={\\\\}]\n        {\n            \\\\\n" + rows + "\n        };\n"


Bgrid = np.linspace(0.01, 200, 300)
Bsim = np.array([0.01, 20, 40, 78, 120, 160, 200])
body = r"""\tikzsetnextfilename{blocking_vs_B}
\begin{tikzpicture}
    \begin{axis}[
    width=\textwidth, height=0.56\textwidth,
    xlabel={Battery capacity $B$ (kWh)}, ylabel={Blocking probability $P_B$},
    xmin=0, xmax=200, ymin=0, ymax=0.75, grid=major, grid style={gray!25, line width=0.3pt},
    legend pos=north east, legend cell align=left, legend style={font=\scriptsize, fill=white, fill opacity=0.85, text opacity=1}, legend columns=3,
    tick label style={font=\footnotesize}, label style={font=\footnotesize},
    extra x ticks={78}, extra x tick labels={$B^*$}, extra x tick style={grid=major, grid style={black, dotted}},
    ]
"""
for lam in LAMBDAS:
    PB = [blocking(lam, b) for b in Bgrid]
    sim = [simulate_PB(lam, b, 100_000) for b in Bsim]
    print(f"lambda={lam}: P_B(B=78) theory {blocking(lam, 78):.4f}, sim {sim[3]:.4f}")
    body += f"    \\addplot[color={COLORS[lam]}, thick]\n" + table(Bgrid, PB)
    body += f"    \\addlegendentry{{{LABELS[lam]}}}\n"
    body += f"    \\addplot[only marks, mark=*, mark size=1.6pt, color={COLORS[lam]}, forget plot]\n" + table(Bsim, sim)
    for rate, sty in [(p1, "dashed"), (p2, "dotted")]:
        v = lam / (lam + mu * rate)
        body += f"    \\addplot[color={COLORS[lam]}, {sty}, forget plot, domain=0:200] {{{v:.6f}}};\n"
body += r"""    \end{axis}
\end{tikzpicture}
"""
with open(os.path.join(OUT, "blocking_vs_B.tex"), "w") as f:
    f.write(body)
print("wrote blocking_vs_B")
