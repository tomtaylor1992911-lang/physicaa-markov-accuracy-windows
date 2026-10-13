"""Reproduce the revised finite-reservoir figures and source data.

Usage: python code/revision_numerics.py [--compute-only] [--plot-only]
Only NumPy and matplotlib are required. Numerical units have hbar=1.
The uniform finite bath is solved in the invariant single-excitation sector.
Gauss-Legendre quadrature approximates the finite-band continuous spectral
integral; it is a numerical reference, not an analytic continuum solution.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path
import time as walltime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG = ROOT / "data", ROOT / "figures"
EVIDENCE = ROOT / "verification"
for folder in (DATA, FIG, EVIDENCE):
    folder.mkdir(parents=True, exist_ok=True)

WIDTH, BANDWIDTH, OMEGA0 = 2.0, 16.0, 10.0
RATIOS = (.01, .04, .1, .2)
TOLERANCES = (.01, .02, .05, .1)
COUNTS = (32, 64, 128, 256)
CUTS = (1., 2., 3., 5.)
BLUE, ORANGE, GREEN, GRAY = "#36648b", "#ba7446", "#54847b", "#666666"
COLORS = [BLUE, GREEN, ORANGE, "#84608c"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .7, "lines.linewidth": 1.15,
    "legend.frameon": False, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8})


def log(message):
    print(message, flush=True)


def write_csv(name, headers, rows):
    with (DATA / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def save(fig, name):
    fig.savefig(FIG / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(FIG / (name + ".svg"), bbox_inches="tight")
    fig.savefig(FIG / (name + ".png"), bbox_inches="tight", dpi=400)
    plt.close(fig)


def star_spectrum(detuning, g2):
    """Eigenvalues and central-state weights of a real arrowhead matrix.

    All roots of x-sum[g_j^2/(x-d_j)] are bracketed between adjacent poles.
    Bisection avoids an unnecessary dense eigenvector matrix at large orders.
    The central weights are 1/(1+sum[g_j^2/(x-d_j)^2]). Results are checked
    against dense eigh for finite-bath sizes in compute().
    """
    detuning = np.asarray(detuning, float)
    g2 = np.asarray(g2, float)
    extent = np.max(np.abs(detuning)) + np.sqrt(g2.sum()) + 1.
    lo = np.concatenate(([-extent], detuning.copy()))
    hi = np.concatenate((detuning.copy(), [extent]))
    for _ in range(56):
        x = (lo + hi) / 2
        with np.errstate(divide="ignore"):
            fn = x - np.sum(g2[None, :] / (x[:, None] - detuning[None, :]), axis=1)
        lo = np.where(fn < 0, x, lo)
        hi = np.where(fn >= 0, x, hi)
    ev = (lo + hi) / 2
    with np.errstate(divide="ignore"):
        wt = 1 / (1 + np.sum(g2[None, :] / (ev[:, None] - detuning[None, :])**2, axis=1))
    if abs(wt.sum() - 1.) > 2e-10:
        raise RuntimeError("Star spectral weights fail normalization")
    return ev, wt


class Bath:
    def __init__(self, n, gamma=.08, scheme="uniform"):
        self.n, self.gamma, self.scheme = int(n), float(gamma), scheme
        if scheme == "uniform":
            self.spacing = BANDWIDTH / n
            self.detuning = -BANDWIDTH/2 + (np.arange(n)+.5)*self.spacing
            integration_weights = np.full(n, self.spacing)
            self.recurrence = 2*np.pi/self.spacing
        elif scheme == "gauss":
            nodes, weights = np.polynomial.legendre.leggauss(n)
            self.detuning = nodes * BANDWIDTH/2
            integration_weights = weights * BANDWIDTH/2
            self.spacing, self.recurrence = np.nan, np.nan
        else:
            raise ValueError(scheme)
        self.g2 = gamma/(2*np.pi)*WIDTH**2/(self.detuning**2+WIDTH**2)*integration_weights
        self.ev, self.wt = star_spectrum(self.detuning, self.g2)
        self.speed_lipschitz = float(self.g2.sum())

    def evaluate(self, time, correlation=True):
        t = np.atleast_1d(time).astype(float)
        c, cd = np.empty(t.size, complex), np.empty(t.size, complex)
        corr = np.empty(t.size, complex) if correlation else None
        for start in range(0, t.size, 1500):
            stop = min(start+1500, t.size)
            phase = np.exp(-1j*self.ev[:, None]*t[None, start:stop])
            c[start:stop] = self.wt @ phase
            cd[start:stop] = (-1j*self.ev*self.wt) @ phase
            if correlation:
                corr[start:stop] = self.g2 @ np.exp(-1j*self.detuning[:, None]*t[None, start:stop])
        return c, cd, corr

    def probability_error(self, t):
        c = np.sum(self.wt*np.exp(-1j*self.ev*t))
        return abs(abs(c)**2 - np.exp(-self.gamma*t))


def grid(horizon, dt):
    return np.linspace(0., horizon, int(np.ceil(horizon/dt))+1)


def crossing(bath, t, population, tolerance):
    error = abs(population-np.exp(-bath.gamma*t))
    hits = np.flatnonzero(error > tolerance)
    if not hits.size:
        return None
    j = int(hits[0])
    if j == 0:
        return 0.
    lo, hi = float(t[j-1]), float(t[j])
    for _ in range(40):
        mid = (lo+hi)/2
        if bath.probability_error(mid) > tolerance:
            hi = mid
        else:
            lo = mid
    return (lo+hi)/2


def cumulative_trapezoid(y, t):
    ans = np.zeros_like(y)
    ans[1:] = np.cumsum((y[:-1]+y[1:])*.5*np.diff(t))
    return ans


def trailing_max(values, points):
    q = deque()
    out = np.empty_like(values)
    for i, value in enumerate(values):
        while q and q[0] < i-points:
            q.popleft()
        while q and values[q[-1]] <= value:
            q.pop()
        q.append(i)
        out[i] = values[q[0]]
    return out


def exp_convolution(y, t, rate):
    """Piecewise-linear quadrature of exp[-rate(t-s)]y(s), stable at late t."""
    dt = t[1]-t[0]
    if abs(rate*dt) < 1e-6:
        # The trapezoid remainder vanishes under grid refinement.
        decay = np.exp(-rate*dt)
        result = np.zeros_like(y)
        for i in range(1, len(y)):
            result[i] = decay*result[i-1] + dt*.5*(decay*y[i-1]+y[i])
        return result
    decay = np.exp(-rate*dt)
    j0 = -np.expm1(-rate*dt)/rate
    j1 = (1-decay*(1+rate*dt))/rate**2
    a1 = j1/dt
    a0 = j0-a1
    result = np.zeros_like(y)
    for i in range(1, len(y)):
        result[i] = decay*result[i-1] + a0*y[i]+a1*y[i-1]
    return result


def residual(bath, t, c, cd, corr, cutoff):
    dt = t[1]-t[0]
    k = int(round(cutoff/dt))
    if abs(t[k]-cutoff) > 1e-10:
        raise ValueError("Residual grid must contain the selected cutoff")
    abs_corr = abs(corr)
    cumulative = cumulative_trapezoid(abs_corr, t)
    moments = cumulative_trapezoid(t*abs_corr, t)
    kappa = cumulative_trapezoid(corr, t)[k]
    # |c''(s)| <= sum_j g_j^2, so half-step margin controls intersample speed.
    speed = trailing_max(abs(cd), k) + bath.speed_lipschitz*dt/2
    variation = speed*moments[np.minimum(np.arange(t.size), k)]
    early = abs(c)*np.maximum(cumulative[k]-cumulative, 0.)
    tail = np.maximum(cumulative-cumulative[k], 0.)
    bound = variation+early+tail
    actual = abs(cd+kappa*c)
    cc = np.exp(-kappa*t)
    amplitude_error = abs(c-cc)
    weighted_actual = exp_convolution(actual, t, float(kappa.real))
    weighted_bound = exp_convolution(bound, t, float(kappa.real))
    prob_cutoff_error = abs(abs(c)**2-abs(cc)**2)
    prob_cutoff_bound = (abs(c)+abs(cc))*weighted_bound
    rate_mismatch = abs(abs(cc)**2-np.exp(-bath.gamma*t))
    markov_bound = prob_cutoff_bound+rate_mismatch
    return {"cutoff": cutoff, "kappa": kappa, "speed": speed,
        "variation": variation, "early": early, "tail": tail,
        "bound": bound, "actual": actual, "amplitude_error": amplitude_error,
        "weighted_actual": weighted_actual, "weighted_bound": weighted_bound,
        "prob_cutoff_error": prob_cutoff_error, "prob_cutoff_bound": prob_cutoff_bound,
        "rate_mismatch": rate_mismatch, "markov_bound": markov_bound}


def contiguous_certification(t, bound, tolerance):
    hits = np.flatnonzero(bound > tolerance)
    if not hits.size:
        return None
    j = int(hits[0])
    if j == 0: return 0.
    return float(np.interp(tolerance, bound[j-1:j+1], t[j-1:j+1]))


def compute():
    started = walltime.time()
    results = {"model": {"Lambda": WIDTH, "W": BANDWIDTH, "omega0": OMEGA0,
        "hbar": 1, "reference_gamma": .08},
        "definitions": {"D": "absolute population error, equal to trace distance for the stated initial state", "t_delta": "first t>=0 with D(t)>delta, contiguous accuracy window starting at0", "continuum": "finite-band Gauss-Legendre quadrature numerical reference", "bound_scope": "a posteriori scalar cutoff time-local replacement; no Born/closure error estimate"}}
    scan_rows, threshold_rows = [], []
    baths = {}
    for ratio in RATIOS:
        for n in COUNTS:
            bath = Bath(n, gamma=ratio*WIDTH)
            baths[(ratio, n)] = bath
            horizon = 1.25*bath.recurrence
            # Both samplings start at 0. Exact root bisection follows first bracket.
            t = grid(horizon, .01)
            c, _, _ = bath.evaluate(t, correlation=False)
            p = abs(c)**2
            t2 = grid(horizon, .005)
            c2, _, _ = bath.evaluate(t2, correlation=False)
            p2 = abs(c2)**2
            for tolerance in TOLERANCES:
                tc, tf = crossing(bath, t, p, tolerance), crossing(bath, t2, p2, tolerance)
                scan_rows.append([n, ratio, bath.gamma, tolerance, bath.spacing, bath.recurrence, horizon, tf, None if tf is None else tf/bath.recurrence])
                threshold_rows.append([n, ratio, tolerance, t[1], t2[1], tc, tf, None if tc is None or tf is None else abs(tc-tf)])
            log(f"Threshold scan gamma/Lambda={ratio:g}, N={n}")
    write_csv("validity_window_scan.csv", ["N_B", "gamma_over_Lambda", "gamma", "tolerance", "delta_omega", "t_recurrence", "horizon", "t_first_exit", "t_first_exit_over_recurrence"], scan_rows)
    write_csv("threshold_convergence.csv", ["N_B", "gamma_over_Lambda", "tolerance", "coarse_dt", "fine_dt", "coarse_crossing", "fine_crossing", "absolute_difference"], threshold_rows)
    results["threshold_scan"] = [dict(zip(["N_B", "gamma_over_Lambda", "gamma", "tolerance", "delta_omega", "recurrence", "horizon", "first_exit", "exit_over_recurrence"], row)) for row in scan_rows]
    results["threshold_grid_max_difference"] = max(row[-1] or 0 for row in threshold_rows)
    # Benchmark retains the source table's mode numbers.
    benchmark_rows = []
    for n in (16,24,32,48,64,96,128,192,256):
        bath = baths.get((.04,n)) or Bath(n)
        t = grid(max(140,1.25*bath.recurrence), .01)
        c, _, _ = bath.evaluate(t, correlation=False)
        error = abs(abs(c)**2-np.exp(-.08*t))
        benchmark_rows.append([n,bath.spacing,bath.recurrence,crossing(bath,t,abs(c)**2,.05),float(error[t<bath.recurrence].max())])
    write_csv("finite_bath_benchmark.csv", ["N_B","delta_omega","t_recurrence","t_first_D_gt_0.05","max_error_before_recurrence"], benchmark_rows)
    results["benchmark"] = benchmark_rows
    # Dense eigensolver independently checks the star-root implementation.
    dense_checks = []
    for n in (32,256):
        bath = baths[(.04,n)]
        h = np.diag(np.r_[0.,bath.detuning])
        h[0,1:],h[1:,0] = np.sqrt(bath.g2),np.sqrt(bath.g2)
        ev, vectors = np.linalg.eigh(h)
        wt = abs(vectors[0])**2
        tc = np.linspace(0,125,2001)
        direct = wt @ np.exp(-1j*ev[:,None]*tc[None,:])
        other = bath.evaluate(tc,correlation=False)[0]
        dense_checks.append({"N_B":n,"max_amplitude_difference":float(abs(direct-other).max())})
    results["dense_eigh_checks"] = dense_checks
    log("Computing finite-band continuous numerical reference")
    t = np.linspace(0,125,25001)
    refs, quad_rows = {}, []
    for order in (512,768,1024):
        bath = Bath(order, scheme="gauss")
        refs[order] = bath.evaluate(t,correlation=False)[0]
        log(f"Gauss-Legendre order {order} complete")
    cref = refs[1024]
    for order in (512,768):
        dif = abs(refs[order]-cref)
        dp = abs(abs(refs[order])**2-abs(cref)**2)
        quad_rows.append(["gauss",order,float(dif.max()),float(dp.max())])
    for n in (512,1024):
        uniform = Bath(n)
        u = uniform.evaluate(t,correlation=False)[0]
        quad_rows.append(["uniform",n,float(abs(u-cref).max()),float(abs(abs(u)**2-abs(cref)**2).max())])
    write_csv("quadrature_convergence.csv",["scheme","order","max_amplitude_difference_to_gauss1024","max_population_difference_to_gauss1024"],quad_rows)
    results["quadrature_convergence"] = quad_rows
    results["continuum_reference_max_time"] = 125.
    c32 = baths[(.04,32)].evaluate(t,correlation=False)[0]
    c256 = baths[(.04,256)].evaluate(t,correlation=False)[0]
    pm = np.exp(-.08*t)
    pref = abs(cref)**2
    write_csv("continuum_reference.csv",["time","P_gauss1024","P_gauss768","P_gauss512","P_N32","P_N256","P_Markov","finite_grid_error_N32","finite_grid_error_N256","finite_memory_markov_error"], zip(t,pref,abs(refs[768])**2,abs(refs[512])**2,abs(c32)**2,abs(c256)**2,pm,abs(abs(c32)**2-pref),abs(abs(c256)**2-pref),abs(pref-pm)))
    results["continuum_markov_max_error_0_to_10"] = float(abs(pref[t<=10]-pm[t<=10]).max())
    results["continuum_markov_max_error_0_to_125"] = float(abs(pref-pm).max())
    results["finite_grid_max_error_0_to_80"] = {str(n):float(abs(abs(cn[t<=80])**2-pref[t<=80]).max()) for n,cn in ((32,c32),(256,c256))}
    # Residual grid contains tau_c=1,2,3,5 exactly.
    tr = np.linspace(0,125,50001)
    bath = baths[(.04,256)]
    cr, cdr, corr = bath.evaluate(tr)
    residuals = {cut:residual(bath,tr,cr,cdr,corr,cut) for cut in CUTS}
    r = residuals[3.]
    write_csv("finite_bath_residual.csv",["time","actual_residual","state_variation_bound","initial_slip_bound","correlation_tail","total_bound","weighted_actual_residual","weighted_total_bound","actual_amplitude_error_to_cutoff","actual_population_error_to_cutoff","population_cutoff_bound","rate_mismatch","population_markov_bound","P_exact","P_cutoff","P_Markov"],zip(tr,r["actual"],r["variation"],r["early"],r["tail"],r["bound"],r["weighted_actual"],r["weighted_bound"],r["amplitude_error"],r["prob_cutoff_error"],r["prob_cutoff_bound"],r["rate_mismatch"],r["markov_bound"],abs(cr)**2,np.exp(-2*r["kappa"].real*tr),np.exp(-.08*tr)))
    residual_summary, cutoff_rows = [], []
    for cut,r in residuals.items():
        item = {"tau_c":cut,"kappa_real":float(r["kappa"].real),"kappa_imag":float(r["kappa"].imag),"population_rate":float(2*r["kappa"].real),
            "max_actual_minus_bound":float((r["actual"]-r["bound"]).max()),
            "max_amplitude_error_minus_weighted_bound":float((r["amplitude_error"]-r["weighted_bound"]).max()),
            "max_population_error_minus_markov_bound":float((abs(abs(cr)**2-np.exp(-.08*tr))-r["markov_bound"]).max()),
            "cutoff_certification_times":{str(d):contiguous_certification(tr,r["prob_cutoff_bound"],d) for d in (.01,.05)},
            "markov_certification_times":{str(d):contiguous_certification(tr,r["markov_bound"],d) for d in (.01,.05)}}
        residual_summary.append(item)
        cutoff_rows.extend(zip(np.repeat(cut,len(tr)),tr,r["actual"],r["bound"],r["amplitude_error"],r["weighted_bound"],r["prob_cutoff_error"],r["prob_cutoff_bound"],r["rate_mismatch"],r["markov_bound"]))
    write_csv("cutoff_sensitivity.csv",["tau_c","time","actual_residual","residual_bound","amplitude_error_to_cutoff","weighted_amplitude_bound","population_error_to_cutoff","population_cutoff_bound","rate_mismatch","population_markov_bound"],cutoff_rows)
    results["residuals"] = residual_summary
    # Quadrature and piecewise-integration error is assessed by halving dt.
    tf = np.linspace(0,125,100001)
    cf,cdf,corrf = bath.evaluate(tf)
    rf = residual(bath,tf,cf,cdf,corrf,3.)
    results["residual_grid_convergence"] = {"coarse_dt":.0025,"fine_dt":.00125,
        "kappa_absolute_difference":float(abs(rf["kappa"]-residuals[3.]["kappa"])),
        "max_weighted_bound_difference":float(abs(rf["weighted_bound"][::2]-residuals[3.]["weighted_bound"]).max()),
        "fine_max_actual_minus_bound":float((rf["actual"]-rf["bound"]).max()),
        "fine_max_amplitude_error_minus_weighted_bound":float((rf["amplitude_error"]-rf["weighted_bound"]).max()),
        "coarse_cutoff_certification_0.05":contiguous_certification(tr,residuals[3.]["prob_cutoff_bound"],.05),
        "fine_cutoff_certification_0.05":contiguous_certification(tf,rf["prob_cutoff_bound"],.05),
        "coarse_markov_certification_0.05":contiguous_certification(tr,residuals[3.]["markov_bound"],.05),
        "fine_markov_certification_0.05":contiguous_certification(tf,rf["markov_bound"],.05)}
    # Save curve samples for the original benchmark at a common grid.
    tb = np.linspace(0,125,12501)
    b32 = baths[(.04,32)].evaluate(tb)
    b256 = baths[(.04,256)].evaluate(tb)
    write_csv("finite_bath_curves.csv",["time","correlation_abs_N32","correlation_abs_N256","P_N32","P_N256","P_Markov","D_N32","D_N256"],zip(tb,abs(b32[2]/b32[2][0]),abs(b256[2]/b256[2][0]),abs(b32[0])**2,abs(b256[0])**2,np.exp(-.08*tb),abs(abs(b32[0])**2-np.exp(-.08*tb)),abs(abs(b256[0])**2-np.exp(-.08*tb))))
    results["walltime_seconds"] = walltime.time()-started
    results["numerical_notes"] = ["Threshold grids begin at0; exact analytic bisection refines the first bracket; halved-grid brackets agree.", "Speed supremum is represented by a causal trailing sample maximum plus a derivative Lipschitz half-step margin.", "Residual integrals and propagation are numerical quadratures; convergence checks support the reported comparison, not a formal machine certificate.", "Quadrature continuum reference retains the finite bandwidth W=16 and Lorentzian width Lambda=2.", "Bounds can lose certification before observed accuracy is lost; their exceeding a tolerance is not a lower-bound proof of failure."]
    checks = {
        "threshold_grid_agreement_1e-8": results["threshold_grid_max_difference"] < 1e-8,
        "independent_dense_eigh_agreement_1e-9": max(x["max_amplitude_difference"] for x in dense_checks) < 1e-9,
        "gauss768_to1024_population_agreement_1e-8": quad_rows[1][3] < 1e-8,
        "residual_bound_with_numerical_tolerance_1e-8": max(x["max_actual_minus_bound"] for x in residual_summary) < 1e-8,
        "amplitude_bound_with_numerical_tolerance_1e-8": max(x["max_amplitude_error_minus_weighted_bound"] for x in residual_summary) < 1e-8,
        "markov_population_bound_with_numerical_tolerance_1e-8": max(x["max_population_error_minus_markov_bound"] for x in residual_summary) < 1e-8,
    }
    results["verification_checks"] = checks
    if not all(checks.values()):
        raise RuntimeError(f"Numerical checks failed: {checks}")
    target = EVIDENCE / "numerical_results.json"
    target.write_text(json.dumps(results,indent=2),encoding="utf-8")
    (DATA / "numerical_results.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    log(f"Results saved: {target}; elapsed {results['walltime_seconds']:.1f}s")


def read_csv(name):
    return np.genfromtxt(DATA/name,delimiter=",",names=True,encoding="utf-8")


def panel(ax, label, title):
    ax.text(-.14,1.07,label,transform=ax.transAxes,fontweight="bold",fontsize=10)
    ax.set_title(title,loc="left",fontsize=9,pad=9)
    ax.grid(alpha=.15,linewidth=.5)


def make_map():
    fig,axes = plt.subplots(2,1,figsize=(7.2,3.8))
    routes = [
        ("Reduced subsystem dynamics", [("Unitary dynamics","exact"),("NZ memory identity","exact projection"),("Second-order memory","weak coupling"),("Time-local Redfield","memory replacement"),("GKSL / Pauli","secular averaging")]),
        ("Many-particle kinetic closure", [("Binary Hamiltonian","exact"),("Correlation hierarchy","exact equations"),("Closed finite-memory kinetics","quasifree / molecular chaos"),("Uehling-Uhlenbeck","gradient + on-shell limits"),("Classical Boltzmann","dilute occupations")])]
    for ax,(route,labels) in zip(axes,routes):
        ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis("off")
        ax.text(0,.95,route,fontsize=8,fontweight="bold")
        xs=np.linspace(.095,.905,5)
        for i,(title,assumption) in enumerate(labels):
            title = title.replace("Unitary dynamics","Unitary\ndynamics").replace("NZ memory identity","NZ memory\nidentity").replace("Second-order memory","Second-order\nmemory").replace("Time-local Redfield","Time-local\nRedfield").replace("Binary Hamiltonian","Binary\nHamiltonian").replace("Correlation hierarchy","Correlation\nhierarchy").replace("Closed finite-memory kinetics","Closed finite-\nmemory kinetics").replace("Uehling-Uhlenbeck","Uehling-\nUhlenbeck").replace("Classical Boltzmann","Classical\nBoltzmann")
            box=FancyBboxPatch((xs[i]-.082,.37),.164,.38,boxstyle="round,pad=.006",fc="#f4f7f9",ec=BLUE,lw=.8)
            ax.add_patch(box)
            ax.text(xs[i],.56,title,ha="center",va="center",fontsize=7)
            ax.text(xs[i],.19,assumption.replace("quasifree / molecular chaos","quasifree /\nmolecular chaos").replace("gradient + on-shell limits","gradient +\non-shell limits"),ha="center",va="center",fontsize=6.4,color=GRAY)
            if i<4:
                ax.add_patch(FancyArrowPatch((xs[i]+.088,.56),(xs[i+1]-.088,.56),arrowstyle="-|>",mutation_scale=7,lw=.8,color=BLUE))
    fig.text(.5,.005,"Separate assumptions; no single error theorem is asserted for all arrows.",ha="center",fontsize=7)
    fig.subplots_adjust(hspace=.25,bottom=.09,left=.025,right=.975,top=.97)
    save(fig,"approximation_map")


def plot():
    result_path=DATA/"numerical_results.json"
    if not result_path.exists(): result_path=EVIDENCE/"numerical_results.json"
    result=json.loads(result_path.read_text(encoding="utf-8"))
    make_map()
    b=read_csv("finite_bath_curves.csv");scan=read_csv("finite_bath_benchmark.csv")
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.0),layout="constrained")
    for ax in axes.flat: ax.set_xlabel("Time, $t$")
    for n,color,ls in ((32,BLUE,"-"),(256,ORANGE,"--")):
        axes[0,0].plot(b["time"],b[f"correlation_abs_N{n}"],color=color,ls=ls,label=f"$N_B={n}$")
        axes[0,1].plot(b["time"],b[f"P_N{n}"],color=color,ls=ls,label=f"finite bath, $N_B={n}$")
        axes[1,0].plot(b["time"],b[f"D_N{n}"],color=color,ls=ls,label=f"$N_B={n}$")
        axes[1,0].axvline(2*np.pi*n/BANDWIDTH,ls=":",lw=.8,color=color)
    axes[0,1].plot(b["time"],b["P_Markov"],color=GRAY,ls=":",label="Markov")
    axes[1,0].axhline(.05,ls="--",lw=.8,color=GRAY,label=r"$\delta=0.05$")
    axes[1,1].plot(scan["N_B"],scan["t_recurrence"],color=GRAY,ls="--",label="$t_{rec}$")
    axes[1,1].plot(scan["N_B"],scan["t_first_D_gt_005"],"o",ms=3,color=BLUE,label="$t_{0.05}$")
    axes[0,0].set(xlim=(0,120),ylabel="$|C_N(t)/C_N(0)|$")
    axes[0,1].set(xlim=(0,120),ylabel="$P_e(t)$")
    axes[1,0].set(xlim=(0,120),ylabel="Trace distance, $D(t)$")
    axes[1,1].set(xlabel="Bath modes, $N_B$",ylabel="Time")
    for ax,label,title in zip(axes.flat,"abcd",["Discrete correlation rephasing","Exact excited-state population","Error at fixed 5% tolerance","First crossing and recurrence scale"]):
        panel(ax,label,title);ax.legend()
    save(fig,"finite_bath_markov_benchmark")
    # Four-coupling scan, logarithmic vertical axis exposes early crossings.
    s=read_csv("validity_window_scan.csv")
    fig,axes=plt.subplots(2,2,figsize=(7.2,4.8),layout="constrained",sharex=True,sharey=True)
    for ax,ratio,label in zip(axes.flat,RATIOS,"abcd"):
        for n,color,marker in zip(COUNTS,COLORS,["o","s","^","D"]):
            rows=s[(s["N_B"]==n)&np.isclose(s["gamma_over_Lambda"],ratio)]
            ax.plot(rows["tolerance"],rows["t_first_exit_over_recurrence"],linestyle="None",marker=marker,ms=4,color=color,label=f"$N_B={n}$")
        ax.axhline(1,color=GRAY,ls=":",lw=.8)
        ax.set_yscale("log");ax.set_xticks(TOLERANCES);ax.set_xticklabels(["1%","2%","5%","10%"])
        panel(ax,label,fr"$\gamma/\Lambda={ratio:g}$")
        ax.set_xlabel(r"Population-error tolerance, $\delta$")
    axes[0,0].legend(ncol=2,loc="lower right")
    axes[0,0].set_ylabel(r"$t_\delta/t_{rec}$");axes[1,0].set_ylabel(r"$t_\delta/t_{rec}$")
    save(fig,"validity_window_scan")
    r=read_csv("finite_bath_residual.csv")
    fig,axes=plt.subplots(2,1,figsize=(7.2,4.6),layout="constrained",sharex=True)
    axes[0].semilogy(r["time"],r["actual_residual"],color=BLUE,label="Actual residual $|r_c|$")
    axes[0].semilogy(r["time"],r["total_bound"],color=ORANGE,ls="--",label="Piecewise bound $b_c$")
    for key,label,color,ls in [("state_variation_bound","Retained-memory variation",BLUE,"-"),("initial_slip_bound","Early initial-slip term",GREEN,"-"),("correlation_tail","Omitted correlation tail",ORANGE,"--")]:
        axes[1].semilogy(r["time"],np.maximum(r[key],1e-10),color=color,ls=ls,label=label)
    for ax in axes:
        ax.axvline(2*np.pi*256/16,color=GRAY,ls=":",lw=.8);ax.legend();ax.set_xlim(0,125)
    axes[0].set_ylabel("Residual magnitude (inverse time)")
    axes[1].set(xlabel="Time, $t$",ylabel="Bound contribution (inverse time)",ylim=(1e-6,1))
    panel(axes[0],"a",r"Causal residual estimate, $N_B=256$, $\tau_c=3$")
    panel(axes[1],"b","Initial slip, retained memory and omitted tail")
    save(fig,"finite_bath_residual_diagnostic")
    c=read_csv("continuum_reference.csv")
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.1),layout="constrained")
    axes[0,0].plot(c["time"],c["P_N256"],color=ORANGE,label="$P_{256}$")
    axes[0,0].plot(c["time"],c["P_gauss1024"],color=BLUE,ls="--",label="$P_W$ (quadrature)")
    axes[0,0].plot(c["time"],c["P_Markov"],color=GRAY,ls=":",label="$P_M$")
    axes[0,1].plot(c["time"],c["finite_memory_markov_error"],color=BLUE,label="$|P_W-P_M|$")
    axes[0,1].plot(c["time"],c["finite_grid_error_N256"],color=ORANGE,label="$|P_{256}-P_W|$")
    axes[1,0].plot(r["time"],r["actual_amplitude_error_to_cutoff"],color=BLUE,label="$|c-c_c|$")
    axes[1,0].plot(r["time"],r["weighted_actual_residual"],color=GREEN,ls=":",label="Integrated actual residual")
    axes[1,0].plot(r["time"],r["weighted_total_bound"],color=ORANGE,ls="--",label="Integrated piecewise bound")
    axes[1,1].plot(r["time"],abs(r["P_exact"]-r["P_Markov"]),color=BLUE,label="$|P_{256}-P_M|$")
    axes[1,1].plot(r["time"],r["population_markov_bound"],color=ORANGE,ls="--",label="Bound incl. rate mismatch")
    axes[1,1].axhline(.05,color=GRAY,ls=":",lw=.8,label="5% tolerance")
    for ax,label,title in zip(axes.flat,"abcd",["Finite bath and finite-band reference","Separating two error mechanisms","Weighted cutoff-amplitude bound","Certification versus observed error"]):
        panel(ax,label,title);ax.legend();ax.set_xlabel("Time, $t$");ax.set_xlim(0,125)
    axes[0,0].set_ylabel("Excited-state population")
    axes[0,1].set_ylabel("Absolute population difference")
    axes[1,0].set_ylabel("Amplitude error / bound");axes[1,0].set_ylim(0,float(r["weighted_total_bound"].max())*1.04)
    axes[1,1].set_ylabel("Population error / bound")
    axes[1,1].set_yscale("log");axes[1,1].set_ylim(1e-5,2)
    save(fig,"continuum_residual_benchmark")
    cut=read_csv("cutoff_sensitivity.csv")
    fig,axes=plt.subplots(1,3,figsize=(8,3.05),layout="constrained")
    summary=result["residuals"]
    axes[0].plot(CUTS,[x["population_rate"] for x in summary],"o-",color=BLUE,ms=3,label=r"$2 Re\,\kappa_c$")
    axes[0].axhline(.08,color=GRAY,ls=":",label=r"$\gamma=0.08$")
    for cutoff,color in zip(CUTS,COLORS):
        rr=cut[np.isclose(cut["tau_c"],cutoff)]
        axes[1].plot(rr["time"],rr["population_error_to_cutoff"],color=color,label=fr"$\tau_c={cutoff:g}$")
    for delta,color,marker in ((.01,BLUE,"o"),(.05,ORANGE,"s")):
        axes[2].plot(CUTS,[x["markov_certification_times"][str(delta)] for x in summary],marker=marker,color=color,ms=3,label=fr"$\delta={delta:g}$")
    axes[0].set(xlabel=r"Cutoff, $\tau_c$",ylabel="Population decay rate")
    axes[1].set(xlabel="Time, $t$",ylabel="$|P_{256}-P_c|$",xlim=(0,20))
    axes[1].set_ylim(0,float(cut["population_error_to_cutoff"][cut["time"]<=20].max())*1.08)
    axes[2].set(xlabel=r"Cutoff, $\tau_c$",ylabel="Certified $P_M$ interval")
    axes[2].set_yscale("log")
    for ax,label,title in zip(axes,"abc",["Cutoff-rate bias","Actual cutoff error","Conservative certification"]):panel(ax,label,title);ax.legend()
    save(fig,"cutoff_sensitivity")
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.9),layout="constrained")
    q=result["quadrature_convergence"]
    gauss=[x for x in q if x[0]=="gauss"];uniform=[x for x in q if x[0]=="uniform"]
    for rows,label,color,marker in ((gauss,"Gauss-Legendre",BLUE,"o"),(uniform,"Uniform midpoints",ORANGE,"s")):
        axes[0].semilogy([x[1] for x in rows],[x[3] for x in rows],marker=marker,color=color,label=label)
    tc=read_csv("threshold_convergence.csv")
    dif=tc["absolute_difference"]
    axes[1].plot(np.arange(len(dif))+1,np.maximum(dif,1e-15),"o",ms=2,color=BLUE)
    axes[1].set_yscale("log");axes[1].set_ylim(1e-15,1e-9)
    axes[1].text(.98,.95,r"Values $<10^{-15}$ shown at floor",transform=axes[1].transAxes,ha="right",va="top",fontsize=7)
    axes[0].set(xlabel="Quadrature order / mode count",ylabel="Max $|P-P_{W,1024}|$",xlim=(480,1056))
    axes[1].set(xlabel="Threshold case (64 combinations)",ylabel="Crossing-time difference")
    panel(axes[0],"a",r"Finite-band reference convergence, $t\leq125$")
    panel(axes[1],"b",r"Halving threshold sampling from $\Delta t\leq0.01$")
    axes[0].legend()
    save(fig,"convergence_checks")
    log("All revised PDF/SVG/PNG figures exported")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute-only",action="store_true")
    parser.add_argument("--plot-only",action="store_true")
    args=parser.parse_args()
    if not args.plot_only: compute()
    if not args.compute_only: plot()


if __name__=="__main__":main()
