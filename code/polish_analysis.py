"""Additional finite-time comparisons for the polished submission.

The reference-only bound uses the prescribed exponential and bare spectral
nodes/weights. It does not diagonalize the Hamiltonian or evaluate its exact
amplitude. Exact trajectories are computed separately to validate that bound.
Run alone to regenerate only new files, or through generate_figures.py for all.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from revision_numerics import Bath

ROOT=Path(__file__).resolve().parents[1]
DATA,FIG,EVIDENCE=ROOT/'data',ROOT/'figures',ROOT/'verification'
RATIOS=(.01,.04,.1,.2)
GAMMAS=tuple(2*x for x in RATIOS)
TOLS=(.01,.02,.05,.1)
WIDTH,BANDWIDTH,N=2.,16.,256
HORIZON,PRE_HORIZON=125.,80.
COLORS=('#36648b','#54847b','#ba7446','#84608c')
BLUE,GREEN,ORANGE,PURPLE=COLORS
GRAY='#666666'


def configure():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
        'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.7,'lines.linewidth':1.4,
        'legend.frameon':False,'legend.fontsize':8,'xtick.labelsize':8,
        'ytick.labelsize':8,'axes.labelsize':9})


def write_csv(name,header,rows):
    with (DATA/name).open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(header);writer.writerows(rows)


def read_csv(name):
    return np.genfromtxt(DATA/name,delimiter=',',names=True,encoding='utf-8')


def cumulative(y,t):
    return np.r_[0.,np.cumsum((y[1:]+y[:-1])*np.diff(t)/2)]


def first_crossing(t,values,delta):
    hits=np.flatnonzero(values>delta)
    if not len(hits): return None
    j=hits[0]
    if j==0:return 0.
    return float(np.interp(delta,values[j-1:j+1],t[j-1:j+1]))


def bare_spectrum(n,gamma):
    """Kernel data only: no eigenproblem or target trajectory."""
    detuning=-BANDWIDTH/2+(np.arange(n)+.5)*BANDWIDTH/n
    g2=gamma/(2*np.pi)*WIDTH**2/(detuning**2+WIDTH**2)*BANDWIDTH/n
    return detuning,g2


def reference_only_bound(n,gamma,t):
    """d_M=m'+C*m, E=int|d_M|, population interval from |c-m|<=E.

    The exact homogeneous Volterra resolvent is c. Unitarity in the
    single-excitation sector gives |c|<=1, hence c-m=-c*d_M and |c-m|<=E.
    This function needs only bare_spectrum(), gamma, and the requested grid.
    """
    detuning,g2=bare_spectrum(n,gamma)
    a=gamma/2;m=np.exp(-a*t)
    d=-a*m.astype(complex)
    for start in range(0,len(t),1500):
        sl=slice(start,start+1500)
        d[sl]+=(g2/(a-1j*detuning))@(
            np.exp(-1j*detuning[:,None]*t[None,sl])-m[None,sl])
    E=cumulative(abs(d),t)
    upper=np.minimum(1.,m+E)**2-m*m
    lower=m*m-np.maximum(0.,m-E)**2
    bound=np.maximum(upper,lower)
    return {'m':m,'residual':abs(d),'amplitude_bound':E,
            'population_bound':bound}


def optimize_maximum(fun,lo,hi):
    """Golden-section local maximum in a sampled maximum's small bracket."""
    phi=(np.sqrt(5)-1)/2
    x1=hi-phi*(hi-lo);x2=lo+phi*(hi-lo)
    y1,y2=fun(x1),fun(x2)
    for _ in range(65):
        if y1<y2:
            lo=x1;x1=x2;y1=y2;x2=lo+phi*(hi-lo);y2=fun(x2)
        else:
            hi=x2;x2=x1;y2=y1;x1=hi-phi*(hi-lo);y1=fun(x1)
    candidates=[(x1,y1),(x2,y2),((lo+hi)/2,fun((lo+hi)/2))]
    return max(candidates,key=lambda item:item[1])


def spectral_first_crossing(bath,t,error,tol):
    hits=np.flatnonzero(error>tol)
    if not len(hits):return None
    j=hits[0];lo,hi=t[max(0,j-1)],t[j]
    for _ in range(48):
        mid=(lo+hi)/2
        if bath.probability_error(mid)>tol:hi=mid
        else:lo=mid
    return float((lo+hi)/2)


def compute():
    for folder in (DATA,FIG,EVIDENCE):folder.mkdir(exist_ok=True,parents=True)
    started=time.perf_counter()
    result={'parameters':{'N_B':N,'Lambda':WIDTH,'W':BANDWIDTH,
        'gamma_over_Lambda':RATIOS,'full_horizon':HORIZON,
        'transient_ceiling_horizon':PRE_HORIZON,
        't_recurrence':2*np.pi*N/BANDWIDTH},
        'scope':'Exact scalar convolution in the unitary single-excitation sector; excited initial state, uniform finite bath. No general superoperator or Born-error claim.',
        'prospective_inputs':'Bare detunings, g_squared, prescribed m(t)=exp(-gamma*t/2). No diagonalization or exact target trajectory in reference_only_bound().',
        'numerical_status':'Analytical inequality evaluated with converged quadrature; no interval-arithmetic certificate.'}
    coarse=np.linspace(0,HORIZON,50001)
    fine=np.linspace(0,HORIZON,100001)
    prospective={};checks=[];window_rows=[]
    start_bound=time.perf_counter()
    for ratio,gamma in zip(RATIOS,GAMMAS):
        bc=reference_only_bound(N,gamma,coarse)
        bf=reference_only_bound(N,gamma,fine)
        prospective[ratio]=(bc,bf)
        item={'gamma_over_Lambda':ratio,'coarse_dt':.0025,'fine_dt':.00125,
            'max_population_bound_difference':float(np.max(abs(bc['population_bound']-bf['population_bound'][::2]))),
            'max_amplitude_bound_difference':float(np.max(abs(bc['amplitude_bound']-bf['amplitude_bound'][::2]))),
            'endpoints':{}}
        for tol in TOLS:
            tc=first_crossing(coarse,bc['population_bound'],tol)
            tf=first_crossing(fine,bf['population_bound'],tol)
            item['endpoints'][str(tol)]={'coarse':tc,'fine':tf,
                'absolute_difference':None if tc is None or tf is None else abs(tc-tf)}
        checks.append(item)
    result['reference_only_compute_seconds']=time.perf_counter()-start_bound
    result['reference_only_convergence']=checks
    # Target trajectory is used below only for validation and actual crossings.
    start_validation=time.perf_counter();baths={};actuals={}
    for ratio,gamma,check in zip(RATIOS,GAMMAS,checks):
        bath=Bath(N,gamma);baths[ratio]=bath
        c=bath.evaluate(coarse,correlation=False)[0]
        actual=abs(abs(c)**2-np.exp(-gamma*coarse));actuals[ratio]=actual
        bound=prospective[ratio][1]['population_bound'][::2]
        check['max_actual_population_error_minus_bound']=float(np.max(actual-bound))
        for tol in TOLS:
            ta=spectral_first_crossing(bath,coarse,actual,tol)
            tf=check['endpoints'][str(tol)]['fine']
            window_rows.append([N,ratio,gamma,tol,HORIZON,tf,ta,
                None if tf is None or ta is None else tf/ta])
        if check['max_actual_population_error_minus_bound']>1e-8:
            raise RuntimeError('Reference-only numerical inequality check failed')
    result['exact_validation_compute_seconds']=time.perf_counter()-start_validation
    write_csv('reference_only_window_scan.csv',
        ['N_B','gamma_over_Lambda','gamma','tolerance','horizon',
         'reference_only_bound_end','observed_first_exit','bound_end_over_observed'],window_rows)
    result['reference_only_windows']=[dict(zip(
        ['N_B','gamma_over_Lambda','gamma','tolerance','horizon','bound_end','observed_exit','ratio'],row)) for row in window_rows]
    old=read_csv('finite_bath_residual.csv')
    if len(old)!=len(coarse) or not np.allclose(old['time'],coarse):
        raise ValueError('Baseline source residual grid does not match expected grid')
    bf=prospective[.04][1];bc=prospective[.04][0]
    simple1=(1+np.sqrt(old['P_cutoff']))*old['weighted_total_bound']+old['rate_mismatch']
    simple2=2*old['weighted_total_bound']+old['rate_mismatch']
    result['baseline_prefactor_comparison']={
        'exact_amplitude_prefactor':first_crossing(coarse,old['population_markov_bound'],.05),
        'one_plus_cutoff_amplitude':first_crossing(coarse,simple1,.05),
        'constant_two':first_crossing(coarse,simple2,.05),
        'note':'Only population prefactor changed; these older cutoff estimates retain the exact-trajectory speed.'}
    write_csv('reference_only_certificate.csv',
        ['time','reference_residual_abs','reference_amplitude_bound',
         'reference_population_bound','reference_population_bound_coarse',
         'actual_population_error','aposteriori_population_bound',
         'aposteriori_bound_one_plus_cc','aposteriori_bound_two'],
        zip(coarse,bf['residual'][::2],bf['amplitude_bound'][::2],
            bf['population_bound'][::2],bc['population_bound'],actuals[.04],
            old['population_markov_bound'],simple1,simple2))
    # Continuous finite-band reference: same pre-rephasing horizon for all gamma.
    # The cutoff at 80 is 0.7958 t_rec(N=256); the refined maxima are all early.
    tpre=np.linspace(0,PRE_HORIZON,8001)
    ceil_rows=[];transition_rows=[];scaling_rows=[];transient_curves=[]
    for ratio,gamma in zip(RATIOS,GAMMAS):
        ref=Bath(1024,gamma,scheme='gauss')
        checkref=Bath(768,gamma,scheme='gauss')
        c=ref.evaluate(tpre,correlation=False)[0]
        ccheck=checkref.evaluate(tpre,correlation=False)[0]
        error=abs(abs(c)**2-np.exp(-gamma*tpre))
        j=int(np.argmax(error))
        peak,ceiling=optimize_maximum(ref.probability_error,tpre[j-1],tpre[j+1])
        bath=baths[ratio]
        finite_c=bath.evaluate(tpre,correlation=False)[0]
        finite_grid_error=float(np.max(abs(abs(finite_c)**2-abs(c)**2)))
        finite_early=actuals[ratio][coarse<=PRE_HORIZON]
        jfinite=int(np.argmax(finite_early))
        finite_peak,finite_ceiling=optimize_maximum(bath.probability_error,
            coarse[jfinite-1],coarse[jfinite+1])
        quad_diff=float(np.max(abs(abs(c)**2-abs(ccheck)**2)))
        ceil_rows.append([ratio,gamma,PRE_HORIZON,peak,ceiling,
            finite_peak,finite_ceiling,quad_diff,finite_grid_error])
        transient_curves.extend(zip(np.repeat(ratio,len(tpre)),tpre,error))
        # Sampling includes the refined early maximum, preventing a missed
        # very narrow near-maximum excursion in this deliberately dense scan.
        tx=np.sort(np.r_[coarse,finite_peak])
        ex=np.insert(actuals[ratio],np.searchsorted(coarse,finite_peak),finite_ceiling)
        factors=np.unique(np.r_[np.linspace(.1,.98,55),
            1-np.geomspace(1e-6,.02,22),1+np.geomspace(1e-6,.02,22),
            np.linspace(1.02,1.5,30)])
        for factor in factors:
            tol=finite_ceiling*factor
            tc=spectral_first_crossing(bath,tx,ex,tol)
            # Different branches are plotted independently, never joined.
            branch=0 if tol<finite_ceiling else 1
            transition_rows.append([N,ratio,gamma,tol,tol/ceiling,
                tc,branch,ceiling,finite_ceiling])
        g2sum=float(bare_spectrum(N,gamma)[1].sum())
        for frac in np.geomspace(1e-4,.2,32):
            tol=ceiling*frac
            tc=spectral_first_crossing(bath,tx,ex,tol)
            scaling_rows.append([ratio,gamma,tol,tol/ceiling,tc,
                gamma*tc/tol,1+(g2sum+gamma**2/2)*tol/gamma**2])
        if quad_diff>1e-9:raise RuntimeError('Continuum ceiling convergence failed')
        print(f'gamma/Lambda={ratio:g}: delta_star={ceiling:.10g}, peak={peak:.6g}, Gauss difference={quad_diff:.2g}',flush=True)
    write_csv('continuum_transient_ceiling.csv',
        ['gamma_over_Lambda','gamma','horizon','peak_time','delta_star',
         'finite_N256_peak_time','finite_N256_delta_star','gauss768_to1024_max_population_difference',
         'finite_N256_to_continuum_max_population_difference'],ceil_rows)
    write_csv('continuum_transient_curves.csv',['gamma_over_Lambda','time','continuum_markov_error'],transient_curves)
    write_csv('dense_tolerance_transition.csv',
        ['N_B','gamma_over_Lambda','gamma','tolerance','tolerance_over_continuum_ceiling',
         't_first_exit','branch_0_early_1_late','continuum_delta_star','finite_N256_delta_star'],transition_rows)
    write_csv('short_time_scaling.csv',
        ['gamma_over_Lambda','gamma','tolerance','tolerance_over_delta_star',
         't_first_exit','gamma_t_exit_over_delta','first_order_ratio'],scaling_rows)
    result['continuum_transient_ceiling']=[dict(zip(
        ['gamma_over_Lambda','gamma','horizon','peak_time','delta_star',
         'finite_peak_time','finite_delta_star','gauss_max_population_difference',
         'finite_N256_to_continuum_max_population_difference_0_to80'],row)) for row in ceil_rows]
    result['short_time_scaling']={'minimum_normalized_tolerance':1e-4,
        'maximum_normalized_tolerance':.2,
        'minimum_tolerance_ratios':{str(r):float(next(row[-2] for row in scaling_rows if row[0]==r)) for r in RATIOS},
        'expansion':'D(t)=gamma*t-(sum(g2)+gamma^2/2)*t^2+O(t^3); t_delta=delta/gamma+O(delta^2).'}
    result['time_seconds']=time.perf_counter()-started
    (EVIDENCE/'numeric_results_polish.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (DATA/'numeric_results_polish.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(f'New analyses completed in {result["time_seconds"]:.2f} s',flush=True)


def panel(ax,letter,title):
    ax.text(-.14,1.08,letter,transform=ax.transAxes,weight='bold',fontsize=10)
    ax.set_title(title,loc='left',fontsize=9.5,pad=10)
    ax.grid(alpha=.15,linewidth=.5)


def save(fig,name):
    for ext in ('pdf','svg','png'):
        fig.savefig(FIG/f'{name}.{ext}',bbox_inches='tight',dpi=400)
    plt.close(fig)


def plot():
    configure()
    data=read_csv('reference_only_certificate.csv')
    windows=read_csv('reference_only_window_scan.csv')
    result=json.loads((DATA/'numeric_results_polish.json').read_text(encoding='utf-8'))
    fig,axes=plt.subplots(2,2,figsize=(8,5.9),layout='constrained')
    for ax in (axes[0,0],axes[0,1],axes[1,0]):
        ax.plot(data['time'],data['actual_population_error'],color=BLUE,label='Observed error')
        ax.plot(data['time'],data['reference_population_bound'],color=GREEN,ls='--',label='Reference-only bound')
        ax.plot(data['time'],data['aposteriori_population_bound'],color=ORANGE,ls=':',label='Cutoff a posteriori bound')
        ax.axhline(.05,color=GRAY,ls='-.',lw=.8)
        ax.set(xlabel='Time, $t$',ylabel='Population error / bound')
    axes[0,0].set(xlim=(0,125),yscale='log',ylim=(1e-5,1.2));axes[0,0].legend(loc='lower left')
    panel(axes[0,0],'a','Full finite-time comparison')
    axes[0,1].set(xlim=(0,3),ylim=(0,.055));axes[0,1].axhline(.01,color=GRAY,ls='--',lw=.7)
    axes[0,1].text(2.85,.0108,'1%',ha='right',color=GRAY,fontsize=8)
    axes[0,1].text(2.85,.0508,'5%',ha='right',color=GRAY,fontsize=8)
    panel(axes[0,1],'b','Initial memory transient')
    axes[1,0].set(xlim=(100,104.2),ylim=(0,.087))
    baseline=[x for x in result['reference_only_windows'] if x['gamma_over_Lambda']==.04 and x['tolerance']==.05][0]
    endpoints=[(result['baseline_prefactor_comparison']['exact_amplitude_prefactor'],ORANGE),
        (baseline['bound_end'],GREEN),(baseline['observed_exit'],BLUE)]
    for x,color in endpoints:axes[1,0].plot(x,.05,'o',color=color,ms=4)
    axes[1,0].text(100.1,.077,'5% endpoints: cutoff 102.58;\nreference-only 102.776; observed 103.387',fontsize=7.5)
    panel(axes[1,0],'c','Resolving the 5% endpoints')
    w=windows[np.isclose(windows['tolerance'],.05)]
    axes[1,1].plot(w['gamma_over_Lambda'],w['observed_first_exit'],'o',ms=6,color=BLUE,label='Observed first exit')
    axes[1,1].plot(w['gamma_over_Lambda'],w['reference_only_bound_end'],'x',ms=7,mew=1.6,color=GREEN,label='Reference-only bound end')
    axes[1,1].set(xlabel=r'Coupling, $\gamma/\Lambda$',ylabel='5% interval endpoint',yscale='log')
    axes[1,1].set_xticks(RATIOS);axes[1,1].set_xticklabels(['0.01','0.04','0.10','0.20'])
    axes[1,1].legend(loc='center right')
    panel(axes[1,1],'d','Comparison across four couplings')
    save(fig,'prospective_certificate')

    trans=read_csv('dense_tolerance_transition.csv')
    ceilings=read_csv('continuum_transient_ceiling.csv')
    fig,axes=plt.subplots(2,2,figsize=(8,5.5),layout='constrained',sharey=True)
    for ax,ratio,color,letter in zip(axes.flat,RATIOS,COLORS,'abcd'):
        rows=trans[np.isclose(trans['gamma_over_Lambda'],ratio)]
        for branch,ls in [(0,'-'),(1,'--')]:
            rr=rows[rows['branch_0_early_1_late']==branch]
            ax.plot(100*rr['tolerance'],rr['t_first_exit'],ls=ls,color=color,lw=1.6,
                label='Initial-transient branch' if branch==0 else 'Rephasing branch')
        cr=ceilings[np.isclose(ceilings['gamma_over_Lambda'],ratio)][0]
        ax.axvline(100*cr['delta_star'],color=GRAY,ls=':',lw=1)
        ax.axhline(2*np.pi*N/BANDWIDTH,color=GRAY,ls=':',lw=.8)
        ax.text(.98,.37,fr'$\delta_*={100*cr["delta_star"]:.3f}\%$',transform=ax.transAxes,ha='right',fontsize=9)
        ax.set(xlabel=r'Population-error tolerance, $\delta$ (%)',yscale='log',ylim=(.025,150))
        panel(ax,letter,fr'$\gamma/\Lambda={ratio:g}$, $N_B=256$')
    axes[0,0].legend(loc='center left',fontsize=7.5)
    axes[0,0].set_ylabel(r'First-exit time, $t_\delta$');axes[1,0].set_ylabel(r'First-exit time, $t_\delta$')
    save(fig,'tolerance_transition')

    curves=read_csv('continuum_transient_curves.csv');sc=read_csv('short_time_scaling.csv')
    fig,axes=plt.subplots(1,2,figsize=(8,3.1),layout='constrained')
    for ratio,color in zip(RATIOS,COLORS):
        c=curves[np.isclose(curves['gamma_over_Lambda'],ratio)]
        axes[0].plot(c['time'],c['continuum_markov_error'],color=color,label=fr'$\gamma/\Lambda={ratio:g}$')
        s=sc[np.isclose(sc['gamma_over_Lambda'],ratio)]
        axes[1].plot(s['tolerance_over_delta_star'],s['gamma_t_exit_over_delta'],color=color)
    axes[0].set(xlim=(0,6),xlabel='Time, $t$',ylabel=r'$|P_W-P_M|$');axes[0].legend()
    axes[1].axhline(1,color=GRAY,ls=':',lw=1)
    axes[1].set(xscale='log',xlabel=r'Relative tolerance, $\delta/\delta_*$',ylabel=r'$\gamma t_\delta/\delta$')
    panel(axes[0],'a','Continuous-reservoir transient')
    panel(axes[1],'b',r'Small-tolerance limit: $t_\delta\sim\delta/\gamma$')
    save(fig,'short_time_scaling')
    print('New PDF/SVG/PNG figures exported',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compute-only',action='store_true')
    parser.add_argument('--plot-only',action='store_true')
    args=parser.parse_args()
    if not args.plot_only:compute()
    if not args.compute_only:plot()


if __name__=='__main__':main()
