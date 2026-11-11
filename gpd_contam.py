import numpy as np, json, sys
from scipy.optimize import brentq, minimize
from numpy.polynomial.laguerre import laggauss
GLx, GLw = laggauss(64)
def gen(name):
    if name=="quad": return (lambda x:x, lambda y:y)
    return (np.arcsinh, np.sinh)        # asinh generator, s = 1
def q1(x,p):                            # type-1 empirical quantile
    xs=np.sort(x); return xs[int(np.ceil(len(x)*p-1e-9))-1]
def emp_sq(x,p,g):
    g1,ginv=g; xs=np.sort(x); m=len(xs); k=int(np.ceil(m*p-1e-9)); q=xs[k-1]
    T=((k-m*p)*g1(q)+g1(xs[k:]).sum())/(m*(1-p)); return q, ginv(T)
def mle(x):
    def nll(par):
        xi,sg=par[0],np.exp(par[1]); xi=1e-6 if abs(xi)<1e-6 else xi
        z=1+xi*x/sg
        if np.any(z<=0): return 1e10
        return len(x)*np.log(sg)+(1/xi+1)*np.log(z).sum()
    best=None
    for s0 in ([0.2,np.log(x.mean()*0.8)],[0.5,np.log(np.median(x))],[0.05,np.log(x.mean())]):
        r=minimize(nll,s0,method="Nelder-Mead",options=dict(maxiter=3000,xatol=1e-9,fatol=1e-11))
        if r.fun<1e9 and (best is None or r.fun<best.fun): best=r
    return (np.nan,np.nan) if best is None else (best.x[0],np.exp(best.x[1]))
def hill(x):
    k=int(np.floor(np.sqrt(len(x)))); xs=np.sort(x[x>0])[::-1]
    return np.mean(np.log(xs[:k]))-np.log(xs[k]), np.nan
def hfun(y,xi): return -np.log(y)/xi-(1+xi)/xi**2*(1-y**xi)
def p_star(xi):
    us=(1+xi)**(-1/xi); return 1-brentq(lambda u:hfun(u,xi),1e-300,us,xtol=1e-14)
def xi_max_for(p):                       # largest xi with p < p*(xi)  (p* decreasing in xi)
    if p_star(0.99)>p: return 0.99
    return brentq(lambda xi:p_star(xi)-p,0.02,0.99)
def scorequant(x,p,ximax):
    r=1-p; qh=q1(x,p)
    def zp(xi):
        y1=brentq(lambda y:hfun(y,xi)-hfun(y+p,xi),1e-12,r-1e-12,xtol=1e-13); return hfun(y1,xi)
    def eq(xi):
        sg=xi*qh/(r**(-xi)-1)
        t2=np.log1p(xi*x/sg)/xi**2-(1+xi)*x/(sg*xi+xi**2*x)
        return q1(t2,p)-zp(xi)
    try: xi=brentq(eq,0.02,ximax-1e-6,xtol=1e-8)
    except Exception: return np.nan,np.nan
    return xi, xi*qh/(r**(-xi)-1)
def model_sq(xi,sg,p,g):
    r=1-p; q=sg/xi*(r**(-xi)-1); s=sg+xi*q
    if g[0](2.0)==2.0: return (sg+q)/(1-xi)
    y=q+s/xi*(np.exp(xi*GLx)-1); return g[1](np.sum(GLw*g[0](y)))
def match(qh,sh,p,g,hi):
    r=1-p
    if qh<=0: return np.nan,np.nan
    sig=lambda xi: xi*qh/(r**(-xi)-1)
    try: xi=brentq(lambda xi:model_sq(xi,sig(xi),p,g)-sh,0.01,hi,xtol=1e-9)
    except Exception: return np.nan,np.nan
    return xi,sig(xi)
def run(xi0,eps,n=200,p=0.8,M=1000,seed=1):
    rng=np.random.default_rng(seed+int(1e4*xi0)+int(100*eps))
    ximax=xi_max_for(p); res={k:[] for k in ["Bregman (asinh) matching","CVaR matching","Hill","MLE","Score-quantile"]}
    gq,ga=gen("quad"),gen("asinh")
    for m in range(M):
        u=rng.random(n); x=(u**(-xi0)-1)/xi0
        if eps>0: x[rng.random(n)<eps]=20.0
        qh,sh=emp_sq(x,p,ga); res["Bregman (asinh) matching"].append(match(qh,sh,p,ga,1.5))
        qh,sh=emp_sq(x,p,gq); res["CVaR matching"].append(match(qh,sh,p,gq,0.98))
        res["Hill"].append(hill(x)); res["MLE"].append(mle(x)); res["Score-quantile"].append(scorequant(x,p,ximax))
    out={}
    for k,v in res.items():
        a=np.array(v,float); ok=np.isfinite(a[:,0])
        e=a[ok,0]-xi0; s=a[ok,1]-1
        out[k]=dict(fail=1-ok.mean(),bias=e.mean(),rmse=np.sqrt((e**2).mean()),
                    rmse_s=np.sqrt(np.nanmean(s**2)) if np.isfinite(s).any() else None)
    return out
if __name__=="__main__":
    p=float(sys.argv[1]) if len(sys.argv)>1 else 0.8
    print("p*(0.1)=%.4f p*(0.5)=%.4f p*(0.99)=%.4f ximax(p)=%.4f"%(p_star(0.1),p_star(0.5),p_star(0.99),xi_max_for(p)))
    R={}
    for eps in (0.0,0.05):
        for xi0 in (0.1,0.5):
            R[f"{eps}_{xi0}"]=run(xi0,eps,p=p); print(eps,xi0,json.dumps(R[f"{eps}_{xi0}"]),flush=True)
    json.dump(R,open(f"contam_p{p}.json","w"),indent=1)
