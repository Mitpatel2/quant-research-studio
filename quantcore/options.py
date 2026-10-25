"""European Black-Scholes-Merton pricing, inversion, Greeks and calibration.

Greeks hold volatility fixed (sticky-strike local partial derivatives). Vega and
rho are per one percentage point; theta is per calendar day (ACT/365).
"""
import math
import numpy as np
import pandas as pd
from scipy.optimize import brentq


def _inputs(s, k, t, r, sigma, q, kind):
    if not all(math.isfinite(float(v)) for v in (s,k,t,r,sigma,q)):
        raise ValueError('All pricing inputs must be finite')
    if s <= 0 or k <= 0 or t < 0 or sigma < 0 or kind not in ('call','put'):
        raise ValueError('Positive spot/strike, nonnegative time/volatility and call/put required')


def cdf(x):
    return .5 * (1 + math.erf(x/math.sqrt(2)))


def price(s, k, t, r, sigma, q=0., kind='call'):
    return black_scholes(s,k,t,r,sigma,q,kind)['price']


def black_scholes(s, k, t, r, sigma, q=0., kind='call'):
    _inputs(s,k,t,r,sigma,q,kind)
    sign = 1 if kind == 'call' else -1
    ds, dk = s*math.exp(-q*t), k*math.exp(-r*t)
    if t == 0 or sigma == 0:
        return dict(price=max(sign*(ds-dk),0.), d1=None,d2=None,
                    delta=None,gamma=None,vega=None,theta=None,rho=None)
    root = math.sqrt(t)
    d1 = (math.log(s/k)+(r-q+.5*sigma*sigma)*t)/(sigma*root)
    d2 = d1-sigma*root
    density = math.exp(-.5*d1*d1)/math.sqrt(2*math.pi)
    value = sign*(ds*cdf(sign*d1)-dk*cdf(sign*d2))
    theta = (-ds*density*sigma/(2*root)
             - sign*r*dk*cdf(sign*d2)+sign*q*ds*cdf(sign*d1))/365
    return dict(price=value,d1=d1,d2=d2,
        delta=sign*math.exp(-q*t)*cdf(sign*d1),
        gamma=math.exp(-q*t)*density/(s*sigma*root),
        vega=ds*density*root*.01,theta=theta,
        rho=sign*dk*t*cdf(sign*d2)*.01)


def implied_vol(target, s, k, t, r, q=0., kind='call'):
    _inputs(s,k,t,r,.2,q,kind)
    if not math.isfinite(target) or t <= 0:
        raise ValueError('Finite option price and time before expiry required')
    lower=price(s,k,t,r,0,q,kind)
    upper=s*math.exp(-q*t) if kind=='call' else k*math.exp(-r*t)
    if target < lower-1e-9 or target >= upper:
        raise ValueError('Quote violates European pricing bounds')
    if abs(target-lower) <= 1e-9:
        return 0.
    f=lambda vol: price(s,k,t,r,vol,q,kind)-target
    if f(5.) < 0:
        raise ValueError('IV exceeds the supported 500% annual volatility bracket')
    return float(brentq(f,1e-10,5.,xtol=1e-12))


QUOTE_COLUMNS=['asof','quote_time','expiry','spot','strike','bid','ask','rate','dividend_yield','kind','exercise_style']


def clean_chain(frame):
    """Never silently coerce American options or future/stale quotes into data."""
    missing=set(QUOTE_COLUMNS)-set(frame.columns)
    if missing:
        raise ValueError('Missing quote columns: '+', '.join(sorted(missing)))
    if frame.empty:
        raise ValueError('No quotes in this file')
    if len(frame)>100000:
        raise ValueError('Use a snapshot with at most 100,000 quotes')
    data=frame.copy()
    for c in ['asof','quote_time','expiry']:
        # Require actual timezone information; do not assume exchange time.
        if not data[c].astype(str).str.contains(r'(?:Z|[+-]\d\d:\d\d)$',regex=True).all():
            raise ValueError(f'{c} must contain timezone-aware ISO timestamps')
        data[c]=pd.to_datetime(data[c],utc=True,errors='raise')
    if data['asof'].nunique()!=1:
        raise ValueError('Upload one as-of snapshot at a time')
    keys=['expiry','strike','kind']
    if data.duplicated(keys).any():
        raise ValueError('Duplicate expiry/strike/option type in snapshot')
    # One spot per snapshot; discount and carry assumptions may vary by expiry.
    for c in ['spot','strike','bid','ask','rate','dividend_yield']:
        data[c]=pd.to_numeric(data[c],errors='raise')
    if data.spot.nunique()!=1:
        raise ValueError('Use a coherent spot assumption for one snapshot')
    for c in ['rate','dividend_yield']:
        if (data.groupby('expiry')[c].nunique()!=1).any():
            raise ValueError(f'Use one coherent {c} assumption per expiry')
    accepted=[]; rejected=[]
    for index,row in data.iterrows():
        try:
            if not np.isfinite(row[['spot','strike','bid','ask','rate','dividend_yield']].astype(float)).all():
                raise ValueError('Nonfinite quote')
            age=(row['asof']-row['quote_time']).total_seconds()
            if age < 0 or age > 60:
                raise ValueError('Quote is future-dated or more than 60 seconds old')
            if row.exercise_style.lower()!='european':
                raise ValueError('Only verified European exercise is supported')
            if row.bid <= 0 or row.ask < row.bid:
                raise ValueError('Nonpositive bid or crossed quote')
            t=(row['expiry']-row['asof']).total_seconds()/(365*86400)
            mid=(row.bid+row.ask)/2
            iv=implied_vol(mid,row.spot,row.strike,t,row.rate,row.dividend_yield,row.kind)
            if iv<=0:
                raise ValueError('Zero-IV boundary is unsuitable for a smooth calibration')
            accepted.append(dict(row,source_row=str(index),years=t,mid=mid,iv=iv,
                                 weight=min(100.,max(.01,mid/(row.ask-row.bid+.01)))))
        except (ValueError,TypeError,AttributeError,OverflowError) as err:
            rejected.append(dict(row=int(index) if isinstance(index,int) else str(index),reason=str(err)))
    valid=pd.DataFrame(accepted)
    drop=[]
    if len(valid):
        for _,pair in valid.groupby(['expiry','strike']):
            if set(pair.kind)=={'call','put'}:
                call=pair.loc[pair.kind=='call'].iloc[0]
                put=pair.loc[pair.kind=='put'].iloc[0]
                expected=call.spot*math.exp(-call.dividend_yield*call.years)-call.strike*math.exp(-call.rate*call.years)
                # A realizable European parity value must fit inside the paired
                # bid/ask interval; comparing midprices alone would overreject.
                if not call.bid-put.ask-1e-7<=expected<=call.ask-put.bid+1e-7:
                    for index,record in pair.iterrows():
                        drop.append(index)
                        rejected.append(dict(row=record.source_row,reason='Put-call parity inconsistent with spot/rate/yield within bid-ask bounds'))
    return valid.drop(index=drop).reset_index(drop=True),pd.DataFrame(rejected)


def basis(strike, years, spot):
    x=np.log(np.asarray(strike,dtype=float)/spot)/.2
    y=np.sqrt(np.asarray(years,dtype=float))/.5
    x,y=np.broadcast_arrays(x,y)
    return np.stack([np.ones_like(x),x,x*x,y,x*y],axis=-1)


def calibrate_legacy(chain, iterations=600, learning_rate=.03, ridge=.001):
    """Weighted log-IV polynomial fitted by gradient descent with backtracking.

    Positive fitted volatility is enforced by exp(). No global absence of static
    arbitrage is promised; report separate diagnostics on observed maturities.
    """
    if len(chain)<10 or chain.expiry.nunique()<2:
        raise ValueError('Need at least 10 valid quotes across two expiries')
    if iterations<1 or not 0<learning_rate<=1 or ridge<0:
        raise ValueError('Invalid gradient descent settings')
    spot=float(chain.spot.iloc[0]); x=basis(chain.strike,chain.years,spot)
    target=np.log(chain.iv.to_numpy()); weights=chain.weight.to_numpy()
    weights=weights/weights.mean(); beta=np.zeros(x.shape[1]);beta[0]=np.mean(target)
    penalty=np.array([0,1,1,1,1.])
    def loss(b):
        return float(np.mean(weights*(x@b-target)**2)+ridge*np.sum(penalty*b*b))
    history=[loss(beta)]
    for _ in range(iterations):
        gradient=2*x.T@(weights*(x@beta-target))/len(x)+2*ridge*penalty*beta
        step=learning_rate
        for attempt in range(30):
            candidate=beta-step*gradient
            if loss(candidate)<=history[-1]+1e-14:break
            step*=.5
        else:break
        beta=candidate;history.append(loss(beta))
        if np.linalg.norm(gradient)<1e-8:break
    fitted=np.exp(x@beta)
    return dict(method='legacy',beta=beta,spot=spot,loss=history,rmse_iv=float(np.sqrt(np.mean((fitted-chain.iv)**2))),
                iterations=len(history)-1,converged=float(np.linalg.norm(gradient))<1e-8,
                gradient_norm=float(np.linalg.norm(gradient)),fitted_iv=fitted)


def _predict_legacy(model, strike, years):
    logvol=basis(strike,years,model['spot'])@np.asarray(model['beta'])
    if np.any(np.abs(logvol)>20):
        raise ValueError('Surface extrapolation outside supported calibration range')
    return np.exp(logvol)


def calibrate(chain, iterations=600, learning_rate=.03, ridge=None, *, method='legacy', degree=4):
    """Select the preserved baseline or conditioned per-expiry smile fit.

    Corrected fitting minimizes a vega/spread-weighted log-IV loss. Full SVD
    preconditioning whitens its augmented design, including ridge. Actual
    gradient steps then optimize that loss; no analytic solution is substituted
    for the reported optimization history. Between expiries, interpolate total
    variance at fixed spot log-moneyness. No global arbitrage guarantee.
    """
    if method=='legacy':
        return calibrate_legacy(chain,iterations,learning_rate,.001 if ridge is None else ridge)
    if method!='corrected':raise ValueError('Unknown surface calibration method')
    ridge=1e-4 if ridge is None else float(ridge)
    if not isinstance(iterations,(int,np.integer)) or iterations<1 or not np.isfinite(learning_rate) or not 0<learning_rate<=1 or not np.isfinite(ridge) or ridge<0:
        raise ValueError('Invalid gradient descent settings')
    if not isinstance(degree,(int,np.integer)) or not 2<=degree<=5:
        raise ValueError('Smile degree must be an integer from 2 to 5')
    if len(chain)<10 or chain.expiry.nunique()<2:
        raise ValueError('Need at least 10 valid quotes across two expiries')
    spot=float(chain.spot.iloc[0]);slices=[];histories=[];norms=[]
    for expiry,rows in chain.groupby('expiry',sort=True):
        if rows.strike.nunique()<degree+2:
            raise ValueError(f'Each expiry needs at least {degree+2} distinct strikes')
        m=np.log(rows.strike.to_numpy()/spot);center=float(m.mean());scale=float(m.std(ddof=1))
        if scale<=0:raise ValueError('Distinct strike moneyness required')
        design=np.polynomial.chebyshev.chebvander((m-center)/scale,degree)
        target=np.log(rows.iv.to_numpy())
        sensitivity=np.array([black_scholes(r.spot,r.strike,r.years,r.rate,r.iv,r.dividend_yield,r.kind)['vega']*100*r.iv for _,r in rows.iterrows()])
        weights=(sensitivity/np.maximum((rows.ask.to_numpy()-rows.bid.to_numpy())/2,.05))**2
        if not np.isfinite(weights).all() or weights.sum()<=0:raise ValueError('Unstable price-sensitivity weights')
        weights=np.maximum(weights/weights.mean(),1e-12)
        penalty=np.diag([0.]+[1.]*degree)
        augmented=np.vstack([design*np.sqrt(weights[:,None]/len(rows)),math.sqrt(ridge)*penalty])
        response=np.r_[target*np.sqrt(weights/len(rows)),np.zeros(degree+1)]
        u,singular,vt=np.linalg.svd(augmented,full_matrices=False)
        if singular[-1]<=singular[0]*1e-12:raise ValueError('Rank-deficient smile design')
        preconditioner=vt.T/singular
        z=augmented@preconditioner
        initial=np.zeros(degree+1);initial[0]=np.average(target,weights=weights)
        gamma=singular*(vt@initial)
        loss=lambda g:float(np.sum((z@g-response)**2))
        history=[loss(gamma)]
        for _ in range(iterations):
            gradient=2*z.T@(z@gamma-response)
            if np.linalg.norm(gradient)<1e-8:break
            step=learning_rate
            for _ in range(30):
                candidate=gamma-step*gradient
                if loss(candidate)<=history[-1]-1e-4*step*float(gradient@gradient)+1e-16:break
                step*=.5
            else:break
            gamma=candidate;history.append(loss(gamma))
        norm=float(np.linalg.norm(2*z.T@(z@gamma-response)))
        unique=np.sort(np.unique(m));spacing=float(np.median(np.diff(unique)))
        slices.append(dict(expiry=str(expiry),years=float(rows.years.iloc[0]),center=center,scale=scale,
            beta=(preconditioner@gamma).tolist(),min_log_moneyness=float(m.min()),max_log_moneyness=float(m.max()),
            edge_allowance=spacing,design_condition=float(singular[0]/singular[-1])))
        histories.append(history);norms.append(norm)
    steps=max(len(h) for h in histories)-1
    history=np.mean([h+[h[-1]]*(steps+1-len(h)) for h in histories],axis=0).tolist()
    model=dict(method='corrected',spot=spot,slices=slices,beta=np.array([s['beta'] for s in slices]),degree=degree,ridge=ridge,
        elapsed_years=0.,loss=history,iterations=steps,converged=bool(max(norms)<1e-8),gradient_norm=max(norms),
        weighting='squared log-vol price sensitivity / max(half spread, 0.05), normalized per expiry',
        optimizer='SVD-preconditioned gradient descent with backtracking',interpolation='linear total variance at fixed spot log-moneyness')
    fitted=predict_surface(model,chain.strike,chain.years)
    model.update(fitted_iv=fitted,rmse_iv=float(np.sqrt(np.mean((fitted-chain.iv.to_numpy())**2))))
    return model


def advance_surface(model, spot, elapsed_years):
    """Roll a frozen surface with observed spot/time, without fitting future data."""
    if not np.isfinite(spot) or spot<=0 or not np.isfinite(elapsed_years) or elapsed_years<0:
        raise ValueError('Positive spot and nonnegative elapsed time required')
    return dict(model,spot=float(spot),elapsed_years=float(model.get('elapsed_years',0.)+elapsed_years))


def predict_surface(model, strike, years):
    if model.get('method','legacy')=='legacy':return _predict_legacy(model,strike,years)
    k,t=np.broadcast_arrays(np.asarray(strike,dtype=float),np.asarray(years,dtype=float))
    if not np.isfinite(k).all() or not np.isfinite(t).all() or np.any(k<=0) or np.any(t<=0):
        raise ValueError('Finite positive strike and maturity required')
    slices=model['slices'];anchors=np.array([s['years'] for s in slices])-model.get('elapsed_years',0.)
    if anchors.min()<=0 or np.any(t<anchors.min()-1e-12) or np.any(t>anchors.max()+1e-12):
        raise ValueError('Surface maturity is outside the observed range')
    m=np.log(k/model['spot']);flat=m.ravel();tf=t.ravel()
    hi=np.clip(np.searchsorted(anchors,tf,side='left'),0,len(anchors)-1)
    exact=np.isclose(tf,anchors[hi],rtol=0,atol=1e-12);lo=np.where(exact,hi,np.maximum(0,hi-1))
    w=[]
    for indices in [lo,hi]:
        values=np.empty(len(flat))
        for j,s in enumerate(slices):
            selected=indices==j
            if not selected.any():continue
            # At most one observed strike spacing of edge extension. Reported
            # validation includes every candidate and all domain rejections.
            if np.any(flat[selected]<s['min_log_moneyness']-s['edge_allowance']-1e-12) or np.any(flat[selected]>s['max_log_moneyness']+s['edge_allowance']+1e-12):
                raise ValueError('Strike is outside calibrated support plus one observed spacing')
            logvol=np.polynomial.chebyshev.chebval((flat[selected]-s['center'])/s['scale'],s['beta'])
            if np.any(np.abs(logvol)>20):raise ValueError('Unstable smile extrapolation')
            values[selected]=np.exp(2*logvol)*anchors[j]
        w.append(values)
    fraction=np.divide(tf-anchors[lo],anchors[hi]-anchors[lo],out=np.zeros_like(tf),where=hi!=lo)
    return np.sqrt(((1-fraction)*w[0]+fraction*w[1])/tf).reshape(k.shape)


def surface_strikes(model, chain, count=35):
    """Common observed strike support, avoiding unobserved corner wings in UI."""
    if model.get('method','legacy')=='legacy':return np.linspace(chain.strike.min(),chain.strike.max(),count)
    lower=max(s['min_log_moneyness'] for s in model['slices'])
    upper=min(s['max_log_moneyness'] for s in model['slices'])
    if lower>=upper:raise ValueError('Expiries have no common observed strike support')
    return np.linspace(model['spot']*math.exp(lower),model['spot']*math.exp(upper),count)


def surface_maturities(chain, count=30):
    """Render every observed expiry as a mesh row, including term-curve kinks."""
    anchors=np.unique(chain.years.to_numpy(dtype=float))
    return np.unique(np.r_[np.linspace(anchors.min(),anchors.max(),count),anchors])


def demo_chain():
    """Analytic fixture, explicitly not observed market quotes."""
    rows=[];s=100.;r=.04;q=.01;asof=pd.Timestamp('2026-06-01T14:00:00Z')
    for days in [14,30,60,90]:
        for k in np.linspace(80,120,13):
            t=days/365;vol=.21-.08*np.log(k/s)+.3*np.log(k/s)**2+.015*math.sqrt(t)
            for kind in ['call','put']:
                mid=price(s,k,t,r,vol,q,kind)
                if mid>.03:
                    width=min(.02,mid*.1)
                    rows.append(dict(asof=asof.isoformat(),quote_time=asof.isoformat(),
                        expiry=(asof+pd.Timedelta(days=days)).isoformat(),spot=s,strike=k,
                        bid=mid-width,ask=mid+width,rate=r,dividend_yield=q,kind=kind,
                        exercise_style='european'))
    return pd.DataFrame(rows)


def carry_at(chain, years):
    """Interpolate supplied expiry rate/carry assumptions; flat inputs stay flat.

    This is a display/pricing convention between observed maturities, not an
    independently observed yield curve. Never extrapolate beyond the chain.
    """
    curve=chain[['years','rate','dividend_yield']].drop_duplicates().sort_values('years')
    t=np.asarray(years,dtype=float)
    if np.any(t<curve.years.min()-1e-12) or np.any(t>curve.years.max()+1e-12):
        raise ValueError('Carry interpolation requires a maturity inside the observed range')
    return np.interp(t,curve.years,curve.rate),np.interp(t,curve.years,curve.dividend_yield)


def surface_diagnostics(model, chain):
    violations=[]
    maturities=sorted(chain.years.unique())
    if model.get('method')=='corrected':
        maturities=sorted(set(maturities)|set(np.linspace(min(maturities),max(maturities),41)))
    strikes=surface_strikes(model,chain,81 if model.get('method')=='corrected' else 41)
    s=model['spot']
    for t in maturities:
        rate,q=map(float,carry_at(chain,t))
        vols=predict_surface(model,strikes,t)
        calls=np.array([price(s,k,t,rate,v,q) for k,v in zip(strikes,vols)])
        if np.any(np.diff(calls)>1e-7):violations.append(f'Call strike monotonicity at {t:.4f} years')
        if np.any(np.diff(calls,2)<-1e-7):violations.append(f'Call convexity at {t:.4f} years')
    # Calendar total-variance check at fixed forward log-moneyness, not fixed strike.
    # Test only common observed support; a wing extrapolation error is not a
    # calendar-arbitrage finding. Preserve the legacy diagnostic for comparison.
    m_low,m_high=-.15,.15
    if model.get('method')=='corrected':
        low=max(s['min_log_moneyness'] for s in model['slices'])
        high=min(s['max_log_moneyness'] for s in model['slices'])
        carry=[(float(carry_at(chain,t)[0])-float(carry_at(chain,t)[1]))*t for t in maturities]
        m_low=max(low-c for c in carry);m_high=min(high-c for c in carry)
        if m_low>=m_high:return sorted(set(violations+['No common forward-moneyness support for calendar diagnostic']))
    for m in np.linspace(m_low,m_high,31 if model.get('method')=='corrected' else 15):
        total=[]
        for t in maturities:
            rate,q=map(float,carry_at(chain,t))
            total.append(float(predict_surface(model,s*math.exp((rate-q)*t+m),t))**2*t)
        if np.any(np.diff(total)<-1e-8):violations.append('Calendar total variance decreases')
    return sorted(set(violations))
