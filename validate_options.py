"""Validate one options snapshot; reports distinguish market fit from Greek math.

Inputs use the same strict European snapshot CSV as the UI. Calibration holds
out complete strike/expiry call-put pairs, so the paired side cannot leak into
the fit. This is a cross-sectional check, not a strategy backtest.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from quantcore.options import clean_chain,calibrate,predict_surface,black_scholes,price,surface_diagnostics

GREEKS=('delta','gamma','vega','theta','rho')


def finite_differences(row,volatility):
    s,k,t,r,q=float(row.spot),float(row.strike),float(row.years),float(row.rate),float(row.dividend_yield)
    v=float(volatility);kind=row.kind
    f=lambda ss=s,tt=t,rr=r,vv=v:price(ss,k,tt,rr,vv,q,kind)
    # Relative spot bump; central maturity bump must remain before expiry.
    h=s*1e-4;eps=1e-5;dt=min(1e-5,t/100)
    return dict(delta=(f(ss=s+h)-f(ss=s-h))/(2*h),
        gamma=(f(ss=s+h)-2*f()+f(ss=s-h))/h**2,
        vega=(f(vv=v+eps)-f(vv=v-eps))/(2*eps)*.01,
        theta=-(f(tt=t+dt)-f(tt=t-dt))/(2*dt)/365,
        rho=(f(rr=r+eps)-f(rr=r-eps))/(2*eps)*.01)


def split_pairs(chain):
    """Every fifth strike held out per expiry; both call and put share a split."""
    if 'validation_split' in chain:
        if not set(chain.validation_split)<= {'train','holdout'}:
            raise ValueError('Unknown validation split')
        if (chain.groupby(['expiry','strike']).validation_split.nunique()>1).any():
            raise ValueError('Paired sides must share a validation split')
        train=chain.loc[chain.validation_split=='train'].copy()
        test=chain.loc[chain.validation_split=='holdout'].copy()
        if train.empty or test.empty:raise ValueError('Both training and holdout quotes required')
        return train,test
    holdout=np.zeros(len(chain),dtype=bool)
    for _,group in chain.groupby('expiry'):
        strikes=sorted(group.strike.unique())
        if len(strikes)<5:raise ValueError('Need five distinct strikes per expiry for a paired holdout')
        selected=set(strikes[2::5])
        holdout[group.index]=group.strike.isin(selected)
    return chain.loc[~holdout].copy(),chain.loc[holdout].copy()


def evaluate(frame,iterations=2000,*,method='legacy',degree=4,ridge=None):
    chain,rejected=clean_chain(frame)
    train,test=split_pairs(chain)
    model=calibrate(train,iterations=iterations,method=method,degree=degree,ridge=ridge)
    results=[]
    for split,rows in [('train',train),('holdout',test)]:
        try:fitted=predict_surface(model,rows.strike,rows.years)
        except ValueError:
            fitted=[]
            for _,row in rows.iterrows():
                try:fitted.append(float(predict_surface(model,row.strike,row.years)))
                except ValueError:fitted.append(np.nan)
        for (_,row),vol in zip(rows.iterrows(),fitted):
            # Quote-specific IV inversion is an arithmetic roundtrip, not an
            # independent market test. Holdout fitted-IV repricing is separate.
            observed=black_scholes(row.spot,row.strike,row.years,row.rate,row.iv,row.dividend_yield,row.kind)
            supported=bool(np.isfinite(vol))
            predicted=price(row.spot,row.strike,row.years,row.rate,float(vol),row.dividend_yield,row.kind) if supported else np.nan
            numeric=finite_differences(row,row.iv)
            fit_greeks=black_scholes(row.spot,row.strike,row.years,row.rate,float(vol),row.dividend_yield,row.kind) if supported else dict.fromkeys(GREEKS,np.nan)
            fit_numeric=finite_differences(row,float(vol)) if supported else dict.fromkeys(GREEKS,np.nan)
            record=dict(source_row=row.source_row,expiry=str(row.expiry),strike=row.strike,kind=row.kind,split=split,
                bid=row.bid,ask=row.ask,mid=row.mid,quote_iv=row.iv,fitted_iv=float(vol),fitted_price=predicted,
                fitted_price_error=predicted-row.mid,inside_spread=bool(row.bid<=predicted<=row.ask),
                supported=supported,
                iv_roundtrip_error=observed['price']-row.mid)
            for name in GREEKS:
                record[name]=observed[name];record[name+'_fd']=numeric[name]
                record[name+'_abs_error']=abs(observed[name]-numeric[name])
                record['fitted_'+name]=fit_greeks[name]
                record['fitted_'+name+'_abs_error']=abs(fit_greeks[name]-fit_numeric[name])
            results.append(record)
    detail=pd.DataFrame(results)
    metrics={}
    for split in ['train','holdout']:
        d=detail[detail.split==split]
        metrics[split]=dict(quotes=len(d),priced_quotes=int(d.supported.sum()),unpriced_quotes=int((~d.supported).sum()),coverage_fraction=float(d.supported.mean()),iv_rmse_vol_points=float(np.sqrt(np.nanmean((d.fitted_iv-d.quote_iv)**2)))*100,
            price_rmse=float(np.sqrt(np.nanmean(d.fitted_price_error**2))),inside_bid_ask_fraction=float(d.inside_spread.mean()))
    summary=dict(status='evaluated_supplied_snapshot',accepted=len(chain),rejected=len(rejected),
        snapshot=str(chain['asof'].iloc[0]),expiries=chain.expiry.nunique(),
        split='Every fifth sorted strike per expiry held out, including both call and put',metrics=metrics,
        iv_roundtrip_max_abs_price_error=float(detail.iv_roundtrip_error.abs().max()),
        greeks_finite_difference_max_abs_error={n:float(detail[n+'_abs_error'].max()) for n in GREEKS},
        fitted_greeks_finite_difference_max_abs_error={n:float(detail['fitted_'+n+'_abs_error'].max()) for n in GREEKS},
        greek_units=dict(delta='per underlying unit',gamma='delta per underlying unit',vega='per 1 vol percentage point',theta='per calendar day elapsed',rho='per 1 rate percentage point'),
        gradient=dict(steps=model['iterations'],converged=model['converged'],initial_loss=model['loss'][0],final_loss=model['loss'][-1]),
        grid_warnings=surface_diagnostics(model,chain),
        calibration_method=method,
        limitations=['Source authenticity and underlying/rate/carry provenance must be verified separately.',
            'Finite differences test model derivatives, not independent market Greek truth.',
            'One snapshot holdout is not walk-forward or strategy performance validation.',
            'No live-trading behavior is changed.'])
    summary['numerical_check_tolerances']=dict(iv_roundtrip_abs_price=1e-7,greek_abs_error=1e-5)
    summary['numerical_checks_passed']=bool(summary['iv_roundtrip_max_abs_price_error']<1e-7 and
        all(error<1e-5 for error in summary['greeks_finite_difference_max_abs_error'].values()) and
        all(error<1e-5 for error in summary['fitted_greeks_finite_difference_max_abs_error'].values()))
    return summary,detail,rejected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('snapshot',type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):p.error('Choose a fresh output directory')
    summary,detail,rejected=evaluate(pd.read_csv(args.snapshot))
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'validation.json').write_text(json.dumps(summary,indent=2))
    detail.to_csv(args.output/'quote-and-greek-checks.csv',index=False)
    rejected.to_csv(args.output/'rejections.csv',index=False)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
