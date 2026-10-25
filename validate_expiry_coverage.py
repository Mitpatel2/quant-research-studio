"""Observed-tenor and leave-expiry-out checks from the existing private cache."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import databento as db
from quantcore.options import clean_chain,calibrate,carry_at,predict_surface,price,surface_diagnostics
from quantcore.databento_options import normalize_snapshot
from validate_options import evaluate,split_pairs
from quantcore.options import advance_surface
from run_surface_retest import predictions_with_domain,price_rows,json_default

# Exclude published 2026 Cboe holidays/early closes and New Year's Day 2027.
# Every remaining weekday in the restricted Nov 6–Jan 15 interval is full RTH.
EXCLUDED={'2026-11-26','2026-11-27','2026-12-24','2026-12-25','2027-01-01'}
HELD_DATES=['2026-11-13','2026-12-04','2026-12-11']


def available_dates(definitions,asof):
    known=definitions.loc[definitions.index<=pd.Timestamp(asof)]
    return sorted(d for d in set(known.expiration.dt.strftime('%Y-%m-%d')) if '2026-11-06'<=d<='2027-01-15' and d not in EXCLUDED and pd.Timestamp(d).dayofweek<5)


def main():
    root=Path(__file__).parent;cache=root/'data/databento/2026-10-23-spxw';out=root/'reports/surface-fit-retest'
    plan=json.loads((out/'selection-plan.json').read_text())
    config=dict(degree=plan['selected_degree'],ridge=plan['selected_ridge'])
    frames={k:db.DBNStore.from_file(cache/f).to_df() for k,f in {'quotes':'opra-pillar-cbbo-1m.dbn.zst','definitions':'opra-pillar-definition.dbn.zst','underlying':'dbix-core-trades.dbn.zst'}.items()}
    diagnostics=[];full={};matched_checks=[]
    for minute in [35,39]:
        asof=f'2026-10-23T14:{minute}:00Z';dates=available_dates(frames['definitions'],asof)
        raw,excluded,provenance=normalize_snapshot(**frames,asof=asof,expiry_dates=dates)
        raw.to_csv(cache/f'snapshot-dense-14{minute}.csv',index=False)
        full[minute]=raw
        details={}
        for method in ['legacy','corrected']:
            summary,detail,rejected=evaluate(raw,method=method,**(config if method=='corrected' else {}))
            summary['provenance']=provenance;summary['normalization_rejections']=len(excluded)
            (out/f'dense-14{minute}-{method}.json').write_text(json.dumps(summary,indent=2))
            detail.to_csv(out/f'dense-14{minute}-{method}.csv',index=False)
            rejected.to_csv(out/f'dense-14{minute}-{method}-rejections.csv',index=False)
            diagnostics.append(dict(asof=asof,method=method,expiries=summary['expiries'],accepted=summary['accepted'],**summary['metrics']['holdout'],converged=summary['gradient']['converged'],grid_warnings=summary['grid_warnings'],numerical_checks_passed=summary['numerical_checks_passed']))
            details[method]=detail
        matched=details['corrected'].loc[(details['corrected'].split=='holdout')&details['corrected'].supported,'source_row']
        entry=dict(asof=asof,matched_quotes=len(matched))
        for method in ['legacy','corrected']:
            d=details[method].loc[details[method].source_row.isin(matched)]
            entry[method]=dict(price_rmse=float(np.sqrt(np.mean(d.fitted_price_error**2))),inside_bid_ask_fraction=float(d.inside_spread.mean()))
        matched_checks.append(entry)
    dense_temporal=[]
    for minute in [31,33,35,37]:
        asof=f'2026-10-23T14:{minute}:00Z';next_asof=f'2026-10-23T14:{minute+1}:00Z'
        dates=available_dates(frames['definitions'],asof)
        raw,training_excluded,provenance=normalize_snapshot(**frames,asof=asof,expiry_dates=dates,skip_unidentifiable=True)
        chain,_=clean_chain(raw);train,_=split_pairs(chain)
        curve={p['expiry']:p for p in provenance['carry']}
        eligible_dates=[pd.Timestamp(p['expiry']).date().isoformat() for p in provenance['carry']]
        later,excluded,_=normalize_snapshot(**frames,asof=next_asof,expiry_dates=eligible_dates,carry_parameters=curve)
        later['years']=(pd.to_datetime(later.expiry,utc=True)-pd.Timestamp(next_asof)).dt.total_seconds()/(365*86400)
        prices={};errors={}
        for method in ['legacy','corrected']:
            m=advance_surface(calibrate(train,2000,method=method,**(config if method=='corrected' else {})),float(later.spot.iloc[0]),1/(365*1440))
            v,errors[method]=predictions_with_domain(m,later);prices[method]=price_rows(later,v)
        matched=np.isfinite(prices['legacy'])&np.isfinite(prices['corrected']);mid=((later.bid+later.ask)/2).to_numpy()
        entry=dict(train_asof=asof,test_asof=next_asof,candidates=len(later),matched_quotes=int(matched.sum()),coverage=float(matched.mean()),domain_rejections=errors['corrected'],training_expiries=len(eligible_dates),excluded_training_expiries=provenance['unsupported_expiries'],training_normalization_rejections=len(training_excluded))
        for method in ['legacy','corrected']:
            v=prices[method][matched]
            entry[method]=dict(price_rmse=float(np.sqrt(np.mean((v-mid[matched])**2))),inside_bid_ask_fraction=float(((v>=later.bid.to_numpy()[matched])&(v<=later.ask.to_numpy()[matched])).mean()))
        dense_temporal.append(entry)
    # Entire target expiry omitted from both rate/carry and volatility fitting.
    # Basic-quality target prices are retained even if interpolated carry would
    # fail parity. No target quote trains the fitted model or its inputs.
    tenor=[];detail=[]
    for date in HELD_DATES:
        train_raw=full[35].loc[~full[35].expiry.str.startswith(date)]
        train,_=clean_chain(train_raw)
        target=full[35].loc[full[35].expiry.str.startswith(date)].copy()
        target['years']=(pd.to_datetime(target.expiry,utc=True)-pd.Timestamp('2026-10-23T14:35Z')).dt.total_seconds()/(365*86400)
        target['rate'],target['dividend_yield']=carry_at(train,target.years)
        prices={};failures={}
        for method in ['legacy','corrected']:
            m=calibrate(train,2000,method=method,**(config if method=='corrected' else {}))
            vols,failures[method]=predictions_with_domain(m,target)
            prices[method]=price_rows(target,vols)
        matched=np.isfinite(prices['legacy'])&np.isfinite(prices['corrected'])
        mid=((target.bid+target.ask)/2).to_numpy()
        result=dict(held_expiry=date,candidates=len(target),matched_candidates=int(matched.sum()),coverage=float(matched.mean()),corrected_domain_rejections=failures['corrected'])
        for method in ['legacy','corrected']:
            pv=prices[method][matched]
            result[method]=dict(price_rmse=float(np.sqrt(np.mean((pv-mid[matched])**2))),inside_bid_ask_fraction=float(((pv>=target.bid.to_numpy()[matched])&(pv<=target.ask.to_numpy()[matched])).mean()))
        result['meets_80_percent_spread_gate']=result['corrected']['inside_bid_ask_fraction']>=.8
        tenor.append(result);detail.append(target.assign(legacy_price=prices['legacy'],corrected_price=prices['corrected'],matched=matched))
    pd.concat(detail).to_csv(out/'leave-expiry-out-predictions.csv',index=False)
    result=dict(config=config,observed_tenor_checks=diagnostics,matched_strike_checks=matched_checks,dense_temporal=dense_temporal,leave_expiry_out=tenor,
        interpolation_validated=all(r['meets_80_percent_spread_gate'] for r in tenor),
        limitations=['Leave-expiry-out prices include interpolated rate/carry error as well as volatility interpolation error.',
            'Observed tenor fits do not justify prices at missing expiries; interpolation remains a modeled approximation.',
            'The extended chain uses only already purchased data; no network or trading action.'],
        calendar_source='https://www.cboe.com/about/hours/us-options',excluded_dates=sorted(EXCLUDED))
    (out/'expiry-coverage.json').write_text(json.dumps(result,indent=2))
    corrected=[r for r in diagnostics if r['method']=='corrected']
    observed_pass=all(r['coverage_fraction']>=.98 and r['inside_bid_ask_fraction']>=.8 and r['converged'] and r['numerical_checks_passed'] and not r['grid_warnings'] for r in corrected)
    (root/'reports/databento-status.json').write_text(json.dumps(dict(status='observed_expiry_fit_improved_interpolation_not_validated',observed_expiry_checks_passed=observed_pass,unseen_expiry_interpolation_validated=result['interpolation_validated'],live_trading_promotion_eligible=False,live_trading_changed=False,config=config,observed_expiry_checks=corrected,report_directory=str(out),new_data_cost_dollars=0),indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
