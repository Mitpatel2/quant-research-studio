"""Offline training-only parameter selection. No network or trading actions."""
import sys,json,argparse
from pathlib import Path
import numpy as np,pandas as pd,databento as db
p=Path(__file__).resolve().parent;sys.path.insert(0,str(p))
parser=argparse.ArgumentParser(description='Reproduce training-only calibration selection from cached data')
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
from quantcore.options import black_scholes,clean_chain,price
from quantcore.databento_options import normalize_snapshot
from validate_options import split_pairs
cache=p/'data/databento/2026-10-23-spxw'
frames={k:db.DBNStore.from_file(cache/f).to_df() for k,f in {'quotes':'opra-pillar-cbbo-1m.dbn.zst','definitions':'opra-pillar-definition.dbn.zst','underlying':'dbix-core-trades.dbn.zst'}.items()}
snapshot,_,prov=normalize_snapshot(**frames,asof='2026-10-23T14:31Z',expiry_dates=['2026-11-06','2026-11-20','2026-12-18','2027-01-15'])
chain,_=clean_chain(snapshot);train,_=split_pairs(chain)
def fit_predict(a,b,degree,ridge):
 vals=[]
 for expiry,rows in a.groupby('expiry'):
  test=b[b.expiry==expiry]
  center=float(np.log(rows.strike/rows.spot).mean());scale=float(np.log(rows.strike/rows.spot).std())
  x=(np.log(rows.strike/rows.spot)-center)/scale;v=np.polynomial.chebyshev.chebvander(x,degree)
  # Linearized price residual in units of half the quoted spread.
  sens=np.array([black_scholes(r.spot,r.strike,r.years,r.rate,r.iv,r.dividend_yield,r.kind)['vega']*100*r.iv for _,r in rows.iterrows()])
  w=(sens/np.maximum((rows.ask-rows.bid)/2,.05))**2;w=np.asarray(w/w.mean())
  penalty=np.diag([0]+[1]*degree)
  A=np.vstack([v*np.sqrt(w[:,None]/len(v)),np.sqrt(ridge)*penalty]);y=np.r_[np.log(rows.iv)*np.sqrt(w/len(v)),np.zeros(degree+1)]
  beta=np.linalg.lstsq(A,y,rcond=None)[0]
  z=(np.log(test.strike/test.spot)-center)/scale
  vols=np.exp(np.polynomial.chebyshev.chebvander(z,degree)@beta)
  for (_,r),vol in zip(test.iterrows(),vols):
   predicted=price(r.spot,r.strike,r.years,r.rate,float(vol),r.dividend_yield,r.kind)
   vals.append((predicted-r.mid, bool(r.bid<=predicted<=r.ask)))
 return vals
# Inner folds keep call-put pairs together. Rates/carry were estimated on the
# outer training cohort; selection is conditional on those training inputs.
results=[]
for degree in [2,3,4,5]:
 for ridge in [0.,1e-6,1e-4]:
  errors=[]
  for fold in range(3):
   mask=pd.Series(False,index=train.index)
   for _,g in train.groupby('expiry'):
    keys=sorted(g.strike.unique());selection=set(keys[fold::3]);mask.loc[g.index]=g.strike.isin(selection)
   errors+=fit_predict(train[~mask],train[mask],degree,ridge)
  errors=np.array(errors);results.append(dict(degree=degree,ridge=ridge,inner_price_rmse=float(np.sqrt(np.mean(errors[:,0]**2))),inside_spread_fraction=float(errors[:,1].mean())))
results.sort(key=lambda r:r['inner_price_rmse']);best=results[0]['inner_price_rmse']
eligible=[x for x in results if x['inner_price_rmse']<=best*1.05]
degree=min(x['degree'] for x in eligible)
chosen=max((x for x in eligible if x['degree']==degree),key=lambda x:x['ridge'])
plan=dict(selection_asof='2026-10-23T14:31:00Z',outer_training_quotes=len(train),inner_folds=3,selected_degree=chosen['degree'],selected_ridge=chosen['ridge'],candidates=results,
 gates={'snapshot_rmse_improvement_at_least':.5,'snapshot_inside_spread_at_least':.8,'temporal_matched_rmse_improvement_at_least':.3,'all_folds_must_improve':True,'new_grid_warnings':0},
 limitations='Previously inspected historical sample; outer holdouts reused. Inner scoring conditional on outer-training parity inputs. No new market data acquired.')
plan['selection_rule']='Lowest degree within 5% of the best pooled training-only CV price RMSE, then strongest tested ridge within that band; frozen before outer/temporal evaluation.'
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(plan,indent=2));print(json.dumps(plan,indent=2))
