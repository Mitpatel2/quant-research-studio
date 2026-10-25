"""Create synthetic M15 candles for a reproducible UI demonstration."""
from pathlib import Path
import numpy as np
import pandas as pd

rng=np.random.default_rng(42)
index=pd.date_range('2026-01-01',periods=360*96,freq='15min',tz='UTC')
close=np.round((20000+np.cumsum(rng.normal(0,1,len(index))))*4)/4
opening=np.r_[close[0],close[:-1]]
spread=rng.integers(1,9,len(index))/4
frame=pd.DataFrame({'datetime':index.astype(str),'open':opening,'high':np.maximum(opening,close)+spread,'low':np.minimum(opening,close)-spread,'close':close,'volume':rng.integers(100,1000,len(index))})
path=Path(__file__).parent/'data/local-futures.csv'
if path.exists():
    raise SystemExit('Existing local candles preserved; move them before generating a demo.')
path.parent.mkdir(exist_ok=True)
frame.to_csv(path,index=False)
print('Created synthetic demonstration candles at data/local-futures.csv. These are not market observations.')
