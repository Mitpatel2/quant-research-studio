"""Strict portable candle imports; timestamps always describe source-bar opens."""
import io
from pathlib import Path
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def load_candles(source):
    df=pd.read_csv(source)
    df.columns=df.columns.str.lower().str.strip()
    col=next((c for c in ['datetime','ts_event','timestamp'] if c in df),None)
    if col is None:raise ValueError('Need datetime, ts_event or timestamp column')
    if not df[col].astype(str).str.contains(r'(?:Z|[+-]\d\d:\d\d)$').all():
        raise ValueError('Candle timestamps must include a timezone')
    df.index=pd.to_datetime(df.pop(col),utc=True,errors='raise')
    if df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError('Candles must be sorted with unique timestamps')
    if not {'open','high','low','close','volume'}<=set(df):
        raise ValueError('Need open, high, low, close and volume columns')
    df=df[['open','high','low','close','volume']].astype(float)
    if df.empty or not np.isfinite(df).all().all() or (df[['open','high','low','close']]<=0).any().any():
        raise ValueError('Need nonempty finite positive OHLC data')
    if (df.high<df[['open','close','low']].max(axis=1)).any() or (df.low>df[['open','close','high']].min(axis=1)).any() or (df.volume<0).any():
        raise ValueError('Malformed OHLC or negative volume')
    if not ((df.index.minute%15==0)&(df.index.second==0)&(df.index.microsecond==0)).all():
        raise ValueError('Use candles on the M15 opening grid')
    return df


def cached_futures():
    path=ROOT/'data/local-futures.csv'
    if not path.exists():raise ValueError('Import a licensed M15 OHLCV CSV to continue')
    return load_candles(path)
