"""Normalize cached Databento SPXW interval quotes without requesting more data.

Exercise style, multiplier and PM expiry closing time come from Cboe SPXW
specifications, not from OPRA's midnight-UTC date-only expiration field.
The caller supplies explicitly verified normal full-session expiry dates.
"""
import math
import re
import numpy as np
import pandas as pd


def normalize_snapshot(quotes,definitions,underlying,asof,expiry_dates,carry_parameters=None,skip_unidentifiable=False):
    asof=pd.Timestamp(asof)
    if asof.tzinfo is None:raise ValueError('Timezone-aware as-of required')
    asof=asof.tz_convert('UTC')
    # Index arrival time must be known, not only an exchange timestamp in future.
    u=underlying.loc[(underlying.index<=asof)&(underlying.ts_event<=asof)&(underlying.symbol=='SPX')].sort_index()
    if u.empty:raise ValueError('No past SPX index observation')
    spot_row=u.iloc[-1];age=(asof-u.index[-1]).total_seconds()
    if age>30 or not np.isfinite(spot_row.price) or spot_row.price<=0:
        raise ValueError('SPX observation is stale or invalid')
    spot=float(spot_row.price)
    known=definitions.loc[definitions.index<=asof].sort_index().drop_duplicates('instrument_id',keep='last')
    # Exact interval boundary, rather than silently forward-filling across gaps.
    snap=quotes.loc[quotes.index==asof].reset_index()
    if snap.empty:raise ValueError('No exact options interval snapshot')
    d=known[['instrument_id','raw_symbol','expiration','strike_price','instrument_class','security_update_action']]
    joined=snap.merge(d,on='instrument_id',how='left',validate='many_to_one')
    rows=[];rejected=[]
    for _,row in joined.iterrows():
        try:
            match=re.fullmatch(r'SPXW\s+(\d{6})([CP])(\d{8})',str(row.raw_symbol))
            if not match:raise ValueError('No verified SPXW definition at snapshot')
            expiry_date=pd.to_datetime(match[1],format='%y%m%d').date().isoformat()
            if expiry_date not in expiry_dates:continue
            if pd.Timestamp(row.expiration).date().isoformat()!=expiry_date:
                raise ValueError('Definition and OCC expiry dates disagree')
            strike=int(match[3])/1000
            if not np.isclose(strike,float(row.strike_price)) or row.instrument_class!=match[2]:
                raise ValueError('Definition and OCC contract terms disagree')
            if row.security_update_action=='D':raise ValueError('Contract definition deleted')
            if abs(strike/spot-1)>.05:continue
            bid,ask=float(row.bid_px_00),float(row.ask_px_00)
            if not all(np.isfinite([bid,ask])) or bid<=0 or ask<bid or ask-bid>5:
                raise ValueError('Invalid, zero or over-five-point-wide quote')
            if row.bid_sz_00<=0 or row.ask_sz_00<=0:raise ValueError('No displayed quote size')
            # Only full-session dates explicitly verified by the caller. DST is
            # resolved by New York time, never a hardcoded UTC hour.
            expiry=pd.Timestamp(expiry_date+' 16:00',tz='America/New_York').tz_convert('UTC')
            if expiry<=asof:raise ValueError('Expired option')
            rows.append(dict(asof=asof.isoformat(),quote_time=asof.isoformat(),expiry=expiry.isoformat(),spot=spot,
                strike=strike,bid=bid,ask=ask,kind='call' if match[2]=='C' else 'put',exercise_style='european',
                instrument_id=int(row.instrument_id),raw_symbol=row.raw_symbol,contract_multiplier=100,
                carry_source='training_pairs_put_call_parity',provider='Databento',dataset='OPRA.PILLAR',schema='cbbo-1m'))
        except (ValueError,TypeError) as err:
            rejected.append(dict(instrument_id=int(row.instrument_id),reason=str(err)))
    out=pd.DataFrame(rows)
    if out.empty:raise ValueError('No valid selected-expiry quotes')
    out=out.sort_values(['expiry','strike','kind']).reset_index(drop=True)
    if out.duplicated(['expiry','strike','kind']).any():raise ValueError('Duplicate contract at snapshot')
    carry=[];unsupported_expiries=[]
    out['validation_split']='train'
    for expiry,group in out.groupby('expiry'):
        strikes=sorted(group.strike.unique());heldout=set(strikes[2::5])
        out.loc[group.index[group.strike.isin(heldout)],'validation_split']='holdout'
        if carry_parameters is not None:
            prior=carry_parameters[expiry]
            out.loc[group.index,'rate']=prior['rate']
            out.loc[group.index,'dividend_yield']=prior['dividend_yield']
            carry.append(dict(prior,source='previous_snapshot_frozen'))
            continue
        train=group.loc[~group.strike.isin(heldout)]
        paired=train.pivot(index='strike',columns='kind',values=['bid','ask']).dropna()
        if len(paired)<10:
            if not skip_unidentifiable:raise ValueError(f'Need ten training parity pairs per expiry: {expiry}')
            unsupported_expiries.append(expiry)
            rejected.extend(dict(instrument_id=int(r.instrument_id),reason='Expiry excluded: fewer than ten training parity pairs') for _,r in group.iterrows())
            continue
        k=paired.index.to_numpy();y=((paired.bid.call+paired.ask.call-paired.bid.put-paired.ask.put)/2).to_numpy()
        width=(paired.ask.call-paired.bid.call+paired.ask.put-paired.bid.put).to_numpy()
        weight=1/np.maximum(width,.05)
        # Centered, scaled strike prevents the high SPX level from making the
        # two-parameter fit ill conditioned. No heldout quote enters this fit.
        x=np.column_stack([np.ones(len(k)),-(k-spot)/spot])
        b=np.linalg.lstsq(x*weight[:,None],y*weight,rcond=None)[0]
        discount=float(b[1]/spot);prepaid=float(b[0]+discount*spot)
        years=(pd.Timestamp(expiry)-asof).total_seconds()/(365*86400)
        if discount<=0 or prepaid<=0:raise ValueError('Invalid parity discount/prepaid forward')
        rate=-math.log(discount)/years;q=-math.log(prepaid/spot)/years
        if not -.1<rate<.3 or not -.1<q<.3:raise ValueError('Unstable parity-implied carry estimate')
        out.loc[group.index,'rate']=rate;out.loc[group.index,'dividend_yield']=q
        pred=prepaid-discount*k
        lo=(paired.bid.call-paired.ask.put).to_numpy();hi=(paired.ask.call-paired.bid.put).to_numpy()
        carry.append(dict(expiry=expiry,training_pairs=len(paired),discount_factor=discount,prepaid_forward=prepaid,
            forward=prepaid/discount,rate=rate,dividend_yield=q,training_parity_inside_fraction=float(((lo<=pred)&(pred<=hi)).mean())))
    out=out.loc[~out.expiry.isin(unsupported_expiries)].copy()
    if out.empty:raise ValueError('No identifiable expiry carry inputs')
    provenance=dict(provider='Databento',asof=asof.isoformat(),spot=spot,spot_record_time=u.index[-1].isoformat(),
        spot_event_time=spot_row.ts_event.isoformat(),spot_age_seconds=age,interval_records=len(snap),normalized_quotes=len(out),
        normalization_rejections=len(rejected),expiry_dates=expiry_dates,carry=carry,unsupported_expiries=unsupported_expiries,
        exercise_source='https://www.cboe.com/available_weeklys/',
        expiry_and_multiplier_source='https://www.cboe.com/tradable-products/sp-500/spx-options/spx-specifications/',
        assumptions=['Rates and carry are inferred jointly from training-only put-call pairs, not independent market rate/dividend data.',
            'Expiry dates explicitly selected as full sessions; 16:00 America/New_York closing, ACT/365.',
            'CBBO timestamp is interval end. Actual last bid/ask update age is unavailable and may exceed a minute.',
            'Spot Greeks hold inferred rate/carry fixed. Contract-dollar values use Cboe multiplier 100.'])
    return out,pd.DataFrame(rejected),provenance
