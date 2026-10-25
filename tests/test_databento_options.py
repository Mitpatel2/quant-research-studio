import unittest
import numpy as np
import pandas as pd
from quantcore.options import price,clean_chain,carry_at,demo_chain
from quantcore.databento_options import normalize_snapshot

ASOF=pd.Timestamp('2026-10-23T14:35:00Z')
EXPIRIES=['2026-11-06','2026-12-18']


def fixture():
    quotes=[];definitions=[];i=0
    for date in EXPIRIES:
        expiry=pd.Timestamp(date+' 16:00',tz='America/New_York').tz_convert('UTC')
        t=(expiry-ASOF).total_seconds()/(365*86400)
        for strike in np.arange(95,105.1,.5):
            for flag,kind in [('C','call'),('P','put')]:
                i+=1;mid=price(100,strike,t,.04,.25,.01,kind)
                symbol='SPXW  '+pd.Timestamp(date).strftime('%y%m%d')+flag+f'{int(strike*1000):08d}'
                quotes.append(dict(instrument_id=i,bid_px_00=mid-.01,ask_px_00=mid+.01,bid_sz_00=2,ask_sz_00=2))
                definitions.append(dict(instrument_id=i,raw_symbol=symbol,expiration=pd.Timestamp(date,tz='UTC'),strike_price=strike,instrument_class=flag,security_update_action='A'))
    q=pd.DataFrame(quotes,index=pd.DatetimeIndex([ASOF]*len(quotes),name='ts_recv'))
    d=pd.DataFrame(definitions,index=pd.DatetimeIndex([ASOF-pd.Timedelta(hours=1)]*len(definitions),name='ts_recv'))
    u=pd.DataFrame([dict(ts_event=ASOF-pd.Timedelta(seconds=1),symbol='SPX',price=100.)],index=pd.DatetimeIndex([ASOF-pd.Timedelta(seconds=1)]))
    return dict(quotes=q,definitions=d,underlying=u)


class DatabentoNormalizationTests(unittest.TestCase):
    def test_contracts_dst_and_training_only_parity_curve(self):
        rows,bad,meta=normalize_snapshot(**fixture(),asof=ASOF,expiry_dates=EXPIRIES)
        self.assertTrue(bad.empty)
        self.assertTrue(rows.expiry.str.endswith('21:00:00+00:00').all())
        np.testing.assert_allclose(rows.rate,.04,atol=1e-10)
        np.testing.assert_allclose(rows.dividend_yield,.01,atol=1e-10)
        for c in meta['carry']:
            count=rows[(rows.expiry==c['expiry'])&(rows.validation_split=='train')].strike.nunique()
            self.assertEqual(c['training_pairs'],count)
        valid,rejected=clean_chain(rows)
        self.assertTrue(rejected.empty);self.assertEqual(len(valid),len(rows))

    def test_heldout_quotes_do_not_change_carry_estimate(self):
        f=fixture();rows,_,a=normalize_snapshot(**f,asof=ASOF,expiry_dates=EXPIRIES)
        heldout=set(rows.loc[rows.validation_split=='holdout','instrument_id'])
        f['quotes'].loc[f['quotes'].instrument_id.isin(heldout),['bid_px_00','ask_px_00']]+=.1
        _,_,b=normalize_snapshot(**f,asof=ASOF,expiry_dates=EXPIRIES)
        self.assertEqual(a['carry'],b['carry'])

    def test_future_definitions_and_index_prices_cannot_change_past_snapshot(self):
        f=fixture();a,_,_=normalize_snapshot(**f,asof=ASOF,expiry_dates=EXPIRIES)
        extra=f['definitions'].copy();extra.index+=pd.Timedelta(hours=2);extra['strike_price']+=1
        f['definitions']=pd.concat([f['definitions'],extra])
        extra=f['underlying'].copy();extra.index+=pd.Timedelta(minutes=2);extra['price']=150
        f['underlying']=pd.concat([f['underlying'],extra])
        b,_,_=normalize_snapshot(**f,asof=ASOF,expiry_dates=EXPIRIES)
        pd.testing.assert_frame_equal(a,b)

    def test_missing_boundary_and_stale_underlying_fail_closed(self):
        f=fixture()
        with self.assertRaises(ValueError):normalize_snapshot(**f,asof=ASOF+pd.Timedelta(seconds=1),expiry_dates=EXPIRIES)
        f['underlying'].index-=pd.Timedelta(minutes=2)
        with self.assertRaises(ValueError):normalize_snapshot(**f,asof=ASOF,expiry_dates=EXPIRIES)

    def test_carry_is_coherent_per_expiry_and_interpolates(self):
        rows,_,_=normalize_snapshot(**fixture(),asof=ASOF,expiry_dates=EXPIRIES)
        later=rows.expiry==rows.expiry.max()
        rows.loc[later,'rate']=.05
        # Prices must be updated with the new rate to preserve parity.
        for idx,row in rows.loc[later].iterrows():
            t=(pd.Timestamp(row.expiry)-ASOF).total_seconds()/(365*86400)
            mid=price(row.spot,row.strike,t,row.rate,.25,row.dividend_yield,row.kind)
            rows.loc[idx,['bid','ask']]=[mid-.01,mid+.01]
        valid,bad=clean_chain(rows);self.assertTrue(bad.empty)
        t=(valid.years.min()+valid.years.max())/2
        rate,_=carry_at(valid,t);self.assertAlmostEqual(float(rate),.045)
        rows.loc[rows.index[0],'rate']=.06
        with self.assertRaises(ValueError):clean_chain(rows)

    def test_sparse_expiry_is_explicitly_rejected_using_training_data_only(self):
        f=fixture();terms=f['definitions']
        # Leave only seven strikes in one expiry: fewer than ten training pairs.
        remove=terms.loc[(terms.expiration==pd.Timestamp(EXPIRIES[0],tz='UTC'))&(terms.strike_price>98),'instrument_id']
        f['quotes']=f['quotes'].loc[~f['quotes'].instrument_id.isin(remove)]
        with self.assertRaisesRegex(ValueError,'ten training parity pairs'):
            normalize_snapshot(**f,asof=ASOF,expiry_dates=EXPIRIES)
        rows,rejections,meta=normalize_snapshot(**f,asof=ASOF,expiry_dates=EXPIRIES,skip_unidentifiable=True)
        self.assertEqual(rows.expiry.nunique(),1)
        self.assertEqual(len(meta['unsupported_expiries']),1)
        self.assertTrue(any('fewer than ten' in s for s in rejections.reason))

if __name__=='__main__':unittest.main()
