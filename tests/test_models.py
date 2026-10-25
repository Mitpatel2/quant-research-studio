import unittest
import math
import numpy as np
import pandas as pd
from quantcore.options import black_scholes,price,implied_vol,demo_chain,clean_chain,calibrate,predict_surface
from quantcore.events import Risk,run,attribute_to_regimes
from quantcore.regimes import features,fit,session_days
from quantcore.data import load_candles


class OptionTests(unittest.TestCase):
    def test_reference_screenshot_inputs(self):
        value=black_scholes(100,100,1,.05,.2)
        self.assertAlmostEqual(value['price'],10.450583572185565,places=10)
        self.assertAlmostEqual(value['d1'],.35,places=12)
        self.assertAlmostEqual(value['d2'],.15,places=12)

    def test_mathworks_reference(self):
        self.assertAlmostEqual(price(100,95,.25,.1,.5),13.6953,places=4)
        self.assertAlmostEqual(price(100,95,.25,.1,.5,kind='put'),6.3497,places=4)
        self.assertAlmostEqual(price(910,980,.25,.02,.25,.025),19.6863,places=4)

    def test_put_call_parity_and_dividends(self):
        for q in [0,.025]:
            c=price(100,105,.6,.04,.3,q)
            p=price(100,105,.6,.04,.3,q,'put')
            self.assertAlmostEqual(c-p,100*math.exp(-q*.6)-105*math.exp(-.04*.6))

    def test_all_greeks_against_price_changes(self):
        for kind in ['call','put']:
            s,k,t,r,v,q=100,103,.7,.04,.28,.02
            out=black_scholes(s,k,t,r,v,q,kind);h=.001
            f=lambda ss=s,tt=t,rr=r,vv=v:price(ss,k,tt,rr,vv,q,kind)
            self.assertAlmostEqual(out['delta'],(f(ss=s+h)-f(ss=s-h))/(2*h),places=7)
            self.assertAlmostEqual(out['gamma'],(f(ss=s+h)-2*f()+f(ss=s-h))/h**2,places=6)
            eps=1e-5
            self.assertAlmostEqual(out['vega'],(f(vv=v+eps)-f(vv=v-eps))/(2*eps)*.01,places=7)
            self.assertAlmostEqual(out['rho'],(f(rr=r+eps)-f(rr=r-eps))/(2*eps)*.01,places=7)
            self.assertAlmostEqual(out['theta'],-(f(tt=t+eps)-f(tt=t-eps))/(2*eps)/365,places=7)

    def test_expiry_and_zero_vol_are_explicit_boundaries(self):
        self.assertEqual(price(110,100,0,.05,.2),10)
        self.assertIsNone(black_scholes(100,100,0,.05,.2)['gamma'])
        self.assertAlmostEqual(price(110,100,1,.05,0),110-100*math.exp(-.05))

    def test_invalid_inputs_rejected(self):
        for values in [(0,100,1,.05,.2),(100,-1,1,.05,.2),(100,100,-1,.05,.2),(100,100,1,.05,-.2),(math.nan,100,1,.05,.2)]:
            with self.assertRaises(ValueError):black_scholes(*values)

    def test_iv_roundtrip_calls_puts_and_dividends(self):
        for kind in ['call','put']:
            for k in [90,100,110]:
                target=price(100,k,.5,.04,.32,.01,kind)
                self.assertAlmostEqual(implied_vol(target,100,k,.5,.04,.01,kind),.32,places=10)

    def test_iv_bad_quote_and_expiry(self):
        for target,t in [(-1,1),(100,1),(10,0)]:
            with self.assertRaises(ValueError):implied_vol(target,100,100,t,.05)

    def test_chain_rejects_future_stale_american_crossed(self):
        d=demo_chain().iloc[:4].copy()
        d.loc[d.index[0],'quote_time']='2026-06-01T14:01:00Z'
        d.loc[d.index[1],'quote_time']='2026-06-01T13:58:00Z'
        d.loc[d.index[2],'exercise_style']='american'
        d.loc[d.index[3],'ask']=0
        valid,bad=clean_chain(d)
        self.assertTrue(valid.empty);self.assertEqual(len(bad),4)

    def test_duplicate_or_naive_snapshot_rejected(self):
        d=demo_chain()
        with self.assertRaises(ValueError):clean_chain(pd.concat([d,d.iloc[:1]]))
        d['asof']='2026-06-01 14:00:00'
        with self.assertRaises(ValueError):clean_chain(d)

    def test_market_pair_parity_is_checked_with_bid_ask_bounds(self):
        d=demo_chain()
        changed=d.copy()
        target=(changed.strike==100)&(changed.kind=='call')
        changed.loc[target,['bid','ask']]+=1
        valid,bad=clean_chain(changed)
        self.assertTrue(all('parity' in reason for reason in bad.reason))
        self.assertEqual(len(bad),8) # both sides of the inconsistent pair at four expiries
        self.assertEqual(len(valid),len(d)-8)

    def test_gradient_descent_decreases_loss_and_fits_observations(self):
        d,bad=clean_chain(demo_chain());self.assertTrue(bad.empty)
        model=calibrate(d)
        self.assertLess(model['loss'][-1],model['loss'][0])
        self.assertTrue(np.all(np.diff(model['loss'])<=1e-12))
        self.assertLess(model['rmse_iv'],.002)
        self.assertTrue(np.all(predict_surface(model,d.strike,d.years)>0))


def bars(n=100):
    return pd.DataFrame(dict(open=np.full(n,100.),high=np.full(n,101.),low=np.full(n,99.),close=np.full(n,100.),volume=100),index=pd.date_range('2026-06-01',periods=n,freq='15min',tz='America/Chicago'))


class EventTests(unittest.TestCase):
    def test_regime_performance_uses_decision_label_not_future_label(self):
        d=bars();out=run(d,signals=self.signal(d))
        observed=pd.DataFrame({'regime':['Lower volatility','Higher volatility']},index=[d.index[34],d.index[35]])
        groups={r['regime']:r for r in attribute_to_regimes(out,observed)}
        self.assertEqual(groups['Lower volatility']['trades'],1)
        self.assertEqual(groups['Higher volatility']['trades'],0)
        self.assertAlmostEqual(groups['Lower volatility']['net'],out['metrics']['net'])

    def test_event_queue_hides_candle_range_until_close(self):
        d=bars();out=run(d,signals=self.signal(d));events=out['events']
        opening=events[0]
        self.assertEqual(opening['event'],'market_open')
        self.assertIn('open',opening)
        self.assertNotIn('high',opening)
        self.assertNotIn('close',opening)
        signal_index=next(i for i,e in enumerate(events) if e['event']=='signal')
        self.assertEqual(events[signal_index-1]['event'],'market_close')
        self.assertEqual(events[signal_index+1]['event'],'order')
        self.assertTrue(all(pd.Timestamp(a['time'])<=pd.Timestamp(b['time']) for a,b in zip(events,events[1:])))

    def signal(self,df):
        s=pd.Series(0,index=df.index);s.iloc[34]=1;return s

    def test_next_bar_entry_and_entry_candle_stop(self):
        d=bars();d.iloc[35,d.columns.get_loc('low')]=90
        out=run(d,signals=self.signal(d));t=out['trades'][0]
        self.assertEqual(t['entry_time'],str(d.index[35]))
        self.assertEqual(t['decision'],str(d.index[35]))
        self.assertEqual(t['reason'],'stop')
        self.assertLessEqual(t['net'],-t['planned_loss']+1e-8)

    def test_gap_cancels_pending(self):
        d=bars();s=self.signal(d);d=d.drop(d.index[35])
        out=run(d,signals=s)
        self.assertEqual(out['trades'],[])
        self.assertTrue(any(e['event']=='cancel' for e in out['events']))

    def test_risk_budget_never_forces_one_contract(self):
        d=bars();out=run(d,Risk(max_trade_loss=1),signals=self.signal(d))
        self.assertEqual(out['trades'],[])
        self.assertTrue(any(e['event']=='reject' for e in out['events']))

    def test_no_warmup_trades(self):
        d=bars();out=run(d,start=d.index[60],signals=self.signal(d))
        self.assertEqual(out['trades'],[])

    def test_cost_reconciliation(self):
        d=bars();out=run(d,signals=self.signal(d));t=out['trades'][0]
        self.assertAlmostEqual(t['net'],(t['exit']-t['entry'])*2-1.54)
        self.assertAlmostEqual(out['equity'][-1]['equity'],50000+out['metrics']['net'])

    def test_future_prices_cannot_change_entry_decision(self):
        d=bars();s=self.signal(d);a=run(d,signals=s)
        changed=d.copy();changed.loc[changed.index[35:],['open','high','low','close']]+=10
        b=run(changed,signals=s)
        signals=lambda result:[e for e in result['events'] if e['event']=='signal']
        self.assertEqual(signals(a),signals(b))

    def test_invalid_risk_rejected(self):
        for risk in [Risk(commission_rt=-1),Risk(quantity=1.5),Risk(slippage_ticks=math.nan)]:
            with self.assertRaises(ValueError):run(bars(),risk)


class RegimeTests(unittest.TestCase):
    def test_sunday_evening_is_monday_futures_session(self):
        ix=pd.DatetimeIndex(['2026-06-21 18:00','2026-06-22 08:30'],tz='America/Chicago')
        self.assertEqual(list(session_days(ix)),[pd.Timestamp('2026-06-22')]*2)

    def sample(self):
        d=bars(960);rng=np.random.default_rng(42)
        c=100+np.cumsum(rng.normal(0,.2,len(d)))
        d['open']=c;d['close']=c;d['high']=c+.5;d['low']=c-.5
        d['volume']=rng.integers(1,300,len(d))
        return d

    def test_future_mutation_cannot_change_features_or_past_labels(self):
        d=self.sample();a=features(d);changed=d.copy();changed.iloc[700:]*=2
        b=features(changed)
        pd.testing.assert_frame_equal(a.loc[a.index<d.index[700]],b.loc[b.index<d.index[700]])
        model=fit(a.loc[a.index<d.index[500]])
        test=a.loc[(a.index>=d.index[500])&(a.index<d.index[700])]
        pd.testing.assert_series_equal(model.predict(test),model.predict(b.loc[test.index]))

    def test_scaler_uses_training_only_and_is_deterministic(self):
        d=features(self.sample());training=d.iloc[:500]
        a=fit(training);b=fit(training)
        np.testing.assert_allclose(a.scaler.mean_,training.mean().values)
        pd.testing.assert_series_equal(a.predict(d.iloc[500:]),b.predict(d.iloc[500:]))

    def test_gap_requires_new_feature_warmup(self):
        d=self.sample().drop(self.sample().index[300]);feat=features(d)
        self.assertNotIn(d.index[301],feat.index)

    def test_training_rows_cannot_be_labeled_as_holdout(self):
        d=features(self.sample());model=fit(d.iloc[:500])
        with self.assertRaises(ValueError):model.predict(d.iloc[:10])

    def test_duplicate_candles_rejected(self):
        import io
        d=self.sample().iloc[:1];bad=pd.concat([d,d]).to_csv(index_label='datetime')
        with self.assertRaises(ValueError):load_candles(io.StringIO(bad))


if __name__=='__main__':unittest.main()
