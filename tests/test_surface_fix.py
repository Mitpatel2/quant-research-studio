import json
import unittest
import numpy as np
from quantcore.options import clean_chain,demo_chain,calibrate,predict_surface,advance_surface,surface_diagnostics,surface_strikes,surface_maturities,price
from validate_options import split_pairs

class CorrectedSurfaceTests(unittest.TestCase):
    def chain(self):return clean_chain(demo_chain())[0]

    def test_conditioned_gradient_converges_and_reduces_actual_loss(self):
        data=self.chain();a=calibrate(data,method='corrected');b=calibrate(data,method='corrected',learning_rate=.1)
        self.assertTrue(a['converged']);self.assertLess(a['gradient_norm'],1e-8)
        self.assertLess(a['iterations'],600)
        self.assertTrue(np.all(np.diff(a['loss'])<=1e-14))
        np.testing.assert_allclose(a['fitted_iv'],b['fitted_iv'],atol=1e-7)
        # Price-sensitive fitting must reproduce prices, including very low-vega
        # wings whose implied volatility is poorly identified by their prices.
        values=np.array([price(r.spot,r.strike,r.years,r.rate,float(v),r.dividend_yield,r.kind) for (_,r),v in zip(data.iterrows(),a['fitted_iv'])])
        self.assertLess(float(np.sqrt(np.mean((values-data.mid)**2))),.001)
        self.assertTrue(((values>=data.bid)&(values<=data.ask)).all())

    def test_backtracking_handles_large_user_step_without_false_convergence(self):
        m=calibrate(self.chain(),method='corrected',learning_rate=1.)
        self.assertTrue(m['converged'])
        self.assertTrue(np.all(np.diff(m['loss'])<=1e-14))
        capped=calibrate(self.chain(),iterations=1,method='corrected',learning_rate=.001)
        self.assertFalse(capped['converged']);self.assertEqual(capped['iterations'],1)

    def test_between_expiries_interpolates_total_variance(self):
        d=self.chain();m=calibrate(d,method='corrected')
        t1,t2=sorted(d.years.unique())[:2];tm=(t1+t2)/2;k=100.
        v1=float(predict_surface(m,k,t1));v2=float(predict_surface(m,k,t2));vm=float(predict_surface(m,k,tm))
        self.assertAlmostEqual(vm*vm*tm,(v1*v1*t1+v2*v2*t2)/2,places=12)
        self.assertTrue(np.isfinite(predict_surface(m,[90,100,110],tm)).all())

    def test_chart_mesh_contains_every_observed_expiry(self):
        d=self.chain();times=surface_maturities(d)
        self.assertTrue(set(d.years.unique())<=set(times))
        self.assertEqual(times.min(),d.years.min());self.assertEqual(times.max(),d.years.max())

    def test_frozen_roll_uses_observed_time_without_refitting_parameters(self):
        d=self.chain();m=calibrate(d,method='corrected');elapsed=1/(365*1440)
        rolled=advance_surface(m,101.,elapsed)
        np.testing.assert_array_equal(m['beta'],rolled['beta'])
        self.assertEqual(m['slices'],rolled['slices']);self.assertEqual(m['elapsed_years'],0.)
        first=float(d.years.min());before=float(predict_surface(m,100,first))
        after=float(predict_surface(rolled,101,first-elapsed))
        self.assertAlmostEqual(before,after,places=12)
        with self.assertRaises(ValueError):predict_surface(m,100,first-elapsed)

    def test_heldout_quote_mutation_cannot_change_fitted_parameters(self):
        d=self.chain();train,test=split_pairs(d)
        a=calibrate(train,method='corrected')
        changed=d.copy();changed.loc[test.index,['iv','bid','ask']]*=1.7
        train2,_=split_pairs(changed);b=calibrate(train2,method='corrected')
        np.testing.assert_array_equal(a['beta'],b['beta'])

    def test_domain_is_bounded_and_common_grid_is_supported(self):
        d=self.chain();m=calibrate(d,method='corrected')
        for k,t in [(1,d.years.min()),(1000,d.years.max()),(100,0),(np.nan,.1)]:
            with self.assertRaises(ValueError):predict_surface(m,k,t)
        k=surface_strikes(m,d)
        for t in np.linspace(d.years.min(),d.years.max(),20):
            self.assertTrue(np.isfinite(predict_surface(m,k,t)).all())
        self.assertFalse(surface_diagnostics(m,d))

    def test_saved_model_reproduces_quotes_and_rejects_bad_settings(self):
        d=self.chain();m=calibrate(d,method='corrected')
        saved=json.loads(json.dumps(m,default=lambda v:v.tolist()))
        np.testing.assert_allclose(predict_surface(saved,d.strike,d.years),m['fitted_iv'])
        for kwargs in [dict(iterations=0),dict(learning_rate=float('nan')),dict(degree=7),dict(ridge=-1)]:
            with self.assertRaises(ValueError):calibrate(d,method='corrected',**kwargs)
        self.assertEqual(calibrate(d)['method'],'legacy')

if __name__=='__main__':unittest.main()
