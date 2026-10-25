import unittest
import numpy as np
from quantcore.options import demo_chain,clean_chain
from validate_options import evaluate,split_pairs

class ValidationTests(unittest.TestCase):
    def test_paired_sides_cannot_leak_into_holdout(self):
        chain,_=clean_chain(demo_chain())
        train,test=split_pairs(chain)
        a=set(zip(train.expiry,train.strike));b=set(zip(test.expiry,test.strike))
        self.assertFalse(a&b)
        self.assertEqual(len(train)+len(test),len(chain))
        self.assertEqual(train.expiry.nunique(),test.expiry.nunique())

    def test_generated_fixture_is_only_a_numerical_check(self):
        summary,detail,bad=evaluate(demo_chain())
        self.assertEqual(summary['status'],'evaluated_supplied_snapshot')
        self.assertTrue(bad.empty)
        self.assertLess(summary['iv_roundtrip_max_abs_price_error'],1e-8)
        for greek,error in summary['greeks_finite_difference_max_abs_error'].items():
            self.assertLess(error,1e-5,greek)
        self.assertIn('holdout',summary['metrics'])
        self.assertTrue(np.isfinite(detail.fitted_price).all())
        self.assertTrue(any('not independent' in text for text in summary['limitations']))

if __name__=='__main__':unittest.main()
