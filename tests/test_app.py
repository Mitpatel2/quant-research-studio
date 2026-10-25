"""Headless widget checks supplement the actual-browser coverage record."""
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]


class AppTests(unittest.TestCase):
    @unittest.skipUnless((ROOT/'data/local-futures.csv').exists(),'Local licensed candles are not included in portable package')
    def test_regime_edits_do_not_relabel_completed_outputs(self):
        a=AppTest.from_file(str(ROOT/'app.py'),default_timeout=20).run()
        a.radio(key='project_select').set_value('Market regimes').run()
        self.assertTrue(any('Completed run: 60 training' in x.value for x in a.info))
        a.number_input[0].set_value(180)
        a.number_input[1].set_value(60)
        a.run()
        self.assertTrue(any('Completed run: 60 training' in x.value for x in a.info))
        # AppTest discards unsubmitted form edits on its explicit run() above;
        # submit the intended values together, as the browser form does.
        a.number_input[0].set_value(180)
        a.number_input[1].set_value(60)
        a.button[0].click().run()
        self.assertFalse(a.exception)
        self.assertTrue(any('Completed run: 180 training sessions; 60 test' in x.value for x in a.info))

    def test_calculator_reference_and_put_switch(self):
        a=AppTest.from_file(str(ROOT/'app.py'),default_timeout=20).run()
        self.assertFalse(a.exception)
        self.assertAlmostEqual(float(a.metric[0].value),10.450583572,places=5)
        a.radio(key='bs_kind').set_value('put').run()
        self.assertAlmostEqual(float(a.metric[0].value),5.573526022,places=5)
        a.number_input(key='bs_q').set_value(2).run()
        self.assertAlmostEqual(float(a.metric[0].value),6.330080627,places=5)

    def test_all_screenshot_inputs_recompute_against_reference(self):
        a=AppTest.from_file(str(ROOT/'app.py'),default_timeout=20).run()
        a.number_input(key='bs_s').set_value(100)
        a.number_input(key='bs_k').set_value(95)
        a.number_input(key='bs_days').set_value(91.25)
        a.number_input(key='bs_vol').set_value(50)
        a.number_input(key='bs_r').set_value(10)
        a.number_input(key='bs_q').set_value(0)
        a.run()
        self.assertFalse(a.exception)
        self.assertAlmostEqual(float(a.metric[0].value),13.6953,delta=.00005)

    def test_boundary_is_explained_and_not_nan(self):
        a=AppTest.from_file(str(ROOT/'app.py'),default_timeout=20).run()
        a.number_input(key='bs_days').set_value(0).run()
        self.assertFalse(a.exception)
        self.assertEqual(float(a.metric[0].value),0)
        self.assertTrue(any('At expiry' in x.value for x in a.info))
        a.number_input(key='bs_days').set_value(30)
        a.number_input(key='bs_vol').set_value(0)
        a.run();self.assertFalse(a.exception)
        self.assertEqual(a.metric[3].value,'—')

    def test_missing_snapshot_clears_surface_metrics(self):
        a=AppTest.from_file(str(ROOT/'app.py'),default_timeout=20).run()
        a.radio(key='quote_source').set_value('Import snapshot CSV').run()
        self.assertFalse(a.exception)
        self.assertFalse(any(m.label=='Accepted quotes' for m in a.metric))
        self.assertTrue(any('Upload a European' in x.value for x in a.info))

    def test_greek_exports_name_the_completed_measure(self):
        a=AppTest.from_file(str(ROOT/'app.py'),default_timeout=20).run()
        for measure in ['Delta','Gamma','Vega','Theta','Rho']:
            a.selectbox[0].set_value(measure).run()
            self.assertFalse(a.exception)
            labels=[b.proto.label for b in a.get('download_button')]
            self.assertIn('Export corrected call '+measure.lower()+' grid',labels)
            self.assertEqual(sum(label.endswith(' grid') for label in labels),1)
        a.radio(key='surface_kind').set_value('put').run()
        self.assertIn('Export corrected put rho grid',[b.proto.label for b in a.get('download_button')])
        a.radio(key='surface_method').set_value('Legacy polynomial').run()
        self.assertIn('Export legacy put rho grid',[b.proto.label for b in a.get('download_button')])

    @unittest.skipUnless((ROOT/'data/databento/2026-10-23-spxw/snapshot-1435.csv').exists(),'Licensed real cache is private')
    def test_real_source_corrected_and_legacy_fit_metrics_match_selected_model(self):
        a=AppTest.from_file(str(ROOT/'app.py'),default_timeout=20).run()
        a.radio(key='quote_source').set_value('Cached Databento SPXW').run()
        values={m.label:m.value for m in a.metric}
        self.assertLess(float(values['Current fit price RMSE (points)']),.1)
        self.assertGreater(float(values['Current fit inside bid/ask'].strip('%')),98)
        self.assertTrue(any('Recorded corrected holdout test' in x.value for x in a.info))
        a.radio(key='surface_method').set_value('Legacy polynomial').run()
        values={m.label:m.value for m in a.metric}
        self.assertGreater(float(values['Current fit price RMSE (points)']),4)
        self.assertTrue(any('Recorded legacy holdout test' in x.value for x in a.warning))


if __name__=='__main__':unittest.main()
