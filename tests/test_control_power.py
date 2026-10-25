import unittest
from core.research_suite import verifier_power_study

class ControlPowerTests(unittest.TestCase):
    def test_power_study_uses_independent_retrain_null(self):
        r=verifier_power_study(seeds=(42,), forget_fraction=.05)
        self.assertEqual(r['n'],25)  # 10 good + 5 legacy + 7 output-patch + 3 radius-patch
        self.assertTrue(any(x['label']=='bad_no_op' for x in r['rows']))
        self.assertTrue(any(x['label']=='good_retrained' for x in r['rows']))
        self.assertEqual(r['n_retrains'],10)
        self.assertIn('roc_auc', r)
        self.assertIn('plain_mia_auc', r)
        self.assertIn('single_axis_auc', r)
        self.assertIsNone(r['false_positive_rate'])

if __name__=='__main__': unittest.main()
