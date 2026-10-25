import unittest
from core.dataset import build_split
from core.model import train_base_model
from core.unlearning import ground_truth_retrain
from core.advanced_audit import run_advanced_audit
from core.research_suite import multi_seed_calibration_benchmark
import numpy as np

class V8ResearchTests(unittest.TestCase):
    def test_representation_probe_is_exposed(self):
        split=build_split(.05, random_state=42)
        X=np.vstack([split.X_retain,split.X_forget]); y=np.concatenate([split.y_retain,split.y_forget])
        original=train_base_model(X,y,random_state=7042,fit_random_state=7042)
        refs=[ground_truth_retrain(split.X_retain,split.y_retain,random_state=52000+i) for i in range(4)]
        audit=run_advanced_audit(refs[0],refs[0],original,split,1.0,1.0,reference_models=refs)
        self.assertTrue(0.0 <= audit.representation_probe_auc <= 1.0)
        self.assertIn('representation_probe_auc', audit.calibration)

    def test_multi_seed_summary(self):
        r=multi_seed_calibration_benchmark(seeds=(42,43), forget_fraction=.05, reference_count=4)
        self.assertEqual(r['summary']['n_seeds'],2)
        self.assertEqual(len(r['rows']),2)
        self.assertIsNotNone(r['summary']['rena_x_bootstrap_ci']['estimate'])

if __name__=='__main__': unittest.main()
