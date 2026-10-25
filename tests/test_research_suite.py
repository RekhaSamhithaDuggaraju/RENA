import unittest, numpy as np
from core.research_suite import bootstrap_ci, cohens_d, rank_methods, experiment_manifest, red_team_evidence

class ResearchSuiteTests(unittest.TestCase):
    def test_bootstrap_ci(self):
        r=bootstrap_ci([1,2,3,4],n_bootstrap=100,seed=1)
        self.assertLessEqual(r['low'],r['estimate']); self.assertLessEqual(r['estimate'],r['high'])
    def test_effect_size(self): self.assertGreater(cohens_d([1,2,3],[0,1,1]),0)
    def test_ranking(self):
        r=rank_methods([{'algorithm':'a','rena_x_worst_case':.8},{'algorithm':'b','rena_x_worst_case':.4}]); self.assertEqual(r[0]['algorithm'],'a')
    def test_manifest_stable_shape(self):
        m=experiment_manifest({'seed':42}); self.assertEqual(len(m['manifest_id']),16)
    def test_red_team(self):
        r={'module_results':[{'module_name':'a','forget_score':.9},{'module_name':'b','forget_score':.9}]}
        out=red_team_evidence(r,(.5,)); self.assertEqual(len(out),2); self.assertTrue(any(x['detected_as_suspicious'] for x in out))

if __name__=='__main__': unittest.main()
