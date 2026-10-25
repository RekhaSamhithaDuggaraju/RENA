import unittest
from core.llm.paired_niah_aggregate import aggregate_paired_niah

class PairedNiahAggregateTests(unittest.TestCase):
    def sample(self, seed, delta):
        return {"candidate":"GA", "reference":"retain_only_retraining", "seed":seed,
                **{k:{"n":4,"candidate_rate":.5+delta,"reference_rate":.5,"difference":delta}
                   for k in ("forget_absent_disclosure","forget_present_retrieval","retain_present_retrieval")}}
    def test_aggregates_paired_seed_differences(self):
        out=aggregate_paired_niah([self.sample(1,.1),self.sample(2,.3)],bootstrap_samples=200)
        self.assertEqual(out['seed_count'],2)
        self.assertAlmostEqual(out['metrics']['forget_absent_disclosure']['mean'],.2)
        self.assertEqual(out['seeds'],[1,2])
    def test_rejects_duplicate_seed(self):
        with self.assertRaises(ValueError): aggregate_paired_niah([self.sample(1,.1),self.sample(1,.2)])
    def test_rejects_different_methods(self):
        b=self.sample(2,.2); b['candidate']='NPO'
        with self.assertRaises(ValueError): aggregate_paired_niah([self.sample(1,.1),b])

if __name__=='__main__': unittest.main()
