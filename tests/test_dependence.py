import unittest
from core.evidence_dependence import witness_dependence
class DependenceTests(unittest.TestCase):
    def test_population_dependence_shape(self):
        r=witness_dependence((7,42,2026),.1,'gradient_ascent')
        self.assertEqual(len(r['scores']),3)
        self.assertEqual(len(r['correlation_matrix']),4)
        self.assertEqual(len(r['correlation_matrix'][0]),4)
if __name__=='__main__': unittest.main()
