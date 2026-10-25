import unittest
from core.research_suite import calibration_control_discrimination

class V8ControlTests(unittest.TestCase):
    def test_control_matrix_shape(self):
        r=calibration_control_discrimination(seeds=(42,), forget_fraction=.05, reference_count=4)
        self.assertEqual(len(r['rows']),4)
        self.assertEqual(set(r['summary']), {'retrain','no_op','gradient_ascent','over_unlearning'})

if __name__=='__main__': unittest.main()
