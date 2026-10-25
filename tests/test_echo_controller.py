import unittest
from core.dataset import build_split
from core.model import train_base_model
from core.echo_controller import EchoConfig, echo_unlearn
class TestEchoController(unittest.TestCase):
 def test_closed_loop_records_candidates_and_limits_iterations(self):
  s=build_split(); m=train_base_model(s.X_train,s.y_train,epochs=3); _,r=echo_unlearn(m,s,EchoConfig(max_iterations=2,min_iterations=1,patience=1))
  self.assertLessEqual(r['iterations'],2); self.assertEqual(r['candidate_evaluations'],r['iterations']*4); self.assertTrue(r['trajectory'])
