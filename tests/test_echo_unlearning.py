import unittest
import numpy as np
from core.dataset import build_split
from core.model import train_base_model
from core.echo_unlearning import OPERATORS, candidate_update
class TestEchoUpdates(unittest.TestCase):
 def test_all_candidates_change_numpy_model(self):
  s=build_split(); m=train_base_model(s.X_train,s.y_train,epochs=3)
  for op in OPERATORS:
   c=candidate_update(m,s,op); self.assertTrue(any(not np.array_equal(a,b) for a,b in zip(m.W,c.W)))
