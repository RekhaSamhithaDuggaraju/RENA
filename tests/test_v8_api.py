import unittest
from backend.app.routers.research import CalibrationRobustnessRequest

class V8ApiTests(unittest.TestCase):
    def test_request_defaults(self):
        r=CalibrationRobustnessRequest()
        self.assertEqual(r.seeds,[42,43,44])
        self.assertEqual(r.reference_retrains,8)

if __name__=='__main__': unittest.main()
