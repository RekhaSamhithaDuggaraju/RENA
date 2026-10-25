import unittest
import importlib.util

class CifarArchitectureTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("torchvision") is not None,
        "Optional CIFAR benchmark requires torch and torchvision",
    )
    def test_cifar_cnn_forward_shape(self):
        import torch
        from core.cifar_benchmark import make_cnn
        model=make_cnn().eval()
        with torch.no_grad(): out=model(torch.zeros(2,3,32,32))
        self.assertEqual(tuple(out.shape),(2,10))

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("torchvision") is not None,
        "Optional CIFAR benchmark requires torch and torchvision",
    )
    def test_resnet18_forward_shape(self):
        import torch
        from core.cifar_benchmark import make_resnet18
        model=make_resnet18().eval()
        with torch.no_grad(): out=model(torch.zeros(2,3,32,32))
        self.assertEqual(tuple(out.shape),(2,10))

    def test_seed_disjointness_contract_is_explicit(self):
        from core.cifar_benchmark import CifarConfig
        cfg=CifarConfig(seed=42,reference_retrains=3)
        candidate_seed=cfg.seed+1
        reference_seeds={cfg.seed+100+i for i in range(cfg.reference_retrains)}
        self.assertNotIn(candidate_seed,reference_seeds)

    def test_invalid_architecture_rejected_before_download(self):
        from core.cifar_benchmark import CifarConfig, run_cifar_reference_benchmark
        with self.assertRaises(ValueError): run_cifar_reference_benchmark(CifarConfig(architecture="unknown"))

if __name__ == '__main__': unittest.main()
