"""Run the real CIFAR-10 reference benchmark. Usage: python scripts_run_cifar_benchmark.py [epochs] [refs] [architecture]"""
from core.cifar_benchmark import CifarConfig, run_cifar_reference_benchmark
import json,sys
cfg=CifarConfig(epochs=int(sys.argv[1]) if len(sys.argv)>1 else 2, reference_retrains=int(sys.argv[2]) if len(sys.argv)>2 else 3, architecture=sys.argv[3] if len(sys.argv)>3 else "both")
print(json.dumps(run_cifar_reference_benchmark(cfg),indent=2))
