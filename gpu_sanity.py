"""Quick GPU sanity: forward-pass latency across both architectures.

Usage:
    python gpu_sanity.py                              # both architectures
    python gpu_sanity.py --architecture anaconda-2001  # one only
"""
import argparse
import time
import torch
from config import NetworkConfig
from neural.network import make_network


def bench_forward(net, batch: int, device: torch.device, n_iters: int = 100) -> float:
    x = torch.randn(batch, 32, device=device)
    for _ in range(10):
        with torch.no_grad():
            _ = net(x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iters):
        with torch.no_grad():
            out = net(x)
        # Force sync every iter (simulates scheduler pattern which calls .cpu()).
        _ = out.cpu()
    return (time.time() - t0) / n_iters


def bench_per_net_pattern(arch: str, device: torch.device,
                          n_nets: int, batch_per_net: int,
                          n_iters: int = 10) -> float:
    """Simulate 'one forward per network per tick' pattern."""
    cfg = NetworkConfig(architecture=arch)
    nets = [make_network(cfg).to(device).eval() for _ in range(n_nets)]
    xs = [torch.randn(batch_per_net, 32, device=device) for _ in range(n_nets)]
    for n, x in zip(nets, xs):
        with torch.no_grad():
            _ = n(x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iters):
        for n, x in zip(nets, xs):
            with torch.no_grad():
                out = n(x)
            _ = out.cpu()
    return (time.time() - t0) / n_iters


def run_arch(arch: str, device: torch.device):
    cfg = NetworkConfig(architecture=arch)
    net = make_network(cfg).to(device).eval()
    n_weights = net.num_weights()
    label = {
        "checkersnet-1999": "1999 CheckersNet",
        "anaconda-2001":    "2001 Anaconda",
    }.get(arch, arch)

    print(f"\n── {label} ({n_weights} weights) ──")
    print("Single forward pass (one net, synced after each):")
    for b in [1, 8, 32, 128, 512, 2048, 9900]:
        t = bench_forward(net, b, device)
        print(f"  batch={b:>5d}: {t*1000:7.3f} ms/pass  ({b/t:>10,.0f} samples/sec)")

    print("\nCurrent scheduler pattern (100 networks, one forward each per tick):")
    for bpn in [1, 8, 99]:
        t = bench_per_net_pattern(arch, device, 100, bpn, n_iters=5)
        print(f"  100 nets x batch={bpn:>3d}: {t*1000:7.1f} ms/tick  "
              f"(total {100*bpn} samples)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--architecture", type=str, default="both",
                        choices=["checkersnet-1999", "anaconda-2001", "both"],
                        help="Which architecture(s) to benchmark.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        raise SystemExit

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Torch: {torch.__version__}")

    archs = (["checkersnet-1999", "anaconda-2001"]
             if args.architecture == "both"
             else [args.architecture])
    for arch in archs:
        run_arch(arch, device)


if __name__ == "__main__":
    main()
