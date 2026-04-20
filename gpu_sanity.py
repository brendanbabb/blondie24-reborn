"""Quick GPU sanity: forward-pass latency on a 1743-param net with various batch sizes."""
import time
import torch
from config import NetworkConfig
from neural.network import CheckersNet


def bench_forward(batch: int, n_iters: int = 100) -> float:
    device = torch.device("cuda")
    net = CheckersNet(NetworkConfig()).to(device).eval()
    x = torch.randn(batch, 32, device=device)
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = net(x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iters):
        with torch.no_grad():
            out = net(x)
        # Force sync every iter (simulates scheduler pattern which calls .cpu().tolist())
        _ = out.cpu()
    return (time.time() - t0) / n_iters


def bench_per_net_pattern(n_nets: int, batch_per_net: int, n_iters: int = 10) -> float:
    """Simulate 'one forward per network per tick' pattern (current scheduler)."""
    device = torch.device("cuda")
    nets = [CheckersNet(NetworkConfig()).to(device).eval() for _ in range(n_nets)]
    xs = [torch.randn(batch_per_net, 32, device=device) for _ in range(n_nets)]
    # Warmup
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


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available"); raise SystemExit

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Torch: {torch.__version__}\n")

    print("Single forward pass (one net, synced after each):")
    for b in [1, 8, 32, 128, 512, 2048, 9900]:
        t = bench_forward(b)
        print(f"  batch={b:>5d}: {t*1000:7.3f} ms/pass  ({b/t:>10,.0f} samples/sec)")

    print("\nCurrent scheduler pattern (100 networks, one forward each per tick):")
    for bpn in [1, 8, 99]:
        t = bench_per_net_pattern(100, bpn, n_iters=5)
        print(f"  100 nets x batch={bpn:>3d}: {t*1000:7.1f} ms/tick  (total {100*bpn} samples)")
