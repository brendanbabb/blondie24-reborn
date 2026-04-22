"""
Minimal mp.Pool repro to isolate the hang.

Test matrix (run each separately):
  python -u -m scripts.mp_repro init-only      # Pool with _mp_worker_init, no task
  python -u -m scripts.mp_repro trivial-task   # Pool with _mp_worker_init + no-op task
  python -u -m scripts.mp_repro no-init        # Pool with NO initializer, trivial task
  python -u -m scripts.mp_repro play-one       # Pool with init + one real game
  python -u -m scripts.mp_repro no-init-play   # Pool with NO initializer + one real game (worker JIT-compiles on first call)
  python -u -m scripts.mp_repro parent-warm-worker-play  # parent warms, worker has no init, plays game (tests cache-reload via game call)
  python -u -m scripts.mp_repro fresh-cache-dir          # parent warms in dir A, workers spawn with NUMBA_CACHE_DIR=<fresh dir B>
  python -u -m scripts.mp_repro fresh-cache-dir-4w       # same as above but 4 workers + 32 games (tests shared-dir write race)
"""
import faulthandler
faulthandler.enable()

import sys
import os
import time
import multiprocessing as mp
import numpy as np


def _noop(x):
    return x * 2


def _one_game(task):
    """Run one real game in the worker — same shape as _mp_play_one."""
    from evolution.tournament import _mp_play_one
    return _mp_play_one(task)


def _build_one_game_task(depth=4):
    """Build one game task (pair_id, black_w, white_w, depth, max_moves) for Anaconda."""
    # Anaconda net has 5048 weights. Random init is fine for timing.
    rng = np.random.default_rng(42)
    black_w = rng.standard_normal(5048).astype(np.float64) * 0.1
    white_w = rng.standard_normal(5048).astype(np.float64) * 0.1
    return (0, black_w, white_w, depth, 200)


def run(mode):
    print(f"[repro] mode={mode} pid={os.getpid()} py={sys.version.split()[0]}")
    t0 = time.time()

    if mode == "init-only":
        from evolution.tournament import _mp_worker_init
        print("[repro] parent warmup starting")
        _mp_worker_init()
        print(f"[repro] parent warmup done in {time.time()-t0:.2f}s")
        t1 = time.time()
        pool = mp.Pool(processes=1, initializer=_mp_worker_init)
        print(f"[repro] pool created in {time.time()-t1:.2f}s")
        time.sleep(3)
        pool.close()
        pool.join()
        print(f"[repro] pool closed after init-only. total={time.time()-t0:.2f}s")

    elif mode == "trivial-task":
        from evolution.tournament import _mp_worker_init
        _mp_worker_init()
        print(f"[repro] parent warmup done {time.time()-t0:.2f}s")
        t1 = time.time()
        pool = mp.Pool(processes=1, initializer=_mp_worker_init)
        print(f"[repro] pool created {time.time()-t1:.2f}s")
        t2 = time.time()
        r = pool.apply_async(_noop, (21,))
        val = r.get(timeout=120)
        print(f"[repro] trivial task returned {val} in {time.time()-t2:.2f}s")
        pool.close()
        pool.join()
        print(f"[repro] done total={time.time()-t0:.2f}s")

    elif mode == "no-init":
        pool = mp.Pool(processes=1)
        print(f"[repro] pool created {time.time()-t0:.2f}s")
        t2 = time.time()
        r = pool.apply_async(_noop, (21,))
        val = r.get(timeout=30)
        print(f"[repro] trivial task returned {val} in {time.time()-t2:.2f}s")
        pool.close()
        pool.join()
        print(f"[repro] done total={time.time()-t0:.2f}s")

    elif mode == "play-one":
        from evolution.tournament import _mp_worker_init
        _mp_worker_init()
        print(f"[repro] parent warmup done {time.time()-t0:.2f}s")
        t1 = time.time()
        pool = mp.Pool(processes=1, initializer=_mp_worker_init)
        print(f"[repro] pool created {time.time()-t1:.2f}s")
        task = _build_one_game_task(depth=4)
        print(f"[repro] submitting game task...")
        t2 = time.time()
        r = pool.apply_async(_one_game, (task,))
        result = r.get(timeout=300)
        print(f"[repro] game result: {result} in {time.time()-t2:.2f}s")
        pool.close()
        pool.join()
        print(f"[repro] done total={time.time()-t0:.2f}s")

    elif mode == "no-init-play":
        pool = mp.Pool(processes=1)
        print(f"[repro] pool created {time.time()-t0:.2f}s")
        task = _build_one_game_task(depth=4)
        print(f"[repro] submitting game task (worker will JIT-compile on first call)...")
        t2 = time.time()
        r = pool.apply_async(_one_game, (task,))
        result = r.get(timeout=300)
        print(f"[repro] game result: {result} in {time.time()-t2:.2f}s")
        pool.close()
        pool.join()
        print(f"[repro] done total={time.time()-t0:.2f}s")

    elif mode == "parent-warm-worker-play":
        # Parent warms (populates disk cache), pool has NO initializer,
        # worker plays a game — exercises the cache via the first game call.
        # If this segfaults, confirms the cache read is what crashes workers,
        # independent of the _mp_worker_init path.
        from evolution.tournament import _mp_worker_init
        _mp_worker_init()
        print(f"[repro] parent warmup done {time.time()-t0:.2f}s")
        t1 = time.time()
        pool = mp.Pool(processes=1)
        print(f"[repro] pool created {time.time()-t1:.2f}s")
        task = _build_one_game_task(depth=4)
        t2 = time.time()
        r = pool.apply_async(_one_game, (task,))
        result = r.get(timeout=300)
        print(f"[repro] game result: {result} in {time.time()-t2:.2f}s")
        pool.close()
        pool.join()
        print(f"[repro] done total={time.time()-t0:.2f}s")

    elif mode == "fresh-cache-dir":
        # Parent warms in default cache dir, then flips NUMBA_CACHE_DIR to a
        # fresh empty dir before spawning workers. Workers inherit the fresh
        # dir, find no cache, compile in-process. Should avoid the segfault
        # if the bug is cross-process cache reload.
        import tempfile
        from evolution.tournament import _mp_worker_init
        _mp_worker_init()
        print(f"[repro] parent warmup done {time.time()-t0:.2f}s")
        fresh = tempfile.mkdtemp(prefix="numba_worker_cache_")
        os.environ["NUMBA_CACHE_DIR"] = fresh
        print(f"[repro] set NUMBA_CACHE_DIR={fresh}")
        t1 = time.time()
        pool = mp.Pool(processes=1, initializer=_mp_worker_init)
        print(f"[repro] pool created {time.time()-t1:.2f}s")
        task = _build_one_game_task(depth=4)
        t2 = time.time()
        r = pool.apply_async(_one_game, (task,))
        result = r.get(timeout=300)
        print(f"[repro] game result: {result} in {time.time()-t2:.2f}s")
        pool.close()
        pool.join()
        print(f"[repro] done total={time.time()-t0:.2f}s")

    elif mode == "fresh-cache-dir-4w":
        # Stress the fresh-dir fix: 4 workers all compile + play games concurrently,
        # all writing to the same fresh NUMBA_CACHE_DIR. If they race-corrupt the
        # cache, we'll see the classic "unresolved symbol" / segfault pattern.
        import tempfile
        from evolution.tournament import _mp_worker_init
        _mp_worker_init()
        print(f"[repro] parent warmup done {time.time()-t0:.2f}s")
        fresh = tempfile.mkdtemp(prefix="numba_worker_cache_")
        os.environ["NUMBA_CACHE_DIR"] = fresh
        print(f"[repro] set NUMBA_CACHE_DIR={fresh}")
        t1 = time.time()
        pool = mp.Pool(processes=4, initializer=_mp_worker_init)
        print(f"[repro] pool created {time.time()-t1:.2f}s")
        tasks = [_build_one_game_task(depth=4) for _ in range(32)]
        print(f"[repro] submitting {len(tasks)} games across 4 workers...")
        t2 = time.time()
        results = pool.map(_one_game, tasks)
        print(f"[repro] {len(results)} games done in {time.time()-t2:.2f}s")
        pool.close()
        pool.join()
        print(f"[repro] done total={time.time()-t0:.2f}s")

    else:
        print(f"[repro] unknown mode {mode!r}")
        sys.exit(2)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "init-only"
    run(mode)
