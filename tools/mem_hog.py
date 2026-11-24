#!/usr/bin/env python3
"""mem_hog - hold host memory until MemAvailable falls to a target, then release it.

The #40 pressure guard is tested by driving MemAvailable under the engine's floor
and checking that a request arriving there is refused (ERR insufficient_memory)
rather than the engine eating its own working set, then that the box serves again
once the hold is released.  Bytes are memset, not just mapped: untouched anonymous
pages are not resident, and the whole point is to take RESIDENT pages away from
the page cache and the engine's mmap'd arena.

  python3 tools/mem_hog.py --target-avail-mib 9216 --hold-seconds 90
"""

import argparse
import time


def avail_mib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise RuntimeError("no MemAvailable in /proc/meminfo")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--target-avail-mib", type=int, default=9216,
                   help="stop growing once MemAvailable is at or under this "
                        "(the qwen4exp floor is 8192 MiB + the tower's claim)")
    p.add_argument("--step-mib", type=int, default=1024)
    p.add_argument("--hold-seconds", type=float, default=90)
    p.add_argument("--max-gib", type=float, default=40, help="give up rather than OOM the box")
    a = p.parse_args()

    pages = []
    while avail_mib() > a.target_avail_mib:
        held = sum(len(b) for b in pages) / 2**30
        if held > a.max_gib:
            print(f"mem_hog: max hold {a.max_gib} GiB reached at {held:.1f} GiB, "
                  f"MemAvailable {avail_mib()} MiB", flush=True)
            break
        b = bytearray(a.step_mib * 2**20)
        b[::4096] = b"\x01" * (len(b) // 4096)   # touch one byte per page: all of it resident
        pages.append(b)
        print(f"mem_hog: held {(held * 2**30 + a.step_mib * 2**20) / 2**30:.1f} GiB, "
              f"MemAvailable {avail_mib()} MiB", flush=True)
    print("mem_hog: target reached, holding", flush=True)
    t0 = time.monotonic()
    while time.monotonic() - t0 < a.hold_seconds:
        time.sleep(1)
    print(f"mem_hog: releasing {sum(len(b) for b in pages) / 2**30:.1f} GiB at exit", flush=True)


if __name__ == "__main__":
    main()
