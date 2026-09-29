import os, sys, re
pid = sys.argv[1]
for fd in os.listdir(f"/proc/{pid}/fd"):
    try:
        t = os.readlink(f"/proc/{pid}/fd/{fd}")
    except OSError:
        continue
    if "card1" in t or "renderD129" in t:
        print("fd", fd, t)
        print(open(f"/proc/{pid}/fdinfo/{fd}").read())
tot = 0; n = 0
for line in open(f"/proc/{pid}/maps"):
    if "card1" in line or "renderD129" in line:
        a, b = line.split()[0].split("-"); tot += int(b, 16) - int(a, 16); n += 1
print("mapped from NPU device:", n, "regions", round(tot / 2**20), "MiB")
