import sys
import numpy as np

fp = sys.argv[1]
method = sys.argv[2]
with open(fp) as f:
    accs = []
    for line_i, line in enumerate(f):
        if not line_i: print(line.strip())
        if method + "  0" in line:
            acc = float(line.split()[-1])
            accs.append(acc)
print(np.average(accs) if len(accs) else "Zero data", f"({len(accs)})")
