"""Readers of the oracle's outputs and of the captured goldens (data/golden)."""
import json
import os
import re

from conftest import DATA

GOLDEN = os.path.join(DATA, "golden")
ORACLE = os.path.join(DATA, "oracle")


def est_pairs(run):
    """[(start, logelf, sigma2, ifault, x_final)] in the C's call order."""
    out = []
    lines = open(os.path.join(GOLDEN, f"{run}.est.txt")).read().splitlines()
    for a, b in zip(lines[0::2], lines[1::2]):
        assert a.startswith("START") and b.startswith("EST")
        start = [float(v) for v in a.split(" x= ")[1].split()]
        hd, xe = b.split(" x= ")
        f = dict(kv.split("=") for kv in hd.split()[1:])
        out.append((start, float(f["logelf"]), float(f["sigma2"]), int(f["ifault"]),
                    [float(v) for v in xe.split()]))
    return out


def specs(run):
    return json.load(open(os.path.join(GOLDEN, "specs.json")))[run]


def out_text(name):
    return open(os.path.join(ORACLE, name)).read()


_ROW = re.compile(r"^  (\S.*?)\s{2,}(-?\d+\.\d+)\s+(-?\d+\.\d+|-)\s+(\S+)\s+(\S+)")


def param_table(text):
    """{label: (estimate, se or None)} from 'Estimated Parameters'."""
    body = text.split("Estimated Parameters and Standard Deviations")[1]
    body = body.split("Signif. codes")[0]
    out = {}
    for line in body.splitlines():
        m = _ROW.match(line)
        if m:
            out[m.group(1).strip()] = (float(m.group(2)),
                                       None if m.group(3) == "-" else float(m.group(3)))
    return out


def number_after(text, key):
    m = re.search(re.escape(key) + r"\s*(-?\d+\.\d+(?:e[-+]\d+)?)", text)
    return float(m.group(1))


def boot_rows(name="rankboot.boot"):
    rows = []
    for line in open(os.path.join(ORACLE, name)):
        if line[0].isdigit():
            h0, rep, st, lr = line.strip().split(",")
            rows.append((int(h0), int(rep), st, float(lr) if lr else None))
    return rows


_DROP = [re.compile(r"^DRVEC .* EML Estimation"), re.compile(r"^Program  *: DRVEC"),
         re.compile(r"^# drvec .* -bootstrap ")]


def golden_filter(text):
    """tools/golden.sh's hash_of filter: the version banner lines dropped and any
    /tmp path removed; everything else counts."""
    out = []
    for line in text.split("\n"):
        if any(p.search(line) for p in _DROP):
            continue
        out.append(re.sub(r"/tmp/[^ ]*", "", line))
    return "\n".join(out)
