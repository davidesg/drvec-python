"""Port defect (drvec-python, found 2026-10-18 by tests.yml on windows-latest,
Python 3.10, job 117699907666, the first run with drvec C on Windows):
c_engine.Job.wait() follows the C's <base>.boot while the C writes it, and on
Windows it reads lines that are not there.

Job.wait opens the .boot in TEXT mode, seeks to `seen` and reads what is new,
then adds len(chunk.encode()) to `seen`. On Windows the C writes its text
files with CRLF, and text mode turns each "\\r\\n" into "\\n": the chunk is one
character shorter per line than the bytes it came from, so `seen` falls
behind the file's offset by one byte per line read. The next read starts
inside lines already delivered, and their tails come out as new lines:
rank_test(engine="auto") failed with `ValueError: not enough values to unpack`
on the line '8938564272', the end of an LR*. It needs the C to write between
two polls, so it is intermittent (Python 3.13 passed in the same run). On
Linux and macOS there is no CR, and the offsets agree.

This repro drives Job.wait with a fake process whose poll() appends the next
piece of a CRLF .boot, as the C on Windows does, and checks that every line
delivered is one the C wrote. Expected: the file read in binary, the offset
counted in bytes, the CR dropped.

    python3 repro.py        # exits 1 while the defect is open
"""
import io
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from drvec import c_engine  # noqa: E402

LINES = ["# drvec 0.13 -bootstrap 4: the rank test, one line per replication",
         "h0,rep,status,lr",
         "0,1,ok,15.744416530514741",
         "0,2,ok,13.915693523537129",
         "0,3,ok,15.71179217705037",
         "0,4,ok,19.728025245693516"]
DATA = "".join(s + "\r\n" for s in LINES).encode()
# the C flushes after each replication; here, two lines at a time and one cut
# in the middle of a number
CUTS = [len("".join(s + "\r\n" for s in LINES[:3])),
        len("".join(s + "\r\n" for s in LINES[:4])) + 7,
        len(DATA)]


class FakeProc:
    def __init__(self, path):
        self.path, self.k, self.done = path, 0, 0

    def poll(self):
        if self.k < len(CUTS):
            with open(self.path, "ab") as f:
                f.write(DATA[self.done:CUTS[self.k]])
            self.done = CUTS[self.k]
            self.k += 1
            return None
        return 0


with tempfile.TemporaryDirectory() as d:
    job = object.__new__(c_engine.Job)
    job.folder, job.base, job.t0 = d, "drvecjob", 0.0
    job._out, job._err = io.StringIO(), io.StringIO()
    job.proc = FakeProc(job.path(".boot"))
    got = []
    job.wait(on_boot_line=lambda s: got.append(s.rstrip("\r")), poll=0)

bad = [s for s in got if s not in LINES]
print("lines written:  ", len(LINES))
print("lines delivered:", len(got), "  not written by the C:", bad)
for s in got:
    try:
        c_engine.parse_boot_line(s)
    except ValueError as e:
        print("  parse_boot_line(%r): %s" % (s, e))
sys.exit(1 if bad or got != LINES else 0)
