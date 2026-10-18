"""Refresh the vendored drvec C sources (csrc/) from an atsw-gui commit.

    python tools/vendor_drvec_c.py --repo PATH/TO/atsw-gui --commit SHA

Reads every file with `git -C REPO show COMMIT:PATH` -- nothing is checked out,
the repository is only read -- writes it under csrc/ at the same path, and
rewrites csrc/VENDORED.json with the commit and each file's sha256, which
tests/test_vendored.py then checks. The list of files is FILES below: drvec's
sources and headers, the lib/ pieces engines/drvec/Makefile links (optim,
diagmv, fuedet), the Makefile itself (the build csrc/build_drvec.py mirrors)
and the program's licence. If the Makefile at the new commit names a source
this list lacks, the tool stops and says which: the list, build_drvec.py's
SOURCES and the manifest move together.

After a refresh: the version the C prints must still be c_engine.C_VERSION,
and the oracle tests (tests/test_bundled.py) decide whether the port still
agrees with it byte for byte.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSRC = os.path.join(ROOT, "csrc")
MANIFEST = os.path.join(CSRC, "VENDORED.json")

FILES = [
    "engines/drvec/LICENSE",
    "engines/drvec/Makefile",
    "engines/drvec/include/diagmvhost.h",
    "engines/drvec/include/fue_bridge.h",
    "engines/drvec/include/fue_pre_reader.h",
    "engines/drvec/include/main.h",
    "engines/drvec/include/optimhost.h",
    "engines/drvec/src/diagnose.c",
    "engines/drvec/src/drvec.c",
    "engines/drvec/src/drvmlest.c",
    "engines/drvec/src/elfvarma.c",
    "engines/drvec/src/fue_bridge.c",
    "engines/drvec/src/fue_pre_reader.c",
    "engines/drvec/src/nlatools.c",
    "engines/drvec/src/qnewtopt.c",
    "lib/diagmv/diagmv.c",
    "lib/diagmv/diagmv.h",
    "lib/fuedet/fuedet.c",
    "lib/fuedet/fuedet.h",
    "lib/optim/lnsrch.c",
]


def git_show(repo, commit, path):
    return subprocess.run(["git", "-C", repo, "show", f"{commit}:{path}"],
                          capture_output=True, check=True).stdout


def makefile_sources(text):
    """The .c files engines/drvec/Makefile compiles into the executable."""
    srcs = set(re.findall(r"\$\(SRC_DIR\)/(\w+)\.c", text))
    objs = set(re.findall(r"\$\(BUILD_DIR\)/(\w+)\.o", text.split("EXEC =")[0]))
    return srcs | objs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", required=True, help="an atsw-gui clone (only read)")
    ap.add_argument("--commit", required=True, help="the commit to vendor")
    a = ap.parse_args(argv)
    commit = subprocess.run(["git", "-C", a.repo, "rev-parse", "--verify",
                             a.commit + "^{commit}"], capture_output=True, text=True,
                            check=True).stdout.strip()
    mk = git_show(a.repo, commit, "engines/drvec/Makefile").decode()
    have = {os.path.basename(f)[:-2] for f in FILES if f.endswith(".c")}
    missing = sorted(makefile_sources(mk) - have)
    if missing:
        sys.exit("engines/drvec/Makefile at %s links %s, which FILES lacks: add them "
                 "here and to csrc/build_drvec.py's SOURCES" % (commit[:7], ", ".join(missing)))
    files = {}
    for path in FILES:
        data = git_show(a.repo, commit, path)
        dest = os.path.join(CSRC, *path.split("/"))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(data)
        files[path] = hashlib.sha256(data).hexdigest()
    man = {
        "what": "drvec C, the oracle of this port, and the lib/ pieces it links, "
                "copied byte for byte from atsw-gui by tools/vendor_drvec_c.py",
        "repository": "https://github.com/davidesg/atsw-gui",
        "commit": commit,
        "files": files,
    }
    with open(MANIFEST, "w") as f:
        json.dump(man, f, indent=2)
        f.write("\n")
    print(f"{len(files)} files from atsw-gui {commit[:7]} into csrc/; manifest rewritten")


if __name__ == "__main__":
    main()
