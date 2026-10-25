"""The vendored drvec C (csrc/, PLAN-vec B5): byte-identical copies of atsw-gui
at the commit csrc/VENDORED.json records.

The C sources live in atsw-gui; csrc/ holds copies, and the wheel's binary
(drvec/_bin/drvec) is built from them. Every file must have the sha256 the
manifest records, and the manifest must name every file csrc/ carries, so that
nothing is edited here and nothing slips in. To update: tools/vendor_drvec_c.py
--repo <atsw-gui> --commit <sha>, which rewrites the files and the manifest
together.

When an atsw-gui clone is at hand (ATSW_GUI, or a sibling directory atsw-gui),
the copies are also compared with `git show <commit>:<path>` in it: read only.

The sdist carries csrc/ (it builds the binary); the installed package does
not, so these tests skip against an installed wheel.
"""
import hashlib
import json
import os
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSRC = os.path.join(ROOT, "csrc")
MANIFEST = os.path.join(CSRC, "VENDORED.json")
# csrc/'s own files, not vendored
OURS = {"VENDORED.json", "build_drvec.py"}

if not os.path.isfile(MANIFEST):
    pytest.skip("no csrc/ (an installed wheel carries the binary, not its sources)",
                allow_module_level=True)

MAN = json.load(open(MANIFEST))


def test_the_manifest_records_the_oracle_commit():
    assert len(MAN["commit"]) == 40
    # drvec C 0.13 with BUG-61 to BUG-64 fixed, -gzero and the I(1, r) check: atsw-gui
    # 800a540 (main after PRs #93-#94; tests/data/oracle/SOURCE)
    assert MAN["commit"].startswith("800a540")


@pytest.mark.parametrize("path", sorted(MAN["files"]))
def test_each_vendored_file_is_atsw_guis(path):
    with open(os.path.join(CSRC, *path.split("/")), "rb") as f:
        got = hashlib.sha256(f.read()).hexdigest()
    assert got == MAN["files"][path], (
        f"csrc/{path} is not atsw-gui {MAN['commit'][:7]}'s: the C lives in atsw-gui; "
        "refresh with tools/vendor_drvec_c.py, never edit the copy")


def test_nothing_else_in_csrc():
    found = set()
    for d, dirs, files in os.walk(CSRC):
        dirs[:] = [x for x in dirs if x != "__pycache__"]
        for f in files:
            rel = os.path.relpath(os.path.join(d, f), CSRC).replace(os.sep, "/")
            if rel not in OURS:
                found.add(rel)
    assert found == set(MAN["files"])


def test_the_build_compiles_what_the_makefile_links():
    """build_drvec.SOURCES are the Makefile's translation units, all vendored."""
    import importlib.util
    import re
    spec = importlib.util.spec_from_file_location("build_drvec",
                                                  os.path.join(CSRC, "build_drvec.py"))
    bd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bd)
    mk = open(os.path.join(CSRC, "engines", "drvec", "Makefile")).read()
    units = set(re.findall(r"\$\(SRC_DIR\)/(\w+)\.c", mk)) | \
        set(re.findall(r"\$\(BUILD_DIR\)/(\w+)\.o", mk.split("EXEC =")[0]))
    built = {os.path.basename(s)[:-2] for s in bd.SOURCES}
    assert built == units
    for s in bd.SOURCES:
        assert s.replace(os.sep, "/") in MAN["files"]
    from drvec.c_engine import C_VERSION
    assert bd.C_VERSION == C_VERSION


def _atsw_gui():
    for c in (os.environ.get("ATSW_GUI", ""), os.path.join(ROOT, "..", "atsw-gui")):
        if c and os.path.isdir(os.path.join(c, ".git")):
            return c
    return None


def test_against_an_atsw_gui_clone():
    repo = _atsw_gui()
    if repo is None:
        pytest.skip("no atsw-gui clone at hand (ATSW_GUI)")
    for path, sha in MAN["files"].items():
        cp = subprocess.run(["git", "-C", repo, "show", f"{MAN['commit']}:{path}"],
                            capture_output=True)
        if cp.returncode != 0:
            pytest.skip(f"the clone at {repo} lacks {MAN['commit'][:7]}")
        assert hashlib.sha256(cp.stdout).hexdigest() == sha, path
