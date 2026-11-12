"""Every Mach-O inside the macOS wheels asks for at most macOS TARGET.

    python tools/check_macos_minos.py 11.0 wheelhouse/*.whl

The wheel's tag (macosx_11_0_*) is a promise that it installs and runs on
that macOS; what decides it is the minimum each binary inside declares
(LC_BUILD_VERSION minos, or LC_VERSION_MIN_MACOSX on older toolchains). A
library linked in from a newer macOS (a Homebrew bottle, say) raises it
silently: pip installs the wheel, and the import fails on the user's Mac.
This stops the workflow instead. Also checked: the tag says TARGET, and the
binaries are of the wheel's architecture. Run on the macOS runner (otool,
lipo come with the Xcode command line tools).
"""
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

MACHO = {bytes.fromhex(m) for m in (
    "feedface", "cefaedfe", "feedfacf", "cffaedfe", "cafebabe", "bebafeca")}


def _v(s):
    return tuple(int(x) for x in s.split("."))


def minos(path):
    """The minimum macOS each architecture slice of PATH declares."""
    out = subprocess.run(["otool", "-l", str(path)], capture_output=True,
                         text=True, check=True).stdout.splitlines()
    found, want = [], None
    for line in out:
        w = line.split()
        if not w:
            continue
        if w[0] == "cmd":
            want = {"LC_BUILD_VERSION": "minos",
                    "LC_VERSION_MIN_MACOSX": "version"}.get(w[1])
        elif want and w[0] == want:
            found.append(w[1])
            want = None
    return found


def main(target, wheels):
    bad = 0
    for whl in wheels:
        name = Path(whl).name
        tag = name[:-4].split("-")[-1]
        arch = tag.split("_", 3)[-1] if tag.startswith("macosx_") else None
        print(f"{name}")
        if not tag.startswith(f"macosx_{target.replace('.', '_')}_"):
            print(f"  TAG: {tag} is not macosx_{target.replace('.', '_')}_*")
            bad += 1
        with tempfile.TemporaryDirectory() as d:
            zipfile.ZipFile(whl).extractall(d)
            n = 0
            for f in sorted(Path(d).rglob("*")):
                if not f.is_file():
                    continue
                with open(f, "rb") as fh:
                    if fh.read(4) not in MACHO:
                        continue
                n += 1
                rel = f.relative_to(d)
                ms = minos(f)
                archs = subprocess.run(["lipo", "-archs", str(f)], capture_output=True,
                                       text=True).stdout.split()
                over = [m for m in ms if _v(m) > _v(target)]
                wrong = arch and arch != "universal2" and archs != [arch]
                flag = "" if ms and not over and not wrong else "  <-- "
                if not ms:
                    flag += "no minimum declared"
                if over:
                    flag += f"asks for macOS {max(over, key=_v)} > {target}"
                if wrong:
                    flag += f"architecture {archs}, the tag says {arch}"
                print(f"  {rel}: minos {','.join(ms) or '?'} [{' '.join(archs)}]{flag}")
                bad += bool(flag)
            if n == 0:
                print("  no Mach-O inside: a binary wheel without its binary")
                bad += 1
    if bad:
        sys.exit(f"{bad} problem(s): the wheels do not hold the macOS {target} promise")
    print(f"OK: every binary asks for at most macOS {target}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
