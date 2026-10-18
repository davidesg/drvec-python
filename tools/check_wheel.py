"""A platform wheel carries drvec C, and it runs (publish.yml, after cibuildwheel).

    python tools/check_wheel.py WHEEL

Checks the tag (py3-none-<platform>), that drvec/_bin/drvec[.exe] is in it with
its executable bit (POSIX), and that the extracted binary prints `drvec 0.13`;
on Linux and macOS it also prints what the binary links (ldd, otool -L), which
must be the C library alone: GSL is static.
"""
import os
import stat
import subprocess
import sys
import tempfile
import zipfile

C_VERSION = "0.13"


def main(path):
    name = os.path.basename(path)
    if "-py3-none-" not in name or name.endswith("-any.whl"):
        sys.exit(f"{name}: not a py3-none-<platform> wheel")
    exe = "drvec.exe" if "win" in name.rsplit("-", 1)[-1] else "drvec"
    member = f"drvec/_bin/{exe}"
    with zipfile.ZipFile(path) as z, tempfile.TemporaryDirectory() as d:
        try:
            info = z.getinfo(member)
        except KeyError:
            sys.exit(f"{name}: no {member}")
        mode = info.external_attr >> 16
        if exe == "drvec" and not mode & stat.S_IXUSR:
            sys.exit(f"{name}: {member} has no executable bit ({oct(mode)})")
        out = z.extract(member, d)
        os.chmod(out, 0o755)
        print(f"{name}: {member}, {info.file_size} bytes")
        v = subprocess.run([out, "--version"], capture_output=True, text=True).stdout
        print(v.strip())
        if v.splitlines()[:1] != [f"drvec {C_VERSION}"]:
            sys.exit(f"{name}: the binary says {v!r}, not drvec {C_VERSION}")
        tool = {"linux": ["ldd"], "darwin": ["otool", "-L"]}.get(sys.platform)
        if tool:
            links = subprocess.run(tool + [out], capture_output=True, text=True).stdout
            print(links)
            if "gsl" in links:
                sys.exit(f"{name}: the binary links GSL dynamically")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
