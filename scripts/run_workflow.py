#!/usr/bin/env python3
"""Run a workflow's `run` steps locally, in order, with CI-like settings.

Why this exists
---------------
CI on this repository runs on GitHub-hosted runners, and a run can sit queued
until it times out when none are available. That makes "did my change pass?" a
question you can only ask by pushing and waiting — which is exactly the loop
this replaces. Every `run:` step in a workflow is a shell script; there is no
good reason to need a server to execute it.

What it does NOT do: `uses:` steps. Those are third-party actions (checkout,
setup-rust, cache) that install toolchains and are already present on a
development machine. They are listed and skipped, with a note, so the omission
is visible rather than silent.

Environment mirrors the runner where it matters:
  - `bash -e {0}` — the Actions **default** shell. Note: *no* `pipefail`.
    That distinction is not academic. GitHub's default is `bash --noprofile
    --norc -e -o pipefail {0}` only when a step says `shell: bash`; with no
    `shell:` key at all (which is the case for every step in ci.yml) the runner
    uses `bash -e {0}` and a pipeline's status is the *last* command's. An
    earlier version of this script always added pipefail and reported a step as
    failing that CI has passed for months: `ionic_new bad.ionic | grep -q X`
    exits 1 under pipefail because the compiler exits 1, even though grep
    matched. Mirroring the real shell is the whole point, so this follows the
    workflow: pipefail only when the step declares `shell: bash`.
  - `$GITHUB_ENV` / `$GITHUB_OUTPUT` are real files, so steps that persist
    values between steps work rather than silently writing to /dev/null.

Usage:
    python3 scripts/run_workflow.py                     # ci.yml, build-and-test
    python3 scripts/run_workflow.py --list              # show steps, run nothing
    python3 scripts/run_workflow.py --from 18           # start at a step index
    python3 scripts/run_workflow.py --only 18 --only 19 # specific steps
    python3 scripts/run_workflow.py --workflow .github/workflows/ci.yml --job build-and-test
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("run_workflow: needs PyYAML (python3 -m pip install pyyaml)")


def load_steps(path, job_name):
    doc = yaml.safe_load(Path(path).read_text())
    jobs = doc.get("jobs") or {}
    if job_name not in jobs:
        sys.exit(f"run_workflow: no job '{job_name}' in {path}; have: {', '.join(jobs)}")
    return jobs[job_name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workflow", default=".github/workflows/ci.yml")
    ap.add_argument("--job", default="build-and-test")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--from", dest="start", type=int, default=0)
    ap.add_argument("--only", type=int, action="append", default=None)
    ap.add_argument("--keep-going", action="store_true",
                    help="run every step even after one fails")
    args = ap.parse_args()

    job = load_steps(args.workflow, args.job)
    steps = job.get("steps") or []

    todo = []
    for i, s in enumerate(steps):
        if args.only is not None and i not in args.only:
            continue
        if args.only is None and i < args.start:
            continue
        todo.append((i, s))

    if args.list:
        for i, s in enumerate(steps):
            kind = "uses" if "uses" in s else "run"
            print(f"{i:2d} {kind:4s} {s.get('name', '(unnamed)')}")
        return 0

    # $GITHUB_ENV / $GITHUB_OUTPUT are real files here, not the runner's magic.
    # Steps that append to them (`echo X=1 >> $GITHUB_ENV`) then actually
    # influence later steps — without this, a green local run can hide a broken
    # step that only works because CI happens to populate something.
    work = Path(tempfile.mkdtemp(prefix="wf_"))
    env_file = work / "env"
    out_file = work / "output"
    env_file.write_text("")
    out_file.write_text("")

    base_env = dict(os.environ)
    base_env.update({
        "CI": "true",
        "GITHUB_ACTIONS": "true",
        "GITHUB_ENV": str(env_file),
        "GITHUB_OUTPUT": str(out_file),
        "GITHUB_WORKSPACE": str(Path.cwd()),
        # Keep the run cheap and honest: no colour in logs, no cargo colour.
        "NO_COLOR": "1",
        "CARGO_TERM_COLOR": "never",
    })

    def apply_github_env(env):
        """Fold anything appended to $GITHUB_ENV into the environment."""
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()

    failures = []
    for i, step in todo:
        name = step.get("name", f"step {i}")

        if "uses" in step:
            print(f"⤼  [{i:2d}] SKIP (uses: {step['uses']}) — {name}")
            continue

        script = step.get("run")
        if not script:
            print(f"⤼  [{i:2d}] SKIP (no run:) — {name}")
            continue

        env = dict(base_env)
        for k, v in (step.get("env") or {}).items():
            env[k] = str(v)
        apply_github_env(env)

        print(f"▶  [{i:2d}] {name}", flush=True)
        # Match the shell the workflow actually gets. GitHub's *default* (no
        # `shell:` key) is `bash -e {0}` — deliberately without pipefail, so a
        # pipeline's status is the last command's. `shell: bash` opts into
        # pipefail. Mirroring this matters: assuming pipefail everywhere made
        # this harness fail a step CI has passed for months.
        shell = step.get("shell")
        argv = ["bash", "--noprofile", "--norc", "-e"]
        if shell == "bash":
            argv += ["-o", "pipefail"]
        argv += ["-c", script]

        proc = subprocess.run(argv, env=env)
        apply_github_env(env)

        if proc.returncode != 0:
            failures.append((i, name, proc.returncode))
            print(f"✗  [{i:2d}] FAILED (exit {proc.returncode}) — {name}\n", flush=True)
            if not args.keep_going:
                break
        else:
            print(f"✓  [{i:2d}] ok\n", flush=True)

    print("=" * 60)
    if failures:
        for i, name, rc in failures:
            print(f"FAIL  [{i:2d}] exit {rc}  {name}")
        return 1
    ran = sum(1 for _, s in todo if "run" in s)
    print(f"PASS  {ran} run step(s) succeeded"
          f"{f' ({len(todo) - ran} skipped)' if ran != len(todo) else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
