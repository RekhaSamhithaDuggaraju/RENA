#!/usr/bin/env python3
"""CI/CD gate and Secret Scanner for RENA."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

SECRET_PATTERNS = [
    (re.compile(r"AIza[0-9A-Za-z\-_]{35}"), "Google API Key"),
    (re.compile(r"sk-[a-zA-Z0-9]{20,}"), "OpenAI/API Key"),
    (re.compile(r"sk-ant-[a-zA-Z0-9]{20,}"), "Anthropic Key"),
    (re.compile(r"(?i)(api[_-]?key|secret|token)\s*[:=]\s*['\"]([A-Za-z0-9_\-]{24,})['\"]"), "Generic API Key/Secret"),
]

EXCLUDED_DIRS = {".git", ".venv", ".venv-llm", "node_modules", "__pycache__", ".pytest_cache", ".next"}
EXCLUDED_FILES = {".env", ".env.local", ".env.production"}

def scan_repository_for_secrets(root_path: Path) -> list[str]:
    violations = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Prune excluded directories in-place
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS and not d.startswith(".venv")]
        for fname in filenames:
            if fname in EXCLUDED_FILES or fname.endswith((".pyc", ".png", ".jpg", ".safetensors", ".bin")):
                continue
            fpath = Path(dirpath) / fname
            try:
                content = fpath.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            for pattern, desc in SECRET_PATTERNS:
                matches = pattern.finditer(content)
                for m in matches:
                    matched_str = m.group(0)
                    # Ignore obvious false-positive placeholders
                    if any(placeholder in matched_str.lower() for placeholder in ["example", "placeholder", "xxx", "your_key", "test_key", "mock"]):
                        continue
                    # Ignore environment variable references like os.environ.get('GEMINI_API_KEY')
                    if "os.environ" in matched_str or "getenv" in matched_str:
                        continue
                    line_no = content[:m.start()].count("\n") + 1
                    violations.append(f"{fpath.relative_to(root_path)}:{line_no} [{desc}] Potential secret detected")
    return violations

def run_ci_gate(audit_id: str, api_url: str, policy: str, output_json: bool) -> int:
    url = f"{api_url.rstrip('/')}/production/ci/gate/{audit_id}?policy={policy}"
    headers = {}
    if os.environ.get("RENA_API_KEY"):
        headers["X-RENA-API-Key"] = os.environ["RENA_API_KEY"]
    if os.environ.get("RENA_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["RENA_TOKEN"]

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
    except Exception as e:
        print(f"RENA CI gate error: {e}", file=sys.stderr)
        return 2

    if output_json:
        print(json.dumps(data, indent=2))
    else:
        print("RENA CI/CD GATE")
        print("─" * 32)
        print(f"Audit: {data.get('audit_id', audit_id)}")
        print(f"Verdict: {data.get('verdict', 'UNKNOWN')}")
        print(f"Policy: {data.get('policy', policy)}")
        print("Result: PASS" if data.get("passed") else "Result: BLOCK")
    return int(data.get("exit_code", 1))

def main():
    p = argparse.ArgumentParser(description="RENA CI Gate & Secret Scanner")
    p.add_argument("audit_id", nargs="?", default=None, help="Audit ID to evaluate")
    p.add_argument("--api", default=os.environ.get("RENA_API_URL", "http://localhost:8000"))
    p.add_argument("--policy", default="strict")
    p.add_argument("--json", action="store_true")
    p.add_argument("--scan-secrets", action="store_true", help="Scan repository for hardcoded API keys and secrets")
    p.add_argument("--scan-path", default=".", help="Root path to scan for secrets")
    args = p.parse_args()

    if args.scan_secrets:
        root = Path(args.scan_path).resolve()
        print(f"[RENA SECRET SCAN] Scanning {root}...")
        violations = scan_repository_for_secrets(root)
        if violations:
            print(f"\n[FAIL] Found {len(violations)} potential secret leaks:")
            for v in violations:
                print(f"  - {v}")
            sys.exit(1)
        else:
            print("[PASS] No secrets or API keys found in scanned files.")
            sys.exit(0)

    if not args.audit_id:
        p.print_help()
        sys.exit(1)

    sys.exit(run_ci_gate(args.audit_id, args.api, args.policy, args.json))

if __name__ == "__main__":
    main()
