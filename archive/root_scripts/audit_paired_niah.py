"""Convert a multi-seed paired NIAH aggregate into a conservative audit decision."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from core.llm.niah_audit_adapter import paired_niah_audit_decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aggregate', required=True, help='JSON from aggregate_paired_niah.py')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.aggregate).read_text(encoding='utf-8'))
    result = paired_niah_audit_decision(payload)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    print(f'Saved audit decision to {output}')

if __name__ == '__main__':
    main()
