"""Run the pre-heldout R2 four-chain scalar diagnostic on four chain reports."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from cf4_r2_chain_diagnostics import four_chain_scalar_diagnostics


def evaluate_reports(report_paths):
    """Load exactly four distinct report files and return a non-promoting gate."""
    paths=[Path(path).resolve() for path in report_paths]
    if len(paths)!=4 or len(set(paths))!=4:
        raise ValueError('exactly four distinct chain report paths are required')
    reports=[]
    inputs=[]
    for path in paths:
        raw=path.read_bytes()
        report=json.loads(raw)
        if not isinstance(report,dict):
            raise ValueError(f'chain report must be a JSON object: {path}')
        reports.append(report)
        inputs.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),
            chain_id=report.get('chain_id')))
    result=four_chain_scalar_diagnostics(reports)
    result['input_reports']=inputs
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports',nargs=4,type=Path,
        help='four completed independent-prior chain result.json files')
    parser.add_argument('--output',type=Path,
        help='new JSON path; existing files are never overwritten (default: stdout)')
    args=parser.parse_args(argv)
    try:
        result=evaluate_reports(args.reports)
        payload=json.dumps(result,indent=2,allow_nan=False)+'\n'
        if args.output is None:
            sys.stdout.write(payload)
        else:
            with args.output.open('x',encoding='utf-8') as handle:
                handle.write(payload)
    except (OSError,json.JSONDecodeError,TypeError,ValueError) as error:
        print(f'four-chain diagnostic refused: {error}',file=sys.stderr)
        return 2
    return 0 if result['scalar_gate']=='PASS' else 2


if __name__=='__main__':
    raise SystemExit(main())
