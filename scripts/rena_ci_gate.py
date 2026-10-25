#!/usr/bin/env python3
"""CI/CD gate for a completed RENA audit."""
import argparse, json, os, sys, urllib.request
p=argparse.ArgumentParser();p.add_argument('audit_id');p.add_argument('--api',default=os.environ.get('RENA_API_URL','http://localhost:8000'));p.add_argument('--policy',default='strict');p.add_argument('--json',action='store_true');a=p.parse_args()
url=f"{a.api.rstrip('/')}/production/ci/gate/{a.audit_id}?policy={a.policy}"
headers={}
if os.environ.get('RENA_API_KEY'): headers['X-RENA-API-Key']=os.environ['RENA_API_KEY']
if os.environ.get('RENA_TOKEN'): headers['Authorization']='Bearer '+os.environ['RENA_TOKEN']
req=urllib.request.Request(url,headers=headers)
try:
    data=json.load(urllib.request.urlopen(req,timeout=30))
except Exception as e:
    print(f'RENA CI gate error: {e}',file=sys.stderr);sys.exit(2)
if a.json: print(json.dumps(data,indent=2))
else:
    print('RENA CI/CD GATE');print('─'*32);print(f"Audit: {data['audit_id']}");print(f"Verdict: {data['verdict']}");print(f"Policy: {data['policy']}");print('Result: PASS' if data['passed'] else 'Result: BLOCK')
sys.exit(int(data['exit_code']))
