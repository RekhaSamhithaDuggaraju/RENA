from app.verdicts import authoritative_verdict, authoritative_score
from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.audit import Audit
from app.auth import current_user, require_audit_access
router=APIRouter(prefix='/compare',tags=['comparison'])
@router.post('')
def compare(body:dict, db:Session=Depends(get_db), authorization:str|None=Header(default=None)):
    user=current_user(authorization,db)
    ids=body.get('audit_ids',[])
    if not isinstance(ids,list) or len(ids)<2: raise HTTPException(400,'Provide at least two audit_ids')
    rows=db.query(Audit).filter(Audit.id.in_(ids)).all(); by={r.id:r for r in rows}
    for a in rows: require_audit_access(a.id,db,user)
    missing=[i for i in ids if i not in by]
    if missing: raise HTTPException(404,detail={'missing':missing})
    out=[]
    for i in ids:
        a=by[i]; out.append({'id':a.id,'algorithm':a.algorithm,'verdict':authoritative_verdict(a),'score':authoritative_score(a),'classical_verdict':a.consensus_verdict,'agreement':a.consensus_agreement,'retain':a.accuracy_retain,'forget':a.accuracy_forget,'test':a.accuracy_test,'runtime':a.runtime_seconds})
    return {'audits':out,'deltas':{'score_range':max(x['score'] for x in out if x['score'] is not None)-min(x['score'] for x in out if x['score'] is not None) if any(x['score'] is not None for x in out) else None}}
