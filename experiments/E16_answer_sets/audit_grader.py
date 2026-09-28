"""Check whether the grading timeout changes any repaired calibration labels."""
import argparse
import json
from math_verify import verify
import data as d
from awv.answers import canonicalize


def run(tag,phase):
    root=d.HERE/tag;audit=json.loads((root/f'{phase}_scoring_audit.json').read_text())
    changed=[];errors=[];checked=0
    for row in audit['changes']:
        for cluster in row['clusters']:
            ans,gold=cluster['representative'],row['gold']
            for digits,key in [(6,'repaired_correct'),(12,'stricter_correct')]:
                checked+=1
                if ans=='<none>':result=False
                else:
                    try:
                        result=bool(verify(d.parsed(gold),d.parsed(ans),float_rounding=digits,
                                           timeout_seconds=5,raise_on_error=True))
                    except BaseException as exc:
                        if isinstance(exc,(KeyboardInterrupt,SystemExit)):raise
                        errors.append(dict(query_id=row['query_id'],answer=ans,digits=digits,error=type(exc).__name__))
                        result=False
                    result=result or canonicalize(ans)==canonicalize(gold)
                if result!=cluster[key]:changed.append(dict(query_id=row['query_id'],answer=ans,digits=digits,
                    cached=cluster[key],longer_timeout=result))
    d.dump(root/f'{phase}_grader_timeout_audit.json',dict(checked=checked,changed=changed,unresolved_errors=errors))
    d.log(tag,phase,'longer grading timeout',checked,'checks,',len(changed),'changes,',len(errors),'unresolved')
    assert not changed,'Repaired labels need rebuilding before using the fitted model'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',required=True);p.add_argument('--phase',default='train')
    args=p.parse_args();run(args.cell,args.phase)
