"""Verify the complete assigned v3 record grid and actual bank file checksums."""
import argparse,json,time
from pathlib import Path
from scripts.refine_strength_bank import sha,atomic


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--host',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text());ph=sha(a.plan);parent=json.loads(Path(plan['parent_index']).read_text())[a.host];completion=json.loads((a.root/'completion.json').read_text())
    assert completion['all_quality_gates_pass'] and completion['completed_trials']==len(parent) and completion['plan_sha256']==ph
    paths={str(p.parent.relative_to(a.root)):p for p in a.root.glob('s*/t*/record.json')};assert set(paths)==set(parent)
    rows={}
    for rel,p in sorted(paths.items()):
        r=json.loads(p.read_text());assert r['passed'] and r['plan_sha256']==ph and r['partition']=='T' and not r['E_set_accessed']
        assert r['parent_record_sha256']==parent[rel]['record_sha256'] and r['parent_bank_sha256']==parent[rel]['bank_sha256']
        assert sha(p.with_name('bank.npz'))==r['bank_sha256']
        rows[rel]={'record_sha256':sha(p),'bank_sha256':r['bank_sha256'],'training_data_sha256':r['training_data_sha256'],'trial_id':r['trial_id']}
    atomic(a.out,dict(accepted=True,plan_sha256=ph,host=a.host,completion=completion,records=rows,audited_unix_time=time.time()))
    print(json.dumps({'host':a.host,'verified_banks':len(rows),'accepted':True}))


if __name__=='__main__':main()
