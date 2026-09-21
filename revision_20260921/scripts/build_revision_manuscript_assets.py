"""Build manuscript tables/figures from complete locked revision records."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.summarize_revision import contrast_statistics

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'results/revision_manuscript_20260921'
TABLES=ROOT/'paper/revision_2026-09/tables'
FIGURES=ROOT/'paper/revision_2026-09/figures'
NAMES={'bci2a':'BCI 2a','bci2b':'BCI 2b','lee2019':'Lee2019','schirrmeister':'HGD'}


def pvalue(p):
    if p < .0001:return '$<0.0001$'
    return f'${p:.4f}$'


def interval(r,key='model_95ci_pp'):
    lo,hi=r[key]
    return f"${r['mean_difference_pp']:+.2f}$ [$ {lo:+.2f}, {hi:+.2f} $]"


def table(name,columns,header,rows):
    content=[r'\begin{tabular}{@{}'+columns+r'@{}}',r'\toprule',
             ' & '.join(header)+r' \\',r'\midrule']
    content+=[' & '.join(row)+r' \\' for row in rows]
    content += [r'\bottomrule',r'\end{tabular}']
    (TABLES/(name+'.tex')).write_text('\n'.join(content)+'\n')


def main(*, from_snapshot=False):
    for p in (EVIDENCE,TABLES,FIGURES):p.mkdir(parents=True,exist_ok=True)
    source=ROOT/'runs/dsp-revision-collected/summary/summary.json'
    if source.exists() and not from_snapshot:
        raw=json.loads(source.read_text())
        if not raw['all_locked_jobs_evaluated'] or raw['evaluated_jobs']!=10760:
            raise RuntimeError('full locked main grid required')
        (EVIDENCE/'main_summary.json').write_text(json.dumps(raw,indent=2)+'\n')
        for dataset,folder in [('schirrmeister','gcn-revision-collected'),('lee2019','gcn-lee-revision-collected')]:
            s=json.loads((ROOT/'runs'/folder/'summary.json').read_text())
            assert s['complete'] and s['fits']==(280 if dataset=='schirrmeister' else 1080)
            (EVIDENCE/(dataset+'_gcn.json')).write_text(json.dumps(s,indent=2)+'\n')
        records={};trained={}
        for host in ('gblinux','gblab146'):
            for p in (ROOT/'runs/dsp-revision-collected'/host/'jobs').glob('*/evaluation.json'):
                d=json.loads(p.read_text());identifier=d['job']['id']
                if identifier in records and records[identifier]!=d:raise RuntimeError('conflicting record')
                records[identifier]=d
            for p in (ROOT/'runs/dsp-revision-collected'/host/'jobs').glob('*/train.json'):
                trained[p.parent.name]=json.loads(p.read_text())
        assert len(records)==10760 and all(d['manifest_sha256']==raw['manifest_sha256'] for d in records.values())
        def matrix(dataset,condition,r=5,rule='primary'):
            n={'bci2a':9,'bci2b':9,'lee2019':54,'schirrmeister':14}[dataset]
            return np.array([[records[f'{dataset}_s{s:02d}_r{seed}_{condition}']['metrics'][rule]['fixed_accuracy']
                              for seed in range(1001,1001+r)] for s in range(1,n+1)])
        matched={}
        for dataset in NAMES:
            cells={c:float(matrix(dataset,c).mean()*100) for c in ['fixed','resampled','real','baseline','zero']}
            matched[dataset]={'cells':cells,'schedule':contrast_statistics(matrix(dataset,'resampled')-matrix(dataset,'fixed'))}
        diagnostics={}
        for dataset in NAMES:
            for condition in ['fixed','resampled','real']:
                rows=[r for r in trained.values() if r['job']['dataset']==dataset and r['job']['condition']==condition]
                assert len(rows)=={'bci2a':90,'bci2b':90,'lee2019':540,'schirrmeister':140}[dataset]
                diagnostics[dataset+':'+condition]={'fits':len(rows),'cap_without_stop':sum(x['hit_maximum_without_stop'] for x in rows),
                    'primary_epoch_median':float(np.median([x['checkpoints']['primary']['epoch'] for x in rows])),
                    'trajectory_epoch_median':float(np.median([x['epochs_completed'] for x in rows]))}
        mc={}
        for dataset in NAMES:
            for condition in ['fixed','resampled']:
                rows=[r['metrics']['primary'] for r in records.values() if r['job']['dataset']==dataset and r['job']['condition']==condition]
                mc[dataset+':'+condition]={key:float(np.mean([r[key] for r in rows])*100)
                                         for key in ['fixed_accuracy','mc_mean_accuracy','probability_ensemble_accuracy']}
        (EVIDENCE/'supplementary_metrics.json').write_text(json.dumps({'matched_five':matched,'epoch_diagnostics':diagnostics,'mc_cells':mc},indent=2)+'\n')
    m=json.loads((EVIDENCE/'main_summary.json').read_text());c=m['contrasts'];cells=m['cells']
    aux=json.loads((EVIDENCE/'supplementary_metrics.json').read_text());matched=aux['matched_five']
    gcn={d:json.loads((EVIDENCE/(d+'_gcn.json')).read_text()) for d in ['lee2019','schirrmeister']}
    rows=[]
    for d,label in NAMES.items():
        r=c[d+':schedule']
        rows.append([label,str(r['n_subjects']),*[f"{cells[d+':'+k]['mean_accuracy_percent']:.2f}" for k in ['fixed','resampled','real']],interval(r),pvalue(r['p_holm'])])
    table('core','lrrrrlr',['Dataset','$n$','$F$ (\\%)','$R$ (\\%)','Real (\\%)','$R-F$ [95\\% CI] (pp)','$p_{\\rm Holm}$'],rows)
    rows=[]
    for d,label in NAMES.items():
        for condition in ['fixed','resampled']:
            r=c[d+':inference:'+condition]
            rows.append([label,'Inference, '+('F' if condition=='fixed' else 'R'),interval(r,'model_90ci_pp'),pvalue(r['p_equivalence_holm'])])
    for d in ['lee2019','schirrmeister']:
        r=c[d+':global_minus_real'];rows.append([NAMES[d],'Global $-$ real',interval(r,'model_90ci_pp'),pvalue(r['p_equivalence_holm'])])
    table('equivalence','lllr',['Dataset','Contrast','$\\Delta$ [90\\% CI] (pp)','TOST $p_{\\rm Holm}$'],rows)
    rows=[]
    for d in ['lee2019','schirrmeister']:
        for norm,label in [('batchnorm','BatchNorm'),('none','None'),('layernorm','LayerNorm'),('frozen','Fixed standardization'),('bn_noaffine','BN, no affine')]:
            if norm=='batchnorm':means=matched[d]['cells'];r=matched[d]['schedule'];ps='---'
            else:
                means={k:cells[d+':'+k+'_'+norm]['mean_accuracy_percent'] for k in ['fixed','resampled']}
                r=c[d+':schedule:'+norm];ps=pvalue(c[d+':normalization_interaction:'+norm]['p_holm'])
            rows.append([NAMES[d],label,f"{means['fixed']:.2f}",f"{means['resampled']:.2f}",interval(r),ps])
    table('normalization','llrrlr',['Dataset','Normalization','$F$','$R$','$R-F$ [95\\% CI] (pp)','$p_H$'],rows)
    rows=[]
    for d in ['lee2019','schirrmeister']:
        for k,label in [('mu','Mu (8--13 Hz)'),('beta','Beta (13--30 Hz)'),('edge','Padded/cropped Hilbert'),('donor','Whole-vector donor')]:
            r=c[d+':schedule:'+k];ps=pvalue(r['p_holm']) if k in ('mu','beta') else '---'
            rows.append([NAMES[d],label,interval(r),ps])
    table('sensitivities','lllr',['Dataset','Condition','$R-F$ [95\\% CI] (pp)','$p_{\\rm Holm}$'],rows)
    rows=[]
    for dimension in [128,512,2048]:
        r=c[f'schirrmeister:dimension:{dimension}'];interaction=c[f'schirrmeister:dimension_interaction:{dimension}']
        means=[np.mean([cells[f'schirrmeister:{arm}_p{dimension}_mask{mask}']['mean_accuracy_percent'] for mask in [1,2,3]]) for arm in ['fixed','resampled']]
        rows.append([f'{dimension:,}',*[f'{x:.2f}' for x in means],interval(r),interval(interaction),pvalue(interaction['p_holm'])])
    rows.append(['8,128',*[f"{matched['schirrmeister']['cells'][arm]:.2f}" for arm in ['fixed','resampled']],interval(matched['schirrmeister']['schedule']),'Reference','---'])
    table('dimensions','rrrllr',['$P$','$F$','$R$','$R-F$ [95\\% CI] (pp)','Change vs full [95\\% CI]','$p_{\\rm Holm}$'],rows)
    rows=[]
    for d,label in NAMES.items():
        for condition in ['baseline','zero']:
            r=c[d+':real_minus_'+condition];rows.append([label,'Real $-$ '+condition,interval(r),pvalue(r['p_holm'])])
    table('anchors','lllr',['Dataset','Contrast','$\\Delta$ [95\\% CI] (pp)','$p_{\\rm Holm}$'],rows)
    anchor_contrasts=(TABLES/'anchors.tex').read_text()
    rows=[[NAMES[d],*[f"{matched[d]['cells'][arm]:.2f}" for arm in ['fixed','resampled','real','baseline','zero']]] for d in NAMES]
    table('anchors','lrrrrr',['Dataset','$F$','$R$','Real','Baseline','Zero'],rows)
    anchor_means=(TABLES/'anchors.tex').read_text()
    (TABLES/'anchors.tex').write_text(anchor_means+'\n\\par\\medskip\n'+anchor_contrasts)
    rows=[]
    for d,s in gcn.items():
        for name,label in [('schedule','$R-F$ (primary)'),('real_minus_identity','Real $-$ identity (secondary)')]:
            r=s['contrasts']['primary:'+name];rows.append([NAMES[d],label,interval(r),pvalue(r['p_two_sided'])])
    table('gcn','lllr',['Dataset','Contrast','$\\Delta$ [95\\% CI] (pp)','$p$'],rows)
    rows=[]
    for norm,label in [('batchnorm','BatchNorm'),('none','None')]:
        rows.append([label,'0','Reference'])
        for q in [.05,.2]:rows.append([label,str(q),interval(c[f'schirrmeister:refresh:{q}:{norm}'])])
        r=matched['schirrmeister']['schedule'] if norm=='batchnorm' else c['schirrmeister:schedule:none']
        rows.append([label,'1',interval(r)])
    table('refresh','lrl',['Normalization','Refresh $q$','Accuracy $-$ fixed [95\\% CI] (pp)'],rows)
    rows=[]
    for d,label in NAMES.items():
        for rule,tag in [('primary','Validation loss'),('accuracy','Validation accuracy'),('fixed_budget','Fixed budget')]:
            r=c[d+':schedule'+('' if rule=='primary' else ':'+rule)]
            rows.append([label,tag,interval(r)])
    table('checkpoint','lll',['Dataset','Checkpoint rule','$R-F$ [95\\% CI] (pp)'],rows)
    rows=[]
    for d,label in NAMES.items():
        vals=c[d+':schedule']['per_replicate_mean_pp']
        rows.append([label]+[f'{v:+.2f}' for v in vals])
    table('replicates','l'+'r'*10,['Dataset']+[str(i) for i in range(1001,1011)],rows)
    rows=[]
    for k,r in aux['epoch_diagnostics'].items():
        d,condition=k.split(':');rows.append([NAMES[d],condition,str(r['fits']),f"{r['primary_epoch_median']:g}",f"{r['trajectory_epoch_median']:g}",str(r['cap_without_stop'])])
    table('epochs','llrrrr',['Dataset','Condition','Fits','Selected median','Trained median','Cap/no stop'],rows)
    rows=[]
    for k,r in aux['mc_cells'].items():
        d,condition=k.split(':');rows.append([NAMES[d],condition]+[f'{v:.3f}' for v in r.values()]+[f"{c[d+':inference:'+condition]['mc_standard_error_of_grand_mean_pp']:.3f}"])
    table('inference_cells','llrrrr',['Dataset','Training','Fixed (\\%)','Draw mean (\\%)','Ensemble (\\%)','MCSE (pp)'],rows)
    rows=[]
    for d,label in NAMES.items():
        r=c[d+':schedule'];lo,hi=r['conditional_subject_bootstrap_95ci_pp']
        rows.append([label,f"{r['between_subject_variance_pp2']:.3f}",f"{r['within_subject_run_variance_pp2']:.3f}",f'[{lo:+.2f}, {hi:+.2f}]'])
    table('uncertainty','lrrl',['Dataset','Between variance','Within-run variance','Conditional bootstrap 95\\% CI'],rows)
    rows=[]
    for d,s in gcn.items():
        for rule,label in [('primary','Validation loss'),('accuracy','Validation accuracy'),('fixed_budget','Fixed budget')]:
            rows.append([NAMES[d],label]+[f"{s['cells'][rule][k]:.2f}" for k in ['fixed','resampled','real','baseline']]+[interval(s['contrasts'][rule+':schedule'])])
    table('gcn_cells','llrrrrl',['Dataset','Checkpoint','$F$','$R$','Real','Identity','$R-F$ [95\\% CI]'],rows)
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(6.4,3.3),layout='constrained')
    for i,(d,label) in enumerate(NAMES.items()):
        r=c[d+':schedule'];v=r['mean_difference_pp'];lo,hi=r['model_95ci_pp']
        ax.errorbar(v,3-i,xerr=[[v-lo],[hi-v]],fmt='o',color='#16697a',capsize=4)
    ax.axvline(0,color='0.6',lw=1);ax.set_yticks(range(4),list(NAMES.values())[::-1]);ax.set_xlabel('Resampled − fixed accuracy (percentage points)')
    fig.savefig(FIGURES/'core_schedule.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8,3.5),layout='constrained')
    for norm,label,color in [('batchnorm','Auxiliary BatchNorm','#16697a'),('none','No auxiliary normalization','#c66332')]:
        qs=[0,.05,.2,1];vals=[0];low=[0];high=[0]
        for q in qs[1:]:
            r=(matched['schirrmeister']['schedule'] if norm=='batchnorm' else c['schirrmeister:schedule:none']) if q==1 else c[f'schirrmeister:refresh:{q}:{norm}']
            v=r['mean_difference_pp'];lo,hi=r['model_95ci_pp'];vals.append(v);low.append(v-lo);high.append(hi-v)
        axes[0].errorbar(range(4),vals,yerr=[low,high],fmt='o-',capsize=3,label=label,color=color)
    axes[0].set_xticks(range(4),['0','.05','.2','1']);axes[0].set_xlabel('Per-trial refresh probability q');axes[0].set_ylabel('Accuracy − fixed (pp)');axes[0].legend(fontsize=8)
    axes[0].set_title('(a) Refresh sensitivity',fontsize=11)
    rows=[c[f'schirrmeister:dimension:{d}'] for d in [128,512,2048]]+[matched['schirrmeister']['schedule']]
    vals=[r['mean_difference_pp'] for r in rows];lo=[v-r['model_95ci_pp'][0] for v,r in zip(vals,rows)];hi=[r['model_95ci_pp'][1]-v for v,r in zip(vals,rows)]
    axes[1].errorbar(range(4),vals,yerr=[lo,hi],fmt='o',capsize=3,color='#16697a');axes[1].set_xticks(range(4),['128','512','2,048','8,128']);axes[1].set_xlabel('Auxiliary coordinates');axes[1].set_ylabel('Resampled − fixed (pp)');axes[1].set_title('(b) Representation size',fontsize=11)
    fig.savefig(FIGURES/'hgd_sensitivities.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,2.9));ax.set_xlim(0,10);ax.set_ylim(0,4);ax.axis('off')
    boxes=[(1.5,2.8,'T partition\n80% fit / 20% validation'),(5,2.8,'Train all declared fits\nLoss stop + 3 checkpoints'),(8.5,2.8,'Verify full-batch seal\nSource + checkpoint hashes'),(8.5,.7,'Open E partition\nFixed + 100-draw evaluation'),(4.3,.7,'Complete paired grids\nSubject/run uncertainty')]
    for x,y,t in boxes:ax.text(x,y,t,ha='center',va='center',fontsize=9,bbox=dict(boxstyle='round,pad=.55',fc='#eef4f5',ec='#16697a'))
    for start,end in [((3,2.8),(3.55,2.8)),((6.5,2.8),(7,2.8)),((8.5,2.15),(8.5,1.35)),((7,.7),(6,.7))]:ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',color='#16697a'))
    fig.savefig(FIGURES/'locked_protocol.pdf',bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    ledger={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in [EVIDENCE,TABLES,FIGURES] for p in folder.iterdir() if p.name!='asset_hashes.json'}
    (EVIDENCE/'asset_hashes.json').write_text(json.dumps(ledger,indent=2)+'\n')
    print('Built',len(list(TABLES.glob('*.tex'))),'tables and 3 figures from 10,760 complete evaluations.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-snapshot', action='store_true', help='Rebuild from versioned evidence without reading run directories')
    main(from_snapshot=parser.parse_args().from_snapshot)
