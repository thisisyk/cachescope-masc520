"""V3 fitting, calibration and one-time fresh-test evaluation.

Previous V1/V2 outcomes are development data. New calibration outcomes select
operating points; final test observations are never opened by train/calibrate.
The deployable hybrid combines neural risk with a separately disclosed
structural guard. It is not described as a pure neural predictor.
"""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,math,sys
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from cachescope import models
from cachescope.features import FEATURE_NAMES,extract_features,preanswer_guard
from cachescope.evaluation import make_decisions,summarize_decisions,group_tasks,grouped_bootstrap
from cachescope.pipeline import read_jsonl,write_jsonl,dump_json

SEEDS=(520,521,522)
SPLIT_SEED=1113520
CALIBRATION_EPSILON=.0125
CALIBRATION_HARM_TIERS=(.0125,.01875,.025)
CALIBRATION_ACCURACY_EPSILON=.0125
BASELINE=ROOT/'assets/v2_baseline'
SELECTION=ROOT/'results/frozen_selection.json'
LEARNED=('mlp_seed_520','mlp_seed_521','mlp_seed_522','mlp_ensemble',
         'hybrid_mlp_ensemble','lr','rf','hybrid_lr','hybrid_rf',
         'hybrid_frozen_v2_mlp')
PRIMARY_CANDIDATES=('hybrid_mlp_ensemble','hybrid_frozen_v2_mlp')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def refresh(rows):
    out=[]
    for source in rows:
        r=dict(source)
        r['features']=extract_features(r['original_context'],r['question'],r['compressed_context'],r['original_tokens'],r['compressed_tokens'])
        out.append(r)
    return out

def validate(rows,allowed,n_tasks=None):
    if not rows or {r['split'] for r in rows}-set(allowed): raise ValueError('Unexpected data partition')
    tasks=group_tasks(rows)
    if n_tasks is not None and len(tasks)!=n_tasks: raise ValueError(f'Expected {n_tasks} tasks; observed {len(tasks)}')
    for task,variants in tasks.items():
        if len(variants)!=2 or {r['candidate_id'] for r in variants}!={'keep_60','keep_30'}: raise ValueError(f'Incomplete task {task}')
        for r in variants:
            if r['original_tokens']!=r['original_backend_prompt_tokens'] or r['compressed_tokens']!=r['compressed_backend_prompt_tokens']: raise ValueError('Tokenizer mismatch')
            if int(r['regression'])!=int(r['original_correct'] and not r['compressed_correct']): raise ValueError('Invalid observed label')
    return tasks

def internal_partitions(rows):
    """Grouped checkpoint-selection split; every old observation remains dev."""
    group_kinds=defaultdict(Counter)
    for r in rows: group_kinds[r['group_id']][r['task_type']]+=1
    groups=defaultdict(set)
    for group,counts in group_kinds.items(): groups[sorted(counts,key=lambda k:(-counts[k],k))[0]].add(group)
    rng=np.random.default_rng(SPLIT_SEED)
    assignment={}
    for kind in sorted(groups):
        names=sorted(groups[kind]); rng.shuffle(names)
        n_hold=max(1,int(round(.2*len(names)))) if len(names)>1 else 0
        for i,name in enumerate(names):
            split='validation' if i<n_hold else 'train'
            if name in assignment and assignment[name]!=split: raise ValueError('A group spans task types with inconsistent assignment')
            assignment[name]=split
    result=[{**r,'split':assignment[r['group_id']]} for r in rows]
    models._validate_groups(result)
    return result,assignment

def save_checkpoint(model,scaler,seed,path):
    import torch
    torch.save({'state_dict':model.state_dict(),'feature_names':list(FEATURE_NAMES),'seed':seed,
      'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),'scaler_var':scaler.var_.tolist(),
      'scaler_n_samples':int(scaler.n_samples_seen_),
      'score_interpretation':'Uncalibrated conditional compression-regression risk; not a probability guarantee.'},path)

def refit_mlp(x,y,seed,epochs):
    """Refit all previous dev using an epoch count selected without fresh cal."""
    import torch
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    model=models._network(x.shape[1]).cpu()
    tx=torch.as_tensor(x,dtype=torch.float32)
    ty=torch.as_tensor(y,dtype=torch.float32).reshape(-1,1)
    criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(y)-int(y.sum()))/int(y.sum())]))
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    history=[]
    for epoch in range(1,epochs+1):
        model.train(); order=rng.permutation(len(y)); total=0.
        for start in range(0,len(order),32):
            ii=order[start:start+32]; optimizer.zero_grad(set_to_none=True)
            loss=criterion(model(tx[ii]),ty[ii]); loss.backward(); optimizer.step()
            total+=float(loss.detach())*len(ii)
        history.append({'epoch':epoch,'train_loss':total/len(y)})
    model.eval()
    return model,history

def train():
    import torch
    torch.set_num_threads(2)
    if SELECTION.exists(): raise RuntimeError('Frozen selection exists: preserve it and start a new experiment rather than retune.')
    path=ROOT/'data/development_candidates.jsonl'
    raw=read_jsonl(path)
    if {r['split'] for r in raw}-{'development','train'}: raise ValueError('Only declared previous development records may fit weights')
    rows=refresh(raw); validate(rows,{'development','train'},960)
    internal,assignment=internal_partitions(rows)
    tr=[r for r in internal if r['split']=='train' and r['original_correct']]
    va=[r for r in internal if r['split']=='validation' and r['original_correct']]
    all_conditional=[r for r in rows if r['original_correct']]
    y=np.asarray([r['regression'] for r in tr],dtype=int)
    yv=np.asarray([r['regression'] for r in va],dtype=int)
    ya=np.asarray([r['regression'] for r in all_conditional],dtype=int)
    if len(np.unique(y))<2 or len(np.unique(ya))<2: raise ValueError('Need genuine examples from both classes')
    out=ROOT/'models'; out.mkdir(exist_ok=True)
    old_scaler=StandardScaler().fit(models.feature_matrix(tr))
    xt=old_scaler.transform(models.feature_matrix(tr)); xv=old_scaler.transform(models.feature_matrix(va))
    scaler=StandardScaler().fit(models.feature_matrix(all_conditional))
    xa=scaler.transform(models.feature_matrix(all_conditional))
    report={'primary_mlp':'mlp_seed_520','feature_names':list(FEATURE_NAMES),'seeds':list(SEEDS),
      'architecture':'d -> 32 ReLU -> 16 ReLU -> 1 logit','internal_split_seed':SPLIT_SEED,
      'old_data_status':'All V1/V2 former tests are now development; none validate V3.',
      'fitting_population':'All previous development candidates whose original answer was correct',
      'epoch_selection':'Stratified group-disjoint 80/20 previous-development split, then fresh initialization and fixed-epoch refit on all previous development.',
      'calibration_used_to_fit_weights':False,'fresh_test_file_opened':False,
      'training_task_records':len({r['task_id'] for r in rows}),'training_candidate_records':len(rows),
      'conditional_training_candidates':len(ya),'conditional_regressions':int(ya.sum()),
      'internal_tasks':{s:len({r['task_id'] for r in internal if r['split']==s}) for s in ('train','validation')},
      'models':{},'versions':{'torch':torch.__version__,'numpy':np.__version__}}
    for seed in SEEDS:
        temporary,history,info=models._train_mlp(xt,y,xv,yv,seed,100,10)
        with torch.inference_mode(): before=torch.sigmoid(temporary(torch.as_tensor(xv,dtype=torch.float32))).numpy().reshape(-1)
        model,refit_history=refit_mlp(xa,ya,seed,info['best_epoch'])
        name=f'mlp_seed_{seed}'
        save_checkpoint(model,scaler,seed,out/f'{name}.pt')
        dump_json(out/f'{name}_history.json',history)
        dump_json(out/f'{name}_refit_history.json',refit_history)
        report['models'][name]={**info,'refit_epochs':info['best_epoch'],
          'internal_validation_metrics_before_full_development_refit':models.classification_metrics(yv,before),
          'parameter_count':sum(p.numel() for p in model.parameters())}
    for name,estimator in [('lr',LogisticRegression(max_iter=2000,class_weight='balanced',random_state=520)),
                          ('rf',RandomForestClassifier(n_estimators=300,max_depth=8,min_samples_leaf=3,class_weight='balanced',random_state=520,n_jobs=1))]:
        estimator.fit(xa,ya)
        joblib.dump({'kind':'sklearn','model':estimator,'scaler':scaler,'feature_names':list(FEATURE_NAMES),
          'score_interpretation':'Uncalibrated conditional risk'},out/f'{name}.joblib')
    dump_json(out/'training_report.json',report)
    dump_json(ROOT/'results/internal_split_manifest.json',{'assignment':assignment,'seed':SPLIT_SEED,'purpose':'checkpoint epoch selection only; all old development is refitted afterward'})
    fit_files={str(p.relative_to(ROOT)):sha(p) for p in out.glob('*') if p.is_file()}
    for name in ['data/development_candidates.jsonl','src/cachescope/features.py','src/cachescope/models.py','train_v3.py']:
        fit_files[name]=sha(ROOT/name)
    dump_json(ROOT/'results/fitted_model_manifest.json',{'fitted_at_utc':datetime.now(timezone.utc).isoformat(),'files_sha256':fit_files,'weights_fit_without_fresh_calibration_or_test':True})
    print(json.dumps({'stage':'trained','training_tasks':report['training_task_records'],'feature_count':len(FEATURE_NAMES),
      'selected_refit_epochs':{n:v['refit_epochs'] for n,v in report['models'].items()}},ensure_ascii=False),flush=True)
    return report

def old_features(rows):
    spec=importlib.util.spec_from_file_location('fixed_v2_features',BASELINE/'features.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    fs=[module.extract_features(r['original_context'],r['question'],r['compressed_context'],r['original_tokens'],r['compressed_tokens']) for r in rows]
    return np.asarray([[f[k] for k in module.FEATURE_NAMES] for f in fs],dtype=float)

def score_records(rows):
    """Scores and structural eligibility use pre-answer fields only."""
    rows=refresh(rows)
    scores=[]
    for seed in SEEDS:
        name=f'mlp_seed_{seed}'
        p=models.predict_scores(models.load_bundle(ROOT/'models',name),rows)
        scores.append(p)
        for r,s in zip(rows,p): r['score_'+name]=float(s)
    maximum=np.max(np.vstack(scores),axis=0)
    average=np.mean(np.vstack(scores),axis=0)
    for r,p,m in zip(rows,maximum,average):
        allowed=bool(preanswer_guard(r['features'],mode='structural'))
        r['structural_guard_allowed']=allowed
        r['score_mlp_ensemble']=float(p)
        r['score_mlp_ensemble_mean_descriptive']=float(m)
        r['score_hybrid_mlp_ensemble']=float(p) if allowed else 1.0
        r['score_guard_only']=0.0 if allowed else 1.0
    for name in ('lr','rf'):
        pp=models.predict_scores(models.load_bundle(ROOT/'models',name),rows)
        for r,p in zip(rows,pp):
            r['score_'+name]=float(p)
            r['score_hybrid_'+name]=float(p) if r['structural_guard_allowed'] else 1.0
    baseline=models.load_bundle(BASELINE,'mlp')
    pp=models.predict_scores(baseline,old_features(rows))
    for r,p in zip(rows,pp):
        r['score_frozen_v2_mlp']=float(p)
        r['score_hybrid_frozen_v2_mlp']=float(p) if r['structural_guard_allowed'] else 1.0
    return rows

def select_threshold(rows,key,epsilon=CALIBRATION_EPSILON):
    """Blocked scores equal one; thresholds are strictly below one."""
    if not rows or {r['split'] for r in rows}!={'calibration'}: raise ValueError('New calibration observations only')
    values=[float(r[key]) for r in rows]
    if any(not math.isfinite(p) or not 0<=p<=1 for p in values): raise ValueError('Invalid scores')
    candidates=[-1.0]+sorted({p for p in values if p<1.0})
    feasible=[]
    for threshold in candidates:
        m=summarize_decisions(make_decisions(rows,'gate',key,threshold))
        if m['harmful_accepted_per_task']<=epsilon+1e-12 and m['accuracy_delta']>=-CALIBRATION_ACCURACY_EPSILON-1e-12:
            feasible.append((threshold,m))
    threshold,m=max(feasible,key=lambda x:(x[1]['input_token_reduction_ratio'],x[1]['accuracy'],-x[0]))
    return {'threshold':threshold,'score_key':key,'selection_split':'calibration','epsilon':epsilon,
      'constraint':'harmful_per_task <= tier epsilon and accuracy_delta >= -0.0125','objective':'maximum complete-input-token reduction',
      'validation_metrics':m,'calibration_metrics':m,'thresholds_examined':len(candidates),'guarantee':False}

def check_hashes(manifest):
    for name,digest in manifest['files_sha256'].items():
        if sha(ROOT/name)!=digest: raise ValueError(f'Frozen artifact changed: {name}')

def verify_freeze(frozen):
    check_hashes(frozen)
    if frozen['primary_policy'] not in PRIMARY_CANDIDATES: raise ValueError('Unknown frozen primary policy')
    if frozen['test_expected_tasks']!=200: raise ValueError('Unexpected final-test size')

def separated(a,b):
    for key in ('task_id','source_id','group_id'):
        if {r[key] for r in a}&{r[key] for r in b}: raise ValueError(f'Partition overlap in {key}')

def calibrate():
    import torch
    torch.set_num_threads(2)
    if SELECTION.exists(): raise RuntimeError('Selection already frozen')
    fitted=json.loads((ROOT/'results/fitted_model_manifest.json').read_text(encoding='utf-8'))
    check_hashes(fitted)
    path=ROOT/'data/calibration_candidates.jsonl'
    rows=read_jsonl(path); validate(rows,{'calibration'},160)
    development=read_jsonl(ROOT/'data/development_candidates.jsonl'); separated(rows,development)
    rows=score_records(rows)
    tier_attempts=[]
    winning=None
    for epsilon in CALIBRATION_HARM_TIERS:
        selection={name:select_threshold(rows,'score_'+name,epsilon) for name in LEARNED}
        feasible=[name for name in PRIMARY_CANDIDATES if selection[name]['calibration_metrics']['input_token_reduction_ratio']>=.2-1e-12]
        tier_attempts.append({'harm_tier':epsilon,'threshold_selection':selection,'feasible_primary_candidates':feasible})
        if winning is None and feasible: winning=(epsilon,selection,feasible)
    write_jsonl(ROOT/'results/calibration_scored_candidates.jsonl',rows)
    attempt={'tier_attempts':tier_attempts,'primary_candidates':list(PRIMARY_CANDIDATES),
      'calibration_tasks':160,'predeclared_harm_tiers':list(CALIBRATION_HARM_TIERS),'accuracy_delta_min':-CALIBRATION_ACCURACY_EPSILON,
      'test_observations_opened':False,'calibration_data_file_sha256':sha(path)}
    dump_json(ROOT/'results/calibration_attempt.json',attempt)
    if winning is None:
        print(json.dumps({'stage':'calibration_not_ready','reason':'No predeclared hybrid candidate saves at least20% at any predeclared safety-buffer tier','tier_attempts':tier_attempts},ensure_ascii=False),flush=True)
        raise RuntimeError('Calibration targets not met; no freeze was written and final test must not run.')
    epsilon,selections,feasible=winning
    primary=max(feasible,key=lambda n:(selections[n]['calibration_metrics']['input_token_reduction_ratio'],selections[n]['calibration_metrics']['accuracy'],n=='hybrid_mlp_ensemble'))
    files={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'models').glob('*') if p.is_file()}
    files.update({str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py') if '__pycache__' not in p.parts})
    files.update({str(p.relative_to(ROOT)):sha(p) for p in BASELINE.glob('*') if p.is_file()})
    for relative in ['data/development_candidates.jsonl','data/calibration_candidates.jsonl','data/fresh_tasks.jsonl','data/source_registry.jsonl','data/generation_audit.json','experiment_protocol.json','protocol.json','results/fitted_model_manifest.json']:
        if (ROOT/relative).exists(): files[relative]=sha(ROOT/relative)
    old=json.loads((BASELINE/'frozen_model_selection.json').read_text(encoding='utf-8'))
    frozen={'version':'3.0','frozen_at_utc':datetime.now(timezone.utc).isoformat(),
      'primary_policy':primary,'primary_is_hybrid':True,'primary_model':primary,
      'primary_description':'Learned risk ranking plus structural guard; the old-V2 hybrid can be selected using calibration if it has better feasible efficiency.',
      'primary_candidates':list(PRIMARY_CANDIDATES),'selection_rule':'Visit predeclared harm tiers strict-to-loose; first tier with hybrid savings>=20% and accuracy_delta>=-0.0125 wins. Within tier maximize savings, then accuracy, then prefer newly trained ensemble.',
      'threshold_selection':selections,'primary_calibration_metrics':selections[primary]['calibration_metrics'],
      'ensemble':'maximum of three predetermined neural-seed scores; no best seed selected',
      'structural_guard':'Reject when original context parses as JSON but compressed context does not; pre-answer only.',
      'feature_names':list(FEATURE_NAMES),'calibration_epsilon_empirical':epsilon,'predeclared_harm_tiers':list(CALIBRATION_HARM_TIERS),
      'calibration_empirical_target_not_guarantee':True,'test_file_opened_during_selection':False,
      'counts':{'development':960,'calibration':160,'test':200},'test_expected_tasks':200,
      'frozen_v2_threshold':old['threshold_selection']['mlp']['threshold'],
      'files_sha256':files,'predeclared_test_success_criteria':{'input_token_reduction_ratio_min':.2,'accuracy_delta_min':-.025,'harmful_accepted_per_task_max':.05}}
    dump_json(SELECTION,frozen)
    print(json.dumps({'stage':'calibrated_and_frozen','primary_policy':primary,'calibration_metrics':frozen['primary_calibration_metrics'],'threshold':selections[primary]['threshold']},ensure_ascii=False),flush=True)
    return frozen

def paired(a,b,repetitions=2000):
    aa={d['task_id']:d for d in a}; bb={d['task_id']:d for d in b}
    if aa.keys()!=bb.keys(): raise ValueError('Unpaired policies')
    wins=sum(aa[k]['correct'] and not bb[k]['correct'] for k in aa)
    losses=sum(bb[k]['correct'] and not aa[k]['correct'] for k in aa)
    d=wins+losses
    p=min(1.,2*sum(math.comb(d,i) for i in range(min(wins,losses)+1))/2**d) if d else 1.
    grouped=defaultdict(list)
    for k,r in aa.items(): grouped[r['group_id']].append(k)
    names=sorted(grouped); rng=np.random.default_rng(SPLIT_SEED); changes=[]
    for _ in range(repetitions):
        sample=[k for g in rng.choice(names,len(names),replace=True) for k in grouped[g]]
        changes.append(sum(int(aa[k]['correct'])-int(bb[k]['correct']) for k in sample)/len(sample))
    return {'n_tasks':len(aa),'a_correct_b_wrong':wins,'a_wrong_b_correct':losses,
      'accuracy_difference_a_minus_b':(wins-losses)/len(aa),
      'accuracy_difference_group_bootstrap_95_ci':np.quantile(changes,[.025,.975]).tolist(),
      'exact_mcnemar_two_sided_p':p,'mcnemar_limitation':'Per-task exact value ignores within-family dependence; group bootstrap is also reported. No multiplicity correction.'}

def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

def evaluate():
    import torch
    torch.set_num_threads(2)
    frozen=json.loads(SELECTION.read_text(encoding='utf-8')); verify_freeze(frozen)
    path=ROOT/'data/test_candidates.jsonl'
    rows=read_jsonl(path); tasks=validate(rows,{'test'},200)
    dev=read_jsonl(ROOT/'data/development_candidates.jsonl')+read_jsonl(ROOT/'data/calibration_candidates.jsonl')
    separated(rows,dev)
    rows=score_records(rows); write_jsonl(ROOT/'results/fresh_test_scored_candidates.jsonl',rows)
    specs=[(n,None,None) for n in ('original','always_60','always_30','simple_rule_gate')]
    specs+=[(n,'score_'+n,frozen['threshold_selection'][n]['threshold']) for n in LEARNED]
    specs+=[('guard_only','score_guard_only',0.),('frozen_v2_mlp','score_frozen_v2_mlp',frozen['frozen_v2_threshold'])]
    report={'version':'3.0','primary_policy':frozen['primary_policy'],'primary_is_hybrid':True,
      'scope':'Fresh synthetic, independent-source and wording/layout-family held-out test within eight known high-level structures.',
      'test_tasks':200,'test_groups':len({r['group_id'] for r in rows}),
      'model_selection_file_sha256':sha(SELECTION),'test_candidate_file_sha256':sha(path),
      'primary_was_selected_on_calibration_only':True,'test_was_not_used_for_training_or_selection':True,
      'success_criteria':frozen['predeclared_test_success_criteria'],'policies':{},'by_task_type':{},'by_length':{},
      'limitations':['All sources are synthetic English facts; one small local answering model.',
      'High-level structures are seen; fresh sources and wording/layout families are held out.',
      'Primary is a disclosed hybrid, not evidence that newly trained neural weights alone improve performance.',
      'Calibration safety target is empirical and cannot guarantee final-test risk.','Token reduction is full prompt input only, not API spending or causal wall-time speedup.']}
    ds_all=[]; policy_decisions={}; flat=[]; kinds=[]; lengths=[]
    bases={k:v[0] for k,v in tasks.items()}
    for name,key,threshold in specs:
        ds=make_decisions(rows,name,key,threshold)
        for d in ds: d['length_stratum']='long' if bases[d['task_id']]['original_context_tokens']>=400 else 'short'
        m=summarize_decisions(ds); flat.append({'policy':name,**m})
        m['paired_group_bootstrap']=grouped_bootstrap(ds,2000,SPLIT_SEED)
        report['policies'][name]=m
        report['by_task_type'][name]={}; report['by_length'][name]={}
        for key2,field,target,collector in [('by_task_type','task_type',report['by_task_type'][name],kinds),('by_length','length_stratum',report['by_length'][name],lengths)]:
            for value in sorted({d[field] for d in ds}):
                sub=summarize_decisions([d for d in ds if d[field]==value]); target[value]=sub
                collector.append({'policy':name,field:value,**sub})
        ds_all+=ds; policy_decisions[name]=ds
    primary=frozen['primary_policy']; m=report['policies'][primary]; c=report['success_criteria']
    report['primary_metrics']=m
    report['primary_success_criteria_observed']={'saves_at_least_20_percent':m['input_token_reduction_ratio']>=c['input_token_reduction_ratio_min']-1e-12,
      'accuracy_delta_at_least_minus_2_5pp':m['accuracy_delta']>=c['accuracy_delta_min']-1e-12,
      'harmful_acceptance_at_most_5_percent':m['harmful_accepted_per_task']<=c['harmful_accepted_per_task_max']+1e-12}
    report['all_predeclared_primary_criteria_met']=all(report['primary_success_criteria_observed'].values())
    report['paired_comparisons']={f'primary_vs_{other}':paired(policy_decisions[primary],policy_decisions[other])
       for other in ('original','always_60','always_30','frozen_v2_mlp','mlp_ensemble','guard_only','hybrid_mlp_ensemble','hybrid_frozen_v2_mlp') if other!=primary}
    conditional=[r for r in rows if r['original_correct']]
    report['conditional_classifier_metrics']={name:models.classification_metrics([r['regression'] for r in conditional],[r['score_'+name] for r in conditional]) for name in LEARNED}
    report['seed_variation']={'seeds':list(SEEDS),'seed_was_not_selected_on_test':True,
      'accuracy_mean':float(np.mean([report['policies'][f'mlp_seed_{s}']['accuracy'] for s in SEEDS])),
      'accuracy_std':float(np.std([report['policies'][f'mlp_seed_{s}']['accuracy'] for s in SEEDS],ddof=1))}
    write_jsonl(ROOT/'results/task_policy_decisions.jsonl',ds_all)
    write_csv(ROOT/'results/policy_results.csv',flat); write_csv(ROOT/'results/by_task_type.csv',kinds); write_csv(ROOT/'results/by_length.csv',lengths)
    write_csv(ROOT/'results/task_policy_decisions.csv',ds_all)
    dump_json(ROOT/'results/metrics.json',report)
    print(json.dumps({'stage':'fresh_test_evaluated','primary_policy':primary,'primary_metrics':m,'all_predeclared_primary_criteria_met':report['all_predeclared_primary_criteria_met']},ensure_ascii=False),flush=True)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['train','calibrate','evaluate']); args=parser.parse_args()
    {'train':train,'calibrate':calibrate,'evaluate':evaluate}[args.stage]()
