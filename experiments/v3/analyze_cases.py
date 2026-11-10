"""Describe locked V3 outcomes; never fit, calibrate or alter a policy.

Run only after the complete final-test evaluation. Required inputs are the
observed metrics, frozen selection, scored final-test candidates and task-level
decisions. No case report is created from absent or incomplete observations.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent
CATEGORY_LABELS={
    'harmful':'原文答对、主方案采用压缩后答错',
    'improved':'原文答错、主方案答对',
    'still_wrong':'原文答错、主方案仍答错',
    'json_guard_selection_change':'JSON结构检查改变选择',
}

def load_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def valid_json_container(text):
    try:value=json.loads(text)
    except (ValueError,TypeError):return False
    return isinstance(value,(dict,list))

def find_entity_record(value,entity):
    if isinstance(value,dict):
        if value.get('project')==entity:return value
        for child in value.values():
            found=find_entity_record(child,entity)
            if found is not None:return found
    elif isinstance(value,list):
        for child in value:
            found=find_entity_record(child,entity)
            if found is not None:return found
    return None

def excerpt(context,question,limit=1600):
    """Choose visible entity-linked text, not an outcome-favoring explanation."""
    match=re.search(r'for project (.+?)\?\s*$',question,re.I)
    entity=match.group(1) if match else None
    if entity and valid_json_container(context):
        record=find_entity_record(json.loads(context),entity)
        if record is not None:
            return json.dumps(record,ensure_ascii=False,indent=2)[:limit]
    if entity:
        paragraphs=[p for p in re.split(r'\n\s*\n',context) if entity in p]
        if paragraphs and max(len(p) for p in paragraphs)<=limit:
            # A prose paragraph normally contains the entire target support.
            return '\n\n'.join(paragraphs)[:limit]
        lines=context.splitlines()
        indices=[i for i,line in enumerate(lines) if entity in line]
        if indices:
            chosen=set()
            for index in indices:
                chosen.update(range(max(0,index-8),min(len(lines),index+11)))
                # A directory mapping and its attribute row can be far apart.
                identifier=re.search(r'\b(?:Ref|Code|Id|Record|Entry|File|Item|Ticket|Key)\s+\d+\b',lines[index])
                if identifier:
                    chosen.update(i for i,line in enumerate(lines) if re.search(r'\b'+re.escape(identifier.group(0))+r'\b',line))
            return '\n'.join(lines[i] for i in sorted(chosen))[:limit]
    return context[:limit]

def choose_without_guard(rows,score_key,threshold):
    eligible=[r for r in rows if float(r[score_key])<=threshold and int(r['compressed_tokens'])<int(r['original_tokens'])]
    return min(eligible,key=lambda r:(int(r['compressed_tokens']),str(r['candidate_id']))) if eligible else None

def candidate_info(row,primary,raw_score_key):
    f=row.get('features',{})
    return {
        'candidate_id':row['candidate_id'],
        'hybrid_score':row.get('score_'+primary),
        'underlying_neural_score':row.get(raw_score_key),
        'structural_guard_allowed':row.get('structural_guard_allowed'),
        'original_json_valid':f.get('original_json_valid'),
        'compressed_json_valid':f.get('compressed_json_valid'),
        'input_tokens':row['compressed_tokens'],
        'context_tokens':row['compressed_context_tokens'],
        'observed_answer':row['compressed_answer'],
        'observed_correct':bool(row['compressed_correct']),
    }

def make_case(category,decision,rows,primary,threshold,raw_score_key,comparison=None):
    base=rows[0]
    selected=next((r for r in rows if r['candidate_id']==decision['candidate_id']),None)
    selected_context=selected['compressed_context'] if selected else base['original_context']
    case={
        'category':category,'task_id':decision['task_id'],'task_type':decision['task_type'],
        'primary_policy':primary,'question':base['question'],'gold':base['answer'],
        'answer_aliases':base.get('answer_aliases',[]),
        'original_answer':base['original_answer'],'selected_answer':decision['answer'],
        'original_correct':bool(decision['original_correct']),'selected_correct':bool(decision['correct']),
        'selected_candidate_id':decision['candidate_id'],'selection_threshold':threshold,
        'selected_hybrid_score':decision.get('risk_score'),
        'selected_underlying_neural_score':selected.get(raw_score_key) if selected else None,
        'original_context_tokens':base['original_context_tokens'],
        'selected_context_tokens':selected['compressed_context_tokens'] if selected else base['original_context_tokens'],
        'original_input_tokens':base['original_tokens'],'selected_input_tokens':decision['input_tokens'],
        'candidates':[candidate_info(r,primary,raw_score_key) for r in sorted(rows,key=lambda r:r['candidate_id'])],
        'source_excerpt':excerpt(base['original_context'],base['question']),
        'selected_excerpt':excerpt(selected_context,base['question']),
        'original_context':base['original_context'],'selected_context':selected_context,
        'observation_note':'Observed paired answers and a deterministic pre-answer selection; the mechanism of an answering error is not established.',
    }
    if comparison is not None:case['guard_comparison']=comparison
    return case

def codeblock(text):
    # All benchmark sources are synthetic, but avoid malformed markdown if a
    # future source contains a triple-backtick sequence.
    fence='````' if '```' in str(text) else '```'
    return fence+'text\n'+str(text)+'\n'+fence

def pct(value):return f'{100*value:.2f}%'

def render(report):
    primary=report['primary_policy']; metrics=report['primary_metrics']
    out=['# CacheScope V3 真实结果与案例',
         f"本报告主方案为 **{primary}**。{report['primary_weights_description']} 该方案结合学习到的风险分数与显式 JSON 结构检查，是混合方案，不能将全部收益归因于重新训练的神经网络。",
         f"在 {report['test_tasks']} 个独立新测试任务上，主方案准确率为 {pct(metrics['accuracy'])}，完整输入 token 节省为 {pct(metrics['input_token_reduction_ratio'])}；原文正确、主方案采用压缩后答错 {metrics['harmful_accepted_count']} 次（占全部任务 {pct(metrics['harmful_accepted_per_task'])}），原文答错后改善 {metrics['improvement_count']} 次。",
         f"预先设定的三项条件：{'全部达到' if report['all_predeclared_primary_criteria_met'] else '尚未全部达到'}。逐项判断与所有策略对照以 results/metrics.json 为准。",
         '所有来源均为合成英文 owner/date/budget 事实。八种高层结构已在开发阶段出现，本轮独立留出新来源和措辞／布局家族；不构成真实文档或完全未见结构的泛化证据。',
         '## 案例选择规则',
         '每类按 task_id 升序选最先两例：有害接受、原文答错后改善、原文答错且仍然答错。另外选最先两个 JSON 结构检查在固定主权重、固定主阈值下改变选择的案例。某类没有符合条件的任务时明确记为零，不补造案例。类别间可能出现同一任务。',
         '“去掉结构检查、保持主阈值”的选择仅用于解释这一项检查如何影响当前决策，是对已存候选回答的描述性重放，未作为另一个经独立选择验证的策略。实际纯模型对照采用其自身已锁定阈值，两者同时列出，避免把阈值差异归因于结构检查。',
         '片段优先展示目标实体及其关联记录，可能截断；完整原文和所选上下文保存在 results/case_studies.json。以下只描述可观察的答案、结构和选择，不给出未经实验建立的错误因果机制。']
    for category,label in CATEGORY_LABELS.items():
        cases=[c for c in report['cases'] if c['category']==category]
        out.extend(['## '+label,f"符合条件的任务数：{report['category_counts'][category]}；展示 {len(cases)} 例。"])
        if not cases:
            out.append('本类为零，没有展示案例。');continue
        for case in cases:
            out.extend([
                f"### {case['task_id']}（{case['task_type']}）",
                '问题：'+case['question'],
                f"Gold：`{case['gold']}`；原文回答：`{case['original_answer']}`；主方案回答：`{case['selected_answer']}`。",
                f"主方案选择 `{case['selected_candidate_id']}`；固定阈值 {case['selection_threshold']:.10g}；所选混合分数 {case['selected_hybrid_score']}；底层神经分数 {case['selected_underlying_neural_score']}。回退原文时所选候选分数为空。",
                f"上下文 token：{case['original_context_tokens']} → {case['selected_context_tokens']}；完整输入 token：{case['original_input_tokens']} → {case['selected_input_tokens']}。",
                '| 候选 | 底层神经分数 | 混合分数 | JSON检查通过 | 实际回答正确 |',
                '|---|---:|---:|---|---|',
            ])
            for row in case['candidates']:
                out.append(f"| {row['candidate_id']} | {row['underlying_neural_score']} | {row['hybrid_score']} | {row['structural_guard_allowed']} | {row['observed_correct']} |")
            if 'guard_comparison' in case:
                comp=case['guard_comparison']; pure=comp['actual_pure_policy']
                out.extend([
                    f"保持相同主权重及阈值、去掉 JSON 检查时，会选择 `{comp['same_threshold_without_guard']['candidate_id']}`，回答 `{comp['same_threshold_without_guard']['answer']}`，实际正确性为 {comp['same_threshold_without_guard']['correct']}。该被选候选的原文为合法 JSON，压缩文本不是合法 JSON 容器，因而被结构检查拒绝。",
                    f"实际纯模型对照 `{pure['policy']}` 使用其自身阈值 {pure['threshold']}，选择 `{pure['candidate_id']}`，回答 `{pure['answer']}`，实际正确性为 {pure['correct']}。这个自身阈值与主阈值可能不同。",
                    '去掉检查时会选择的候选片段：',codeblock(comp['same_threshold_without_guard']['excerpt']),
                ])
            out.extend(['原文关键片段：',codeblock(case['source_excerpt']),
                        '主方案所选上下文片段：',codeblock(case['selected_excerpt']),
                        '观察范围：这些记录确定了本次输入、候选选择和回答的变化；单个案例不能证明模型为何改变回答，也不能证明该规则对所有未来请求都有效。'])
    out.extend(['## 原始证据与限制',
                '本报告只读冻结后的真实结果，未改动模型、阈值、候选回答或标签。风险分数是未校准的排序分数，不能直接解释成保证成立的失败概率。经验校准约束也不能保证未来数据的风险上限。',
                '[完整指标](../results/metrics.json) · [完整案例 JSON](../results/case_studies.json) · [测试前冻结选择](../results/frozen_selection.json)'])
    return '\n\n'.join(out)+'\n'

def main():
    results=ROOT/'results'
    frozen_path=results/'frozen_selection.json'
    paths={
        'metrics':results/'metrics.json','frozen_selection':frozen_path,
        'scored_candidates':results/'fresh_test_scored_candidates.jsonl',
        'policy_decisions':results/'task_policy_decisions.jsonl',
    }
    for path in paths.values():
        if not path.exists():raise RuntimeError('Complete final-test evaluation first; missing '+str(path))
    metrics,frozen=load_json(paths['metrics']),load_json(frozen_path)
    primary=metrics['primary_policy']
    if primary!=frozen['primary_policy']:raise ValueError('Primary differs from frozen selection')
    if not frozen.get('primary_is_hybrid'):raise ValueError('V3 primary must be disclosed as a hybrid')
    if metrics.get('model_selection_file_sha256')!=sha(frozen_path):raise ValueError('Selection hash differs from evaluation')
    n=metrics['test_tasks']
    if n!=frozen['test_expected_tasks']:raise ValueError('Unexpected final-test size')
    rows=load_jsonl(paths['scored_candidates'])
    grouped=defaultdict(list)
    for row in rows:
        if row['split']!='test':raise ValueError('Only final-test records are allowed')
        grouped[row['task_id']].append(row)
    if len(grouped)!=n or any(len(v)!=2 for v in grouped.values()):raise ValueError('Incomplete test candidates')
    all_decisions=load_jsonl(paths['policy_decisions'])
    decisions={policy:{d['task_id']:d for d in all_decisions if d['policy']==policy} for policy in {d['policy'] for d in all_decisions}}
    if primary not in decisions or set(decisions[primary])!=set(grouped):raise ValueError('Incomplete primary decisions')
    threshold=float(frozen['threshold_selection'][primary]['threshold'])
    if primary=='hybrid_mlp_ensemble':
        raw_score_key='score_mlp_ensemble';pure_name='mlp_ensemble'
        weights='底层使用本轮重新训练的三个 MLP（seed 520/521/522），候选风险取三者最大值。'
    elif primary=='hybrid_frozen_v2_mlp':
        raw_score_key='score_frozen_v2_mlp';pure_name='frozen_v2_mlp'
        weights='底层使用冻结 V2 MLP 权重；本轮重新校准阈值并加入 JSON 检查。新训练权重仍有独立对照结果，不能声称主方案采用了它们。'
    else:raise ValueError('Unknown predeclared primary policy')
    ds=decisions[primary]
    conditions={
        'harmful':lambda d:bool(d['harmful']),
        'improved':lambda d:bool(d['improved']),
        'still_wrong':lambda d:not d['original_correct'] and not d['correct'],
    }
    cases=[];counts={}
    for category,predicate in conditions.items():
        chosen=[task for task in sorted(ds) if predicate(ds[task])]
        counts[category]=len(chosen)
        cases.extend(make_case(category,ds[task],grouped[task],primary,threshold,raw_score_key) for task in chosen[:2])
    guard_changes=[]
    for task in sorted(ds):
        unguarded=choose_without_guard(grouped[task],raw_score_key,threshold)
        if unguarded is None or unguarded.get('structural_guard_allowed',True):continue
        if unguarded['candidate_id']==ds[task]['candidate_id']:continue
        if not valid_json_container(unguarded['original_context']) or valid_json_container(unguarded['compressed_context']):continue
        pure=decisions[pure_name][task]
        pure_threshold=(frozen['frozen_v2_threshold'] if pure_name=='frozen_v2_mlp' else frozen['threshold_selection'][pure_name]['threshold'])
        comparison={
            'same_threshold_without_guard':{
                'candidate_id':unguarded['candidate_id'],'threshold':threshold,'answer':unguarded['compressed_answer'],
                'correct':bool(unguarded['compressed_correct']),'risk_score':unguarded[raw_score_key],
                'excerpt':excerpt(unguarded['compressed_context'],unguarded['question']),
                'compressed_context':unguarded['compressed_context'],
            },
            'actual_pure_policy':{'policy':pure_name,'threshold':pure_threshold,
                'candidate_id':pure['candidate_id'],'answer':pure['answer'],'correct':bool(pure['correct'])},
            'interpretation':'The same-threshold replay isolates the deterministic guard selection. The actual pure-policy comparison also changes the calibrated operating point.',
        }
        guard_changes.append((task,comparison))
    counts['json_guard_selection_change']=len(guard_changes)
    for task,comparison in guard_changes[:2]:
        cases.append(make_case('json_guard_selection_change',ds[task],grouped[task],primary,threshold,raw_score_key,comparison))
    report={
        'primary_policy':primary,'primary_is_hybrid':True,'primary_weights_description':weights,
        'test_tasks':n,'primary_metrics':metrics['policies'][primary],
        'all_predeclared_primary_criteria_met':metrics['all_predeclared_primary_criteria_met'],
        'selection_rule':'First two task IDs in each explicitly defined observed-outcome category; no padding for empty categories.',
        'category_counts':counts,'cases':cases,
        'inputs_sha256':{name:sha(path) for name,path in paths.items()},
        'synthetic_sources_only':True,'models_thresholds_or_observations_changed':False,
    }
    text=render(report)
    (ROOT/'docs').mkdir(exist_ok=True)
    (results/'case_studies.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (ROOT/'docs/FAILURE_ANALYSIS.md').write_text(text,encoding='utf-8')
    print(json.dumps({'built':['docs/FAILURE_ANALYSIS.md','results/case_studies.json'],
                     'primary_policy':primary,'category_counts':counts,'displayed_cases':len(cases)},ensure_ascii=False))

if __name__=='__main__':main()
