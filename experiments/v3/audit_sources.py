"""Audit frozen rendered V3 sources without any answering-model observations.

Only fresh_tasks.jsonl, source_registry.jsonl and generation_audit.json are
read. No sibling V1/V2 project, model, candidate, response or score is needed.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent
FIRST='Ada|Mia|Nora|Lena|Iris|Clara|Owen|Evan|Noah|Liam|Leo|Finn|Hugo|Maya|Zoe|Elena'
LAST='Carter|Reed|Brooks|Ward|Hayes|Bennett|Morgan|Stone|Parker|Wells|Ross|Turner|Foster|Grant|Blake|Cole'
PATTERNS={
    'name':re.compile(r'\b(?:'+FIRST+r') (?:'+LAST+r')\b'),
    'date':re.compile(r'\b2028-\d{2}-\d{2}\b'),
    'amount':re.compile(r'\bUSD \d{5}\b'),
}
# These four fixed contrast forms state the accepted value before the rejected
# value. Other contrast forms, and every update form, state the current value
# last. This audits source semantics; it is never a risk-feature input.
CONTRAST_CURRENT_FIRST={0,2,4,7}

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def json_record(value,entity):
    if isinstance(value,dict):
        if value.get('project')==entity:return value
        for child in value.values():
            found=json_record(child,entity)
            if found is not None:return found
    elif isinstance(value,list):
        for child in value:
            found=json_record(child,entity)
            if found is not None:return found
    return None

def json_field(value,field):
    if isinstance(value,dict):
        if field in value:return value[field]
        for child in value.values():
            found=json_field(child,field)
            if found is not None:return found
    return None

def recover_current(task,source):
    """Recover the queried value from the emitted source, not template inputs."""
    kind,text=task['task_type'],task['context']
    entity,field=source['entity'],source['requested_field']
    if kind=='table':
        lines=[line for line in text.splitlines() if line.startswith('|')]
        assert len(lines)>=3
        header=[cell.strip() for cell in lines[0].strip('|').split('|')]
        parsed=[dict(zip(header,[cell.strip() for cell in line.strip('|').split('|')])) for line in lines[2:]]
        targets=[row for row in parsed if row.get('Project')==entity]
        assert len(targets)==1
        # The table emits all three fields, so audit all three bindings too.
        assert all(targets[0][key]==value for key,value in source['current_attributes'].items())
        return targets[0][field]
    if kind=='nested_json':
        record=json_record(json.loads(text),entity)
        assert record is not None
        assert all(json_field(record,key)==value for key,value in source['current_attributes'].items())
        return json_field(record,field)
    if kind=='paired_directory':
        mapping=[line for line in text.splitlines() if entity in line]
        assert len(mapping)==1
        match=re.search(r'\b(Ref|Code|Id|Record|Entry|File|Item|Ticket|Key) (\d+)\b',mapping[0])
        assert match is not None
        identifier=match.group(0)
        attribute_rows=[line for line in text.splitlines() if line.startswith(identifier+' | ')]
        assert len(attribute_rows)==1
        fields=dict(part.split(': ',1) for part in attribute_rows[0].split(' | ')[1:])
        assert all(fields[key]==value for key,value in source['current_attributes'].items())
        return fields[field]
    assert kind in ('direct','header','update','contrast','reference')
    paragraphs=[p for p in text.split('\n\n') if entity in p]
    assert len(paragraphs)==1
    values=PATTERNS[task['answer_type']].findall(paragraphs[0])
    if kind in ('update','contrast'):
        assert len(values)==2 and values[0]!=values[1]
        current_index=0 if kind=='contrast' and task['template_family'] in CONTRAST_CURRENT_FIRST else 1
        assert values[1-current_index]==source['obsolete_or_rejected'][field]
        return values[current_index]
    assert len(values)==1
    return values[0]

def main():
    data=ROOT/'data'
    tasks=read_jsonl(data/'fresh_tasks.jsonl')
    records=read_jsonl(data/'source_registry.jsonl')
    registry={row['task_id']:row for row in records}
    generation=json.loads((data/'generation_audit.json').read_text(encoding='utf-8'))
    for name,expected in generation['sha256'].items():
        assert hashlib.sha256((data/name).read_bytes()).hexdigest()==expected,('Frozen source changed',name)
    assert len(tasks)==len(records)==len(registry)==360
    assert {t['task_id'] for t in tasks}==set(registry)
    assert len({t['source_id'] for t in tasks})==360
    assert Counter(t['split'] for t in tasks)=={'calibration':160,'test':200}
    groups=defaultdict(set)
    for task in tasks:groups[task['group_id']].add(task['split'])
    assert len(groups)==72 and all(len(splits)==1 for splits in groups.values())
    counts=Counter()
    for task in tasks:
        source=registry[task['task_id']]
        current=recover_current(task,source)
        assert current==source['current_attributes'][source['requested_field']],task['task_id']
        assert current.removeprefix('USD ')==task['answer'],task['task_id']
        assert task['answer'] in task['answer_aliases']
        assert all(d[source['requested_field']]!=current for d in source['distractors'])
        assert source['obsolete_or_rejected'][source['requested_field']]!=current
        counts[task['task_type']]+=1
    assert len(counts)==8 and set(counts.values())=={45}
    report={
        'pass':True,'requested_current_values_independently_recovered':360,
        'structured_sources_all_three_current_fields_verified':135,
        'prose_sources_queried_current_value_verified':225,
        'structure_counts':dict(counts),'calibration_tasks':160,'test_tasks':200,
        'family_splits_disjoint':True,'wording_layout_families':72,
        'frozen_source_hashes_match':True,'model_outputs_read':False,
        'sibling_projects_required':False,
        'old_new_disjointness':'Recorded by frozen generation_audit.json; not rechecked against absent sibling datasets.',
        'source_sha256':generation['sha256'],
    }
    output=ROOT/'results/source_audit.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
