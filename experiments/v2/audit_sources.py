"""Independent rendered-source audit; never reads model responses or scores."""
from pathlib import Path
import json,re,hashlib
ROOT=Path(__file__).resolve().parent
tasks=[json.loads(x) for x in (ROOT/'data/new_tasks.jsonl').read_text(encoding='utf-8').splitlines()]
registry={r['task_id']:r for r in [json.loads(x) for x in (ROOT/'data/source_registry.jsonl').read_text(encoding='utf-8').splitlines()]}
first='Ada|Mia|Nora|Lena|Iris|Clara|Owen|Evan|Noah|Liam|Leo|Finn|Hugo|Maya|Zoe|Elena'
last='Carter|Reed|Brooks|Ward|Hayes|Bennett|Morgan|Stone|Parker|Wells|Ross|Turner|Foster|Grant|Blake|Cole'
patterns={'name':re.compile(r'\b(?:'+first+r') (?:'+last+r')\b'),'date':re.compile(r'\b2027-\d{2}-\d{2}\b'),'amount':re.compile(r'\bUSD \d{5}\b')}
counts={}

def json_record(value,entity):
    if isinstance(value,dict):
        if value.get('project')==entity: return value
        for child in value.values():
            found=json_record(child,entity)
            if found is not None: return found
    elif isinstance(value,list):
        for child in value:
            found=json_record(child,entity)
            if found is not None: return found
    return None

def find_field(value,field):
    if isinstance(value,dict):
        if field in value: return value[field]
        for child in value.values():
            found=find_field(child,field)
            if found is not None: return found
    return None

for task in tasks:
    source=registry[task['task_id']]
    entity,field=source['entity'],source['requested_field']
    text,kind=task['context'],task['task_type']
    if kind=='table':
        lines=[x for x in text.splitlines() if x.strip().startswith('|')]
        header=[x.strip() for x in lines[0].strip('|').split('|')]
        records=[dict(zip(header,[x.strip() for x in line.strip('|').split('|')])) for line in lines[2:]]
        matches=[r for r in records if r['Project']==entity]
        assert len(matches)==1
        current=matches[0][field]
    elif kind=='nested_json':
        record=json_record(json.loads(text),entity)
        assert record is not None
        current=find_field(record,field)
    elif kind=='paired_directory':
        matches=[line for line in text.splitlines() if line.endswith(': '+entity)]
        assert len(matches)==1
        prefix=re.match(r'^([A-Za-z]+ \d+): ',matches[0]).group(1)
        attributes=[line for line in text.splitlines() if line.startswith(prefix+': ') and '=' in line]
        assert len(attributes)==1
        current=dict(x.split('=',1) for x in attributes[0].split(': ',1)[1].split('; '))[field]
    else:
        matches=[p for p in text.split('\n\n') if entity in p]
        assert len(matches)==1
        values=patterns[task['answer_type']].findall(matches[0])
        if kind in ['direct','header','reference']:
            assert len(values)==1
            current=values[0]
        else:
            assert len(values)==2 and values[0]!=values[1]
            position=0 if kind=='contrast' and task['template_family'] in (1,4,7) else 1
            current=values[position]
            assert values[1-position]==source['obsolete_or_rejected'][field]
    assert current==source['current_attributes'][field],task['task_id']
    assert current.removeprefix('USD ')==task['answer'],task['task_id']
    counts[kind]=counts.get(kind,0)+1
assert sum(counts.values())==360
report={'pass':True,'rendered_source_current_values_independently_parsed':360,'counts':counts,'model_outputs_read':False,'data_sha256':hashlib.sha256((ROOT/'data/new_tasks.jsonl').read_bytes()).hexdigest()}
(ROOT/'results/source_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report))
