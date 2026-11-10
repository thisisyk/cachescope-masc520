"""Freeze independent V3 calibration/test sources before answering-model calls.

All eight high-level structures were observed during development. New source
records and wording/layout families are held out. This is not structural OOD.
The generator reads no model responses, features, labels or policy outcomes.
"""
from __future__ import annotations
import calendar
from collections import Counter, defaultdict
from datetime import date
import hashlib
import json
from pathlib import Path
import random
import re

SEED = 1113520
KINDS = ('direct', 'header', 'update', 'contrast', 'reference', 'table', 'nested_json', 'paired_directory')
LABELS = ('owner', 'release date', 'approved budget')
FIRST = ('Ada','Mia','Nora','Lena','Iris','Clara','Owen','Evan','Noah','Liam','Leo','Finn','Hugo','Maya','Zoe','Elena')
LAST = ('Carter','Reed','Brooks','Ward','Hayes','Bennett','Morgan','Stone','Parker','Wells','Ross','Turner','Foster','Grant','Blake','Cole')
PROJECT = ('Lumen','Beacon','Horizon','Drift','Glade','Aquila','Beryl','Lagoon','Prairie','Nettle','Mistral','Cypress')

# These target wording inventories do not reuse V1 or V2 target templates.
DIRECT = (
    'The completed file for {entity} certifies a {label} of {value}.',
    '{value} is the {label} entered on the certified card for project {entity}.',
    'An official ledger line reads: project {entity}; {label}: {value}.',
    'Project {entity} carries this approved attribute in the master file: {label} {value}.',
    'The signed data card pairs project {entity} with {value} in its {label} field.',
    'A check of the completed project file finds {label} {value} for {entity}.',
    'For the project named {entity}, the certified sheet specifies {value} under {label}.',
    'The master record associates the {label} {value} directly with project {entity}.',
    'The current file contains one explicit assertion about {entity}: its {label} equals {value}.',
)
HEADER = (
    'Section title: project {entity}. The attribute recorded beneath this title is {label}: {value}.',
    'A project card is headed {entity}. On that card, {value} occupies the {label} field.',
    'This entry is filed as {entity}. One detail follows for this entry. The {label} equals {value}.',
    'Open the folder labelled {entity}. The attribute shown inside is {label} {value}.',
    'A new project heading names {entity}. Beneath the heading, the {label} is entered as {value}.',
    'The subject of this block is {entity}. The block states {value} for the {label}.',
    'Project identifier for this section: {entity}. The section gives its {label} as {value}.',
    'An individual card now begins for {entity}. Its sole recorded attribute is {label}: {value}.',
    'The document labels this paragraph {entity}. The paragraph assigns {value} to its {label}.',
)
UPDATE = (
    'The superseded card for {entity} recorded {label} {old}. The newly signed card replaces that entry with {value}.',
    'At first the {label} of project {entity} was recorded as {old}. A subsequent authorized edit sets it to {value}.',
    'A retired file named {old} as the {label} for {entity}. Its active replacement names {value}.',
    'For project {entity}, an unapproved draft showed {label} {old}. The approved release of the record shows {value}.',
    'The project card for {entity} was amended: its former {label} was {old}. Its current {label} is {value}.',
    'An archived entry for {entity} contains {old} in the {label} field. The operative version contains {value} instead.',
    'Project {entity} initially carried {label} {old}. The final signed update changes the recorded value to {value}.',
    'The first sheet associated {entity} with {old} for the {label}. The replacement sheet supersedes this with {value}.',
    'For the {label} of {entity}, the earlier value {old} has been withdrawn. The latest approved value is {value}.',
)
CONTRAST = (
    'The certified {label} for {entity} is {value}; the alternative {old} is marked invalid.',
    'An incorrect transcription gave {old} as the {label} for {entity}. The checked transcription gives {value}.',
    'Project {entity} must use {value} for its {label}, not the rejected proposal {old}.',
    'The examiner ruled out {old} in the {label} field of {entity}. The accepted field value is {value}.',
    'In the card for {entity}, {value} is the valid {label} and {old} is explicitly incorrect.',
    'The register warns that the {label} of {entity} is not {old}. It expressly confirms {value}.',
    'A rejected claim about {entity} supplied {label} {old}. The verified record supplies {value}.',
    'For project {entity}, the accepted {label} reads {value}. A competing reading of {old} was disallowed.',
    'Do not take {old} as the {label} of {entity}. The certification identifies {value} as correct.',
)
REFERENCE = (
    'The retrieval code for project {entity} is {key}. The linked card {key} contains {label} {value}.',
    'Look up {entity} through identifier {key}. Under identifier {key}, the {label} is entered as {value}.',
    'The project index assigns record {key} to {entity}. Reading record {key} reveals {label} {value}.',
    'A pointer from {entity} leads to file {key}. That file {key} specifies {value} for the {label}.',
    'The directory lists {key} as the key belonging to project {entity}. The keyed record {key} reports {label} {value}.',
    'Project {entity} can be retrieved using reference {key}. The entry for reference {key} certifies {value} as the {label}.',
    'In the project index, {entity} is connected to ticket {key}. The ticket labelled {key} contains {label} {value}.',
    'The lookup table binds {entity} to code {key}. The separate code {key} record states {value} for the {label}.',
    'An index entry for project {entity} names record key {key}. Record key {key} carries {label} {value}.',
)
TEXT_TEMPLATES = dict(zip(KINDS[:5], (DIRECT, HEADER, UPDATE, CONTRAST, REFERENCE)))
CONTRAST_CURRENT_FIRST = {0, 2, 4, 7}
CAPTIONS = (
    'CERTIFIED ATTRIBUTE ROSTER', 'OPERATIVE PROJECT CATALOGUE', 'CHECKED PROJECT DATASHEET',
    'PROJECT MASTER CARD INDEX', 'ACTIVE ATTRIBUTE DIRECTORY', 'CERTIFICATION OFFICE LEDGER',
    'RELEASED PROJECT FACT SHEET', 'SIGNED SOURCE CARD COLLECTION', 'PROJECT RECORD RECONCILIATION',
)
ROOT_KEYS = ('source_cards','operative_catalogue','certified_roster','master_cards','active_entries','checked_files','source_inventory','released_cards','verified_collection')
DETAIL_KEYS = ('certified_state','operative_attributes','attribute_sheet','released_values','active_facts','signed_fields','checked_attributes','source_values','certified_details')
FIELD_ORDERS = ((0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0),(0,1,2),(1,2,0),(2,0,1))
PREFIXES = ('Ref','Code','Id','Record','Entry','File','Item','Ticket','Key')

def facts(rng):
    owner = f'{rng.choice(FIRST)} {rng.choice(LAST)}'
    day = date(2028,rng.randint(1,12),rng.randint(1,28))
    amount = rng.randint(12000,98000)
    values = [owner,day.isoformat(),f'USD {amount}']
    aliases = [[owner],[day.isoformat(),f'{calendar.month_name[day.month]} {day.day}, {day.year}'],
               [str(amount),f'{amount:,}',f'USD {amount}',f'USD {amount:,}',f'${amount}',f'{amount} USD']]
    return values,aliases

def render(kind,family,entity,values,old,key,distractors,rng,field):
    if kind in TEXT_TEMPLATES:
        target = TEXT_TEMPLATES[kind][family].format(entity=entity,label=LABELS[field],value=values[field],old=old[field],key=key)
        paragraphs = [f'The project card for {name} gives owner {v[0]}, release date {v[1]}, and approved budget {v[2]}.' for name,v in distractors]
        paragraphs.insert(rng.randrange(len(paragraphs)+1),target)
        if kind=='reference':
            paragraphs.insert(rng.randrange(len(paragraphs)+1),f'A separate key {key}X gives {LABELS[field]} {old[field]}; it belongs to a different project.')
        return '\n\n'.join(paragraphs)
    records = list(distractors)
    records.insert(rng.randrange(len(records)+1),(entity,values))
    order = FIELD_ORDERS[family]
    if kind=='table':
        # Project is in an interior column in all new layouts. V2 used its
        # first or last column. The nine resulting column layouts are unique.
        columns=[LABELS[j] for j in order]
        columns.insert(1 if family<6 else 2,'Project')
        lines=[CAPTIONS[family], '| '+' | '.join(columns)+' |','| '+' | '.join('---' for _ in columns)+' |']
        for name,v in records:
            mapping={'Project':name,**dict(zip(LABELS,v))}
            lines.append('| '+' | '.join(mapping[c] for c in columns)+' |')
        lines.append('Read each row by its column headings; every row is a separate project card.')
        return '\n'.join(lines)
    if kind=='nested_json':
        entries=[]
        for name,v in records:
            payload={DETAIL_KEYS[family]:{LABELS[j]:v[j] for j in order}}
            item={'project':name,'card':payload} if family%2==0 else {'card':payload,'project':name}
            entries.append(item)
        return json.dumps({'certification':{ROOT_KEYS[family]:entries}},indent=2,ensure_ascii=False)
    if kind=='paired_directory':
        prefix=PREFIXES[family]
        # Identifier ranges are unique per family. Linking is still a visible
        # key join, but the mapping and field layouts differ from V2's colon
        # mapping plus equals-separated attribute row.
        codes=[family*100+101+i for i in range(len(records))]
        forms=(
            lambda n,k:f'{prefix} {k} -> {n}',
            lambda n,k:f'{n} = {prefix} {k}',
            lambda n,k:f'Project {n} uses {prefix} {k}.',
            lambda n,k:f'{prefix} {k} corresponds to {n}.',
            lambda n,k:f'{n} :: {prefix} {k}',
            lambda n,k:f'Project {n} is linked with {prefix} {k}.',
            lambda n,k:f'{prefix} {k} identifies project {n}.',
            lambda n,k:f'Lookup for {n}: {prefix} {k}.',
            lambda n,k:f'The directory assigns {prefix} {k} to {n}.',
        )
        mappings=[forms[family](n,k) for (n,_),k in zip(records,codes)]
        attributes=[f'{prefix} {k} | '+' | '.join(f'{LABELS[j]}: {v[j]}' for j in order) for (_,v),k in zip(records,codes)]
        if family%2: attributes.reverse()
        return (f'{CAPTIONS[family]} — PROJECT LOOKUP\n'+'\n'.join(mappings)+
                f'\n\n{CAPTIONS[family]} — ATTRIBUTE CARDS\n'+'\n'.join(attributes)+
                '\nA shared identifier connects each project lookup to its attribute card.')
    raise ValueError(kind)

def parse_rendered_current(task,source):
    """Audit emitted source text rather than trust its template arguments."""
    kind,text,entity,field=task['task_type'],task['context'],source['entity'],source['requested_field']
    if kind in TEXT_TEMPLATES:
        target=[p for p in text.split('\n\n') if entity in p]
        assert len(target)==1
        patterns={
            'name':rf"\b(?:{'|'.join(FIRST)}) (?:{'|'.join(LAST)})\b",
            'date':r'\b2028-\d{2}-\d{2}\b', 'amount':r'\bUSD \d{5}\b'}
        values=re.findall(patterns[task['answer_type']],target[0])
        if kind in ('update','contrast'):
            assert len(values)==2 and values[0]!=values[1]
            position=0 if kind=='contrast' and task['template_family'] in CONTRAST_CURRENT_FIRST else 1
            assert values[1-position]==source['obsolete_or_rejected'][field]
            return values[position]
        assert len(values)==1
        return values[0]
    if kind=='table':
        lines=[line for line in text.splitlines() if line.startswith('|')]
        header=[x.strip() for x in lines[0].strip('|').split('|')]
        records=[dict(zip(header,[x.strip() for x in line.strip('|').split('|')])) for line in lines[2:]]
        matches=[r for r in records if r['Project']==entity]
        assert len(matches)==1
        return matches[0][field]
    if kind=='nested_json':
        def find_record(value):
            if isinstance(value,dict):
                if value.get('project')==entity: return value
                for child in value.values():
                    found=find_record(child)
                    if found is not None:return found
            elif isinstance(value,list):
                for child in value:
                    found=find_record(child)
                    if found is not None:return found
            return None
        def find_field(value):
            if isinstance(value,dict):
                if field in value:return value[field]
                for child in value.values():
                    found=find_field(child)
                    if found is not None:return found
            return None
        return find_field(find_record(json.loads(text)))
    if kind=='paired_directory':
        mappings=[line for line in text.splitlines() if entity in line]
        assert len(mappings)==1
        match=re.search(r'\b(Ref|Code|Id|Record|Entry|File|Item|Ticket|Key) (\d+)\b',mappings[0])
        assert match is not None
        prefix=match.group(0)
        attributes=[line for line in text.splitlines() if line.startswith(prefix+' | ')]
        assert len(attributes)==1
        return dict(part.split(': ',1) for part in attributes[0].split(' | ')[1:])[field]
    raise ValueError(kind)

def generate():
    split_rng,rng=random.Random(SEED),random.Random(SEED+1)
    tasks,sources,assignments=[],[],{}
    for kind in KINDS:
        family_order=list(range(9));split_rng.shuffle(family_order)
        assignments[kind]={f:'calibration' if i<4 else 'test' for i,f in enumerate(family_order)}
        for family in range(9):
            for instance in range(5):
                index=len(tasks);field=index%3
                values,aliases=facts(rng);old,_=facts(rng)
                while old[field]==values[field]:old,_=facts(rng)
                entity=f'{rng.choice(PROJECT)} Vela-{index:04d}'
                key=f'R{930000+index}'
                band='long' if (family+instance)%2 else 'short'
                short,long={'table':(4,14),'nested_json':(2,7),'paired_directory':(2,9)}.get(kind,(3,11))
                count=long if band=='long' else short
                distractors=[]
                for j in range(count):
                    v,_=facts(rng)
                    while v[field]==values[field]:v,_=facts(rng)
                    distractors.append((f'{rng.choice(PROJECT)} Peripheral-{index:04d}-{j:02d}',v))
                text=render(kind,family,entity,values,old,key,distractors,rng,field)
                task={'task_id':f'v3-fresh-{index:05d}','source_id':f'v3-source-{index:05d}',
                      'group_id':f'v3-{kind}-family-{family:02d}','split':assignments[kind][family],
                      'task_type':kind,'answer_type':('name','date','amount')[field],
                      'context':text,'question':f'What is the current {LABELS[field]} for project {entity}?',
                      'answer':aliases[field][0],'answer_aliases':aliases[field],
                      'length_band':band,'template_family':family,'template_version':'3.0',
                      'data_origin':'synthetic controlled source facts; AI-assisted generator'}
                source={'task_id':task['task_id'],'entity':entity,'requested_field':LABELS[field],
                        'current_attributes':dict(zip(LABELS,values)),
                        'obsolete_or_rejected':dict(zip(LABELS,old)),
                        'distractors':[{'project':name,**dict(zip(LABELS,v))} for name,v in distractors]}
                parsed=parse_rendered_current(task,source)
                assert parsed==source['current_attributes'][LABELS[field]],task['task_id']
                assert parsed.removeprefix('USD ')==task['answer']
                source['independent_rendered_current_audit_pass']=True
                tasks.append(task);sources.append(source)
    return tasks,sources,assignments

def main():
    root=Path(__file__).resolve().parent;data=root/'data';data.mkdir(exist_ok=True)
    tasks,sources,assignments=generate()
    assert generate()==(tasks,sources,assignments),'Determinism failure'
    assert len(tasks)==360 and len({t['task_id'] for t in tasks})==360 and len({t['source_id'] for t in tasks})==360
    split_counts=Counter(t['split'] for t in tasks)
    assert split_counts=={'calibration':160,'test':200}
    groups=defaultdict(set)
    for t in tasks:groups[t['group_id']].add(t['split'])
    assert len(groups)==72 and all(len(s)==1 for s in groups.values())
    for kind in KINDS:
        assert Counter(t['split'] for t in tasks if t['task_type']==kind)=={'calibration':20,'test':25}
        for split in split_counts:
            assert {t['length_band'] for t in tasks if t['task_type']==kind and t['split']==split}=={'short','long'}
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(root/'assets/tokenizer.json'))
    counts=[len(tokenizer.encode(t['context'],add_special_tokens=False).ids) for t in tasks]
    assert max(counts)<1300
    old_tasks=[]
    for version,names in [('CacheScope_MASC520',['pilot','main','structural_holdout','length_stress']),('CacheScope_MASC520_V2',['new'])]:
        for name in names:
            path=root.parent/version/'data'/f'{name}_tasks.jsonl'
            if path.exists():old_tasks.extend(json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip())
    for key in ('task_id','source_id','context'):
        assert not ({t[key] for t in tasks}&{t[key] for t in old_tasks})
    for filename,rows in [('fresh_tasks.jsonl',tasks),('source_registry.jsonl',sources)]:
        (data/filename).write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows),encoding='utf-8')
    audit={'seed':SEED,'tasks':360,'split_counts':dict(split_counts),'unique_sources':360,
           'wording_layout_families':72,'calibration_families':32,'test_families':40,'tasks_per_family':5,
           'family_splits_disjoint':True,'wording_inventories_new_to_v3':True,
           'old_new_task_source_context_disjoint':True,'old_task_records_compared':len(old_tasks),
           'independent_rendered_gold_audit_pass':360,'no_model_outputs_read':True,
           'structure_counts':dict(Counter(t['task_type'] for t in tasks)),
           'length_band_counts':dict(Counter(t['length_band'] for t in tasks)),
           'context_tokens':{band:{'min':min(n for t,n in zip(tasks,counts) if t['length_band']==band),
                                  'max':max(n for t,n in zip(tasks,counts) if t['length_band']==band)} for band in ('short','long')},
           'context_tokens_by_structure':{kind:{band:{'min':min(n for t,n in zip(tasks,counts) if t['length_band']==band and t['task_type']==kind),
                                                      'max':max(n for t,n in zip(tasks,counts) if t['length_band']==band and t['task_type']==kind)} for band in ('short','long')} for kind in KINDS},
           'family_assignments':assignments,
           'scope':'Known eight high-level structures; independent new sources and new wording/layout families. Synthetic English owner/date/budget extraction only; not unseen-structure OOD or real long-document benchmark.',
           'sha256':{name:hashlib.sha256((data/name).read_bytes()).hexdigest() for name in ('fresh_tasks.jsonl','source_registry.jsonl')}}
    (data/'generation_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('family_assignments','context_tokens_by_structure')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
