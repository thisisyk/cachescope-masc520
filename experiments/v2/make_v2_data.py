"""Generate the frozen V2 synthetic benchmark, before observing model answers.

Eight known high-level structures occur in each split. Wording/layout families,
source records and entities are disjoint. This is not a structural OOD test.
Canonical source facts are written separately for a reproducible gold audit;
neither the compressor nor the risk features receive that registry.
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

SEED = 1111520
KINDS = ('direct', 'header', 'update', 'contrast', 'reference', 'table', 'nested_json', 'paired_directory')
LABELS = ('owner', 'release date', 'approved budget')
FIRST = ('Ada', 'Mia', 'Nora', 'Lena', 'Iris', 'Clara', 'Owen', 'Evan', 'Noah', 'Liam', 'Leo', 'Finn', 'Hugo', 'Maya', 'Zoe', 'Elena')
LAST = ('Carter', 'Reed', 'Brooks', 'Ward', 'Hayes', 'Bennett', 'Morgan', 'Stone', 'Parker', 'Wells', 'Ross', 'Turner', 'Foster', 'Grant', 'Blake', 'Cole')
PROJECT = ('Aster', 'Kestrel', 'Quartz', 'Redwood', 'Larch', 'Sequoia', 'Acacia', 'Wren', 'Elm', 'Tern', 'Spruce', 'Pine')

# Each list is a family inventory, never selected using measured outcomes.
DIRECT = (
    'The approved registry states that project {entity} has {label} {value}.',
    'For project {entity}, the official {label} entry is {value}.',
    'Project {entity}: the confirmed {label} is {value}.',
    'A signed project sheet names {value} as the {label} for {entity}.',
    'The {label} attached to project {entity} in the final register reads {value}.',
    'In its authorized project file, {entity} records {value} for the {label}.',
    'One finalized record specifies the following for {entity}: {label} = {value}.',
    'The verified {label} of project {entity} was entered as {value}.',
    'Under project {entity}, the registry confirms this attribute: {label}, {value}.',
)
HEADER = (
    'Project {entity}. This section concerns that project alone. Its {label} is {value}.',
    'The next section is about {entity}. For this section, the {label} is {value}.',
    'File heading: {entity}. The filed {label} reads {value}.',
    'Here begins the individual entry for {entity}. Its approved {label} equals {value}.',
    'The record now turns to {entity}. A separate sentence records its {label}: {value}.',
    'Dedicated project: {entity}. The following attribute belongs to it. The {label} is {value}.',
    'This subsection names project {entity}. Within it, {value} is listed as the {label}.',
    'The project described in this paragraph is {entity}. Its verified {label} is recorded as {value}.',
    'Individual folder for {entity}. In that folder the current {label} is {value}.',
)
UPDATE = (
    'Project {entity} once listed {old} as its {label}. The final approved correction replaces it with {value}.',
    'The draft {label} for {entity} was {old}. Its current authorized value is {value}.',
    'An earlier sheet assigned {old} to the {label} of {entity}. A signed amendment changes that entry to {value}.',
    'For {entity}, the provisional {label} read {old}. The accepted revision now reads {value}.',
    'Project {entity} had a historical {label} of {old}. The latest verified register instead records {value}.',
    'The obsolete project entry for {entity} gave the {label} {old}. Its replacement gives {value}.',
    'A first version named {old} as the {label} for {entity}. That version was withdrawn and the final value is {value}.',
    'The former {label} entry for {entity} was {old}. The project office approved a correction to {value}.',
    'For project {entity}, {old} was the original {label}. After approval of the update, the current value became {value}.',
)
CONTRAST = (
    'For project {entity}, the {label} is not {old}. The verified value is {value}.',
    'The {label} for {entity} is {value}, rather than the incorrectly proposed {old}.',
    'A claim of {old} as the {label} of {entity} was rejected. The valid entry is {value}.',
    'Project {entity} explicitly excludes {old} as its {label}. Its confirmed value is {value}.',
    'An audit of {entity} accepted {value} for the {label} and rejected {old}.',
    'For the {label} of {entity}, {old} is an error. The correct transcription is {value}.',
    'Do not use {old} for the {label} of {entity}. The approved record specifies {value}.',
    'Two values for the {label} of {entity} were examined. Only {value} was valid; {old} was marked incorrect.',
    'The disputed {label} for {entity} was {old}. Verification ruled out that claim and established {value}.',
)
REFERENCE = (
    'Project {entity} is indexed under {key}. Record {key} gives the {label} {value}.',
    'The directory connects {entity} with entry {key}. Entry {key} records its {label} as {value}.',
    'To find project {entity}, use key {key}. The {label} in the file for {key} reads {value}.',
    'Project {entity} has registry identifier {key}. Registry identifier {key} points to the {label} {value}.',
    'The assigned ledger code for {entity} is {key}. Ledger code {key} specifies {value} as the {label}.',
    'Project {entity} belongs to file {key}. File {key} contains the current {label}: {value}.',
    'An index lookup maps {entity} to ticket {key}. Ticket {key} confirms the {label} {value}.',
    'The source record associated with {entity} is {key}. Source record {key} lists {value} in the {label} field.',
    'A project directory entry says {entity} uses reference {key}. Reference {key} assigns the {label} {value}.',
)
TEXT_TEMPLATES = {'direct': DIRECT, 'header': HEADER, 'update': UPDATE, 'contrast': CONTRAST, 'reference': REFERENCE}
CAPTIONS = (
    'APPROVED PROJECT REGISTER', 'CURRENT PROJECT RECORDS', 'VERIFIED PROJECT LEDGER',
    'SIGNED ATTRIBUTE REGISTER', 'FINAL PROJECT INVENTORY', 'AUTHORIZED RECORD CATALOG',
    'PROJECT OFFICE RECORD BOOK', 'CONFIRMED PROJECT DIRECTORY', 'ACCEPTED PROJECT DATA',
)
ROOT_KEYS = ('records', 'project_register', 'approved_entries', 'catalog', 'project_files', 'ledger', 'inventory', 'confirmed_data', 'directory')
DETAIL_KEYS = ('details', 'attributes', 'approved_values', 'current_fields', 'record', 'facts', 'verified_fields', 'project_data', 'properties')
FIELD_ORDERS = ((0,1,2), (2,0,1), (1,2,0), (0,2,1), (2,1,0), (1,0,2), (0,1,2), (1,2,0), (2,0,1))


def facts(rng):
    owner = f'{rng.choice(FIRST)} {rng.choice(LAST)}'
    day = date(2027, rng.randint(1,12), rng.randint(1,28))
    amount = rng.randint(12000, 98000)
    values = [owner, day.isoformat(), f'USD {amount}']
    aliases = [[owner], [day.isoformat(), f'{calendar.month_name[day.month]} {day.day}, {day.year}'],
               [str(amount), f'{amount:,}', f'USD {amount}', f'USD {amount:,}', f'${amount}', f'{amount} USD']]
    return values, aliases


def render(kind, family, entity, values, old, key, distractors, rng):
    """Render only predefined facts; no answering model is involved."""
    if kind in TEXT_TEMPLATES:
        field = render.field
        target = TEXT_TEMPLATES[kind][family].format(entity=entity, label=LABELS[field], value=values[field], old=old[field], key=key)
        units = [f'Project {name} has owner {v[0]}, release date {v[1]}, and approved budget {v[2]}.' for name,v in distractors]
        units.insert(rng.randrange(len(units)+1), target)
        if kind == 'reference':
            units.insert(rng.randrange(len(units)+1), f'An unrelated record {key}X has the {LABELS[field]} {old[field]}.')
        return '\n\n'.join(units), None
    records = list(distractors)
    target_pos = rng.randrange(len(records)+1)
    records.insert(target_pos, (entity, values))
    order = FIELD_ORDERS[family]
    if kind == 'table':
        # A varying placement of the project column is part of the layout family.
        columns = ['Project'] + [LABELS[j] for j in order]
        if family >= 6:
            columns = columns[1:] + columns[:1]
        lines = [CAPTIONS[family], '| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join('---' for _ in columns) + ' |']
        for name,v in records:
            mapping = {'Project': name, **dict(zip(LABELS,v))}
            lines.append('| ' + ' | '.join(mapping[c] for c in columns) + ' |')
        lines.append('Each row describes one project; the headings identify its attributes.')
        text = '\n'.join(lines)
        # Parse the emitted header and row rather than trust the answer variable.
        parsed = [x.strip() for x in next(line for line in lines if entity in line).strip('|').split('|')]
        parsed_fields = dict(zip(columns, parsed))
        return text, [parsed_fields[label] for label in LABELS]
    if kind == 'nested_json':
        root, detail = ROOT_KEYS[family], DETAIL_KEYS[family]
        entries = []
        for name,v in records:
            attributes = {LABELS[j]: v[j] for j in order}
            item = {'project':name, detail:attributes} if family % 2 == 0 else {detail:attributes, 'project':name}
            entries.append(item)
        text = json.dumps({root:entries}, indent=2, ensure_ascii=False)
        parsed = next(r[detail] for r in json.loads(text)[root] if r['project']==entity)
        return text, [parsed[label] for label in LABELS]
    if kind == 'paired_directory':
        prefixes = ('Item', 'Entry', 'Record', 'Slot', 'Index', 'Unit', 'File', 'Ticket', 'Key')
        prefix = prefixes[family]
        mappings = [f'{prefix} {i+1}: {name}' for i,(name,_) in enumerate(records)]
        attributes = [f'{prefix} {i+1}: ' + '; '.join(f'{LABELS[j]}={v[j]}' for j in order) for i,(_,v) in enumerate(records)]
        if family % 2:
            attributes.reverse()
        headings = (f'{CAPTIONS[family]}: PROJECT INDEX', f'{CAPTIONS[family]}: ATTRIBUTE INDEX')
        text = headings[0]+'\n'+'\n'.join(mappings)+'\n\n'+headings[1]+'\n'+'\n'.join(attributes)+f'\nThe {prefix.lower()} numbers link the project index to the attribute index.'
        item_number = next(re.match(r'\w+ (\d+): ', line).group(1) for line in mappings if line.endswith(': '+entity))
        attribute_line = next(line for line in attributes if line.startswith(f'{prefix} {item_number}: '))
        parsed = dict(part.split('=',1) for part in attribute_line.split(': ',1)[1].split('; '))
        return text, [parsed[label] for label in LABELS]
    raise ValueError(kind)


def generate():
    split_rng = random.Random(SEED)
    rng = random.Random(SEED + 1)
    rows, registry = [], []
    assignments = {}
    for kind in KINDS:
        family_order = list(range(9))
        split_rng.shuffle(family_order)
        assignments[kind] = {family: ('train' if i<3 else 'validation' if i<5 else 'test') for i,family in enumerate(family_order)}
        for family in range(9):
            for instance in range(5):
                index = len(rows)
                field = index % 3
                render.field = field
                values, aliases = facts(rng)
                old,_ = facts(rng)
                while old[field] == values[field]:
                    old,_ = facts(rng)
                entity = f'{rng.choice(PROJECT)} Nova-{index:04d}'
                key = f'R{710000+index}'
                band = 'long' if (instance + family) % 2 else 'short'
                # More structured syntax consumes more tokens per record.
                counts = {'nested_json':(2,8), 'paired_directory':(3,11), 'table':(3,13)}
                short_count,long_count = counts.get(kind,(3,11))
                count = long_count if band=='long' else short_count
                distractors = [(f'{rng.choice(PROJECT)} Side-{index:04d}-{j:02d}', facts(rng)[0]) for j in range(count)]
                # Ensure every stored current target value is unique among rows.
                for j,(name,v) in enumerate(distractors):
                    while v[field] == values[field]:
                        v,_ = facts(rng)
                    distractors[j] = (name,v)
                context, parsed_values = render(kind,family,entity,values,old,key,distractors,rng)
                if parsed_values is not None:
                    assert parsed_values == values, (kind,index,parsed_values,values)
                split = assignments[kind][family]
                task_id = f'v2-new-{index:05d}'
                row = {'task_id':task_id, 'split':split, 'group_id':f'v2-{kind}-family-{family:02d}',
                       'source_id':f'v2-source-{index:05d}', 'task_type':kind,
                       'answer_type':('name','date','amount')[field], 'context':context,
                       'question':f'What is the current {LABELS[field]} for project {entity}?',
                       'answer':aliases[field][0], 'answer_aliases':aliases[field],
                       'length_band':band, 'data_origin':'synthetic controlled source records; AI-assisted generator',
                       'template_version':'2.0', 'template_family':family}
                source = {'task_id':task_id,'entity':entity,'requested_field':LABELS[field],
                          'current_attributes':dict(zip(LABELS,values)), 'obsolete_or_rejected':dict(zip(LABELS,old)),
                          'distractors':[{'project':n, **dict(zip(LABELS,v))} for n,v in distractors],
                          'parsed_structured_fields_match':parsed_values==values if parsed_values is not None else None}
                assert row['answer'] == (source['current_attributes'][LABELS[field]].removeprefix('USD ') if field==2 else source['current_attributes'][LABELS[field]])
                assert entity in context and values[field] in context
                assert all(n != entity and v[field] != values[field] for n,v in distractors)
                rows.append(row)
                registry.append(source)
    return rows,registry,assignments


def main():
    root = Path(__file__).resolve().parent
    data = root/'data'
    data.mkdir(exist_ok=True)
    rows,registry,assignments = generate()
    groups = defaultdict(set)
    for row in rows:
        groups[row['group_id']].add(row['split'])
    assert len(rows)==360 and len({r['task_id'] for r in rows})==360
    assert len({r['source_id'] for r in rows})==360 and len(groups)==72
    assert all(len(splits)==1 for splits in groups.values())
    counts = Counter(r['split'] for r in rows)
    assert counts=={'train':120,'validation':80,'test':160}
    assert Counter(len([r for r in rows if r['group_id']==group]) for group in groups)=={5:72}
    for kind in KINDS:
        assert Counter(r['split'] for r in rows if r['task_type']==kind)=={'train':15,'validation':10,'test':20}
        for split in counts:
            assert {r['length_band'] for r in rows if r['task_type']==kind and r['split']==split}=={'short','long'}
    tokens = None
    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(root/'assets/tokenizer.json'))
        tokens = [len(tokenizer.encode(r['context'], add_special_tokens=False).ids) for r in rows]
        assert max(tokens)<1300
    except ImportError:
        pass
    def write_jsonl(path, records):
        path.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in records), encoding='utf-8')
    write_jsonl(data/'new_tasks.jsonl',rows)
    write_jsonl(data/'source_registry.jsonl',registry)
    audit = {'seed':SEED, 'tasks':len(rows), 'unique_source_ids':360,'unique_task_ids':360,
             'families':72,'tasks_per_family':5,'family_splits_disjoint':True,'split_counts':dict(counts),
             'structure_counts':dict(Counter(r['task_type'] for r in rows)),
             'length_band_counts':dict(Counter(r['length_band'] for r in rows)),
             'structure_split_length_counts':dict(Counter(f"{r['task_type']}/{r['split']}/{r['length_band']}" for r in rows)),
             'max_characters':max(len(r['context']) for r in rows),'max_words':max(len(r['context'].split()) for r in rows),
             'context_tokens':{'min':min(tokens),'max':max(tokens),
                               'by_band':{band:{'min':min(t for r,t in zip(rows,tokens) if r['length_band']==band),
                                               'max':max(t for r,t in zip(rows,tokens) if r['length_band']==band)} for band in ('short','long')}} if tokens else None,
             'gold_audit':{'canonical_registry_checked':360,'structured_context_independently_parsed':135,
                           'prose_checked_against_explicit_canonical_registry':225,
                           'target_current_value_differs_from_distractor_values':True,
                           'no_llm_answers_observed_during_generation':True},
             'scope':'Controlled synthetic factual extraction; eight known structures occur in all splits; wording/layout families and sources are disjoint. Not a structural OOD or real-document benchmark.',
             'family_assignments':assignments,
             'sha256':{name:hashlib.sha256((data/name).read_bytes()).hexdigest() for name in ('new_tasks.jsonl','source_registry.jsonl')}}
    (data/'new_data_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('family_assignments','structure_split_length_counts')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
