"""Reproducible synthetic factual-extraction tasks with family-disjoint splits.

No text in this module is a measured model response. Gold answers originate from
the explicit source facts. Pilot tasks are separate sources using train-family
forms. The structural holdout is synthetic, not human-authored or real-world.
"""
from __future__ import annotations

import calendar
from datetime import date
import json
from pathlib import Path
import random


DIRECT = [
    "The {label} for project {entity} is {value}.",
    "Project {entity} lists {value} as its {label}.",
    "The project register assigns the following {label} to {entity}: {value}.",
    "According to the signed record, {entity} has the {label} {value}.",
    "For {entity}, the recorded {label} reads {value}.",
    "A review of project {entity} confirms its {label} as {value}.",
    "In the approved project register, the {label} of {entity} is recorded as {value}.",
    "The official entry for {entity} gives its {label}: {value}.",
    "The project summary identifies {value} as the {label} for {entity}.",
    "One confirmed detail about project {entity} is its {label}, which is {value}.",
]
HEADER = [
    "Project {entity} record. The following sentence belongs only to this project. Its {label} is {value}.",
    "The next entry concerns {entity}. This entry records the {label} as {value}.",
    "A separate section is reserved for project {entity}. Within this section, the {label} is {value}.",
    "Now consider the record for {entity}. The {label} in that record is {value}.",
    "The register opens a new subsection for {entity}. It contains one confirmed {label}: {value}.",
    "The heading of this entry is project {entity}. Under that heading, the {label} reads {value}.",
    "This paragraph describes only {entity}. The {label} specified here is {value}.",
    "An individual file has been opened for {entity}. The file reports a {label} of {value}.",
    "The document turns to project {entity}. Its one verified {label} is {value}.",
    "The next two statements refer to {entity}. A {label} was verified. The verified value is {value}.",
]
UPDATE = [
    "An old entry gives {old} as the {label} for {entity}. A later approved correction replaces that value with {value}.",
    "Project {entity} previously had the {label} {old}. Its final authorized revision changes this to {value}.",
    "The initial {label} for {entity} was {old}. The current register instead specifies {value}.",
    "An obsolete project sheet assigns {old} as the {label} of {entity}. The replacement sheet records {value}.",
    "The {label} of {entity} appeared as {old} in a draft. After review, the accepted value became {value}.",
    "Project {entity} had a provisional {label} of {old}. The signed amendment supersedes it with {value}.",
    "The first record for {entity} reports the {label} {old}. The latest approved record replaces it with {value}.",
    "A historical summary names {old} as the {label} for {entity}. The updated summary confirms {value} instead.",
    "The draft {label} entry for {entity} reads {old}. That draft was withdrawn; the final entry reads {value}.",
    "Project {entity} was originally listed with {old} as its {label}. A correction was accepted, making {value} the current value.",
]
CONTRAST = [
    "The {label} for {entity} is not {old}. The confirmed value is {value}.",
    "A claim that the {label} of {entity} is {old} was rejected. The verified entry is {value}.",
    "Project {entity} has the {label} {value}, rather than the incorrectly suggested {old}.",
    "Do not use {old} as the {label} of {entity}. Its authorized value is {value}.",
    "An audit rejected {old} as the {label} for {entity} and accepted {value}.",
    "Two candidate values were checked for the {label} of {entity}. The value {old} was invalid; {value} was confirmed.",
    "The project register for {entity} excludes {old} as its {label}. It confirms {value}.",
    "For the {label} of {entity}, {old} is an erroneous transcription. The correct reading is {value}.",
    "A disputed entry lists {old} as the {label} for {entity}. An explicit verification establishes {value}.",
    "Only {value} is valid as the {label} of {entity}; {old} is explicitly marked incorrect.",
]
REFERENCE = [
    "Project {entity} uses the internal record key {key}. The record with key {key} has the {label} {value}.",
    "The file assigned to {entity} is {key}. In file {key}, the recorded {label} is {value}.",
    "To locate {entity}, consult ledger entry {key}. Entry {key} gives {value} as the {label}.",
    "The index maps project {entity} to reference {key}. Reference {key} specifies its {label}: {value}.",
    "Project {entity} is tracked under ticket {key}. That ticket records the {label} as {value}.",
    "The register identifies {entity} by code {key}. A separate record for code {key} reports the {label} {value}.",
    "The relevant record identifier for {entity} is {key}. The identifier {key} is associated with {value} in the {label} field.",
    "The project-to-file directory links {entity} with {key}. The file named {key} lists its {label} as {value}.",
    "Project {entity} belongs to registry item {key}. Registry item {key} confirms the {label} {value}.",
    "A lookup for {entity} returns entry {key}. Reading entry {key} yields the {label} {value}.",
]
FAMILIES = {"direct": DIRECT, "header": HEADER, "update": UPDATE,
            "contrast": CONTRAST, "reference": REFERENCE}
FIRST = ["Ada", "Mia", "Nora", "Lena", "Iris", "Clara", "Owen", "Evan", "Noah", "Liam", "Leo", "Finn", "Hugo", "Maya", "Zoe", "Elena"]
LAST = ["Carter", "Reed", "Brooks", "Ward", "Hayes", "Bennett", "Morgan", "Stone", "Parker", "Wells", "Ross", "Turner", "Foster", "Grant", "Blake", "Cole"]
PROJECT = ["Falcon", "Cedar", "Juniper", "Harbor", "Meadow", "Willow", "Orchid", "Summit", "Maple", "Granite", "Cobalt", "Pelican", "Birch", "Sparrow", "Coral", "Aspen"]
LABELS = ["owner", "release date", "approved budget"]


def _facts(rng: random.Random):
    owner = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    day = date(2026, rng.randint(1, 12), rng.randint(1, 28))
    amount = rng.randint(10000, 99999)
    return [owner, day.isoformat(), f"USD {amount}"], [
        [owner],
        [day.isoformat(), f"{calendar.month_name[day.month]} {day.day}, {day.year}",
         f"{calendar.month_abbr[day.month]} {day.day}, {day.year}",
         f"{day.day} {calendar.month_name[day.month]} {day.year}"],
        [str(amount), f"{amount:,}", f"USD {amount}", f"USD {amount:,}", f"${amount}", f"${amount:,}", f"{amount} USD"],
    ]


def _split_map(seed: int):
    """Split at template-family level, stratifying the five structural types."""
    result = {}
    rng = random.Random(seed)
    for kind in FAMILIES:
        ids = list(range(10))
        rng.shuffle(ids)
        for position, family_id in enumerate(ids):
            result[(kind, family_id)] = ("train" if position < 6 else "validation" if position < 8 else "test")
    return result


def _distractor(rng, index, excluded):
    entity = f"{rng.choice(PROJECT)} Auxiliary-{index:04d}"
    values, _ = _facts(rng)
    return (f"Project {entity} has owner {values[0]}, release date {values[1]}, "
            f"and approved budget {values[2]}.")


def _task(rng, index, kind, family_id, split, prefix):
    field = index % 3
    values, aliases = _facts(rng)
    other, _ = _facts(rng)
    while other[field] == values[field]:
        other, _ = _facts(rng)
    entity = f"{rng.choice(PROJECT)} Main-{index:04d}"
    key = f"R{rng.randint(10000, 99999)}"
    target = FAMILIES[kind][family_id].format(
        label=LABELS[field], entity=entity, value=values[field], old=other[field], key=key)
    distractors = [_distractor(rng, index * 10 + j, entity) for j in range(4)]
    # Unrelated, explicitly separate records remain nearby to test attribution.
    distractors.append("Each paragraph above is an independent project record. Values from one record do not apply to another project.")
    insertion = rng.randrange(5)
    distractors.insert(insertion, target)
    if kind == "reference":
        distractors.insert(rng.randrange(len(distractors) + 1),
            f"An unrelated entry R{int(key[1:]) + 1} has the {LABELS[field]} {other[field]}.")
    question = f"What is the current {LABELS[field]} for project {entity}?"
    return {
        "task_id": f"{prefix}-{index:05d}", "split": split,
        "group_id": f"{prefix if prefix == 'pilot' else 'main'}-{kind}-{family_id:02d}",
        "task_type": kind, "answer_type": ["name", "date", "amount"][field],
        "context": "\n\n".join(distractors), "question": question,
        "answer": aliases[field][0], "answer_aliases": aliases[field],
        "data_origin": "synthetic controlled facts", "template_version": "1.0",
        "source_id": f"{prefix}-source-{index:05d}",
    }


def generate_tasks(seed=520, n_tasks=1000, pilot=False):
    """Generate independent source records; same seed and arguments reproduce bytes.

    All variants of a template family have the same main split. Pilot uses only
    training-family sentence forms and distinct source records/IDs.
    """
    if n_tasks < 1:
        raise ValueError("n_tasks must be positive")
    split_map = _split_map(seed)
    families = [(kind, fid) for kind in FAMILIES for fid in range(10)
                if not pilot or split_map[(kind, fid)] == "train"]
    rng = random.Random(seed + (99991 if pilot else 0))
    rng.shuffle(families)
    tasks = []
    for index in range(n_tasks):
        kind, fid = families[index % len(families)]
        split = "pilot" if pilot else split_map[(kind, fid)]
        tasks.append(_task(rng, index, kind, fid, split, "pilot" if pilot else "task"))
    return tasks


def generate_ood_tasks(seed=520, n_tasks=50):
    """Separate synthetic structural holdout: tables, JSON, and paired lists.

    These are explicitly authored generation rules, not human-written examples.
    All source records are held out. No model or threshold may be fitted on them.
    """
    rng = random.Random(seed + 271828)
    tasks = []
    for index in range(n_tasks):
        field = index % 3
        entity = f"{rng.choice(PROJECT)} Holdout-{index:04d}"
        values, aliases = _facts(rng)
        others = [_facts(rng)[0] for _ in range(3)]
        names = [f"Auxiliary Holdout-{index:04d}-{j}" for j in range(3)]
        names.insert(index % 4, entity)
        rows = list(others)
        rows.insert(index % 4, values)
        style = (index // 3) % 3
        if style == 0:
            context = "| Project | Owner | Release date | Approved budget |\n| --- | --- | --- | --- |\n"
            context += "\n".join(f"| {name} | {row[0]} | {row[1]} | {row[2]} |" for name, row in zip(names, rows))
            context += "\nEach table row describes one project. The column headings identify the fields."
        elif style == 1:
            context = json.dumps({"records": [
                {"project": name, "details": {"owner": row[0], "release date": row[1], "approved budget": row[2]}}
                for name, row in zip(names, rows)]}, indent=2)
        else:
            mapping = "\n".join(f"Item {j + 1}: {name}" for j, name in enumerate(names))
            facts = "\n".join(f"Item {j + 1}: owner={row[0]}; release date={row[1]}; approved budget={row[2]}" for j, row in enumerate(rows))
            context = "PROJECT DIRECTORY\n" + mapping + "\n\nATTRIBUTE DIRECTORY\n" + facts + "\nThe item numbers join the two directories."
        tasks.append({
            "task_id": f"ood-{index:05d}", "split": "ood", "group_id": f"ood-structure-{style}",
            "task_type": ["table", "nested_json", "paired_directory"][style],
            "answer_type": ["name", "date", "amount"][field], "context": context,
            "question": f"What is the current {LABELS[field]} for project {entity}?",
            "answer": aliases[field][0], "answer_aliases": aliases[field],
            "data_origin": "synthetic structural holdout", "template_version": "1.0",
            "source_id": f"ood-source-{index:05d}",
        })
    return tasks


def write_dataset(path, tasks):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n")

