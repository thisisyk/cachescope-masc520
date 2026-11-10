"""Inventory publishable repository files without exposing suspected secrets."""
from __future__ import annotations
import hashlib,json,re
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={'.git','__pycache__','.cache','.venv','venv','reproductions','tmp'}
GENERATED={'upload_manifest.json','publication_scan.json'}
SECRET_PATTERNS={
    'github_token':re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})'),
    'openai_style_key':re.compile(rb'sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{25,}'),
    'huggingface_token':re.compile(rb'hf_[A-Za-z0-9]{30,}'),
    'aws_access_key':re.compile(rb'(?:AKIA|ASIA)[A-Z0-9]{16}'),
    'private_key_header':re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
}
EMAIL=re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
LOCAL_PATH=re.compile(r'(?:[A-Za-z]:[\\/]{1,2}Users[\\/]{1,2}[A-Za-z0-9._-]+[\\/]|/(?:Users|home)/[A-Za-z0-9._-]+/)')

def eligible(path):
    relative=path.relative_to(ROOT)
    if any(part in EXCLUDED for part in relative.parts):return False
    if len(relative.parts)>1 and relative.parts[0]=='reports' and relative.parts[1] in {'qa','rendered'}:return False
    return path.name not in {'upload_manifest.json','.DS_Store','Thumbs.db'} and not path.name.startswith('~$') and path.suffix not in {'.pyc','.tmp'}

def inventory():
    entries=[]
    for path in sorted(p for p in ROOT.rglob('*') if p.is_file() and eligible(p)):
        data=path.read_bytes()
        entries.append({'path':path.relative_to(ROOT).as_posix(),'bytes':len(data),
          'sha256':hashlib.sha256(data).hexdigest(),
          'git_blob_sha1':hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()})
    return entries

def scan():
    credentials=[];emails=[];local_paths=[];suspicious_names=[]
    for path in sorted(p for p in ROOT.rglob('*') if p.is_file() and eligible(p) and p.name not in GENERATED):
        relative=path.relative_to(ROOT).as_posix()
        if path.name=='.env' or path.suffix.lower() in {'.pem','.key','.p12','.pfx'} or path.name in {'id_rsa','id_ed25519'}:
            suspicious_names.append(relative)
        data=path.read_bytes()
        for category,pattern in SECRET_PATTERNS.items():
            count=len(pattern.findall(data))
            if count:credentials.append({'path':relative,'category':category,'matches':count})
        try:text=data.decode('utf-8')
        except UnicodeDecodeError:continue
        count=len(EMAIL.findall(text))
        if count:emails.append({'path':relative,'category':'email_like_text','matches':count})
        count=len(LOCAL_PATH.findall(text))
        if count:local_paths.append({'path':relative,'category':'historical_machine_path','matches':count})
    return {'credential_candidates':credentials,'credential_candidate_files':len({x['path'] for x in credentials}),
      'email_like_files':emails,'historical_machine_path_files':local_paths,'suspicious_sensitive_filenames':suspicious_names,
      'suspected_values_printed':False,
      'interpretation':'Pattern scan, not a security guarantee. Synthetic people/project facts are deliberate dataset content. Historical machine paths are retained for byte-preserved provenance; portable commands are in README. Inspect any credential or real-contact candidate before publication.'}

def main():
    scanned=scan()
    (ROOT/'publication_scan.json').write_text(json.dumps(scanned,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    entries=inventory()
    assert all(x['bytes']<100_000_000 for x in entries),'A file exceeds the GitHub standard single-file limit'
    unique={x['git_blob_sha1']:x['bytes'] for x in entries}
    summary={'files':len(entries),'total_bytes':sum(x['bytes'] for x in entries),
      'unique_git_blobs':len(unique),'unique_blob_bytes':sum(unique.values()),
      'duplicate_content_bytes':sum(x['bytes'] for x in entries)-sum(unique.values()),
      'largest_file':max(entries,key=lambda x:x['bytes']),
      'version_file_counts':dict(Counter(x['path'].split('/')[1] for x in entries if x['path'].startswith('experiments/'))),
      'all_files_below_100MB':True}
    manifest={'format':'CacheScope submission inventory v1','hash_scope':'Every publishable file except upload_manifest.json itself; raw bytes without line-ending normalization.',
      'summary':summary,'files':entries}
    (ROOT/'upload_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'summary':summary,'credential_candidate_files':scanned['credential_candidate_files'],
      'email_like_file_count':len(scanned['email_like_files']),'historical_machine_path_file_count':len(scanned['historical_machine_path_files']),
      'suspicious_sensitive_filenames':scanned['suspicious_sensitive_filenames']},ensure_ascii=False))

if __name__=='__main__':main()
