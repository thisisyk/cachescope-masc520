"""Copy preserved versions under their original sibling names for regeneration."""
from pathlib import Path
import hashlib,json,shutil,tempfile

ROOT=Path(__file__).resolve().parents[1]
NAMES={'v1':'CacheScope_MASC520','v2':'CacheScope_MASC520_V2','v3':'CacheScope_MASC520_V3'}

def main():
    parent=(ROOT/'reproductions').resolve();parent.mkdir(exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='run-',dir=parent)).resolve()
    assert stage.parent==parent
    count=0
    for version,name in NAMES.items():
        source=ROOT/'experiments'/version
        manifest=json.loads((source/'artifact_manifest.json').read_text(encoding='utf-8'))
        for relative in list(manifest['sha256'])+['artifact_manifest.json']:
            path=source/relative
            if relative!='artifact_manifest.json':assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['sha256'][relative]
            target=stage/name/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
            assert target.read_bytes()==path.read_bytes();count+=1
    print(json.dumps({'staged_workspace':str(stage),'files':count,'original_deliveries_modified':False,
      'note':'Use this separate copy for generation workflows requiring historical sibling folder names. Install requirements first. Frozen original inputs and model-selection rules still apply.'},ensure_ascii=False))

if __name__=='__main__':main()
