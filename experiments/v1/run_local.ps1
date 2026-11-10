param(
    [ValidateSet('train','verify','test','predict','pilot','label','prepare')]
    [string]$Stage = 'verify',
    [string]$ContextFile,
    [string]$Question
)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$localPython = 'C:\Users\HP\Documents\Codex\2026-10-29\den\work\venv\Scripts\python.exe'
$localPackages = @(
    (Join-Path $projectRoot 'src'),
    'C:\Users\HP\Documents\Codex\2026-11-05\ni-h\work\runtime_extra',
    'C:\Users\HP\Documents\Codex\2026-10-29\den\work\torch_compatible',
    'C:\Users\HP\Documents\Codex\2026-10-29\den\work\science_compatible',
    'C:\Users\HP\Documents\Codex\2026-10-29\den\work\packages'
)
if (-not (Test-Path -LiteralPath $localPython)) {
    throw 'The pre-existing local Python environment is unavailable. Install requirements.txt in a new environment and use the README commands.'
}
$env:PYTHONPATH = $localPackages -join [IO.Path]::PathSeparator
$env:MPLCONFIGDIR = Join-Path $projectRoot '.cache\matplotlib'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:PYTHONIOENCODING = 'utf-8'
$env:TEMP = Join-Path $projectRoot '.cache\temp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
Push-Location -LiteralPath $projectRoot
try {
    switch ($Stage) {
        'verify' { & $localPython (Join-Path $projectRoot 'verify_results.py') }
        'test' { & $localPython -m unittest discover -s tests -v }
        'predict' {
            if (-not $ContextFile -or -not $Question) { throw 'predict requires -ContextFile and -Question' }
            & $localPython (Join-Path $projectRoot 'predict.py') --context-file $ContextFile --question $Question
        }
        default { & $localPython (Join-Path $projectRoot 'run_experiment.py') $Stage }
    }
    if ($LASTEXITCODE -ne 0) { throw "Python exited with code $LASTEXITCODE" }
} finally { Pop-Location }
