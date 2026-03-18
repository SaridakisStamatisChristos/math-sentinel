[CmdletBinding()]
param(
    [switch]$Yes,
    [switch]$KeepResults
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir

$targets = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)

function Add-Target {
    param([string]$Path)
    if (Test-Path -LiteralPath $Path) {
        [void]$targets.Add($Path)
    }
}

Add-Target (Join-Path $repoRoot 'checkpoints')
Add-Target (Join-Path $repoRoot 'logs')
Add-Target (Join-Path $repoRoot '.tmp-benchmarks')
Add-Target (Join-Path $repoRoot '.tmp-tests')
Add-Target (Join-Path $repoRoot '.pytest_cache')
Add-Target (Join-Path $repoRoot 'memory\replay.jsonl')
Add-Target (Join-Path $repoRoot 'memory\hard_cases.json')
Add-Target (Join-Path $repoRoot 'memory\lemma_store.json')
Add-Target (Join-Path $repoRoot 'memory\tactic_stats.json')

if (-not $KeepResults) {
    Add-Target (Join-Path $repoRoot 'results\benchmark_ledger.jsonl')
    Add-Target (Join-Path $repoRoot 'results\campaigns')
    $resultFiles = Get-ChildItem (Join-Path $repoRoot 'results') -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension -in @('.json', '.jsonl') -and $_.Name -ne 'README.md' }
    foreach ($file in $resultFiles) {
        Add-Target $file.FullName
    }
}

$rootPycache = Join-Path $repoRoot '__pycache__'
Add-Target $rootPycache

$pycacheRoots = @(
    (Join-Path $repoRoot 'benchmarks'),
    (Join-Path $repoRoot 'curriculum'),
    (Join-Path $repoRoot 'domains'),
    (Join-Path $repoRoot 'engine'),
    (Join-Path $repoRoot 'memory'),
    (Join-Path $repoRoot 'plugins'),
    (Join-Path $repoRoot 'proof'),
    (Join-Path $repoRoot 'search'),
    (Join-Path $repoRoot 'sentinel'),
    (Join-Path $repoRoot 'tests'),
    (Join-Path $repoRoot 'tools')
)
foreach ($root in $pycacheRoots) {
    if (-not (Test-Path -LiteralPath $root)) {
        continue
    }
    $pycacheDirs = Get-ChildItem $root -Directory -Recurse -Force -Filter __pycache__ -ErrorAction SilentlyContinue
    foreach ($dir in $pycacheDirs) {
        Add-Target $dir.FullName
    }
}

Write-Host 'Targets to remove:'
foreach ($target in $targets) {
    Write-Host " - $target"
}

if (-not $Yes) {
    $answer = Read-Host 'Proceed and remove these files/directories? (y/N)'
    if ($answer -notin @('y', 'Y', 'yes', 'YES')) {
        Write-Host 'Aborted by user.'
        exit 0
    }
}

foreach ($target in $targets) {
    if (-not (Test-Path -LiteralPath $target)) {
        Write-Host "Not found: $target"
        continue
    }

    try {
        Remove-Item -LiteralPath $target -Recurse -Force -ErrorAction Stop
        Write-Host "Removed: $target"
    }
    catch {
        Write-Warning "Skipped locked path: $target"
    }
}

New-Item -ItemType Directory -Path (Join-Path $repoRoot 'checkpoints') -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $repoRoot 'logs') -Force | Out-Null

Write-Host 'Clean complete.'
