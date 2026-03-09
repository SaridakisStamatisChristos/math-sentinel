[CmdletBinding()]
param(
    [switch]$Yes
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir

$targets = @(
    (Join-Path $repoRoot 'checkpoints'),
    (Join-Path $repoRoot 'logs'),
    (Join-Path $repoRoot 'memory\replay.jsonl'),
    (Join-Path $repoRoot 'memory\hard_cases.json'),
    (Join-Path $repoRoot 'memory\lemma_store.json'),
    (Join-Path $repoRoot 'memory\tactic_stats.json')
)

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

    Remove-Item -LiteralPath $target -Recurse -Force
    Write-Host "Removed: $target"
}

New-Item -ItemType Directory -Path (Join-Path $repoRoot 'checkpoints') -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $repoRoot 'logs') -Force | Out-Null

Write-Host 'Clean complete.'
