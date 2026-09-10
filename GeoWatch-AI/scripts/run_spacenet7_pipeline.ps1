[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DatasetRoot,
    [string]$Python = "python",
    [int]$Epochs = 20,
    [int]$Batch = 4,
    [int]$PatchSize = 256
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$datasetPath = (Resolve-Path -LiteralPath $DatasetRoot).Path
$manifestPath = Join-Path $projectRoot "data\spacenet7\pairs.json"
$modelDir = Join-Path $projectRoot "models\spacenet7_change_baseline"
$metricsPath = Join-Path $projectRoot "data\runs\spacenet7_metrics.json"
$demoCasesPath = Join-Path $projectRoot "data\demo_cases"

function Invoke-PythonStep {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python step failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

Push-Location $projectRoot
try {
    Invoke-PythonStep -Arguments @("-c", "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print(torch.cuda.get_device_name(0))")
    Invoke-PythonStep -Arguments @("scripts\prepare_spacenet7_pairs.py", "--source", $datasetPath, "--output", $manifestPath, "--hash-files")
    Invoke-PythonStep -Arguments @("scripts\train_spacenet7_change.py", "--manifest", $manifestPath, "--output", $modelDir, "--epochs", "$Epochs", "--patch-size", "$PatchSize", "--samples-per-pair", "8", "--batch", "$Batch", "--device", "cuda")
    Invoke-PythonStep -Arguments @("scripts\evaluate_spacenet7_change.py", "--manifest", $manifestPath, "--checkpoint", (Join-Path $modelDir "best.pt"), "--output", $metricsPath, "--split", "test", "--batch", "$Batch", "--device", "cuda")
    Invoke-PythonStep -Arguments @("scripts\build_real_demo_cases.py", "--pairs", $manifestPath, "--output", $demoCasesPath)
    Write-Host "SpaceNet 7 pipeline completed."
    Write-Host "Metrics: $metricsPath"
    Write-Host "Demo cases: $demoCasesPath"
}
finally {
    Pop-Location
}
