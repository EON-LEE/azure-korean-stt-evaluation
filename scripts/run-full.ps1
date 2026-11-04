<#
.SYNOPSIS
    Runs `kstt supervise` for a long evaluation, restarts it after a crash and backs up the run.

.DESCRIPTION
    kstt exit codes 0 (done), 2 (configuration), 3 (incomplete), 4 (STOP file), 5 (run locked) and
    130 (Ctrl+C) end this script with the same code. Any other exit is treated as a crash: the script
    waits -RestartDelaySeconds and starts the same command again, which resumes the run. After
    -MaxCrashes crashes in a row without new final items it gives up with exit code 1.

    With -BackupDir, runs\<RunId> is copied every -BackupEveryMinutes minutes and at the end. The
    shared result cache (runs\cache) is copied every -CacheBackupEveryHours hours and at the end.
    The lock and STOP files are not copied.

    To stop: create runs\<RunId>\STOP. The current requests finish, then kstt exits with code 4.

.EXAMPLE
    .\scripts\run-full.ps1 -RunId full-1 -Routes "fast-azure,sdk-azure" -MaxAudioHours 55 -BackupDir D:\kstt-backup
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string] $RunId,
    [Parameter(Mandatory = $true)][double] $MaxAudioHours,
    [string] $Routes = "all",
    [ValidateSet("smoke", "partial", "full")][string] $Subset = "full",
    [string] $RunsDir = "runs",
    [string] $BackupDir,
    [int] $BackupEveryMinutes = 60,
    [int] $CacheBackupEveryHours = 24,
    [int] $MaxCrashes = 5,
    [int] $RestartDelaySeconds = 60,
    # Extra `kstt supervise` options as one string, e.g. -ExtraArgs "--resource-limit speech-kc=8"
    [string] $ExtraArgs = "",
    [string] $UvPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
if (-not $UvPath) { $UvPath = (Get-Command uv -CommandType Application | Select-Object -First 1).Source }
$runDir = Join-Path $RunsDir $RunId
$finalCodes = @(0, 2, 3, 4, 5, 130)

function Write-Log([string] $Message) {
    Write-Host ("[{0}] run-full: {1}" -f (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"), $Message)
}

function Format-Arg([string] $Value) {
    if ($Value -match '[\s"]') { return '"' + ($Value -replace '"', '\"') + '"' }
    return $Value
}

function Get-FinalCount {
    # Native stderr must not become a terminating error in Windows PowerShell 5.1.
    $ErrorActionPreference = "Continue"
    $out = & $UvPath run --no-sync kstt --runs-dir $RunsDir status --run $RunId --json 2>$null
    if ($LASTEXITCODE -ne 0) { return -1 }
    try { $st = ($out | Out-String) | ConvertFrom-Json } catch { return -1 }
    $sum = 0
    foreach ($p in $st.routes.PSObject.Properties) { $sum += [int]$p.Value.final }
    return $sum
}

function Backup-Run([bool] $WithCache) {
    if (-not $BackupDir) { return }
    $ErrorActionPreference = "Continue"
    $targets = @(@{ Src = $runDir; Dst = (Join-Path $BackupDir $RunId) })
    if ($WithCache) { $targets += @{ Src = (Join-Path $RunsDir "cache"); Dst = (Join-Path $BackupDir "cache") } }
    foreach ($t in $targets) {
        if (-not (Test-Path -LiteralPath $t.Src)) { continue }
        & robocopy $t.Src $t.Dst /E /R:1 /W:1 /XF lock STOP /NFL /NDL /NJH /NJS /NP | Out-Null
        if ($LASTEXITCODE -ge 8) { Write-Log "backup of $($t.Src) failed (robocopy exit $LASTEXITCODE)" }
        else { Write-Log "backed up $($t.Src) to $($t.Dst)" }
    }
    $global:LASTEXITCODE = 0
}

$kstt = @("run", "--no-sync", "kstt", "--runs-dir", $RunsDir, "supervise", "--run-id", $RunId,
          "--routes", $Routes, "--subset", $Subset,
          "--max-audio-hours", $MaxAudioHours.ToString([System.Globalization.CultureInfo]::InvariantCulture)) +
        @($ExtraArgs -split '\s+' | Where-Object { $_ })
$crashes = 0
$bestFinal = Get-FinalCount
$lastCacheBackup = Get-Date

while ($true) {
    if (Test-Path -LiteralPath (Join-Path $runDir "STOP")) {
        Write-Log "STOP file present in $runDir; not starting. Delete it to continue."
        exit 4
    }
    Write-Log ("start: uv " + ($kstt -join " "))
    $p = Start-Process -FilePath $UvPath -ArgumentList ($kstt | ForEach-Object { Format-Arg $_ }) -NoNewWindow -PassThru
    $null = $p.Handle  # keeps ExitCode readable after the process ends
    $nextBackup = (Get-Date).AddMinutes($BackupEveryMinutes)
    while (-not $p.WaitForExit(60000)) {
        if ((Get-Date) -ge $nextBackup) {
            $withCache = ((Get-Date) - $lastCacheBackup).TotalHours -ge $CacheBackupEveryHours
            Backup-Run $withCache
            if ($withCache) { $lastCacheBackup = Get-Date }
            $nextBackup = (Get-Date).AddMinutes($BackupEveryMinutes)
        }
    }
    $p.WaitForExit()
    $code = $p.ExitCode
    Write-Log "kstt supervise exited with code $code"
    if ($finalCodes -contains $code) {
        Backup-Run $true
        exit $code
    }

    $final = Get-FinalCount
    if ($final -gt $bestFinal) {
        $crashes = 0
        $bestFinal = $final
    }
    $crashes++
    if ($crashes -ge $MaxCrashes) {
        Write-Log "$crashes crash(es) in a row without new final items; giving up. Check $runDir\logs."
        Backup-Run $true
        exit 1
    }
    Write-Log "crash $crashes/$MaxCrashes (final items: $final); restarting in $RestartDelaySeconds s"
    Start-Sleep -Seconds $RestartDelaySeconds
}
