param([string]$Exe, [int[]]$Seqs = (0..15), [string]$Raw = "", [int]$Sub = 64)
# 多帧数据集：逐段导出（720p 一次、2K 一次）→ convert_seq.py 转格式 → 删原始文件（每段原始文件约 2.8 GB，转完就删）
#   powershell -File tools\export\export_seqs.ps1 -Exe <打了 patch_seq.py 的副本>\x64\Debug\LearnDX12.exe [-Seqs 12,13]
$ErrorActionPreference = 'Stop'
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent (Split-Path -Parent $Here)
$py = "$Root\.venv\Scripts\python.exe"
if ($Raw -eq "") { $Raw = "$Root\data\seq\raw" } # 和数据集同一个盘：转换时直接移动文件
$splits = @{}
Import-Csv "$Root\data\seq\seqs.csv" | ForEach-Object { $splits[[int]$_.seq] = $_.split }
foreach ($k in $Seqs) {
    $cams = "$Root\data\seq\cams_{0:D2}.txt" -f $k
    $lr = "$Raw\lr_$k"; $gt = "$Raw\gt_$k"
    Remove-Item -Recurse -Force $lr, $gt -ErrorAction SilentlyContinue
    "== seq $k ($($splits[$k]))"
    & "$Here\run_seq.ps1" -Exe $Exe -Cams $cams -OutDir $lr -Mode lr -Split $splits[$k] | ForEach-Object { "  $_" }
    & "$Here\run_seq.ps1" -Exe $Exe -Cams $cams -OutDir $gt -Mode gt -Split $splits[$k] -Sub $Sub | ForEach-Object { "  $_" }
    & $py "$Root\tools\convert_seq.py" $k $lr $gt
    if ($LASTEXITCODE -ne 0) { throw "convert_seq.py failed on seq $k" }
    Remove-Item -Recurse -Force $lr, $gt
}
