param([string]$Exe, [string]$Cams, [string]$OutDir, [string]$Mode = "lr", [string]$Split = "train", [int]$Sub = 64,
      [int]$Warmup = 30, [string]$WinPos = "2560,0", [int]$Timeout = 900, [int]$Gbv = 0)
# 多帧数据集导出的一次运行（s193）：打了 patch_seq.py 的 LearnDX12 副本，按 $Cams 画一段序列，读回写成 .npy
#   -Exe   副本里的 x64\Debug\LearnDX12.exe（工作目录 = 副本根目录，运行时编译 HLSL）
#   -Mode  lr = 1280×720 带抖动 | gt = 2560×1440，每帧冻住时间画 1 + $Sub 次
# 收集调试层输出（DBWIN），最后打印 [SEQ] 日志和 D3D12 错误 / 警告的条数
$ErrorActionPreference = 'Stop'
$enc = [Text.UTF8Encoding]::new($false)
if (-not ("DbgCapS" -as [type])) {
Add-Type -TypeDefinition @'
using System; using System.Collections.Generic; using System.IO.MemoryMappedFiles; using System.Runtime.InteropServices; using System.Text; using System.Threading;
public static class DbgCapS {
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode)] static extern IntPtr CreateEvent(IntPtr a, bool manual, bool init, string name);
    [DllImport("kernel32.dll")] static extern bool SetEvent(IntPtr h);
    [DllImport("kernel32.dll")] static extern uint WaitForSingleObject(IntPtr h, uint ms);
    public static List<string> Lines = new List<string>(); static volatile bool stop; static Thread t;
    public static void Start() {
        stop = false; lock (Lines) Lines.Clear();
        IntPtr ready = CreateEvent(IntPtr.Zero, false, false, "DBWIN_BUFFER_READY");
        IntPtr data = CreateEvent(IntPtr.Zero, false, false, "DBWIN_DATA_READY");
        var view = MemoryMappedFile.CreateOrOpen("DBWIN_BUFFER", 4096).CreateViewAccessor();
        t = new Thread(() => { SetEvent(ready); while (!stop) { if (WaitForSingleObject(data, 200) == 0) {
            byte[] buf = new byte[4092]; view.ReadArray(4, buf, 0, buf.Length); int n = Array.IndexOf(buf, (byte)0); if (n < 0) n = buf.Length;
            lock (Lines) Lines.Add(Encoding.UTF8.GetString(buf, 0, n).TrimEnd()); SetEvent(ready); } } });
        t.IsBackground = true; t.Start();
    }
    public static void Stop() { stop = true; t.Join(); }
}
'@
}
$names = "SEQ_CAMS", "SEQ_DIR", "SEQ_MODE", "SEQ_SPLIT", "SEQ_SUB", "SEQ_WARMUP", "RENDER_SIZE", "DEFER", "IBL", "SPHERES", "TAA_DT60", "NO_UI", "WIN_POS", "BLOOM", "HDR_GBV"
try {
    New-Item -ItemType Directory -Force $OutDir | Out-Null
    $env:SEQ_CAMS = (Resolve-Path $Cams).Path; $env:SEQ_DIR = (Resolve-Path $OutDir).Path; $env:SEQ_MODE = $Mode; $env:SEQ_SPLIT = $Split
    $env:SEQ_SUB = "$Sub"; $env:SEQ_WARMUP = "$Warmup"
    $env:RENDER_SIZE = $(if ($Mode -eq "gt") { "2560,1440" } else { "1280,720" })
    $env:DEFER = "1"; $env:IBL = "1"; $env:SPHERES = "1"; $env:TAA_DT60 = "1"; $env:NO_UI = "1"; $env:BLOOM = "1"
    $env:WIN_POS = $(if ($WinPos -ne "") { $WinPos } else { $null })
    $env:HDR_GBV = $(if ($Gbv) { "1" } else { $null })
    [DbgCapS]::Start()
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $Exe))
    $p = Start-Process $Exe -WorkingDirectory $root -PassThru
    $exited = $p.WaitForExit($Timeout * 1000)
    if (-not $exited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Milliseconds 300
    [DbgCapS]::Stop()
}
finally {
    foreach ($n in $names) { Remove-Item "env:$n" -ErrorAction SilentlyContinue }
}
"$Mode exited=$exited exit=$(if ($exited) { '0x{0:X}' -f $p.ExitCode } else { 'killed' })  $([int]$sw.Elapsed.TotalSeconds) s  files: $((Get-ChildItem $OutDir -Filter *.npy).Count)"
$lines = [DbgCapS]::Lines
$lines | Where-Object { $_ -match '^\[(SEQ|GBV)\]' } | Select-Object -First 3 | ForEach-Object { "   " + $_ }
$lines | Where-Object { $_ -match '^\[SEQ\]' } | Select-Object -Last 1 | ForEach-Object { "   " + $_ }
$d3d = $lines | Where-Object { $_ -match 'D3D12 (ERROR|WARNING|CORRUPTION)|DXGI (ERROR|WARNING)' }
"d3d errors/warnings: $($d3d.Count)   captured lines: $($lines.Count)"
$d3d | ForEach-Object { ($_ -replace '0x[0-9A-Fa-f]{8,}', 'PTR') } | Select-Object -Unique | Select-Object -First 4 | ForEach-Object { "   " + $_.Substring(0, [Math]::Min(400, $_.Length)) }
