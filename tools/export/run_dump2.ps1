param([string]$Cams, [string]$OutDir, [string]$Size = "2560,1440", [int]$Warmup = 30, [string]$Dir = "nsr", [string]$Prefix = "a", [int]$Defer = 1, [int]$View = 0, [int]$Ssao = 1, [int]$Blur = 2,
      [int]$Many = 0, [int]$Volume = 1, [int]$Bug = 0, [string]$Cam = "", [int]$Frame = 60, [int]$Gbv = 0, [int]$Resize = 0,
      [int]$Spheres = 1, [double]$SunEl = -1, [int]$Bloom = 1, [int]$Timeout = 900, [string]$WinPos = "2560,0")
# SSAO 沙盒的一次截图：第 $Frame 帧读回后台缓冲 → PNG；收集调试层输出、[DEFER] / [T] / [GBV] 日志
#   HLSL 的错法（1～9）临时改 Deferred.hlsl 文本（运行时编译），跑完还原；C++ 的错法（10～14）走环境变量 SSAO_BUG
#   1 不翻到法线一侧 | 2 不用随机向量 | 3 投影后 y 没翻 | 4 没除以 w | 5 半分辨率没乘 2 | 6 模糊不看边
#   7 遮蔽乘到了直接光上 | 8 忘了 1 减 | 9 像素中心没加 0.5
#   10 视口没换成半分辨率 | 11 画完视口没换回来 | 12 遮蔽图没转成渲染目标 | 13 PSO 输出格式没改 | 14 竖的那趟读了正在写的那张
$ErrorActionPreference = 'Stop'
$S = Split-Path -Parent $MyInvocation.MyCommand.Path
$inv = [Globalization.CultureInfo]::InvariantCulture
$enc = [Text.UTF8Encoding]::new($false)
Add-Type -AssemblyName System.Drawing
if (-not ("DbgCapD" -as [type])) {
Add-Type -TypeDefinition @'
using System; using System.Collections.Generic; using System.IO.MemoryMappedFiles; using System.Runtime.InteropServices; using System.Text; using System.Threading;
public static class DbgCapD {
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
$hlsl = "$S\$Dir\HLSL\Deferred.hlsl"
$orig = [IO.File]::ReadAllText($hlsl, $enc)
$hlslBugs = @{
    1 = @("        float flip = sign(dot(offset, n));", "        float flip = 1.0f;")
    2 = @("        float3 offset = reflect(kAoOffsets[i], randVec);", "        float3 offset = kAoOffsets[i];")
    3 = @("qH.xy / qH.w * float2(0.5f, -0.5f) + 0.5f;", "qH.xy / qH.w * float2(0.5f, 0.5f) + 0.5f;")
    4 = @("qH.xy / qH.w * float2(0.5f, -0.5f) + 0.5f;", "qH.xy * float2(0.5f, -0.5f) + 0.5f;")
    5 = @("    int2 pixel = int2(pin.PosH.xy) * 2;", "    int2 pixel = int2(pin.PosH.xy);")
    6 = @("        if (i == 0 || (dot(neighborN, centerN) >= 0.8f && abs(neighborZ - centerZ) <= 0.2f))", "        if (true)")
    7 = @("    float3 color = ambient + direct;", "    float3 color = ambient + direct * gAoMap.SampleLevel(gsamLinearClamp, pin.UV, 0);")
    8 = @("    float access = 1.0f - occlusionSum / kAoSamples;", "    float access = occlusionSum / kAoSamples;")
    9 = @("    float2 uv = (pixel + 0.5f) / float2(width, height);", "    float2 uv = pixel / float2(width, height);")
}
$bmp = "$S\${Prefix}.bmp"
Remove-Item $bmp, "$S\${Prefix}.png" -ErrorAction SilentlyContinue
try {
    if ($hlslBugs.ContainsKey($Bug)) {
        $old = $hlslBugs[$Bug][0]; $new = $hlslBugs[$Bug][1]
        if (-not $orig.Contains($old)) { throw "bug $Bug pattern not found" }
        [IO.File]::WriteAllText($hlsl, $orig.Replace($old, $new), $enc)
    }
    New-Item -ItemType Directory -Force $OutDir | Out-Null
    $env:DUMP_CAMS = (Resolve-Path $Cams).Path; $env:DUMP_DIR = (Resolve-Path $OutDir).Path; $env:DUMP_WARMUP = "$Warmup"; $env:RENDER_SIZE = $Size
    $env:SHADOW_CAM = $(if ($Cam -ne "") { $Cam } else { $null })
    $env:DEFER = "$Defer"; $env:GBVIEW = "$View"; $env:MANY = "$Many"; $env:VOLUME = "$Volume"
    $env:SSAO = "$Ssao"; $env:AOBLUR = "$Blur"; $env:SSAO_BUG = "$Bug"
    $env:RESIZE_AT = $(if ($Resize -gt 0) { "$Resize" } else { $null })
    $env:HDR_GBV = $(if ($Gbv) { "1" } else { $null })
    $env:SPHERES = "$Spheres"; $env:BLOOM = "$Bloom"
    $env:WIN_POS = $(if ($WinPos -ne "") { $WinPos } else { $null })
    $env:SUN_EL = $(if ($SunEl -ge 0) { $SunEl.ToString($inv) } else { $null })
    [DbgCapD]::Start()
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $p = Start-Process "$S\$Dir\x64\Debug\LearnDX12.exe" -WorkingDirectory "$S\$Dir" -PassThru
    $exited = $p.WaitForExit($Timeout * 1000)
    if (-not $exited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Milliseconds 300
    [DbgCapD]::Stop()
}
finally {
    [IO.File]::WriteAllText($hlsl, $orig, $enc)
    foreach ($n in "DUMP_CAMS", "DUMP_DIR", "DUMP_WARMUP", "RENDER_SIZE", "SNAP_FRAME", "SHADOW_CAM", "DEFER", "GBVIEW", "MANY", "VOLUME", "SSAO", "AOBLUR", "SSAO_BUG", "RESIZE_AT", "HDR_GBV", "SPHERES", "BLOOM", "SUN_EL", "WIN_POS") {
        Remove-Item "env:$n" -ErrorAction SilentlyContinue
    }
}
"exited=$exited exit=$(if ($exited) { '0x{0:X}' -f $p.ExitCode } else { 'killed' })  $([int]$sw.Elapsed.TotalSeconds) s  files: $((Get-ChildItem $OutDir -Filter *.npy).Count)"
if (Test-Path $bmp) {
    $img = [System.Drawing.Bitmap]::FromFile($bmp); $img.Save("$S\${Prefix}.png", [System.Drawing.Imaging.ImageFormat]::Png); $img.Dispose()
    Remove-Item $bmp
    "snap ok -> ${Prefix}.png"
} else { "NO snap" }
$lines = [DbgCapD]::Lines
$lines | Where-Object { $_ -match '^\[(DEFER|GBV|DATA)\]' } | ForEach-Object { "   " + $_ }
$t = $lines | Where-Object { $_ -match '^\[T\]' } | Select-Object -Last 1
if ($t) { "   " + $t }
$d3d = $lines | Where-Object { $_ -match 'D3D12 (ERROR|WARNING|CORRUPTION|MESSAGE)|DXGI (ERROR|WARNING)' }
"d3d errors/warnings: $($d3d.Count)"
$d3d | ForEach-Object { ($_ -replace '0x[0-9A-Fa-f]{8,}', 'PTR') } | Select-Object -Unique | Select-Object -First 4 | ForEach-Object { "   " + $_.Substring(0, [Math]::Min(460, $_.Length)) }
"captured lines: $($lines.Count)"
