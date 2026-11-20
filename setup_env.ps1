# ============================================================
#  小模型训练环境一键安装（Windows / PowerShell 5.1 兼容）
#  目标环境: Windows 10/11 x64 + 支持 CUDA 12.8 的 NVIDIA 显卡
#
#  安装位置:  默认脚本所在目录（可用 -Root 指定）
#      python312\     独立 Python 3.12（不写注册表、不动你系统里的 3.14）
#      venv\          虚拟环境
#      cache\uv       uv 下载缓存
#      wheels\torch   下载好的 torch 安装包（可断点续传，可复用）
#  日志:      <安装目录>\setup_log.txt
#
#  用法:
#      powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
#      powershell -ExecutionPolicy Bypass -File .\setup_env.ps1 -IndexBase https://pypi.tuna.tsinghua.edu.cn/simple
#      powershell -ExecutionPolicy Bypass -File .\setup_env.ps1 -SkipTorch      # 只装基础环境
# ============================================================

param(
    [string]$Root      = '',
    [string]$IndexBase = 'https://mirrors.aliyun.com/pypi/simple',
    [string]$TorchMirror = '',        # 例如 https://mirror.sjtu.edu.cn/pytorch-wheels/cu130
    [switch]$SkipTorch,
    [switch]$ReinstallTorch
)

$ErrorActionPreference = 'Continue'
if ([string]::IsNullOrWhiteSpace($Root)) { $Root = $PSScriptRoot }
$ProgressPreference    = 'SilentlyContinue'
$OutputEncoding        = New-Object System.Text.UTF8Encoding $false
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$PyDir    = Join-Path $Root 'python312'
$VenvDir  = Join-Path $Root 'venv'
$UvCache  = Join-Path $Root 'cache\uv'
$WhlDir   = Join-Path $Root 'wheels\torch'
$LogFile  = Join-Path $Root 'setup_log.txt'
$PyExe    = Join-Path $PyDir 'python.exe'
$VenvPy   = Join-Path $VenvDir 'Scripts\python.exe'
$UvExe    = Join-Path $VenvDir 'Scripts\uv.exe'

New-Item -ItemType Directory -Force -Path $Root, $UvCache, $WhlDir | Out-Null

$script:Step = 0
function Log {
    param([string]$Msg, [string]$Level = 'INFO')
    $t = Get-Date -Format 'HH:mm:ss'
    $prefix = switch ($Level) {
        'OK'    { '[ OK ]' }
        'WARN'  { '[WARN]' }
        'ERR'   { '[FAIL]' }
        default { '[INFO]' }
    }
    $line = "$t $prefix $Msg"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line -Encoding UTF8
}
function Section {
    param([string]$Title)
    $script:Step++
    $bar = '=' * 62
    Write-Host ''
    Write-Host $bar -ForegroundColor Cyan
    $head = " 第 $($script:Step) 步 / $Title"
    Write-Host $head -ForegroundColor Cyan
    Write-Host $bar -ForegroundColor Cyan
    Add-Content -Path $LogFile -Value "`n$bar`n$head`n$bar" -Encoding UTF8
}

# ---------- 运行外部命令，输出实时落到终端 + 日志 ----------
function Invoke-Logged {
    param([string]$Exe, [string[]]$Arguments, [string]$WhatIfFailed = '命令失败')
    Add-Content -Path $LogFile -Value ">>> `"$Exe`" $($Arguments -join ' ')" -Encoding UTF8
    & $Exe @Arguments 2>&1 | ForEach-Object {
        $s = $_.ToString()
        Write-Host $s
        Add-Content -Path $LogFile -Value $s -Encoding UTF8
    }
    return $LASTEXITCODE
}

# ============================================================
Section '环境自检'
# ============================================================
$os = Get-CimInstance Win32_OperatingSystem
Log ("系统: " + $os.Caption + " / Build " + $os.BuildNumber)
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if ($isAdmin) { Log '当前是管理员权限（本脚本不需要管理员，但也能跑）' 'WARN' } else { Log '当前是普通用户权限（本脚本不需要管理员，正常）' }

try {
    $sm = Get-CimInstance Win32_ComputerSystem
    $ramGB = [math]::Round($sm.TotalPhysicalMemory / 1GB, 1)
    Log "内存: $ramGB GB"
} catch { Log '内存信息读取失败（不影响安装）' 'WARN' }

$freeD = $null
try {
    $d = Get-PSDrive -Name 'D' -ErrorAction Stop
    $freeD = [math]::Round($d.Free / 1GB, 1)
    Log "D 盘可用空间: $freeD GB"
} catch { Log '读不到 D 盘' 'WARN' }

if ($freeD -ne $null -and $freeD -lt 20) {
    Log "D 盘只剩 $freeD GB，torch 本体就要 3GB，加上缓存可能不够。建议先清理到 20GB 以上。" 'ERR'
    $go = Read-Host '仍要继续吗？输入 y 继续，其它键退出'
    if ($go -ne 'y') { exit 1 }
}
Log "Python 3.14 (系统里那个) 不能用于训练：PyTorch 官方没有 cp314 的 Windows 轮子。" 'WARN'
Log "本脚本会另装一个独立的 Python 3.12，放在 $PyDir，不会影响你现有的 Python 3.14。"

# 检查 VC++ 运行库（torch 依赖它，缺了会报 DLL 加载失败）
$vcFound = $false
foreach ($p in @("$env:SystemRoot\System32\vcruntime140.dll", "$env:SystemRoot\System32\vcruntime140_1.dll")) {
    if (Test-Path $p) { $vcFound = $true }
}
if ($vcFound) { Log '系统已有 VC++ 运行库（vcruntime140.dll）' 'OK' }
else { Log '未找到 vcruntime140.dll，torch 可能加载失败；如报 DLL 错误请装 VC++ 2015-2022 x64 运行库：winget install Microsoft.VCRedist.2015+.x64' 'WARN' }

# ============================================================
Section '安装/定位 uv（负责装 Python 与包，比 pip 快很多且支持断点续传）'
# ============================================================
function Get-Uv {
    # 先刷新本进程 PATH：刚用 pip 装的东西在旧 PATH 快照里是看不见的
    try {
        $m = [Environment]::GetEnvironmentVariable('Path', 'Machine')
        $u = [Environment]::GetEnvironmentVariable('Path', 'User')
        if ($m -or $u) { $env:PATH = ($m, $u -ne $null | Where-Object { $_ }) -join ';' }
    } catch { }

    $c = Get-Command uv -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    if (Test-Path $UvExe) { return $UvExe }

    # pip install --user 会装到 %APPDATA%\Python\<py版本>\Scripts，这个目录默认不在 PATH 里
    foreach ($p in @(
        "$env:APPDATA\Python\Python314\Scripts\uv.exe",
        "$env:APPDATA\Python\Python313\Scripts\uv.exe",
        "$env:APPDATA\Python\Python312\Scripts\uv.exe",
        "$env:APPDATA\Python\Scripts\uv.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python314\Scripts\uv.exe",
        "$env:USERPROFILE\.local\bin\uv.exe",
        "$env:LOCALAPPDATA\uv\uv.exe",
        "$env:CARGO_HOME\bin\uv.exe"
    )) {
        if ($p -and (Test-Path $p)) { return $p }
    }
    return $null
}

function Find-UvExe {
    # 兜底：在常见根目录里递归找（uv.exe 约 40MB）
    $roots = @("$env:APPDATA\Python", "$env:LOCALAPPDATA\Programs", "$env:USERPROFILE\.local", "$env:LOCALAPPDATA")
    foreach ($r in $roots) {
        if (-not (Test-Path $r)) { continue }
        $hit = Get-ChildItem -Path $r -Filter 'uv.exe' -Recurse -ErrorAction SilentlyContinue -Force | Select-Object -First 1
        if ($hit) { return $hit.FullName }
    }
    return $null
}

$uv = Get-Uv
if (-not $uv) { $uv = Find-UvExe }
if ($uv) {
    Log "已找到 uv: $uv" 'OK'
} else {
    Log '未找到 uv，开始安装（pip --user 或官方脚本，都不需要管理员）'
    # 不写死路径：先问 PATH 里的 python，再退到已知安装位置
    $py314 = $null
    $pc = Get-Command python -ErrorAction SilentlyContinue
    if ($pc) { $py314 = $pc.Source }
    if (-not $py314) {
        foreach ($p in @(
            "$env:LOCALAPPDATA\Programs\Python\Python314\python.exe",
            "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
            "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
        )) { if (Test-Path $p) { $py314 = $p; break } }
    }
    Log ("用来装 uv 的 Python: " + $(if ($py314) { $py314 } else { '没找到' }))

    $rc = 1
    if ($py314) {
        $rc = Invoke-Logged -Exe $py314 -Arguments @('-m', 'pip', 'install', '--user', 'uv', '-i', $IndexBase)
        if ($rc -ne 0) { $rc = Invoke-Logged -Exe $py314 -Arguments @('-m', 'pip', 'install', '--user', 'uv', '--index-url', 'https://pypi.org/simple') }
    }
    if ($rc -ne 0) {
        Log 'pip 安装 uv 失败，改用官方安装脚本'
        try {
            $env:UV_INSTALL_DIR = $VenvDir
            Invoke-RestMethod -Uri 'https://astral.sh/uv/install.ps1' -UseBasicParsing -TimeoutSec 60 | Invoke-Expression
            $rc = 0
        } catch {
            Log ('uv 安装脚本也失败: ' + $_.Exception.Message) 'ERR'
            $rc = 1
        }
    }
    if ($rc -ne 0) {
        Log 'uv 装不上。请手动执行（把路径换成你系统里 Python 3.14 的实际位置）:' 'ERR'
        Log '    py -3.14 -m pip install --user uv' 'ERR'
        Log '装好后重新运行本脚本即可（脚本是幂等的，重复跑没问题）' 'ERR'
        exit 1
    }
    $uv = Get-Uv
    if (-not $uv) { $uv = Find-UvExe }
    if ($uv) { Log "uv 安装完成: $uv" 'OK' } else { Log 'uv 装了但找不到，请重开一个 PowerShell 再跑本脚本' 'ERR'; exit 1 }
}

# ============================================================
Section '安装独立 Python 3.12 到 D 盘'
# ============================================================
$env:UV_PYTHON_INSTALL_DIR = $PyDir
$env:UV_CACHE_DIR          = $UvCache
$env:UV_PYTHON_DOWNLOADS   = 'automatic'
Log "UV_PYTHON_INSTALL_DIR = $PyDir"

if (Test-Path $PyExe) {
    $v = & $PyExe --version 2>&1
    Log "已存在独立 Python: $v" 'OK'
} else {
    Log '开始下载 Python 3.12（约 30-60MB，来自 Astral 官方 CDN，不走 PyPI 镜像）'
    $rc = Invoke-Logged -Exe $uv -Arguments @('python', 'install', '3.12')
    if ($rc -ne 0) { Log 'Python 3.12 下载失败，检查网络后重跑本脚本' 'ERR'; exit 1 }
    if (Test-Path $PyExe) { Log ("Python 3.12 就绪: " + (& $PyExe --version 2>&1)) 'OK' }
    else {
        # uv 的目录结构在不同版本里可能是 <PyDir>\python.exe 或 <PyDir>\cpython-3.12.x-windows-x86_64-none\python.exe
        $foundPy = $null
        if (Test-Path $PyDir) {
            $hit = Get-ChildItem -Path $PyDir -Filter 'python.exe' -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($hit) { $foundPy = $hit.FullName }
        }
        if ($foundPy) {
            Log "在 $foundPy 找到 python.exe，把它复制到 $PyExe 以统一路径" 'WARN'
            Copy-Item -Force $foundPy $PyExe
        }
        if (Test-Path $PyExe) {
            Log ("Python 3.12 就绪（已归位）: " + (& $PyExe --version 2>&1)) 'OK'
        } else {
            Log "没在 $PyExe 找到 python.exe，列出目录内容以便排查:"
            if (Test-Path $PyDir) { Get-ChildItem -Recurse -Depth 3 $PyDir | Select-Object -First 40 -ExpandProperty FullName | ForEach-Object { Log "  $_" } }
            Log '请在日志里找 python.exe 的实际位置，把该目录整体改为 python312，或把 -PyDir 参数指过去' 'ERR'
            exit 1
        }
    }
}

# ============================================================
Section '创建虚拟环境'
# ============================================================
# 关键坑（实测踩过）：uv 装出来的 Python 3.12 不带 pip，
# 所以 venv 建好后 `python -m pip` 会报 "No module named pip"。
# 而 pip 不通会让后面「装 torch → 测试 → 换版本」陷入
# 「每次白下载 1.8GB 再失败」的循环。所以这里先把 pip 弄通。
function Test-PipWorks {
    param([string]$Py)
    $null = & $Py -m pip --version 2>$null
    return ($LASTEXITCODE -eq 0)
}

if (Test-Path $VenvPy) {
    Log "虚拟环境已存在: $VenvDir" 'OK'
} else {
    # 优先用 uv venv（快），并带上 --seed 让 venv 自带 pip
    $rc = Invoke-Logged -Exe $uv -Arguments @('venv', $VenvDir, '--python', '3.12', '--seed')
    if ($rc -ne 0 -or -not (Test-Path $VenvPy)) {
        Log 'uv venv --seed 失败，改用 python -m venv --seed' 'WARN'
        $rc = Invoke-Logged -Exe $PyExe -Arguments @('-m', 'venv', '--seed', $VenvDir)
    }
    if (Test-Path $VenvPy) { Log "虚拟环境就绪: $VenvPy" 'OK' } else { Log '虚拟环境创建失败' 'ERR'; exit 1 }
}

$env:VIRTUAL_ENV = $VenvDir
$env:UV_PYTHON   = $VenvPy
Log ("venv 里的 Python: " + (& $VenvPy --version 2>&1))

# 确保 pip 可用（坏的话重建一次带 --seed 的 venv）
if (Test-PipWorks -Py $VenvPy) {
    Log 'venv 自带 pip，正常' 'OK'
} else {
    Log 'venv 里没有 pip（uv 装的 Python 常见问题），重建一次带 --seed 的环境' 'WARN'
    Remove-Item -Recurse -Force $VenvDir -ErrorAction SilentlyContinue
    $rc = Invoke-Logged -Exe $uv -Arguments @('venv', $VenvDir, '--python', '3.12', '--seed')
    if ($rc -ne 0 -or -not (Test-Path $VenvPy)) {
        $rc = Invoke-Logged -Exe $PyExe -Arguments @('-m', 'venv', '--seed', $VenvDir)
    }
    if (Test-PipWorks -Py $VenvPy) {
        Log 'pip 修好了' 'OK'
    } else {
        Log 'pip 仍然不行。不要紧 —— 后续全部改用 uv pip（uv 自带安装能力，不需要 venv 里有 pip）' 'WARN'
    }
}

# ============================================================
Section '安装基础依赖'
# ============================================================
# 统一走 uv pip：既快，又不依赖 venv 里的 pip。
# （注意：不要用 $args 当变量名，它是 PowerShell 的自动变量，会出怪问题。）
function Install-Pkgs {
    param([string[]]$Packages, [string]$FromIndex = '')
    if ($uv) {
        $uvArgs = @('pip', 'install', '--python', $VenvPy) + $Packages
        if ($FromIndex -ne '') {
            # uv 的源参数：新版叫 --default-index，老版叫 --index-url。先试新版。
            $withIdx = $uvArgs + @('--default-index', $FromIndex)
            $rc = Invoke-Logged -Exe $uv -Arguments $withIdx
            if ($rc -eq 0) { return 0 }
            Log '  --default-index 不被这个 uv 版本支持，改用 --index-url' 'WARN'
            $withIdx = $uvArgs + @('--index-url', $FromIndex)
            $rc = Invoke-Logged -Exe $uv -Arguments $withIdx
            if ($rc -eq 0) { return 0 }
            # 镜像不好使就退回官方源
            return (Invoke-Logged -Exe $uv -Arguments $uvArgs)
        }
        return (Invoke-Logged -Exe $uv -Arguments $uvArgs)
    }
    if (-not (Test-PipWorks -Py $VenvPy)) {
        Log '既没有 uv，venv 里也没有 pip —— 没法装任何东西。请先装 uv：py -3.14 -m pip install --user uv' 'ERR'
        return 1
    }
    $pipArgs = @('-m', 'pip', 'install') + $Packages
    if ($FromIndex -ne '') { $pipArgs += @('-i', $FromIndex, '--timeout', '120') }
    return (Invoke-Logged -Exe $VenvPy -Arguments $pipArgs)
}

Log "pip 源: $IndexBase"
# 前置检查：确认至少有一样能装包的工具，避免后面白忙一场
$uvOk = $false
if ($uv) {
    $null = & $uv --version 2>$null
    $uvOk = ($LASTEXITCODE -eq 0)
}
Log ("可用安装器: " + $(if ($uvOk) { "uv ($uv)" } elseif (Test-PipWorks -Py $VenvPy) { "venv 自带 pip" } else { "无（会失败）" }))
if (-not $uvOk -and -not (Test-PipWorks -Py $VenvPy)) {
    Log 'uv 和 pip 都不可用，无法继续。请先执行: py -3.14 -m pip install --user uv' 'ERR'
    exit 1
}

$rc = Install-Pkgs -Packages @('numpy', 'matplotlib', 'tqdm') -FromIndex $IndexBase
if ($rc -ne 0) {
    Log '换回官方 PyPI 再试一次'
    $rc = Install-Pkgs -Packages @('numpy', 'matplotlib', 'tqdm')
}
if ($rc -eq 0) { Log '基础依赖安装完成 (numpy / matplotlib / tqdm)' 'OK' } else { Log '基础依赖安装失败（不影响训练核心，可继续）' 'WARN' }

# ============================================================
Section '安装 PyTorch（关键一步：自动挑能真正跑起来的版本）'
# ============================================================
$SmokeFile = Join-Path $PSScriptRoot 'gpu_smoke_test.py'
$SmokeScript = Join-Path $Root 'gpu_smoke.py'

# 内置一份烟测脚本作为兜底（万一工作区里的 gpu_smoke_test.py 被挪走）
$SmokeFallback = @'
import sys, os, io, traceback
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ok = True
def line(s): print(s, flush=True)
try:
    import torch
except Exception as e:
    print("FAIL_IMPORT", e); sys.exit(3)
line("torch=%s cuda_build=%s" % (torch.__version__, torch.version.cuda))
if not torch.cuda.is_available():
    print("FAIL_NO_CUDA"); sys.exit(4)
try:
    p = torch.cuda.get_device_properties(0)
    line("gpu=%s cc=%d.%d vram=%.2fGB" % (p.name, p.major, p.minor, p.total_memory/1024**3))
except Exception as e:
    line("props warn: %s" % e)
try:
    x = torch.randn(256, 256, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    line("KERNEL_OK sum=%.4f" % float(y.sum().item()))
except Exception as e:
    print("FAIL_KERNEL", repr(e)); sys.exit(5)
for dt in (torch.bfloat16, torch.float16):
    try:
        a = torch.randn(256, 256, device="cuda", dtype=dt)
        b = (a @ a).float().sum().item()
        torch.cuda.synchronize()
        line("DTYPE_OK %s sum=%.4f" % (dt, b))
    except Exception as e:
        line("DTYPE_FAIL %s : %r" % (dt, e)); ok = False
try:
    from torch.nn import functional as F
    x = torch.randn(2, 8, 64, 8, 64, device="cuda", dtype=torch.bfloat16)
    o = F.scaled_dot_product_attention(x, x, x, is_causal=True)
    torch.cuda.synchronize()
    line("SDPA_OK shape=%s" % (tuple(o.shape),))
except Exception as e:
    line("SDPA_FAIL %r" % (e,)); ok = False
print("SMOKE_RESULT=%s" % ("PASS" if ok else "PARTIAL"))
sys.exit(0 if ok else 6)
'@

if (Test-Path $SmokeFile) {
    Copy-Item -Force $SmokeFile $SmokeScript
    Log "使用工作区里的 gpu_smoke_test.py"
} else {
    Set-Content -Path $SmokeScript -Value $SmokeFallback -Encoding UTF8
    Log "工作区里没找到 gpu_smoke_test.py，已写入内置简版烟测脚本"
}

function Test-TorchWorks {
    # 返回 $true / $false。顺带把子进程退出码放进 $script:LastSmokeExit
    param([string]$Py, [string]$Script)
    $out = & $Py $Script 2>&1
    $script:LastSmokeExit = $LASTEXITCODE
    $text = ($out | Out-String)
    $text -split "`r?`n" | Where-Object { $_ -ne '' } | ForEach-Object {
        Write-Host "    $_"
        Add-Content -Path $LogFile -Value "    $_" -Encoding UTF8
    }
    if ($text -match 'SMOKE_RESULT=PASS') { return $true }
    return $false
}

if ($SkipTorch) {
    Log '按参数要求跳过 PyTorch 安装' 'WARN'
} else {
    # 候选顺序：越靠前越新。cu130 对应 CUDA 13.x 驱动与较新的 GPU 架构。
    $Candidates = @(
        @{ v = '2.14.0'; cu = 'cu130' },
        @{ v = '2.13.0'; cu = 'cu130' },
        @{ v = '2.12.1'; cu = 'cu130' },
        @{ v = '2.11.0'; cu = 'cu130' },
        @{ v = '2.11.0'; cu = 'cu128' },
        @{ v = '2.10.0'; cu = 'cu130' },
        @{ v = '2.10.0'; cu = 'cu128' },
        @{ v = '2.9.1';  cu = 'cu130' },
        @{ v = '2.9.1';  cu = 'cu128' }
    )

    $script:LastSmokeExit = -999
    $existing = ''
    $probe = & $VenvPy -c "import torch,sys; sys.stdout.write(torch.__version__)" 2>$null
    $probeExit = $LASTEXITCODE
    if ($probe) { $existing = ($probe | Out-String).Trim() }
    if ($probeExit -ne 0) {
        Log 'venv 里还没有 torch，直接从最新候选开始装'
    } elseif ($existing -and -not $ReinstallTorch) {
        Log "venv 里已有 torch $existing，先验证它能不能真算"
        if (Test-TorchWorks -Py $VenvPy -Script $SmokeScript) {
            Log "现有 torch $existing 工作正常，跳过安装" 'OK'
            $SkipTorch = $true
        } else {
            Log "现有 torch $existing 算不起来（很可能就是 较新的 GPU 架构 内核缺失），开始换版本" 'WARN'
        }
    }

    $found = $false
    foreach ($c in $Candidates) {
        if ($SkipTorch) { $found = $true; break }
        $ver = $c.v; $cu = $c.cu
        $whlName = "torch-$ver+$cu-cp312-cp312-win_amd64.whl"
        $whlPath = Join-Path $WhlDir $whlName
        # 注意：wheel 文件名里的 + 号在 URL 里必须写成 %2B，别用 EscapeDataString（会把 % 再编码一层）
        $url = "https://download.pytorch.org/whl/$cu/torch-$ver%2B$cu-cp312-cp312-win_amd64.whl"
        if ($TorchMirror -ne '') { $url = "$TorchMirror/$whlName" }

        Log "---- 尝试 torch $ver + $cu ----"
        if ($ReinstallTorch) { & $VenvPy -m pip uninstall -y torch 2>&1 | Out-Null }

        if (Test-Path $whlPath) {
            Log "本地已有安装包，跳过下载: $whlPath"
        } else {
            Log "下载: $url"
            $okDl = $false
            for ($attempt = 1; $attempt -le 3; $attempt++) {
                try {
                    # curl.exe 支持断点续传，比 Invoke-WebRequest 稳
                    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
                    if ($curl) {
                        & curl.exe -L --fail --retry 3 --retry-delay 3 -C - --connect-timeout 20 -o $whlPath $url
                        if ($LASTEXITCODE -eq 0 -and (Test-Path $whlPath)) { $okDl = $true; break }
                    } else {
                        Invoke-WebRequest -Uri $url -OutFile $whlPath -UseBasicParsing -TimeoutSec 1800
                        $okDl = $true; break
                    }
                } catch {
                    Log ("下载第 $attempt 次失败: " + $_.Exception.Message) 'WARN'
                    Start-Sleep -Seconds 3
                }
            }
            if (-not $okDl) {
                if (Test-Path $whlPath) { Remove-Item -Force $whlPath -ErrorAction SilentlyContinue }
                Log "这个版本下载不下来，试下一个候选" 'WARN'
                continue
            }
        }

        $sizeGB = [math]::Round((Get-Item $whlPath).Length / 1GB, 2)
        Log "安装包大小 $sizeGB GB，开始安装（这一步最慢，请耐心）"
        # 记录安装前的 torch 版本，用来判断「到底是装上了还是没装上」。
        # 之前用 `python -m pip` 时 pip 坏了，每次都是下载 1.8GB 后装失败，
        # 白白浪费十几个 GB 流量 —— 现在改成装完必须能 import，否则立刻换下一个。
        $before = ''
        try { $before = ((& $VenvPy -c "import torch,sys;sys.stdout.write(torch.__version__)" 2>$null) | Out-String).Trim() } catch { $before = '' }

        $rc = Install-Pkgs -Packages @('--no-deps', '--force-reinstall', $whlPath)
        if ($rc -ne 0) { Log "安装失败，试下一个候选" 'WARN'; continue }

        # torch 的运行依赖必须单独装（上面用了 --no-deps）
        Log "补装 torch 的运行依赖"
        $rc = Install-Pkgs -Packages @('filelock', 'typing-extensions', 'sympy', 'networkx', 'jinja2', 'fsspec') -FromIndex $IndexBase
        if ($rc -ne 0) { Log '依赖装失败，换官方 PyPI 重试' 'WARN'; $rc = Install-Pkgs -Packages @('filelock', 'typing-extensions', 'sympy', 'networkx', 'jinja2', 'fsspec') }

        # 前置确认：装完能不能 import torch？不能就别浪费时间跑冒烟测试
        $after = ''
        try { $after = ((& $VenvPy -c "import torch,sys;sys.stdout.write(torch.__version__)" 2>$null) | Out-String).Trim() } catch { $after = '' }
        if (-not $after) {
            Log "torch $ver + $cu 装完了但 import 不进来（依赖没补齐），试下一个候选" 'WARN'
            continue
        }
        Log "已成功安装 torch $after，开始冒烟测试" 'OK'

        Log "开始 GPU 冒烟测试：真做一次矩阵乘法 + bf16 + 用 6 层/384 维 真训 20 步"
        if (Test-TorchWorks -Py $VenvPy -Script $SmokeScript) {
            Log "★ 成功！torch $ver + $cu 在这台机器上可以真正跑 GPU 计算" 'OK'
            Set-Content -Path (Join-Path $Root 'WORKING_TORCH.txt') -Value "torch==$ver+$cu`nwhl=$whlPath`nverified=$(Get-Date -Format s)" -Encoding UTF8
            $found = $true
            break
        } else {
            switch ($script:LastSmokeExit) {
                3  { Log "torch $ver + $cu 装上了但 import 失败（多半缺 VC++ 运行库或依赖没补齐）" 'WARN' }
                4  { Log "torch $ver + $cu 看不到 CUDA 设备（可能装成了 CPU 版）" 'WARN' }
                5  { Log "torch $ver + $cu 能装上但计算核跑不起来 —— 正是 issue 174731 那个症状（较新的 GPU 架构 内核缺失）" 'WARN' }
                6  { Log "torch $ver + $cu 基本计算能用，但有子项失败（多半是 bf16），试下一个候选" 'WARN' }
                default { Log "torch $ver + $cu 冒烟测试未通过（退出码 $script:LastSmokeExit）" 'WARN' }
            }
        }
    }

    if (-not $found) {
        Log '所有稳定版候选都没通过。下一步建议试 nightly 构建（新架构/新驱动通常要 nightly 构建）:' 'ERR'
        Log "  & `"$VenvPy`" -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130" 'ERR'
        Log "  装完再跑:  & `"$VenvPy`" `"$SmokeScript`"" 'ERR'
        Log '另外可尝试 --torch-mirror 指向镜像（例如 https://mirror.sjtu.edu.cn/pytorch-wheels/cu130）加速下载' 'ERR'
    }
}

# ============================================================
Section '完成 / 后续命令'
# ============================================================
Log "安装根目录:   $Root"
Log "Python:       $PyExe"
Log "虚拟环境:     $VenvDir"
Log "torch 安装包: $WhlDir"
Log "日志:         $LogFile"
Write-Host ''
Write-Host '以后每次要用这个环境，先激活：' -ForegroundColor Green
Write-Host "    & `"$VenvDir\Scripts\Activate.ps1`"" -ForegroundColor Yellow
Write-Host '或者不激活，直接用解释器全路径：' -ForegroundColor Green
Write-Host "    & `"$VenvPy`" 你的脚本.py" -ForegroundColor Yellow
Write-Host ''
Log '把本脚本最后的输出（尤其 SMOKE_RESULT 那一行）贴回给 AI，即可锁定版本开始搭引擎。'
