# ============================================================
#  灵语引擎启动器（仓库自带，跨机器可用）
#
#  用法（在任何目录都能跑；脚本按自身位置定位仓库，不写死盘符）：
#      & "<本仓库>\lingyu.ps1" doctor
#      & "<本仓库>\lingyu.ps1" train --config configs/small.ini
#
#  嫌路径长，可在自己的 PowerShell 里加别名（只对当前窗口有效）：
#      function ly { & "<本仓库>\lingyu.ps1" @args }
#      ly doctor
#
#  想永久生效，把上面那行 function 加到 $PROFILE 里。
# ============================================================

param(
    # 让所有参数原样传给 lingyu，不再被 PowerShell 的参数绑定拦住
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$LingYuArgs
)

$ErrorActionPreference = 'Stop'

# 仓库根 = 本脚本所在目录（按自身位置定位，不依赖开发机的固定盘符）
$EngineDir = $PSScriptRoot
$VenvPy    = Join-Path $EngineDir 'venv\Scripts\python.exe'
$SrcPkg    = Join-Path $EngineDir 'src\lingyu'

# ---- 检查解释器 ----
if (-not (Test-Path $VenvPy)) {
    Write-Host "[错误] 找不到 Python 解释器: $VenvPy" -ForegroundColor Red
    Write-Host "       请先在仓库根目录运行 setup_env.ps1，或手动创建 venv 并 pip install -e ." -ForegroundColor Red
    exit 1
}

# ---- 检查引擎代码（v0.3 起是 src/lingyu，不再是根目录 lingyu/）----
if (-not (Test-Path (Join-Path $SrcPkg 'cli.py'))) {
    Write-Host "[错误] 在 $EngineDir 里找不到 src\lingyu\cli.py" -ForegroundColor Red
    Write-Host "       本脚本必须和仓库根目录放在一起（仓库结构见 README）" -ForegroundColor Red
    exit 1
}

# ---- 切到仓库根（相对路径的 configs/、data/ 以它为基准）----
Set-Location -LiteralPath $EngineDir

# ---- 没给参数就直接看总帮助 ----
if (-not $LingYuArgs -or $LingYuArgs.Count -eq 0) {
    & $VenvPy -m lingyu --help
    $code = $LASTEXITCODE
    Write-Host ""
    Write-Host "常用命令：" -ForegroundColor Cyan
    Write-Host "  ly doctor                                            # 体检环境"
    Write-Host "  ly scan    --raw raw                                 # 看语料"
    Write-Host "  ly prepare --raw raw --data data                     # 准备数据"
    Write-Host "  ly plan    --config configs/small.ini                # 预估显存"
    Write-Host "  ly train   --config configs/small.ini                # 训练"
    Write-Host "  ly sft     --config configs/sft.ini                  # 微调"
    Write-Host "  ly chat    --config configs/sft.ini                  # 聊天"
    Write-Host "  ly sample  --config configs/small.ini --prompt 他抬起头  # 看效果"
    exit $code
}

# ---- 转发执行 ----
& $VenvPy -m lingyu @LingYuArgs
exit $LASTEXITCODE
