# Scratch driver for tinkering with cmc.exe on the CM test kernel.
# Edit the variables below and re-run:  .\scratch_cmc.ps1
#
# Goal: find ONE working command line that turns 200_CM_Gemm.cpp into
# SPIR-V / ISA. Once that works, the same flags go into CMCompiler.compile().
#
# FINDINGS SO FAR (2026-06-15):
#  * CM headers (<cm/cm.h>) are BUILT IN to cmc.exe — no CM_ROOT / -I needed.
#  * /arch:<x>            -> wrong flag (x86 cl flag), gives "invalid arch name".
#  * -Qxcm_jit_target=skl -> compiles, but skl has no LSC, so cm_store fails with
#                            "Not supported feature for this platform". The
#                            -Qxcm_jit_target list (hsw..icl) is all PRE-Xe.
#    => Need an Xe/LSC-capable target. Try an ocloc/-device style target flag.
#  * Kernel error: 'bfloat16' is not a CM builtin type name (needs the right
#    CM bf16 type/typedef). Tinker in 200_CM_Gemm.cpp.

$ErrorActionPreference = "Continue"

$cmc = Join-Path $PSScriptRoot "third_party\cmc\cmc.exe"
$src = Join-Path $PSScriptRoot "test_kernels\200_CM_Gemm.cpp"
$out = Join-Path $PSScriptRoot "test_kernels\200_CM_Gemm.spv"

# --- THINGS TO TINKER WITH ---------------------------------------------------
# Canonical invocation (from reference compile_file_cmc):
#   cmc <src> -o <out> -emit-spirv -mcpu=<PLATFORM> <options>
# Platform via -mcpu=  (BMG / DG2 / MTL / TGLLP).
$PLATFORM = "BMG"

# Extra kernel options. NOTE: -mdump_asm / -Qxcm_jit_target force the native-ISA
# path through ocloc64.dll (not present here). Plain -emit-spirv stops at SPIR-V
# and does NOT need ocloc, so leave the asm-dump flags OFF unless you have ocloc.
$OPTIONS = @()
# -----------------------------------------------------------------------------

$argsList = @($src, "-o", $out, "-emit-spirv", "-mcpu=$PLATFORM") + $OPTIONS

Write-Host "cmc:  $cmc"
Write-Host "args: $($argsList -join ' ')`n" -ForegroundColor Cyan

# Ensure cmc's own DLLs (clangFEWrapper.dll, etc.) are found, mirroring the
# reference function's LD_LIBRARY_PATH=./ — prepend the cmc dir to PATH.
$env:PATH = (Split-Path $cmc) + ";" + $env:PATH

& $cmc @argsList
Write-Host "`nexit code: $LASTEXITCODE" -ForegroundColor Yellow
if (Test-Path $out) {
    Write-Host "OUTPUT: $out ($((Get-Item $out).Length) bytes)" -ForegroundColor Green
}
