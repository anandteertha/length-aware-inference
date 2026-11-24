$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$build = Join-Path $repo 'tmp/pdfs/neurips2024'
$style = Join-Path $build 'Styles'
$output = Join-Path $repo 'output/pdf'
New-Item -ItemType Directory -Force $build, $output | Out-Null

if (-not (Test-Path (Join-Path $style 'neurips_2024.sty'))) {
    $archive = Join-Path $build 'Styles.zip'
    Invoke-WebRequest -Uri 'https://media.neurips.cc/Conferences/NeurIPS2024/Styles.zip' -OutFile $archive
    Expand-Archive -LiteralPath $archive -DestinationPath $build -Force
}

Push-Location -LiteralPath $repo
try {
    $latexArgs = @('-interaction=nonstopmode', '-halt-on-error',
        "-include-directory=$style", "-output-directory=$build", 'report/final-report.tex')
    for ($pass = 0; $pass -lt 2; $pass++) {
        & pdflatex @latexArgs
        if ($LASTEXITCODE -ne 0) {
            throw "LaTeX compilation failed. See $build/final-report.log."
        }
    }
    Copy-Item -LiteralPath (Join-Path $build 'final-report.pdf') -Destination $output
} finally {
    Pop-Location
}
