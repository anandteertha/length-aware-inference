# Builds an editable project deck from NC State's official 16:9 PowerPoint template.
# Source: https://brand.ncsu.edu/downloads/
# Run from the repository root: powershell -File presentation/build_deck.ps1

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$build = Join-Path $root '.codex-build\presentation'
$render = Join-Path $build 'render-v2'
$template = Join-Path $build 'ncstate-template.potx'
$output = Join-Path $PSScriptRoot 'nc-state-etdl-project.pptx'

if (Get-Process POWERPNT -ErrorAction SilentlyContinue) {
    throw 'Close PowerPoint before building the deck; this script uses an isolated PowerPoint COM session.'
}
if (Test-Path -LiteralPath $output) {
    throw "Output already exists: $output"
}
New-Item -ItemType Directory -Path $build, $render -Force | Out-Null
if (-not (Test-Path -LiteralPath $template)) {
    Invoke-WebRequest -Uri 'https://drive.google.com/uc?export=download&id=1ecVcMtVcEAK90-GOG2tEe2ay_PzTA8QA' -OutFile $template
}

function Rgb([int]$r, [int]$g, [int]$b) {
    return [int]($r + ($g -shl 8) + ($b -shl 16))
}
$red = Rgb 204 0 0
$black = Rgb 0 0 0
$dark = Rgb 51 51 51
$mid = Rgb 90 90 90
$gray = Rgb 128 128 128
$light = Rgb 242 242 242
$line = Rgb 205 205 205
$white = Rgb 255 255 255

function Add-Text($slide, [string]$name, [string]$value,
                  [double]$x, [double]$y, [double]$w, [double]$h,
                  [double]$size, [int]$color, [bool]$bold = $false,
                  [int]$align = 1) {
    $shape = $slide.Shapes.AddTextbox(1, $x, $y, $w, $h)
    $shape.Name = $name
    $shape.TextFrame.MarginLeft = 0
    $shape.TextFrame.MarginRight = 0
    $shape.TextFrame.MarginTop = 0
    $shape.TextFrame.MarginBottom = 0
    $shape.TextFrame.WordWrap = -1
    $shape.TextFrame.AutoSize = 0
    $range = $shape.TextFrame.TextRange
    $range.Text = $value
    $range.Font.Name = 'Arial'
    $range.Font.Size = $size
    $range.Font.Bold = $(if ($bold) { -1 } else { 0 })
    $range.Font.Color.RGB = $color
    $range.ParagraphFormat.Alignment = $align
    return $shape
}

function Add-Rect($slide, [string]$name, [double]$x, [double]$y,
                  [double]$w, [double]$h, [int]$color) {
    $shape = $slide.Shapes.AddShape(1, $x, $y, $w, $h)
    $shape.Name = $name
    $shape.Fill.Solid()
    $shape.Fill.ForeColor.RGB = $color
    $shape.Line.Visible = 0
    $shape.Shadow.Visible = 0
    return $shape
}

function Add-Line($slide, [string]$name, [double]$x1, [double]$y1,
                  [double]$x2, [double]$y2, [int]$color, [double]$width = 1) {
    $shape = $slide.Shapes.AddLine($x1, $y1, $x2, $y2)
    $shape.Name = $name
    $shape.Line.ForeColor.RGB = $color
    $shape.Line.Weight = $width
    return $shape
}

function Set-Notes($slide, [string]$value) {
    foreach ($shape in $slide.NotesPage.Shapes) {
        try {
            if ($shape.PlaceholderFormat.Type -eq 2) {
                $shape.TextFrame.TextRange.Text = $value
                return
            }
        } catch { }
    }
}

function Set-Title($slide, [string]$value) {
    $shape = $slide.Shapes.Title
    $shape.TextFrame.TextRange.Text = $value
    $shape.TextFrame.TextRange.Font.Name = 'Arial'
    $shape.TextFrame.TextRange.Font.Size = 27
    $shape.TextFrame.TextRange.Font.Bold = -1
    $shape.TextFrame.TextRange.Font.Color.RGB = $black
}

function Add-ContentSlide($deck, [string]$title, [string]$notes) {
    $layout = $deck.SlideMaster.CustomLayouts.Item(6) # NC State's Title Only layout
    $slide = $deck.Slides.AddSlide($deck.Slides.Count + 1, $layout)
    Set-Title $slide $title
    Set-Notes $slide $notes
    return $slide
}

function Add-Source($slide, [string]$value) {
    Add-Text $slide 'Source note' $value 36 355 648 15 8 $mid | Out-Null
}

$app = $null
$deck = $null
try {
    $app = New-Object -ComObject PowerPoint.Application
    $deck = $app.Presentations.Open($template, -1, -1, 0)
    if ($deck.PageSetup.SlideWidth -ne 720 -or $deck.PageSetup.SlideHeight -ne 405) {
        throw 'Unexpected template slide dimensions.'
    }

    # Cover (template slide 1)
    $s = $deck.Slides.Item(1)
    $s.Shapes.Title.TextFrame.TextRange.Text = 'When is a smaller model enough?'
    $s.Shapes.Title.TextFrame.TextRange.Font.Name = 'Arial'
    $s.Shapes.Title.TextFrame.TextRange.Font.Size = 32
    $s.Shapes.Title.TextFrame.TextRange.Font.Bold = -1
    $s.Shapes.Title.TextFrame.TextRange.Font.Color.RGB = $black
    $subtitle = $s.Shapes.Placeholders.Item(2)
    $subtitle.TextFrame.TextRange.Text = "Training and routing for efficient CPU sentiment inference`rAnandteertha Rao | ECE 592 / CSC 591 / ECE 492`rNovember 23, 2026"
    $subtitle.TextFrame.TextRange.Font.Name = 'Arial'
    $subtitle.TextFrame.TextRange.Font.Size = 18
    $subtitle.TextFrame.TextRange.Font.Color.RGB = $dark
    Add-Text $s 'Repository link' 'github.com/anandteertha/length-aware-inference' 108 336 504 18 11 $red $true 2 | Out-Null
    Set-Notes $s '10-minute course presentation. Repository: https://github.com/anandteertha/length-aware-inference. NC State template: https://brand.ncsu.edu/downloads/.'

    # 1. Problem
    $s = Add-ContentSlide $deck 'The accuracy–latency tradeoff' 'Explain the service-level goal: keep sentiment accuracy close to the six-layer teacher while reducing CPU request time. Source: docs/student-training-results.md; docs/gate-results.md.'
    Add-Text $s 'Teacher heading' 'Six-layer teacher' 40 138 285 28 21 $black $true | Out-Null
    Add-Text $s 'Teacher accuracy' '91.06%' 40 175 285 53 42 $red $true | Out-Null
    Add-Text $s 'Teacher label' 'SST-2 validation accuracy' 40 225 285 24 15 $dark | Out-Null
    Add-Line $s 'Divider' 356 135 356 284 $line 1.2 | Out-Null
    Add-Text $s 'Student heading' 'Four-layer student' 395 138 270 28 21 $black $true | Out-Null
    Add-Text $s 'Student accuracy' '87.73%' 395 175 270 53 42 $black $true | Out-Null
    Add-Text $s 'Student label' 'Less work, more mistakes' 395 225 270 24 15 $dark | Out-Null
    Add-Rect $s 'Goal rule' 40 282 640 55 $light | Out-Null
    Add-Text $s 'Goal text' 'Goal: ≤1 point accuracy loss  •  ≥15% lower median time  •  <1,536 MiB resident memory' 52 294 614 36 15 $black $true | Out-Null
    Add-Source $s 'SST-2 validation n=872; target thresholds fixed in project protocol.'

    # 2. Prior approaches
    $s = Add-ContentSlide $deck 'Existing ways to reduce inference work' 'Prior work: Hinton et al., Distilling the Knowledge in a Neural Network (2015), https://arxiv.org/abs/1503.02531; Xin et al., DeeBERT (ACL 2020), https://aclanthology.org/2020.acl-main.204/; Li et al., CascadeBERT (EMNLP Findings 2021), https://aclanthology.org/2021.findings-emnlp.43/. These methods are established; we do not claim to invent distillation, early exit, or cascades.'
    $prior = @(
        @('Fixed input shapes', 'Use a few compiled sequence lengths; padding can add work.'),
        @('Knowledge distillation', 'Train a smaller student to imitate a larger teacher.'),
        @('Early exits and cascades', 'Spend extra computation only on uncertain inputs.')
    )
    for ($i=0; $i -lt 3; $i++) {
        $y = 135 + 62*$i
        Add-Rect $s "Prior accent $i" 39 $y 5 48 $red | Out-Null
        Add-Text $s "Prior name $i" $prior[$i][0] 57 $y 220 28 19 $black $true | Out-Null
        Add-Text $s "Prior description $i" $prior[$i][1] 286 $y 380 48 16 $dark | Out-Null
    }
    Add-Text $s 'Study question' 'Our study: does length help beyond confidence alone?' 40 321 640 25 17 $red $true | Out-Null
    Add-Source $s 'Hinton et al. 2015; DeeBERT 2020; CascadeBERT 2021.'

    # 3. Original limitation
    $s = Add-ContentSlide $deck 'Fixed shapes did not pay off on real text' 'The original project tested verified fixed 16/32/64/128-token DistilBERT variants and a real-text break-even map at 11 exact lengths. No length passed the prespecified screen for a fixed-route confirmation. See docs/final-results.md and docs/breakeven-map-results.md. This is a bounded CPU finding, not a statement that fixed shapes never help.'
    Add-Text $s 'Input token count' '17 tokens' 42 151 180 42 28 $black $true | Out-Null
    Add-Text $s 'Arrow' '→' 227 150 60 42 30 $red $true 2 | Out-Null
    Add-Text $s 'Bucket size' '32-token slot' 297 151 280 42 28 $black $true | Out-Null
    Add-Rect $s 'Real tokens' 40 213 330 30 $red | Out-Null
    Add-Rect $s 'Padding tokens' 370 213 290 30 $light | Out-Null
    Add-Text $s 'Real token label' '17 real tokens' 46 218 318 23 15 $white $true 2 | Out-Null
    Add-Text $s 'Padding label' '15 padded positions' 378 218 274 23 15 $dark $true 2 | Out-Null
    Add-Text $s 'Screen result' '0 of 11 tested lengths passed the fixed-route screen' 40 277 640 41 23 $red $true | Out-Null
    Add-Source $s 'Fixed-route screen: 11 real-text lengths, 9–48 tokens; dynamic ONNX retained.'

    # 4. Challenges
    $s = Add-ContentSlide $deck 'Why student-first routing is difficult' 'A teacher fallback incurs both the student and teacher on that request. The teacher can correct or harm a student prediction. The teacher cost rises with token length. Tail latency therefore may differ from median latency. See docs/gate-results.md.'
    $challenges = @(
        @('Accuracy', 'The student misses some cases the teacher gets right.'),
        @('Recomputation', 'Fallback requests run both models, not just the teacher.'),
        @('Length', 'Teacher execution is costlier on longer text.'),
        @('Tail latency', 'Rare fallbacks can dominate slow requests.')
    )
    for ($i=0; $i -lt 4; $i++) {
        $y = 131 + 51*$i
        Add-Text $s "Challenge number $i" ('0' + ($i+1)) 41 $y 45 34 22 $red $true | Out-Null
        Add-Text $s "Challenge name $i" $challenges[$i][0] 94 $y 180 32 20 $black $true | Out-Null
        Add-Text $s "Challenge description $i" $challenges[$i][1] 278 $y 390 36 16 $dark | Out-Null
        if ($i -lt 3) { Add-Line $s "Challenge rule $i" 40 ($y+39) 672 ($y+39) $line | Out-Null }
    }
    Add-Source $s 'Measured on one laptop; request latency includes tokenization, routing and inference.'

    # 5. Training method
    $s = Add-ContentSlide $deck 'Train a smaller student' 'Teacher checkpoint: distilbert-base-uncased-finetuned-sst-2-english, pinned revision 714eb0fa89d2f80546fda750413ed43d93601a13. Student initialization copies embeddings, the first three or four transformer layers and classifier. 8,192 fit rows, 1,024 development rows, seed 2027, two CPU epochs, AdamW 2e-5, batch 16. Distillation temperature 2 and equal CE/KL weight. See docs/student-training-results.md and results/student-study/manifest.json.'
    Add-Text $s 'Teacher label' 'Teacher' 44 140 150 28 20 $black $true | Out-Null
    Add-Text $s 'Student label' 'Student' 397 140 150 28 20 $black $true | Out-Null
    for ($i=0; $i -lt 6; $i++) { Add-Rect $s "Teacher layer $i" (44+45*$i) 181 36 34 $red | Out-Null }
    for ($i=0; $i -lt 4; $i++) { Add-Rect $s "Student layer $i" (397+45*$i) 181 36 34 $black | Out-Null }
    Add-Text $s 'Teacher layers' '6 transformer layers' 44 228 286 25 15 $dark | Out-Null
    Add-Text $s 'Student layers' 'Copy first 4, then update weights' 397 228 272 25 15 $dark | Out-Null
    Add-Line $s 'Method divider' 42 265 677 265 $line | Out-Null
    Add-Text $s 'Objective formula' 'Training loss = ½ label loss + ½ teacher-matching loss' 44 279 630 35 21 $red $true | Out-Null
    Add-Text $s 'Objective detail' 'Teacher-matching uses softened logits (temperature 2).' 44 316 630 25 15 $dark | Out-Null
    Add-Source $s 'SST-2 train partitions only; best checkpoint chosen on development accuracy.'

    # 6. Routing method
    $s = Add-ContentSlide $deck 'Freeze the teacher fallback rule' 'Four-layer student logits and token length form gate features. A ridge fit predicts teacher incremental correctness on 2,048 gate-fit rows. Cost-aware score divides predicted benefit by measured teacher time. Thresholds are selected on a separate 1,024-row gate-calibration partition, then applied unchanged to 872 validation rows. Confidence-only is the simple control. The released teacher was trained on SST-2 training data, which makes gate training/calibration optimistic. See docs/gate-results.md.'
    Add-Rect $s 'Student node' 40 153 145 56 $light | Out-Null
    Add-Text $s 'Student node text' 'Student' 49 164 127 34 21 $black $true 2 | Out-Null
    Add-Text $s 'Gate arrow 1' '→' 193 163 44 37 26 $red $true 2 | Out-Null
    Add-Rect $s 'Gate node' 244 153 177 56 $light | Out-Null
    Add-Text $s 'Gate node text' 'Gate: confidence + length' 252 162 160 40 17 $black $true 2 | Out-Null
    Add-Text $s 'Gate arrow 2' '→' 429 163 42 37 26 $red $true 2 | Out-Null
    Add-Rect $s 'Teacher node' 478 153 195 56 $light | Out-Null
    Add-Text $s 'Teacher node text' 'Teacher if selected' 485 164 180 34 19 $black $true 2 | Out-Null
    Add-Text $s 'Fit label' 'FIT' 45 252 132 24 14 $red $true | Out-Null
    Add-Text $s 'Fit rows' '2,048 gate rows' 45 277 179 26 17 $black $true | Out-Null
    Add-Text $s 'Cal label' 'CALIBRATE' 262 252 150 24 14 $red $true | Out-Null
    Add-Text $s 'Cal rows' '1,024 other rows' 262 277 180 26 17 $black $true | Out-Null
    Add-Text $s 'Eval label' 'EVALUATE' 488 252 150 24 14 $red $true | Out-Null
    Add-Text $s 'Eval rows' '872 validation rows' 488 277 190 26 17 $black $true | Out-Null
    Add-Source $s 'Thresholds frozen before validation; teacher pretraining on SST-2 train limits calibration.'

    # 7. Training result
    $s = Add-ContentSlide $deck 'Training matters more than truncation' 'Exact 872-row SST-2 validation counts: untrained 3-layer 645 (73.97%); supervised 3-layer 735 (84.29%); distilled 3-layer 739 (84.75%); distilled 4-layer 765 (87.73%); full teacher 794 (91.06%). Distillation adds only four examples under one seed. See docs/student-training-results.md.'
    $rows = @(
        @('3-layer, no updates',73.97,$gray),
        @('3-layer supervised',84.29,$dark),
        @('3-layer distilled',84.75,$dark),
        @('4-layer distilled',87.73,$red),
        @('6-layer teacher',91.06,$black)
    )
    for ($i=0; $i -lt 5; $i++) {
        $y = 137 + 39*$i
        Add-Text $s "Model $i" $rows[$i][0] 40 $y 176 28 15 $black $(($i -eq 3) -or ($i -eq 4)) | Out-Null
        Add-Rect $s "Bar backdrop $i" 223 ($y+3) 355 20 $light | Out-Null
        Add-Rect $s "Accuracy bar $i" 223 ($y+3) (3.55*$rows[$i][1]) 20 $rows[$i][2] | Out-Null
        Add-Text $s "Accuracy value $i" ('{0:N2}%' -f $rows[$i][1]) 589 $y 82 28 17 $black $true | Out-Null
    }
    Add-Text $s 'Training takeaway' 'Supervised training adds 90 correct examples over plain truncation.' 40 326 640 24 17 $red $true | Out-Null
    Add-Source $s 'Validation n=872; one seed and two-epoch CPU training budget.'

    # 8. Gate result
    $s = Add-ContentSlide $deck 'Confidence routing meets the observed target' 'Both ONNX sessions were resident for every cascade policy. Three shuffled process blocks, four ORT threads, natural validation text. Confidence gate routed 161/872 requests to teacher; 789 correct versus teacher 794, a 0.57-point observed loss. Matched block median reductions were 10.1%, 24.1%, 30.3%; central 24.1%. Peak RSS about 852 MiB; process peak working set about 928 MiB. Confidence accuracy paired bootstrap 95% difference vs teacher: -1.61 to +0.46 points, so no population noninferiority proof. p95 mixed. See docs/gate-results.md and results/student-study/cascade-timing/summary.json.'
    Add-Rect $s 'Header line' 40 137 635 2 $red | Out-Null
    $headers = @(@('Policy',42),@('Accuracy',292),@('Fallbacks',401),@('Median',499),@('Paired gain',587))
    foreach($h in $headers){ Add-Text $s ("Header " + $h[0]) $h[0] $h[1] 147 100 25 13 $mid $true | Out-Null }
    $table = @(
        @('Teacher only','91.06%','872','27.92 ms','—'),
        @('Student only','87.73%','0','18.32 ms','33.9%'),
        @('Length + cost gate','89.33%','121','20.45 ms','23.2%'),
        @('Confidence gate','90.48%','161','21.20 ms','24.1%')
    )
    for($i=0;$i -lt 4;$i++){
        $y=179+37*$i
        if($i -eq 3){ Add-Rect $s 'Selected policy highlight' 40 ($y-3) 635 35 (Rgb 255 235 235) | Out-Null }
        Add-Text $s "Policy $i" $table[$i][0] 42 $y 243 29 15 $(if($i -eq 3){$red}else{$black}) $(($i -eq 0) -or ($i -eq 3)) | Out-Null
        Add-Text $s "Policy accuracy $i" $table[$i][1] 292 $y 100 29 15 $black $(($i -eq 0) -or ($i -eq 3)) | Out-Null
        Add-Text $s "Policy fallbacks $i" $table[$i][2] 401 $y 93 29 15 $black | Out-Null
        Add-Text $s "Policy median $i" $table[$i][3] 499 $y 88 29 15 $black | Out-Null
        Add-Text $s "Policy reduction $i" $table[$i][4] 587 $y 87 29 15 $black $(($i -eq 3)) | Out-Null
        Add-Line $s "Table rule $i" 40 ($y+30) 675 ($y+30) $line | Out-Null
    }
    Add-Text $s 'Gate takeaway' 'Within 0.57 accuracy points; 24.1% paired median reduction; ~852 MiB RSS.' 40 326 635 23 16 $red $true | Out-Null
    Add-Source $s 'SST-2 n=872; three paired CPU blocks. p95 direction mixed; accuracy interval exceeds 1-point loss.'

    # 9. Transfer result
    $s = Add-ContentSlide $deck 'Cost-weighted training did not transfer' 'Protocol and evaluator were committed before external prediction. Same four-layer model, initialization, data, seed and budget; only KL loss weights changed. On reused SST-2 validation uniform 765/872, weighted 770/872 (exploratory). On pinned, balanced IMDb test n=1000, teacher 895, uniform 843, weighted 832; paired weighted-minus-uniform difference -1.1 points, 95% bootstrap interval [-2.6,+0.4]. The weighted student lost three correct examples in the targeted longest SST-2 stratum. IMDb median 211 tokens, 855/1000 129–512 tokens. Weighted checkpoint was not benchmarked for latency. See docs/cost-weighted-extension-results.md.'
    Add-Text $s 'SST2 heading' 'SST-2 validation (reused)' 40 140 290 28 20 $black $true | Out-Null
    Add-Text $s 'SST2 comparison' '765  →  770' 40 180 290 47 31 $red $true | Out-Null
    Add-Text $s 'SST2 descriptor' 'uniform       weighted   (+5)' 40 226 290 29 15 $dark | Out-Null
    Add-Line $s 'Transfer divider' 357 141 357 273 $line 1.2 | Out-Null
    Add-Text $s 'IMDb heading' 'IMDb test (new domain)' 391 140 290 28 20 $black $true | Out-Null
    Add-Text $s 'IMDb comparison' '843  →  832' 391 180 290 47 31 $black $true | Out-Null
    Add-Text $s 'IMDb descriptor' 'uniform       weighted   (−11)' 391 226 290 29 15 $dark | Out-Null
    Add-Rect $s 'Transfer result band' 40 282 640 53 $light | Out-Null
    Add-Text $s 'Transfer result text' 'External difference: −1.1 points  |  95% interval: −2.6 to +0.4' 52 293 616 33 17 $red $true | Out-Null
    Add-Source $s 'IMDb n=1,000 balanced, ≤512 tokens. No weighted-model ONNX latency measured.'

    # Closing
    $s = Add-ContentSlide $deck 'What the evidence supports' 'Answer the success question directly. The course project has a positive observed efficiency result: confidence-gated four-layer student gets 789/872 versus teacher 794/872 with median paired latency reduction 24.1% and resident RSS about 852 MiB under 1,536 MiB. The proposed length/cost gate misses one-point accuracy target and cost-weighted training does not improve external accuracy. These are bounded observations on one laptop, one SST-2 validation split, one training seed and one external sample. Population noninferiority is unproved and p95 timing is mixed. Distillation and cascades are prior art, so claim a careful experimental study rather than a novel algorithm. Source: docs/gate-results.md; docs/cost-weighted-extension-results.md.'
    Add-Text $s 'Conclusion 1' 'Yes: an observed CPU efficiency win with a trained student and simple gate.' 41 140 637 55 22 $red $true | Out-Null
    Add-Line $s 'Close rule 1' 41 204 678 204 $line | Out-Null
    Add-Text $s 'Conclusion 2' 'The length-and-cost gate missed the accuracy target; weighted training did not improve external accuracy.' 41 220 637 60 19 $black | Out-Null
    Add-Line $s 'Close rule 2' 41 292 678 292 $line | Out-Null
    Add-Text $s 'Conclusion 3' 'Scope: one laptop, one seed, mixed p95, and no general algorithm novelty claim.' 41 307 637 42 18 $dark | Out-Null
    Add-Source $s 'Code and full protocols: github.com/anandteertha/length-aware-inference'

    if ($deck.Slides.Count -ne 11) { throw "Expected 11 slides, got $($deck.Slides.Count)" }
    for ($i=1; $i -le $deck.Slides.Count; $i++) {
        $deck.Slides.Item($i).Export((Join-Path $render ('slide-{0:D2}.png' -f $i)), 'PNG', 1600, 900)
    }
    $deck.SaveAs($output, 24) # ppSaveAsOpenXMLPresentation
    Write-Output "Saved $output with $($deck.Slides.Count) slides."
}
finally {
    if ($deck) { $deck.Close() }
    if ($app) { $app.Quit() }
}
