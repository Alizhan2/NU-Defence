$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$outDir = Join-Path $root 'submission'
$deckPath = Join-Path $outDir 'GeoWatch_AI_Competition_Deck_Final.pptx'
$pdfPath = Join-Path $outDir 'GeoWatch_AI_Competition_Deck_Final.pdf'
$dashboard = Join-Path $outDir 'screenshots\dashboard.png'
$temporal = Join-Path $outDir 'screenshots\temporal_workspace.png'

function Rgb([int]$r, [int]$g, [int]$b) { return $r + 256 * $g + 65536 * $b }
$navy = Rgb 6 24 39
$ink = Rgb 16 38 57
$muted = Rgb 82 103 120
$teal = Rgb 22 198 183
$blue = Rgb 49 91 254
$orange = Rgb 255 150 90
$paper = Rgb 246 248 252
$white = Rgb 255 255 255
$line = Rgb 218 226 237

$ppt = New-Object -ComObject PowerPoint.Application
$ppt.Visible = -1
$presentation = $ppt.Presentations.Add()
$presentation.PageSetup.SlideSize = 15 # wide 16:9
$presentation.PageSetup.SlideWidth = 1280
$presentation.PageSetup.SlideHeight = 720

function Add-Box($slide, [float]$x, [float]$y, [float]$w, [float]$h, [int]$color, [int]$radius = 0) {
  $type = if ($radius -eq 1) { 5 } else { 1 }
  $shape = $slide.Shapes.AddShape($type, $x, $y, $w, $h)
  $shape.Fill.ForeColor.RGB = $color
  $shape.Line.Visible = 0
  return $shape
}
function Add-Text($slide, [string]$text, [float]$x, [float]$y, [float]$w, [float]$h, [int]$size, [int]$color, [bool]$bold = $false, [string]$font = 'Aptos') {
  $shape = $slide.Shapes.AddTextbox(1, $x, $y, $w, $h)
  $shape.TextFrame.MarginLeft = 0
  $shape.TextFrame.MarginRight = 0
  $shape.TextFrame.MarginTop = 0
  $shape.TextFrame.MarginBottom = 0
  $shape.TextFrame.WordWrap = -1
  $shape.TextFrame.TextRange.Text = $text
  $shape.TextFrame.TextRange.Font.Name = $font
  $shape.TextFrame.TextRange.Font.Size = $size
  $shape.TextFrame.TextRange.Font.Bold = if ($bold) { -1 } else { 0 }
  $shape.TextFrame.TextRange.Font.Color.RGB = $color
  return $shape
}
function Add-Header($slide, [string]$title, [int]$number) {
  Add-Text $slide 'GEOWATCH AI' 54 30 200 24 15 $teal $true | Out-Null
  Add-Text $slide $title 54 82 1100 48 28 $ink $true | Out-Null
  Add-Text $slide ('{0:00}' -f $number) 1174 32 52 20 13 $muted $true | Out-Null
  $rule = $slide.Shapes.AddLine(54, 145, 1225, 145)
  $rule.Line.ForeColor.RGB = $line
  $rule.Line.Weight = 1
}
function New-Slide {
  $slide = $presentation.Slides.Add($presentation.Slides.Count + 1, 12)
  $slide.Background.Fill.ForeColor.RGB = $paper
  return $slide
}
function Add-Card($slide, [float]$x, [float]$y, [float]$w, [float]$h) {
  $card = Add-Box $slide $x $y $w $h $white 1
  $card.Shadow.Visible = -1
  $card.Shadow.ForeColor.RGB = Rgb 202 212 226
  $card.Shadow.Transparency = 0.75
  return $card
}

# 1. Cover
$slide = New-Slide
$slide.Background.Fill.ForeColor.RGB = $navy
Add-Box $slide 0 0 1280 720 $navy | Out-Null
Add-Box $slide 760 0 520 720 (Rgb 9 49 67) | Out-Null
Add-Box $slide 823 122 304 304 $teal 1 | Out-Null
Add-Box $slide 862 161 226 226 $navy 1 | Out-Null
Add-Text $slide '◉' 905 187 160 150 100 $teal $true | Out-Null
Add-Text $slide 'GeoWatch AI' 70 116 630 70 47 $white $true | Out-Null
Add-Text $slide 'Проверяемая аналитика изменений на открытых космических снимках' 70 205 580 62 25 (Rgb 207 225 233) $false | Out-Null
Add-Text $slide 'NU STeP × Defence Tech Challenge' 70 317 460 26 17 $teal $true | Out-Null
Add-Text $slide "Бижан Әлижан  Team Lead`nҚанымбек Марат  Backend`nҚұлтас Баубек  Frontend" 70 452 470 112 20 $white $false | Out-Null
Add-Text $slide 'Конкурсная презентация' 70 637 310 20 14 (Rgb 156 185 196) $false | Out-Null

# 2. Problem
$slide = New-Slide; Add-Header $slide 'Проблема и задача аналитика' 2
Add-Text $slide 'Сравнение снимков «до / после» вручную занимает время и не оставляет удобной истории решений.' 54 178 950 48 24 $ink $true | Out-Null
Add-Card $slide 54 275 348 248 | Out-Null
Add-Text $slide '01' 82 302 60 28 20 $blue $true | Out-Null
Add-Text $slide 'Много визуального шума' 82 346 265 32 20 $ink $true | Out-Null
Add-Text $slide 'Облачность, сезонность, угол съёмки и разные разрешения создают ложные различия.' 82 394 270 88 16 $muted $false | Out-Null
Add-Card $slide 466 275 348 248 | Out-Null
Add-Text $slide '02' 494 302 60 28 20 $teal $true | Out-Null
Add-Text $slide 'Результат нужно проверить' 494 346 270 32 20 $ink $true | Out-Null
Add-Text $slide 'Автоматический вывод без evidence и решения эксперта нельзя использовать как окончательный.' 494 394 270 88 16 $muted $false | Out-Null
Add-Card $slide 878 275 348 248 | Out-Null
Add-Text $slide '03' 906 302 60 28 20 $orange $true | Out-Null
Add-Text $slide 'Нет единого контура' 906 346 270 32 20 $ink $true | Out-Null
Add-Text $slide 'Файлы, события, комментарии и отчёты часто находятся в разных инструментах.' 906 394 270 88 16 $muted $false | Out-Null

# 3. Solution
$slide = New-Slide; Add-Header $slide 'Решение GeoWatch AI' 3
Add-Text $slide 'Локальный MVP переводит снимки и temporal-пары в очередь наблюдений для аналитика.' 54 180 820 38 23 $ink $true | Out-Null
Add-Text $slide 'Источник / AOI' 82 305 150 28 18 $muted $true | Out-Null
Add-Text $slide 'Проверка качества' 287 305 188 28 18 $muted $true | Out-Null
Add-Text $slide 'AI-анализ' 535 305 126 28 18 $muted $true | Out-Null
Add-Text $slide 'Экспертная проверка' 737 305 184 28 18 $muted $true | Out-Null
Add-Text $slide 'Кейс и экспорт' 1004 305 160 28 18 $muted $true | Out-Null
foreach ($x in @(82, 287, 535, 737, 1004)) { Add-Box $slide $x 356 116 116 $teal 1 | Out-Null }
foreach ($x in @(198, 403, 651, 853)) { $l = $slide.Shapes.AddLine($x, 414, $x + 89, 414); $l.Line.ForeColor.RGB = $blue; $l.Line.Weight = 2 }
Add-Text $slide 'Фото и GeoTIFF' 68 495 146 30 16 $ink $true | Out-Null
Add-Text $slide 'Gate по совместимости пары' 256 495 184 42 16 $ink $true | Out-Null
Add-Text $slide 'YOLO, тайлы, NMS, события' 495 495 185 42 16 $ink $true | Out-Null
Add-Text $slide 'confirmed / rejected / needs_review' 704 495 212 42 16 $ink $true | Out-Null
Add-Text $slide 'JSON, CSV, HTML, PDF, ZIP' 982 495 206 42 16 $ink $true | Out-Null
Add-Text $slide 'Каждое событие требует статуса review. GeoWatch AI не принимает автономных решений.' 54 610 1050 28 18 $muted $false | Out-Null

# 4. Product screenshot
$slide = New-Slide; Add-Header $slide 'Рабочий экран: анализ и evidence' 4
if (Test-Path $dashboard) { $slide.Shapes.AddPicture($dashboard, 0, -1, 54, 178, 743, 418) | Out-Null }
Add-Text $slide 'Интерфейс помогает' 854 198 310 34 24 $ink $true | Out-Null
Add-Text $slide "• загрузить снимок и выбрать порог confidence`n• увидеть bounding boxes, класс и score`n• открыть evidence card конкретного объекта`n• направить результат на review" 854 260 340 190 18 $muted $false | Out-Null
Add-Box $slide 854 500 304 72 (Rgb 226 248 245) 1 | Out-Null
Add-Text $slide "Модель работает локально.`nСекреты не попадают в UI." 877 518 270 42 16 $ink $true | Out-Null

# 5. Temporal workspace
$slide = New-Slide; Add-Header $slide 'Хронология изменений «до / после»' 5
Add-Text $slide 'Отдельный workspace показывает две даты, слои объектов и события appeared, disappeared, stable.' 54 178 1050 36 22 $ink $true | Out-Null
if (Test-Path $temporal) { $slide.Shapes.AddPicture($temporal, 0, -1, 54, 245, 730, 410) | Out-Null }
Add-Text $slide 'Что проверяет система' 846 252 320 34 22 $ink $true | Out-Null
Add-Text $slide "• качество входных изображений`n• overlap, CRS и grid для GeoTIFF`n• безопасную регистрацию JPG/PNG`n• связь событий с review и кейсом" 846 312 340 180 18 $muted $false | Out-Null
Add-Box $slide 846 540 330 74 (Rgb 255 239 228) 1 | Out-Null
Add-Text $slide "Демо temporal-flow синтетическое.`nОн показывает product flow, не метрику change model." 868 558 292 45 15 $ink $true | Out-Null

# 6. Architecture
$slide = New-Slide; Add-Header $slide 'Архитектура MVP' 6
Add-Text $slide 'Единый pipeline используется в Streamlit и FastAPI, а результаты и review хранятся локально.' 54 178 1060 34 22 $ink $true | Out-Null
$arch = @(
  @('Streamlit UI', 'Анализ, изменения, review, кейсы'),
  @('FastAPI', 'Интеграционный API и OpenAPI'),
  @('Core pipeline', 'preprocessing, tiles, YOLO, matching, reporting'),
  @('SQLite + artifacts', 'история, jobs, exports, evidence')
)
$pos = @(54, 350, 646, 942)
for ($i = 0; $i -lt $arch.Count; $i++) {
  Add-Card $slide $pos[$i] 300 242 195 | Out-Null
  Add-Box $slide ($pos[$i] + 24) 327 42 42 $(if ($i -eq 0) { $blue } elseif ($i -eq 1) { $teal } elseif ($i -eq 2) { $orange } else { $navy }) 1 | Out-Null
  Add-Text $slide ('{0}' -f ($i + 1)) ($pos[$i] + 38) 339 22 22 14 $white $true | Out-Null
  Add-Text $slide $arch[$i][0] ($pos[$i] + 24) 390 194 28 18 $ink $true | Out-Null
  Add-Text $slide $arch[$i][1] ($pos[$i] + 24) 433 190 56 15 $muted $false | Out-Null
}
Add-Text $slide 'Optional: Google Earth Engine adapter. Для реальных сцен нужны Cloud Project и авторизация.' 54 606 900 26 17 $muted $false | Out-Null

# 7. Model and evidence
$slide = New-Slide; Add-Header $slide 'ML-модель и проверяемость' 7
Add-Card $slide 54 190 510 390 | Out-Null
Add-Text $slide 'Активный detection baseline' 84 224 410 30 21 $ink $true | Out-Null
Add-Text $slide "YOLOv8n-OBB, DOTA4`nКлассы: aircraft, ship, small vehicle, large vehicle" 84 282 420 72 18 $muted $false | Out-Null
Add-Text $slide 'Изолированный scene-separated test split' 84 392 390 24 17 $teal $true | Out-Null
Add-Text $slide "Precision  0.8741`nRecall     0.8354`nF1             0.8543`nmAP50      0.8915`nmAP50-95 0.6719" 84 438 320 122 18 $ink $true | Out-Null
Add-Card $slide 628 190 598 390 | Out-Null
Add-Text $slide 'Честные границы' 658 224 400 30 21 $ink $true | Out-Null
Add-Text $slide 'Метрики относятся к DOTA4 и четырём указанным классам. Они не доказывают автоматическое выявление строительства.' 658 282 500 76 18 $muted $false | Out-Null
Add-Text $slide 'SpaceNet 7 change detection' 658 393 320 25 18 $orange $true | Out-Null
Add-Text $slide 'Подготовлен pipeline и AOI-separated split. Реальное обучение и независимая test-оценка остаются следующим этапом.' 658 435 500 84 18 $muted $false | Out-Null
Add-Text $slide 'Checkpoint hash, seed, dataset manifest и метрики лежат в репозитории рядом с моделью.' 54 625 1100 24 16 $muted $false | Out-Null

# 8. Requirements
$slide = New-Slide; Add-Header $slide 'Соответствие ключевым требованиям' 8
$rows = @(
  @('Форматы JPG, PNG, GeoTIFF', 'Готово', 'preprocessing + quality gate'),
  @('Детекция, классы, bbox, confidence', 'Готово', 'YOLO-OBB, tiling, class-aware NMS'),
  @('Пара «до / после» и chronology', 'Частично', 'events + review; реальная change-model в следующем этапе'),
  @('Экспертная проверка и история', 'Готово', 'review queue, SQLite persistence, cases'),
  @('Отчёты и интеграция', 'Готово', 'JSON, CSV, HTML, PDF, ZIP и FastAPI'),
  @('Google Earth Engine', 'Частично', 'metadata adapter; нужен Cloud Project и auth')
)
Add-Box $slide 54 188 1170 46 $navy 1 | Out-Null
Add-Text $slide 'Требование' 78 202 415 22 16 $white $true | Out-Null
Add-Text $slide 'Статус' 540 202 150 22 16 $white $true | Out-Null
Add-Text $slide 'Реализация / ограничение' 722 202 450 22 16 $white $true | Out-Null
for ($i=0; $i -lt $rows.Count; $i++) {
  $y = 235 + $i * 58
  Add-Box $slide 54 $y 1170 55 $(if (($i % 2) -eq 0) { $white } else { Rgb 239 244 249 }) | Out-Null
  Add-Text $slide $rows[$i][0] 78 ($y+14) 420 24 15 $ink $false | Out-Null
  $c = if ($rows[$i][1] -eq 'Готово') { $teal } else { $orange }
  Add-Text $slide $rows[$i][1] 540 ($y+14) 140 24 15 $c $true | Out-Null
  Add-Text $slide $rows[$i][2] 722 ($y+14) 450 27 14 $muted $false | Out-Null
}

# 9. Demo
$slide = New-Slide; Add-Header $slide 'Сценарий live demo: 4–5 минут' 9
$steps = @(
  @('01', 'Контекст', 'Показываем задачу аналитика и правило human review'),
  @('02', 'Temporal demo', 'Открываем пару «до / после», выбираем событие и evidence'),
  @('03', 'Review → case', 'Сохраняем confirmed / rejected / needs_review, создаём кейс'),
  @('04', 'Export', 'Скачиваем ZIP-пакет: manifest, results, reviews, evidence')
)
for ($i=0; $i -lt $steps.Count; $i++) {
  $y = 188 + $i * 103
  Add-Box $slide 70 $y 70 70 $(if ($i % 2 -eq 0) { $blue } else { $teal }) 1 | Out-Null
  Add-Text $slide $steps[$i][0] 87 ($y+22) 38 22 17 $white $true | Out-Null
  Add-Text $slide $steps[$i][1] 177 ($y+5) 260 28 21 $ink $true | Out-Null
  Add-Text $slide $steps[$i][2] 177 ($y+41) 720 28 17 $muted $false | Out-Null
}
Add-Box $slide 927 204 242 320 (Rgb 226 248 245) 1 | Out-Null
Add-Text $slide 'Резервный режим' 955 235 180 30 20 $ink $true | Out-Null
Add-Text $slide 'Если сеть, GPU или Earth Engine недоступны, показываем локальный demo-кейс. Модель не подменяет результат фиктивными detections.' 955 292 180 175 16 $muted $false | Out-Null

# 10. Team and submission
$slide = New-Slide; Add-Header $slide 'Команда и комплект для отправки' 10
Add-Text $slide 'GeoWatch AI готов к показу как честный конкурсный MVP с локальным demo-flow и проверяемыми артефактами.' 54 180 1080 34 22 $ink $true | Out-Null
Add-Card $slide 54 270 350 204 | Out-Null
Add-Text $slide 'Бижан Әлижан' 82 310 250 30 23 $ink $true | Out-Null
Add-Text $slide 'Team Lead' 82 353 160 25 18 $teal $true | Out-Null
Add-Text $slide 'Продукт, ML-пайплайн, интеграция, demo' 82 399 260 44 16 $muted $false | Out-Null
Add-Card $slide 465 270 350 204 | Out-Null
Add-Text $slide 'Қанымбек Марат' 493 310 250 30 23 $ink $true | Out-Null
Add-Text $slide 'Backend' 493 353 160 25 18 $blue $true | Out-Null
Add-Text $slide 'API, persistence, review и exports' 493 399 260 44 16 $muted $false | Out-Null
Add-Card $slide 876 270 350 204 | Out-Null
Add-Text $slide 'Құлтас Баубек' 904 310 250 30 23 $ink $true | Out-Null
Add-Text $slide 'Frontend' 904 353 160 25 18 $orange $true | Out-Null
Add-Text $slide 'Интерфейс, визуализация и пользовательский flow' 904 399 270 44 16 $muted $false | Out-Null
Add-Box $slide 54 550 1170 74 $navy 1 | Out-Null
Add-Text $slide 'Вложить в письмо: презентацию (.pptx / .pdf), исходники (.zip), 2 скриншота и ссылку на GitHub.' 82 575 1060 25 17 $white $false | Out-Null

if (Test-Path $deckPath) { Remove-Item -LiteralPath $deckPath -Force }
if (Test-Path $pdfPath) { Remove-Item -LiteralPath $pdfPath -Force }
$presentation.SaveAs($deckPath, 24)
$presentation.SaveAs($pdfPath, 32)
$presentation.Close()
$ppt.Quit()
[Runtime.InteropServices.Marshal]::ReleaseComObject($presentation) | Out-Null
[Runtime.InteropServices.Marshal]::ReleaseComObject($ppt) | Out-Null
[GC]::Collect(); [GC]::WaitForPendingFinalizers()
Write-Host "Created $deckPath"


