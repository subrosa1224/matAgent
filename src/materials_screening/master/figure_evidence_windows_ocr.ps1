# Production file-reader only. No UI automation, network, or output file writes.
param([Parameter(Mandatory=$true)][string]$ImagePath)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
function Await-WinRT($Operation, [Type]$Type) {
    $converter = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
        $_.Name -eq 'AsTask' -and $_.IsGenericMethodDefinition -and
        $_.GetGenericArguments().Count -eq 1 -and $_.GetParameters().Count -eq 1 -and
        $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
    } | Select-Object -First 1
    $pending = $converter.MakeGenericMethod($Type).Invoke($null, @($Operation))
    if (-not $pending.Wait(30000)) { throw 'OCR_TIMEOUT' }
    return $pending.Result
}
try {
    $phase = 'INITIALIZATION'
    Add-Type -AssemblyName System.Runtime.WindowsRuntime
    [Windows.Storage.StorageFile,Windows.Storage,ContentType=WindowsRuntime] | Out-Null
    [Windows.Storage.FileAccessMode,Windows.Storage,ContentType=WindowsRuntime] | Out-Null
    [Windows.Storage.Streams.IRandomAccessStream,Windows.Storage.Streams,ContentType=WindowsRuntime] | Out-Null
    [Windows.Graphics.Imaging.BitmapDecoder,Windows.Graphics.Imaging,ContentType=WindowsRuntime] | Out-Null
    [Windows.Graphics.Imaging.SoftwareBitmap,Windows.Graphics.Imaging,ContentType=WindowsRuntime] | Out-Null
    [Windows.Graphics.Imaging.BitmapPixelFormat,Windows.Graphics.Imaging,ContentType=WindowsRuntime] | Out-Null
    [Windows.Graphics.Imaging.BitmapAlphaMode,Windows.Graphics.Imaging,ContentType=WindowsRuntime] | Out-Null
    [Windows.Media.Ocr.OcrEngine,Windows.Foundation,ContentType=WindowsRuntime] | Out-Null
    [Windows.Media.Ocr.OcrResult,Windows.Foundation,ContentType=WindowsRuntime] | Out-Null
    $inputFile = (Resolve-Path -LiteralPath $ImagePath).Path
    if ([IO.Path]::GetExtension($inputFile) -ne '.png' -or (Get-Item -LiteralPath $inputFile).Length -gt 10485760) {
        throw 'IMAGE_REJECTED'
    }
    $languages = @([Windows.Media.Ocr.OcrEngine]::AvailableRecognizerLanguages | ForEach-Object {$_.LanguageTag})
    $reader = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
    if ($null -eq $reader) {
        Write-Output (@{status='unavailable';engine='Windows.Media.Ocr';error_code='NO_INSTALLED_OCR_LANGUAGE';installed_languages=$languages} | ConvertTo-Json -Compress)
        exit 0
    }
    $phase = 'OPEN_FILE'
    $file = Await-WinRT ([Windows.Storage.StorageFile]::GetFileFromPathAsync($inputFile)) ([Windows.Storage.StorageFile])
    $phase = 'OPEN_STREAM'
    $stream = Await-WinRT ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
    try {
        $phase = 'DECODE'
        $decoder = Await-WinRT ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
        $phase = 'BITMAP'
        $bitmap = Await-WinRT ($decoder.GetSoftwareBitmapAsync([Windows.Graphics.Imaging.BitmapPixelFormat]::Bgra8,[Windows.Graphics.Imaging.BitmapAlphaMode]::Premultiplied)) ([Windows.Graphics.Imaging.SoftwareBitmap])
        try {
            if ($bitmap.PixelWidth -gt [Windows.Media.Ocr.OcrEngine]::MaxImageDimension -or
                $bitmap.PixelHeight -gt [Windows.Media.Ocr.OcrEngine]::MaxImageDimension -or
                [long]$bitmap.PixelWidth * $bitmap.PixelHeight -gt 4000000) { throw 'IMAGE_LIMIT' }
            $phase = 'RECOGNIZE'
            $text = Await-WinRT ($reader.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
            $phase = 'OUTPUT'
            if ($text.Text.Length -gt 10000) { throw 'TEXT_LIMIT' }
            $words = @($text.Lines | ForEach-Object { $_.Words | ForEach-Object {
                @{text=$_.Text;x=$_.BoundingRect.X;y=$_.BoundingRect.Y;width=$_.BoundingRect.Width;height=$_.BoundingRect.Height}
            }})
            if ($words.Count -gt 10000) { throw 'WORD_LIMIT' }
            # Do not depend on PowerShell module auto-loading inherited from a
            # Python host. Hash the input using the runtime library directly.
            $hasher = [Security.Cryptography.SHA256]::Create()
            $hashStream = [IO.File]::OpenRead($inputFile)
            try {
                $imageHash = [BitConverter]::ToString($hasher.ComputeHash($hashStream)).Replace('-', '').ToLowerInvariant()
            } finally { $hashStream.Dispose(); $hasher.Dispose() }
            $response = @{status='ok';engine='Windows.Media.Ocr';language=$reader.RecognizerLanguage.LanguageTag;
                installed_languages=$languages;raw_text=$text.Text;words=$words;
                image_sha256=$imageHash}
            $serialized = $response | ConvertTo-Json -Depth 8 -Compress
            if ([Text.Encoding]::UTF8.GetByteCount($serialized) -gt 1048576) { throw 'OUTPUT_LIMIT' }
            Write-Output $serialized
        } finally { $bitmap.Dispose() }
    } finally { $stream.Dispose() }
} catch {
    # Do not expose private filesystem paths or exception/provider details.
    Write-Output (@{status='failed';engine='Windows.Media.Ocr';error_code=('WINDOWS_OCR_'+$phase)} | ConvertTo-Json -Compress)
}
