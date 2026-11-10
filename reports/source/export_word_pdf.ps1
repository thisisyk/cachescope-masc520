param([string]$DocumentPath,[string]$PdfPath)
$ErrorActionPreference='Stop'
if(-not $DocumentPath){$DocumentPath=Join-Path $PSScriptRoot '..\CacheScope_MASC520_Final_Report.docx'}
if(-not $PdfPath){$PdfPath=Join-Path $PSScriptRoot '..\CacheScope_MASC520_Final_Report.pdf'}
$inputDocument=[IO.Path]::GetFullPath($DocumentPath)
$outputPdf=[IO.Path]::GetFullPath($PdfPath)
$wordApp=$null
$openedDocument=$null
try {
    $wordApp=New-Object -ComObject Word.Application
    $wordApp.Visible=$false
    $wordApp.DisplayAlerts=0
    $wordApp.AutomationSecurity=3
    $openedDocument=$wordApp.Documents.Open($inputDocument,$false,$true,$false)
    $openedDocument.Repaginate()
    $pageCount=$openedDocument.ComputeStatistics(2)
    $openedDocument.ExportAsFixedFormat($outputPdf,17)
    [ordered]@{pdf=$outputPdf;pages=$pageCount;backend='Microsoft Word ExportAsFixedFormat';opened_read_only=$true;author_block='omitted'} | ConvertTo-Json -Compress
} finally {
    if($null -ne $openedDocument){$openedDocument.Close(0);[void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($openedDocument)}
    if($null -ne $wordApp){$wordApp.Quit(0);[void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($wordApp)}
    [GC]::Collect();[GC]::WaitForPendingFinalizers()
}
