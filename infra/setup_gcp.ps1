param(
  [string]$Project = $env:GCP_PROJECT,
  [string]$Dataset = $env:BQ_DATASET,
  [string]$Location = "US",
  [string]$CsvPath = "..\yellow_tripdata_2025-01.csv",  # optional seed
  [switch]$LoadCsv
)

if (-not $Project -or -not $Dataset) {
  throw "Set -Project and -Dataset or define env vars GCP_PROJECT / BQ_DATASET."
}

Write-Host "Project=$Project  Dataset=$Dataset  Location=$Location" -ForegroundColor Cyan

# 1) Create dataset if it doesn't exist
$bqList = bq --location=$Location ls -d "$Project"
if ($LASTEXITCODE -ne 0) { throw "bq not available; install the Cloud SDK and auth first." }
if ($bqList -notmatch "$Project:$Dataset") {
  Write-Host "Creating dataset $Project:$Dataset..."
  bq --location=$Location mk -d "$Project:$Dataset" | Out-Null
} else {
  Write-Host "Dataset already exists."
}

# 2) Optionally load the CSV locally into BigQuery
if ($LoadCsv) {
  if (-not (Test-Path $CsvPath)) { throw "CSV not found: $CsvPath" }
  Write-Host "Loading $CsvPath into $Project:$Dataset.yellow_2025_01 ..."
  bq --location=$Location load --replace --autodetect --source_format=CSV `
    "$Project:$Dataset.yellow_2025_01" "$CsvPath"
}

# Helper to run a SQL file with ${PROJECT}/${DATASET} substitution
function Invoke-BqSqlFile {
  param([string]$Path)
  $sql = Get-Content $Path -Raw
  $sql = $sql.Replace('${PROJECT}', $Project).Replace('${DATASET}', $Dataset)
  bq query --use_legacy_sql=false "$sql"
}

# 3) Apply views / models
Push-Location ..\sql
Invoke-BqSqlFile ".\02_feature_view.sql"       | Out-Null
# Only if you want BQML:
if (Test-Path ".\03_bqml_models.sql") {
  Invoke-BqSqlFile ".\03_bqml_models.sql"     | Out-Null
}
Pop-Location

Write-Host "BigQuery setup complete." -ForegroundColor Green
