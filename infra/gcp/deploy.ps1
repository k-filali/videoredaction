[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidatePattern('^[a-z][a-z0-9-]{4,28}[a-z0-9]$')]
    [string] $ProjectId,

    [ValidatePattern('^[a-z]+-[a-z]+\d+$')]
    [string] $Region = 'us-central1',

    [ValidatePattern('^[a-z0-9][a-z0-9._-]{0,62}$')]
    [string] $Repository = 'clearframe',

    [ValidatePattern('^[a-z0-9][a-z0-9._-]{0,127}$')]
    [string] $ImageTag = '',

    [ValidatePattern('^[a-z0-9][a-z0-9._-]{0,62}$')]
    [string] $BucketName = '',

    [ValidateRange(1, 10240)]
    [int] $MaxUploadMb = 8192,

    [string] $NeonPooledUrlFile = '',

    [string] $NeonDirectUrlFile = '',

    [string] $AccessTokenFile = '',

    [switch] $RotateSecrets,

    [switch] $SkipBuild,

    [switch] $NonInteractive
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:ProjectId = $ProjectId
$script:Region = $Region
$script:TemporaryFiles = [System.Collections.Generic.List[string]]::new()

$Names = @{
    ApiService = 'clearframe-api'
    WebService = 'clearframe-web'
    CpuJob = 'clearframe-cpu-worker'
    GpuJob = 'clearframe-gpu-worker'
    MigrationJob = 'clearframe-migrate'
    ReconcileJob = 'clearframe-reconcile'
    SchedulerJob = 'clearframe-reconcile'
    ApiAccount = 'clearframe-api'
    WebAccount = 'clearframe-web'
    WorkerAccount = 'clearframe-worker'
    MigrationAccount = 'clearframe-migrate'
    SchedulerAccount = 'clearframe-scheduler'
    DatabaseSecret = 'clearframe-database-url'
    MigrationDatabaseSecret = 'clearframe-migration-database-url'
    AccessTokenSecret = 'clearframe-access-token'
}

function Write-Step {
    param([Parameter(Mandatory)][string] $Message)
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Invoke-Gcloud {
    param(
        [Parameter(Mandatory)]
        [string[]] $Arguments,
        [switch] $Capture
    )

    if ($Capture) {
        $result = & gcloud @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "gcloud failed: gcloud $($Arguments -join ' ')"
        }
        return ($result -join "`n").Trim()
    }

    & gcloud @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "gcloud failed: gcloud $($Arguments -join ' ')"
    }
}

function Test-GcloudResource {
    param([Parameter(Mandatory)][string[]] $Arguments)
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & gcloud @Arguments *> $null
    }
    finally {
        $ErrorActionPreference = $previous
    }
    return $LASTEXITCODE -eq 0
}

function New-TemporaryJson {
    param([Parameter(Mandatory)][object] $Value)
    $path = [System.IO.Path]::Combine(
        [System.IO.Path]::GetTempPath(),
        "clearframe-$([guid]::NewGuid().ToString('N')).json"
    )
    $json = ConvertTo-Json -InputObject $Value -Depth 10
    [System.IO.File]::WriteAllText(
        $path,
        $json,
        [System.Text.UTF8Encoding]::new($false)
    )
    $script:TemporaryFiles.Add($path)
    return $path
}

function Read-PlainTextSecret {
    param(
        [Parameter(Mandatory)][string] $Prompt,
        [string] $File = ''
    )

    if ($File) {
        $resolved = Resolve-Path -LiteralPath $File -ErrorAction Stop
        $value = [System.IO.File]::ReadAllText($resolved.Path).Trim()
        if (-not $value) {
            throw "Secret file is empty: $File"
        }
        return $value
    }
    if ($NonInteractive) {
        throw "$Prompt is required. Supply its corresponding secret file."
    }

    $secure = Read-Host $Prompt -AsSecureString
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try {
        return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    }
}

function New-RandomAccessToken {
    $bytes = [byte[]]::new(48)
    $rng = [Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        $rng.GetBytes($bytes)
    }
    finally {
        $rng.Dispose()
    }
    return [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
}

function Test-Secret {
    param([Parameter(Mandatory)][string] $Name)
    return Test-GcloudResource @(
        'secrets', 'describe', $Name,
        "--project=$script:ProjectId",
        '--quiet'
    )
}

function Set-SecretValue {
    param(
        [Parameter(Mandatory)][string] $Name,
        [Parameter(Mandatory)][string] $Value
    )

    if (-not (Test-Secret $Name)) {
        Invoke-Gcloud @(
            'secrets', 'create', $Name,
            '--replication-policy=automatic',
            "--project=$script:ProjectId",
            '--quiet'
        )
    }

    $path = [System.IO.Path]::Combine(
        [System.IO.Path]::GetTempPath(),
        "clearframe-secret-$([guid]::NewGuid().ToString('N'))"
    )
    try {
        [System.IO.File]::WriteAllText(
            $path,
            $Value,
            [System.Text.UTF8Encoding]::new($false)
        )
        Invoke-Gcloud @(
            'secrets', 'versions', 'add', $Name,
            "--data-file=$path",
            "--project=$script:ProjectId",
            '--quiet'
        )
    }
    finally {
        if (Test-Path -LiteralPath $path) {
            [System.IO.File]::WriteAllText($path, '')
            Remove-Item -LiteralPath $path -Force
        }
    }
}

function Initialize-Secrets {
    $databaseExists = Test-Secret $Names.DatabaseSecret
    if ($RotateSecrets -or -not $databaseExists) {
        $pooledUrl = Read-PlainTextSecret `
            -Prompt 'Paste the Neon pooled PostgreSQL URL' `
            -File $NeonPooledUrlFile
        try {
            if ($pooledUrl -notmatch '^postgres(ql)?://') {
                throw 'The Neon pooled URL must start with postgresql://.'
            }
            Set-SecretValue -Name $Names.DatabaseSecret -Value $pooledUrl
        }
        finally {
            $pooledUrl = $null
        }
    }

    $migrationDatabaseExists = Test-Secret $Names.MigrationDatabaseSecret
    if ($RotateSecrets -or -not $migrationDatabaseExists) {
        $directUrl = Read-PlainTextSecret `
            -Prompt 'Paste the Neon direct PostgreSQL URL' `
            -File $NeonDirectUrlFile
        try {
            if ($directUrl -notmatch '^postgres(ql)?://') {
                throw 'The Neon direct URL must start with postgresql://.'
            }
            Set-SecretValue -Name $Names.MigrationDatabaseSecret -Value $directUrl
        }
        finally {
            $directUrl = $null
        }
    }

    $accessTokenExists = Test-Secret $Names.AccessTokenSecret
    if ($RotateSecrets -or -not $accessTokenExists) {
        if ($AccessTokenFile) {
            $token = Read-PlainTextSecret `
                -Prompt 'ClearFrame access token' `
                -File $AccessTokenFile
        }
        else {
            $token = New-RandomAccessToken
        }
        try {
            if ($token.Length -lt 32) {
                throw 'The ClearFrame access token must contain at least 32 characters.'
            }
            Set-SecretValue -Name $Names.AccessTokenSecret -Value $token
        }
        finally {
            $token = $null
        }
    }
}

function Ensure-ServiceAccount {
    param(
        [Parameter(Mandatory)][string] $Name,
        [Parameter(Mandatory)][string] $DisplayName
    )
    $email = "$Name@$script:ProjectId.iam.gserviceaccount.com"
    if (-not (Test-GcloudResource @(
        'iam', 'service-accounts', 'describe', $email,
        "--project=$script:ProjectId",
        '--quiet'
    ))) {
        Invoke-Gcloud @(
            'iam', 'service-accounts', 'create', $Name,
            "--display-name=$DisplayName",
            "--project=$script:ProjectId",
            '--quiet'
        )
    }
    return $email
}

function Grant-SecretAccess {
    param(
        [Parameter(Mandatory)][string] $Secret,
        [Parameter(Mandatory)][string] $ServiceAccount
    )
    Invoke-Gcloud @(
        'secrets', 'add-iam-policy-binding', $Secret,
        "--member=serviceAccount:$ServiceAccount",
        '--role=roles/secretmanager.secretAccessor',
        "--project=$script:ProjectId",
        '--quiet'
    )
}

function Grant-BucketRole {
    param(
        [Parameter(Mandatory)][string] $ServiceAccount,
        [Parameter(Mandatory)][string] $Role
    )
    Invoke-Gcloud @(
        'storage', 'buckets', 'add-iam-policy-binding', "gs://$BucketName",
        "--member=serviceAccount:$ServiceAccount",
        "--role=$Role",
        "--project=$script:ProjectId",
        '--quiet'
    )
}

function Grant-JobRole {
    param(
        [Parameter(Mandatory)][string] $Job,
        [Parameter(Mandatory)][string] $Member,
        [Parameter(Mandatory)][string] $Role
    )
    Invoke-Gcloud @(
        'run', 'jobs', 'add-iam-policy-binding', $Job,
        "--member=$Member",
        "--role=$Role",
        "--region=$script:Region",
        "--project=$script:ProjectId",
        '--quiet'
    )
}

function Deploy-WorkerJobs {
    param(
        [Parameter(Mandatory)][string] $ApiImage,
        [Parameter(Mandatory)][string] $GpuImage,
        [Parameter(Mandatory)][string] $WorkerAccount,
        [Parameter(Mandatory)][string] $MigrationAccount
    )

    $workerEnvironment = @(
        'CLEARFRAME_ENV=production',
        'CLEARFRAME_API_HOST=127.0.0.1',
        'CLEARFRAME_JOB_BACKEND=cloud_run',
        "CLEARFRAME_GCP_PROJECT_ID=$script:ProjectId",
        "CLEARFRAME_GCP_REGION=$script:Region",
        "CLEARFRAME_CLOUD_RUN_CPU_JOB=$($Names.CpuJob)",
        "CLEARFRAME_CLOUD_RUN_GPU_JOB=$($Names.GpuJob)",
        'CLEARFRAME_STORAGE_BACKEND=gcs',
        "CLEARFRAME_GCS_BUCKET=$BucketName",
        'CLEARFRAME_STORAGE_SCRATCH_ROOT=/tmp/clearframe',
        'CLEARFRAME_MODEL_REGISTRY_PATH=/app/configs/models/registry.yaml',
        "CLEARFRAME_MAX_UPLOAD_MB=$MaxUploadMb",
        'CLEARFRAME_MAX_DURATION_MINUTES=240',
        "CLEARFRAME_BUILD_ID=$ImageTag",
        'CLEARFRAME_LOG_LEVEL=INFO'
    ) -join ','

    Invoke-Gcloud @(
        'run', 'jobs', 'deploy', $Names.CpuJob,
        "--image=$ApiImage",
        "--region=$script:Region",
        "--service-account=$WorkerAccount",
        '--cpu=8',
        '--memory=16Gi',
        '--tasks=1',
        '--parallelism=1',
        '--max-retries=0',
        '--task-timeout=3600s',
        '--command=python',
        '--args=-m,clearframe.worker',
        "--set-env-vars=$workerEnvironment",
        "--set-secrets=CLEARFRAME_DATABASE_URL=$($Names.DatabaseSecret):latest",
        '--labels=app=clearframe,component=cpu-worker',
        "--project=$script:ProjectId",
        '--quiet'
    )

    Invoke-Gcloud @(
        'run', 'jobs', 'deploy', $Names.GpuJob,
        "--image=$GpuImage",
        "--region=$script:Region",
        "--service-account=$WorkerAccount",
        '--cpu=4',
        '--memory=16Gi',
        '--gpu=1',
        '--gpu-type=nvidia-l4',
        '--no-gpu-zonal-redundancy',
        '--tasks=1',
        '--parallelism=1',
        '--max-retries=0',
        '--task-timeout=3600s',
        '--command=python',
        '--args=-m,clearframe.worker',
        "--set-env-vars=$workerEnvironment,CLEARFRAME_REQUIRE_CUDA=true",
        "--set-secrets=CLEARFRAME_DATABASE_URL=$($Names.DatabaseSecret):latest",
        '--labels=app=clearframe,component=gpu-worker',
        "--project=$script:ProjectId",
        '--quiet'
    )

    Invoke-Gcloud @(
        'run', 'jobs', 'deploy', $Names.MigrationJob,
        "--image=$ApiImage",
        "--region=$script:Region",
        "--service-account=$MigrationAccount",
        '--cpu=1',
        '--memory=1Gi',
        '--tasks=1',
        '--parallelism=1',
        '--max-retries=0',
        '--task-timeout=600s',
        '--command=alembic',
        '--args=upgrade,head',
        '--set-env-vars=CLEARFRAME_ENV=production',
        "--set-secrets=CLEARFRAME_DATABASE_URL=$($Names.MigrationDatabaseSecret):latest",
        '--labels=app=clearframe,component=migration',
        "--project=$script:ProjectId",
        '--quiet'
    )

    Invoke-Gcloud @(
        'run', 'jobs', 'deploy', $Names.ReconcileJob,
        "--image=$ApiImage",
        "--region=$script:Region",
        "--service-account=$WorkerAccount",
        '--cpu=1',
        '--memory=1Gi',
        '--tasks=1',
        '--parallelism=1',
        '--max-retries=1',
        '--task-timeout=300s',
        '--command=python',
        '--args=-m,clearframe.reconcile',
        "--set-env-vars=$workerEnvironment",
        "--set-secrets=CLEARFRAME_DATABASE_URL=$($Names.DatabaseSecret):latest",
        '--labels=app=clearframe,component=reconciler',
        "--project=$script:ProjectId",
        '--quiet'
    )
}

function Deploy-Api {
    param(
        [Parameter(Mandatory)][string] $ApiImage,
        [Parameter(Mandatory)][string] $ApiAccount,
        [Parameter(Mandatory)][string] $WebOrigin
    )
    $environment = @(
        'CLEARFRAME_ENV=production',
        'CLEARFRAME_API_HOST=0.0.0.0',
        "CLEARFRAME_WEB_ORIGIN=$WebOrigin",
        'CLEARFRAME_JOB_BACKEND=cloud_run',
        "CLEARFRAME_GCP_PROJECT_ID=$script:ProjectId",
        "CLEARFRAME_GCP_REGION=$script:Region",
        "CLEARFRAME_CLOUD_RUN_CPU_JOB=$($Names.CpuJob)",
        "CLEARFRAME_CLOUD_RUN_GPU_JOB=$($Names.GpuJob)",
        'CLEARFRAME_STORAGE_BACKEND=gcs',
        "CLEARFRAME_GCS_BUCKET=$BucketName",
        "CLEARFRAME_GCS_SIGNING_SERVICE_ACCOUNT=$ApiAccount",
        'CLEARFRAME_STORAGE_SCRATCH_ROOT=/tmp/clearframe',
        "CLEARFRAME_MAX_UPLOAD_MB=$MaxUploadMb",
        'CLEARFRAME_MAX_DURATION_MINUTES=240',
        'CLEARFRAME_MAX_API_BODY_KB=1024',
        "CLEARFRAME_BUILD_ID=$ImageTag",
        'CLEARFRAME_LOG_LEVEL=INFO',
        'CLEARFRAME_RUN_MIGRATIONS=false'
    ) -join ','

    Invoke-Gcloud @(
        'run', 'deploy', $Names.ApiService,
        "--image=$ApiImage",
        "--region=$script:Region",
        '--platform=managed',
        "--service-account=$ApiAccount",
        '--cpu=1',
        '--memory=2Gi',
        '--concurrency=20',
        '--min=0',
        '--max=3',
        '--timeout=300s',
        '--port=8000',
        '--allow-unauthenticated',
        "--set-env-vars=$environment",
        "--set-secrets=CLEARFRAME_DATABASE_URL=$($Names.DatabaseSecret):latest,CLEARFRAME_ACCESS_TOKEN=$($Names.AccessTokenSecret):latest",
        '--labels=app=clearframe,component=api',
        "--project=$script:ProjectId",
        '--quiet'
    )
}

function Deploy-Web {
    param(
        [Parameter(Mandatory)][string] $WebImage,
        [Parameter(Mandatory)][string] $WebAccount,
        [Parameter(Mandatory)][string] $ApiUrl
    )
    Invoke-Gcloud @(
        'run', 'deploy', $Names.WebService,
        "--image=$WebImage",
        "--region=$script:Region",
        '--platform=managed',
        "--service-account=$WebAccount",
        '--cpu=1',
        '--memory=512Mi',
        '--concurrency=80',
        '--min=0',
        '--max=2',
        '--timeout=300s',
        '--port=8080',
        '--allow-unauthenticated',
        "--set-env-vars=CLEARFRAME_API_URL=$ApiUrl,NGINX_ENVSUBST_FILTER=CLEARFRAME_",
        "--set-secrets=CLEARFRAME_ACCESS_TOKEN=$($Names.AccessTokenSecret):latest",
        '--labels=app=clearframe,component=web',
        "--project=$script:ProjectId",
        '--quiet'
    )
}

function Set-BucketCors {
    param([Parameter(Mandatory)][string] $WebOrigin)
    $cors = @(
        @{
            origin = @($WebOrigin, 'http://localhost:5173')
            method = @('GET', 'HEAD', 'POST', 'PUT')
            responseHeader = @(
                'Content-Type',
                'Content-Range',
                'ETag',
                'Range',
                'x-goog-generation',
                'x-goog-hash',
                'x-guploader-uploadid'
            )
            maxAgeSeconds = 3600
        }
    )
    $corsFile = New-TemporaryJson $cors
    Invoke-Gcloud @(
        'storage', 'buckets', 'update', "gs://$BucketName",
        "--cors-file=$corsFile",
        '--uniform-bucket-level-access',
        '--public-access-prevention',
        "--project=$script:ProjectId",
        '--quiet'
    )
}

function Set-ReconcileSchedule {
    param([Parameter(Mandatory)][string] $SchedulerAccount)
    $uri = "https://run.googleapis.com/v2/projects/$script:ProjectId/locations/$script:Region/jobs/$($Names.ReconcileJob):run"
    $common = @(
        "--location=$script:Region",
        '--schedule=*/5 * * * *',
        '--time-zone=Etc/UTC',
        "--uri=$uri",
        '--http-method=POST',
        "--oauth-service-account-email=$SchedulerAccount",
        '--oauth-token-scope=https://www.googleapis.com/auth/cloud-platform',
        '--message-body={}',
        '--attempt-deadline=60s',
        '--max-retry-attempts=3',
        '--min-backoff=10s',
        '--max-backoff=60s',
        "--project=$script:ProjectId",
        '--quiet'
    )
    if (Test-GcloudResource @(
        'scheduler', 'jobs', 'describe', $Names.SchedulerJob,
        "--location=$script:Region",
        "--project=$script:ProjectId",
        '--quiet'
    )) {
        Invoke-Gcloud (
            @('scheduler', 'jobs', 'update', 'http', $Names.SchedulerJob) +
            $common +
            @('--update-headers=Content-Type=application/json')
        )
    }
    else {
        Invoke-Gcloud (
            @('scheduler', 'jobs', 'create', 'http', $Names.SchedulerJob) +
            $common +
            @('--headers=Content-Type=application/json')
        )
    }
}

try {
    Write-Step 'Checking Google Cloud prerequisites'
    if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
        throw 'Google Cloud CLI is not installed or is not on PATH.'
    }
    $activeAccount = Invoke-Gcloud -Capture @(
        'auth', 'list',
        '--filter=status:ACTIVE',
        '--format=value(account)'
    )
    if (-not $activeAccount) {
        throw 'No active gcloud account. Run gcloud auth login first.'
    }
    if (-not (Test-GcloudResource @(
        'projects', 'describe', $ProjectId,
        "--project=$ProjectId",
        '--quiet'
    ))) {
        throw "Google Cloud project '$ProjectId' does not exist or is not accessible."
    }

    $projectNumber = Invoke-Gcloud -Capture @(
        'projects', 'describe', $ProjectId,
        '--format=value(projectNumber)',
        "--project=$ProjectId"
    )
    if (-not $BucketName) {
        $BucketName = "$ProjectId-clearframe-media"
    }
    if ($BucketName.Length -gt 63) {
        throw 'BucketName cannot exceed 63 characters.'
    }
    if (-not $ImageTag) {
        $ImageTag = "manual-$([DateTime]::UtcNow.ToString('yyyyMMddHHmmss'))"
    }

    Write-Host "Account: $activeAccount"
    Write-Host "Project: $ProjectId ($projectNumber)"
    Write-Host "Region:  $Region"
    Write-Host "Bucket:  $BucketName"
    Write-Host "Image:   $ImageTag"

    Write-Step 'Enabling required Google Cloud APIs'
    $services = @(
        'artifactregistry.googleapis.com',
        'cloudbuild.googleapis.com',
        'cloudscheduler.googleapis.com',
        'compute.googleapis.com',
        'iam.googleapis.com',
        'iamcredentials.googleapis.com',
        'logging.googleapis.com',
        'monitoring.googleapis.com',
        'run.googleapis.com',
        'secretmanager.googleapis.com',
        'serviceusage.googleapis.com',
        'storage.googleapis.com'
    )
    Invoke-Gcloud (@(
        'services', 'enable'
    ) + $services + @(
        "--project=$ProjectId",
        '--quiet'
    ))

    Write-Step 'Creating Artifact Registry and storage'
    if (-not (Test-GcloudResource @(
        'artifacts', 'repositories', 'describe', $Repository,
        "--location=$Region",
        "--project=$ProjectId",
        '--quiet'
    ))) {
        Invoke-Gcloud @(
            'artifacts', 'repositories', 'create', $Repository,
            '--repository-format=docker',
            "--location=$Region",
            '--description=ClearFrame container images',
            "--project=$ProjectId",
            '--quiet'
        )
    }

    $lifecycle = @{
        rule = @(
            @{
                action = @{ type = 'Delete' }
                condition = @{
                    age = 2
                    matchesPrefix = @('tmp/uploads/')
                }
            }
        )
    }
    $lifecycleFile = New-TemporaryJson $lifecycle
    if (-not (Test-GcloudResource @(
        'storage', 'buckets', 'describe', "gs://$BucketName",
        "--project=$ProjectId",
        '--quiet'
    ))) {
        Invoke-Gcloud @(
            'storage', 'buckets', 'create', "gs://$BucketName",
            "--location=$Region",
            '--default-storage-class=STANDARD',
            '--uniform-bucket-level-access',
            '--public-access-prevention',
            "--lifecycle-file=$lifecycleFile",
            "--project=$ProjectId",
            '--quiet'
        )
    }
    else {
        $bucketLocation = Invoke-Gcloud -Capture @(
            'storage', 'buckets', 'describe', "gs://$BucketName",
            '--format=value(location)',
            "--project=$ProjectId"
        )
        if ($bucketLocation.ToLowerInvariant() -ne $Region.ToLowerInvariant()) {
            throw "Existing bucket '$BucketName' is in '$bucketLocation', not '$Region'."
        }
        Invoke-Gcloud @(
            'storage', 'buckets', 'update', "gs://$BucketName",
            '--uniform-bucket-level-access',
            '--public-access-prevention',
            "--lifecycle-file=$lifecycleFile",
            "--project=$ProjectId",
            '--quiet'
        )
    }

    Write-Step 'Creating service identities and secrets'
    $apiAccount = Ensure-ServiceAccount $Names.ApiAccount 'ClearFrame API'
    $webAccount = Ensure-ServiceAccount $Names.WebAccount 'ClearFrame web proxy'
    $workerAccount = Ensure-ServiceAccount $Names.WorkerAccount 'ClearFrame workers'
    $migrationAccount = Ensure-ServiceAccount $Names.MigrationAccount 'ClearFrame migrations'
    $schedulerAccount = Ensure-ServiceAccount $Names.SchedulerAccount 'ClearFrame scheduler'
    Initialize-Secrets

    Grant-SecretAccess $Names.DatabaseSecret $apiAccount
    Grant-SecretAccess $Names.DatabaseSecret $workerAccount
    Grant-SecretAccess $Names.MigrationDatabaseSecret $migrationAccount
    Grant-SecretAccess $Names.AccessTokenSecret $apiAccount
    Grant-SecretAccess $Names.AccessTokenSecret $webAccount

    foreach ($account in @($apiAccount, $workerAccount)) {
        Grant-BucketRole $account 'roles/storage.objectUser'
        Grant-BucketRole $account 'roles/storage.bucketViewer'
    }
    Invoke-Gcloud @(
        'iam', 'service-accounts', 'add-iam-policy-binding', $apiAccount,
        "--member=serviceAccount:$apiAccount",
        '--role=roles/iam.serviceAccountTokenCreator',
        "--project=$ProjectId",
        '--quiet'
    )

    $deployerType = if ($activeAccount.EndsWith('.gserviceaccount.com')) {
        'serviceAccount'
    }
    else {
        'user'
    }
    foreach ($account in @(
        $apiAccount,
        $webAccount,
        $workerAccount,
        $migrationAccount,
        $schedulerAccount
    )) {
        Invoke-Gcloud @(
            'iam', 'service-accounts', 'add-iam-policy-binding', $account,
            "--member=${deployerType}:$activeAccount",
            '--role=roles/iam.serviceAccountUser',
            "--project=$ProjectId",
            '--quiet'
        )
    }

    $buildAccount = Invoke-Gcloud -Capture @(
        'builds', 'get-default-service-account',
        "--project=$ProjectId",
        '--format=value(serviceAccountEmail)'
    )
    if ($buildAccount) {
        Invoke-Gcloud @(
            'projects', 'add-iam-policy-binding', $ProjectId,
            "--member=serviceAccount:$buildAccount",
            '--role=roles/cloudbuild.builds.builder',
            '--condition=None',
            '--quiet'
        ) | Out-Null
        Invoke-Gcloud @(
            'artifacts', 'repositories', 'add-iam-policy-binding', $Repository,
            "--location=$Region",
            "--member=serviceAccount:$buildAccount",
            '--role=roles/artifactregistry.writer',
            "--project=$ProjectId",
            '--quiet'
        )
    }
    $runServiceAgent = "service-$projectNumber@serverless-robot-prod.iam.gserviceaccount.com"
    Invoke-Gcloud @(
        'artifacts', 'repositories', 'add-iam-policy-binding', $Repository,
        "--location=$Region",
        "--member=serviceAccount:$runServiceAgent",
        '--role=roles/artifactregistry.reader',
        "--project=$ProjectId",
        '--quiet'
    )

    $imagePrefix = "$Region-docker.pkg.dev/$ProjectId/$Repository"
    $apiImage = "$imagePrefix/api:$ImageTag"
    $gpuImage = "$imagePrefix/gpu-worker:$ImageTag"
    $webImage = "$imagePrefix/web:$ImageTag"

    if (-not $SkipBuild) {
        Write-Step 'Building and publishing application images'
        $root = Resolve-Path (Join-Path $PSScriptRoot '..\..')
        Invoke-Gcloud @(
            'builds', 'submit', $root.Path,
            "--config=$PSScriptRoot\cloudbuild.yaml",
            "--substitutions=_API_IMAGE=$apiImage,_GPU_IMAGE=$gpuImage,_WEB_IMAGE=$webImage",
            '--timeout=2400s',
            "--project=$ProjectId",
            '--quiet'
        )
    }
    else {
        foreach ($image in @($apiImage, $gpuImage, $webImage)) {
            if (-not (Test-GcloudResource @(
                'artifacts', 'docker', 'images', 'describe', $image,
                "--project=$ProjectId",
                '--quiet'
            ))) {
                throw "Image does not exist: $image"
            }
        }
    }

    Write-Step 'Deploying worker, migration, and reconciliation jobs'
    Deploy-WorkerJobs $apiImage $gpuImage $workerAccount $migrationAccount

    foreach ($job in @($Names.CpuJob, $Names.GpuJob)) {
        Grant-JobRole $job "serviceAccount:$apiAccount" 'roles/run.developer'
        Grant-JobRole $job "serviceAccount:$workerAccount" 'roles/run.developer'
    }
    Grant-JobRole $Names.ReconcileJob "serviceAccount:$schedulerAccount" 'roles/run.invoker'

    Write-Step 'Applying database migrations'
    Invoke-Gcloud @(
        'run', 'jobs', 'execute', $Names.MigrationJob,
        "--region=$Region",
        '--wait',
        "--project=$ProjectId",
        '--quiet'
    )

    Write-Step 'Deploying API and web services'
    Deploy-Api $apiImage $apiAccount 'http://localhost:5173'
    $apiUrl = Invoke-Gcloud -Capture @(
        'run', 'services', 'describe', $Names.ApiService,
        "--region=$Region",
        '--format=value(status.url)',
        "--project=$ProjectId"
    )
    Deploy-Web $webImage $webAccount $apiUrl
    $webUrl = Invoke-Gcloud -Capture @(
        'run', 'services', 'describe', $Names.WebService,
        "--region=$Region",
        '--format=value(status.url)',
        "--project=$ProjectId"
    )
    Deploy-Api $apiImage $apiAccount $webUrl
    Set-BucketCors $webUrl

    Write-Step 'Scheduling reconciliation'
    Set-ReconcileSchedule $schedulerAccount

    Write-Step 'Verifying the deployed stack'
    $ready = Invoke-RestMethod -Uri "$apiUrl/api/ready" -TimeoutSec 60
    if ($ready.status -ne 'ready') {
        throw "API readiness check failed: $($ready | ConvertTo-Json -Compress)"
    }
    $capability = Invoke-RestMethod `
        -Uri "$webUrl/api/uploads/capability" `
        -TimeoutSec 60
    if ($capability.mode -ne 'resumable') {
        throw "Expected resumable cloud uploads, received '$($capability.mode)'."
    }

    Write-Host "`nClearFrame cloud deployment is ready." -ForegroundColor Green
    Write-Host "Web: $webUrl"
    Write-Host "API: $apiUrl"
    Write-Host "Storage: gs://$BucketName"
    Write-Host "Image tag: $ImageTag"
}
finally {
    foreach ($path in $script:TemporaryFiles) {
        if (Test-Path -LiteralPath $path) {
            Remove-Item -LiteralPath $path -Force
        }
    }
}
