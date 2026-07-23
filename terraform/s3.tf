resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "aws_s3_bucket" "artifacts" {
  bucket = "${var.project_name}-artifacts-${random_id.bucket_suffix.hex}"
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Package the app source (code + reference data JSON that compliance.db and
# code_reference.py are built from) — excludes .git history, regenerable
# caches/output, uploaded runtime notes, and .env (secrets go through Secrets
# Manager instead).
data "archive_file" "app_source" {
  type        = "zip"
  source_dir  = "${path.module}/.."
  output_path = "${path.module}/.build/app-source.zip"

  excludes = [
    ".git",
    ".git/**",
    "terraform",
    "terraform/**",
    "data/result_cache",
    "data/result_cache/**",
    "data/qdrant_store",
    "data/qdrant_store/**",
    # compliance.db is a build artifact (rebuilt/refreshed from the JSON
    # sources on the instance) — shipping a laptop-built copy both bloats
    # the zip and lets `unzip -o` on redeploy REGRESS the instance's DB to
    # whatever vintage the deploying machine happened to have, undoing any
    # cron-refreshed data. The instance builds/updates its own.
    "data/compliance.db",
    # doctors_notes is runtime data uploaded directly to the instance, not
    # app source. Bundling it meant every redeploy's `unzip -o` rewrote
    # every PDF on EC2, firing inotify events that made note-watcher.service
    # kick off its own full batch run on every code deploy — colliding with
    # whatever run was already in progress and silently dropping/overwriting
    # result files.
    "doctors_notes",
    "doctors_notes/**",
    "output",
    "output/**",
    "logs",
    "logs/**",
    ".env",
    ".DS_Store",
  ]
}

resource "aws_s3_object" "app_source" {
  bucket = aws_s3_bucket.artifacts.id
  key    = "app-source-${data.archive_file.app_source.output_md5}.zip"
  source = data.archive_file.app_source.output_path
  etag   = data.archive_file.app_source.output_md5
}
