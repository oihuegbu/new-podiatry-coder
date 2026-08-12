# Single JSON secret holding the full runtime .env — simpler to rotate/fetch
# than one Secrets Manager entry per variable, and keeps API keys out of
# user_data (which is visible via the EC2 instance metadata API to anyone
# with describe-instance-attribute permissions).
#
# THIS MAP MUST BE THE COMPLETE RUNTIME .env, not a subset. Every apply writes
# a brand-new secret version and moves AWSCURRENT to it, so a key that is live
# but absent here is DELETED from the deployed configuration by the next apply
# +`refresh-secrets.sh` — silently, since a missing environment variable reads
# as "use the default" almost everywhere. That had already happened: the live
# secret carried 22 keys and this file encoded 10, so any apply would have
# dropped model routing, consistency and audit-pass configuration. The 12
# missing ones are now declared in variables.tf with the live values as
# defaults. When adding a runtime setting, add it in BOTH places.

resource "aws_secretsmanager_secret" "app_env" {
  name                    = "${var.project_name}-app-env"
  description             = "Runtime .env for the podiatry medical coder pipeline"
  recovery_window_in_days = 0 # fully delete on destroy, no 7-30 day hold
}

resource "aws_secretsmanager_secret_version" "app_env" {
  secret_id = aws_secretsmanager_secret.app_env.id
  secret_string = jsonencode({
    ANTHROPIC_API_KEY     = var.anthropic_api_key
    OPENAI_API_KEY        = var.openai_api_key
    STEDI_API_KEY         = var.stedi_api_key
    STEDI_ELIGIBILITY_URL = var.stedi_eligibility_url
    LLM_PROVIDER          = var.llm_provider
    CLAUDE_MODEL          = var.claude_model
    CLAUDE_EFFORT         = var.claude_effort
    LOG_LEVEL             = var.log_level
    QDRANT_URL            = "http://qdrant:6333"
    NOTES_DIR             = "/app/attachments"

    # Independent-execution / multi-pass runtime configuration. These were live
    # in the secret but missing from this map (see the header note); they are
    # rendered as STRINGS because every value in this secret is a string -- the
    # secret is materialised into a shell .env line by line, and a JSON number
    # here would change the encoded value and make every apply rewrite it.
    OPENAI_MODEL                  = var.openai_model
    OPENAI_REASONING_EFFORT       = var.openai_reasoning_effort
    AUTHORIZED_MODEL_PROVIDERS    = join(",", var.authorized_model_providers)
    CODING_EXECUTION_PROFILES     = jsonencode(var.coding_execution_profiles)
    MIN_INDEPENDENT_MODEL_DOMAINS = tostring(var.min_independent_model_domains)
    CONSISTENCY_MODE              = var.consistency_mode
    CONSISTENCY_RUNS              = tostring(var.consistency_runs)
    CONSISTENCY_WORKERS           = tostring(var.consistency_workers)
    CODER_ADJUDICATOR_MODEL       = var.coder_adjudicator_model
    CODER_ADJUDICATION_PASSES     = tostring(var.coder_adjudication_passes)
    CLINICAL_AUDITOR_MODEL        = var.clinical_auditor_model
    CLINICAL_AUDIT_PASSES         = tostring(var.clinical_audit_passes)

    # Terminal-head checkpoint anchor (claude_coder/checkpoint.py, issue #6
    # F6-R4-A). DERIVED from the bucket resource, never a literal: the bucket
    # name carries a random suffix, so a hand-copied value would point a clean
    # `terraform apply` at a bucket that no longer exists -- and because an
    # unreachable anchor correctly fails closed, that would present as a total
    # release outage far from its cause. Reading it from the resource means the
    # runtime is always pointed at the bucket this same config created.
    #
    # The matching PROVENANCE_CHECKPOINT_REQUIRED=1 is deliberately NOT here:
    # it lives in docker-compose.yml so a stale/unrefreshed .env can only ever
    # produce "required but unanchored" (which holds the release), never
    # "silently unanchored".
    #
    # The prefix must be one the app role is granted (s3_checkpoint.tf scopes
    # its PutObject/GetObject to checkpoints/*); a bucket-root spec is refused
    # at configuration time by S3CheckpointAnchor.
    PROVENANCE_CHECKPOINT_ANCHOR = "s3://${aws_s3_bucket.provenance_checkpoint.bucket}/checkpoints?region=${var.aws_region}"
  })
}
