# Single JSON secret holding the full runtime .env — simpler to rotate/fetch
# than one Secrets Manager entry per variable, and keeps API keys out of
# user_data (which is visible via the EC2 instance metadata API to anyone
# with describe-instance-attribute permissions).

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
  })
}
