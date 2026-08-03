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
    ANTHROPIC_API_KEY             = var.anthropic_api_key
    OPENAI_API_KEY                = var.openai_api_key
    STEDI_API_KEY                 = var.stedi_api_key
    STEDI_ELIGIBILITY_URL         = var.stedi_eligibility_url
    LLM_PROVIDER                  = var.llm_provider
    OPENAI_MODEL                  = var.openai_model
    CLAUDE_MODEL                  = var.claude_model
    CLAUDE_EFFORT                 = var.claude_effort
    AUTHORIZED_MODEL_PROVIDERS    = join(",", sort(tolist(var.authorized_model_providers)))
    CODING_EXECUTION_PROFILES     = length(var.coding_execution_profiles) > 0 ? jsonencode(var.coding_execution_profiles) : ""
    MIN_INDEPENDENT_MODEL_DOMAINS = tostring(var.min_independent_model_domains)
    CODER_ADJUDICATOR_MODEL       = var.coder_adjudicator_model
    CLINICAL_AUDITOR_MODEL        = var.clinical_auditor_model
    CLINICAL_AUDIT_PASSES         = "2"
    CODER_ADJUDICATION_PASSES     = "2"
    LOG_LEVEL                     = var.log_level
    QDRANT_URL                    = "http://qdrant:6333"
    NOTES_DIR                     = "/app/attachments"
  })

  lifecycle {
    precondition {
      condition = length(setsubtract(
        toset([for profile in var.coding_execution_profiles : profile.provider]),
        var.authorized_model_providers,
      )) == 0
      error_message = "Every coding execution profile provider must be explicitly authorized."
    }

    precondition {
      condition = (
        length(var.coding_execution_profiles) == 0 ||
        length(toset([for profile in var.coding_execution_profiles : profile.provider])) >= var.min_independent_model_domains
      )
      error_message = "Configured coding profiles must span the required number of independent provider domains."
    }

    precondition {
      condition = (
        var.clinical_auditor_model != var.coder_adjudicator_model &&
        alltrue([
          for profile in var.coding_execution_profiles :
          profile.provider != var.llm_provider || profile.model != var.clinical_auditor_model
        ])
      )
      error_message = "The clinical auditor model must differ from primary-provider coding and adjudication models."
    }
  }
}
