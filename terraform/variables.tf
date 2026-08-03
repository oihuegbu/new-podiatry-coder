variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Name prefix for all resources"
  type        = string
  default     = "podiatry-coder"
}

variable "instance_type" {
  description = "EC2 instance type for the pipeline host. r6i.xlarge = 4 vCPU / 32GB, ~2x headroom over the ~8-9GB measured peak memory."
  type        = string
  default     = "r6i.xlarge"
}

variable "root_volume_gb" {
  description = "Root EBS volume size in GB (holds Docker images, HF model cache, compliance.db, Qdrant store)"
  type        = number
  default     = 100
}

variable "ssh_allowed_cidr" {
  description = "CIDR block allowed to SSH into the instance (your IP, /32)"
  type        = string
}

# --- secrets (sensitive, provide via terraform.tfvars which is gitignored) ---

variable "anthropic_api_key" {
  description = "Anthropic API key for Claude Opus vision extraction + coding"
  type        = string
  sensitive   = true
}

variable "openai_api_key" {
  description = "OpenAI API key for explicitly authorized independent coding profiles"
  type        = string
  sensitive   = true
  default     = ""
}

variable "stedi_api_key" {
  description = "Stedi API key for eligibility checks"
  type        = string
  sensitive   = true
  default     = ""
}

# --- non-secret app config (mirrors .env defaults) ---

variable "llm_provider" {
  type    = string
  default = "claude"
}

variable "openai_model" {
  description = "OpenAI model used by an explicitly configured OpenAI execution profile"
  type        = string
  default     = "gpt-5.6-sol"
}

variable "openai_reasoning_effort" {
  description = "Reasoning effort for OpenAI coding and adjudication calls"
  type        = string
  default     = "high"

  validation {
    condition     = contains(["none", "low", "medium", "high", "xhigh", "max"], var.openai_reasoning_effort)
    error_message = "openai_reasoning_effort must be none, low, medium, high, xhigh, or max."
  }
}

variable "authorized_model_providers" {
  description = "Providers explicitly approved to receive clinical-note data"
  type        = set(string)
  default     = ["claude"]

  validation {
    condition = (
      length(var.authorized_model_providers) > 0 &&
      alltrue([for provider in var.authorized_model_providers : contains(["claude", "openai"], provider)])
    )
    error_message = "authorized_model_providers must contain only claude and/or openai."
  }
}

variable "coding_execution_profiles" {
  description = "Ordered, explicitly authorized provider/model profiles used for independent coding runs"
  type = list(object({
    profile_id = string
    provider   = string
    model      = string
  }))
  default = []

  validation {
    condition = alltrue([
      for profile in var.coding_execution_profiles :
      contains(["claude", "openai"], profile.provider) &&
      trimspace(profile.profile_id) != "" &&
      trimspace(profile.model) != ""
    ])
    error_message = "Every coding profile must name a claude/openai provider, profile_id, and model."
  }
}

variable "min_independent_model_domains" {
  description = "Minimum provider-level independence domains required for autonomous release"
  type        = number
  default     = 2

  validation {
    condition     = var.min_independent_model_domains >= 2
    error_message = "Autonomous release requires at least two independent model domains."
  }
}

variable "coder_adjudicator_model" {
  description = "Primary-provider model used for deterministic-dispute adjudication"
  type        = string
  default     = "claude-fable-5"
}

variable "clinical_auditor_model" {
  description = "Primary-provider model reserved for whole-claim clinical audit"
  type        = string
  default     = "claude-sonnet-5"
}

variable "claude_model" {
  type    = string
  default = "claude-opus-4-8"
}

variable "claude_effort" {
  type    = string
  default = "high"
}

variable "log_level" {
  type    = string
  default = "INFO"
}

variable "stedi_eligibility_url" {
  type    = string
  default = "https://healthcare.us.stedi.com/2024-04-01/change/medicalnetwork/eligibility/v3"
}
