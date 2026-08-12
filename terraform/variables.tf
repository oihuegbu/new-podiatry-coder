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
  description = "OpenAI API key (fallback/compat, may be unused)"
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

variable "claude_model" {
  type    = string
  default = "claude-sonnet-5"
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

# --- independent-execution / multi-pass config ---
#
# These were being set on the deployed instance through the runtime secret but
# were never declared here, so `terraform.tfvars` carried values for undeclared
# variables (terraform validate warned about each) and `secrets.tf` encoded a
# STRICT SUBSET of what the live secret actually held. The consequence was a
# latent landmine: any `terraform apply` would have written a new secret version
# from the 10 keys in config, silently dropping the 12 below -- and the next
# `refresh-secrets.sh` would have erased them from /opt/app/.env, downgrading
# model routing, consistency and audit-pass configuration with no error. Found
# while wiring the checkpoint anchor through the same secret (issue #6 F6-R4-A);
# the defaults here are the values the live secret holds today, so a clean
# `terraform apply` now REPRODUCES the running configuration instead of
# truncating it.

variable "openai_model" {
  type    = string
  default = "gpt-5.6-sol"
}

variable "openai_reasoning_effort" {
  type    = string
  default = "high"
}

variable "authorized_model_providers" {
  description = "Providers explicitly authorized to receive PHI, as an allowlist."
  type        = list(string)
  default     = ["claude", "openai"]
}

variable "coding_execution_profiles" {
  description = "Independent execution identities used for corroboration."
  type = list(object({
    profile_id = string
    provider   = string
    model      = string
  }))
  default = [
    { profile_id = "primary", provider = "claude", model = "claude-opus-4-8" },
    { profile_id = "corroborator", provider = "openai", model = "gpt-5.6-sol" },
  ]
}

variable "min_independent_model_domains" {
  type    = number
  default = 2
}

variable "consistency_mode" {
  type    = string
  default = "adaptive"
}

variable "consistency_runs" {
  type    = number
  default = 3
}

variable "consistency_workers" {
  type    = number
  default = 12
}

variable "coder_adjudicator_model" {
  type    = string
  default = "claude-fable-5"
}

variable "coder_adjudication_passes" {
  type    = number
  default = 2
}

variable "clinical_auditor_model" {
  type    = string
  default = "claude-sonnet-5"
}

variable "clinical_audit_passes" {
  type    = number
  default = 2
}
