# Terraform-capable access for infra work done from the EC2 box, WITHOUT
# widening the app's own always-on runtime identity.
#
# History: this file originally attached PowerUserAccess + self-IAM-management
# directly to aws_iam_role.app (the role the running application always has,
# via instance profile). Codex's round-4 re-review (issue #6, F6-R4-A finding
# B) correctly identified that as a real hole, not a theoretical one: the same
# role the checkpoint-anchor Deny (s3_checkpoint.tf) exists to constrain could
# just delete or rewrite that Deny's own policy (iam:DeleteRolePolicy /
# PutRolePolicy on itself), then use PowerUserAccess's S3 delete rights to
# erase the checkpoint objects it was supposed to be unable to touch. A Deny
# is not a boundary against a principal that can edit the policy containing
# the Deny.
#
# Fix: a SEPARATE role, not attached to the instance profile, assumable only
# via an explicit human bootstrap action -- never by the app's own
# instance-metadata-sourced credentials, which are not a trusted principal in
# its trust policy. The always-on app process can never reach this role's
# permissions no matter what code runs under it.
#
# CORRECTED TWICE, both times by a live negative-AND-positive test the first
# version skipped (issue #6, F6-R4-A -> finding B -> F6-R4-B):
#
# 1. The original trust policy named the account-root ARN as Principal,
#    intending "root-user-only." Per AWS's own docs
#    (reference_policies_elements_principal.html) that Principal form
#    delegates trust to the ACCOUNT, not literally the root user -- any
#    principal later granted sts:AssumeRole on this role's ARN in their own
#    policy could assume it too. Not exploitable in practice (nothing else
#    held that grant), but not the boundary claimed.
# 2. Tightening it with an `aws:PrincipalArn == root` condition (this file's
#    prior revision) turned out to fix nothing, because verifying it live
#    (root actually attempting the assume, not just confirming the app role
#    is refused) surfaced a harder fact: **AWS unconditionally refuses to let
#    the root user assume ANY role** ("Roles may not be assumed by root
#    accounts") -- not a trust-policy-configurable restriction, and true even
#    for a session-token derived from root (tested live). The entire
#    root-assumable design could never have worked; only the negative half
#    (app role refused) had ever actually been checked.
#
# Real fix: a dedicated IAM USER whose ONLY permission is sts:AssumeRole on
# this role -- a valid principal type for AssumeRole, unlike root. The user's
# own long-term access key is deliberately narrow (assume-only, nothing
# else), so a leaked key's blast radius is "can start a <=1h PowerUserAccess
# session," not "has PowerUserAccess directly." To run terraform from the
# box: fetch that key ONCE (`terraform output -raw terraform_operator_access_key_id`
# / `-raw terraform_operator_secret_access_key`, store it in a password
# manager, not in this repo or on the box), then from local:
# `AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... aws sts assume-role
# --role-arn <this role's arn> --role-session-name terraform`, and export
# the resulting SHORT-LIVED credentials into one SSH session only -- never
# written to disk on the box, never available to the always-running app.

data "aws_caller_identity" "current" {}

resource "aws_iam_user" "terraform_operator" {
  name = "${var.project_name}-terraform-operator-user"
}

resource "aws_iam_access_key" "terraform_operator" {
  user = aws_iam_user.terraform_operator.name
}

# Deliberately the ONLY permission this user has: mint a session on the
# actually-privileged role below. Nothing else -- this user is a bootstrap
# key, not an admin identity in its own right.
data "aws_iam_policy_document" "terraform_operator_user_assume_only" {
  statement {
    actions   = ["sts:AssumeRole"]
    resources = [aws_iam_role.terraform_operator.arn]
  }
}

resource "aws_iam_user_policy" "terraform_operator_assume_only" {
  name   = "${var.project_name}-terraform-operator-assume-only"
  user   = aws_iam_user.terraform_operator.name
  policy = data.aws_iam_policy_document.terraform_operator_user_assume_only.json
}

data "aws_iam_policy_document" "terraform_operator_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "AWS"
      identifiers = [aws_iam_user.terraform_operator.arn]
    }
  }
}

resource "aws_iam_role" "terraform_operator" {
  name                 = "${var.project_name}-terraform-operator"
  assume_role_policy   = data.aws_iam_policy_document.terraform_operator_assume.json
  max_session_duration = 3600
}

resource "aws_iam_role_policy_attachment" "terraform_operator_poweruser" {
  role       = aws_iam_role.terraform_operator.name
  policy_arn = "arn:aws:iam::aws:policy/PowerUserAccess"
}

# PowerUserAccess deliberately excludes IAM. Scope IAM management to exactly
# the principals this project's terraform config manages (the app role and
# this operator role itself) -- not blanket iam:* over every principal in the
# account.
data "aws_iam_policy_document" "terraform_operator_iam_management" {
  statement {
    sid = "ManageProjectRoles"
    actions = [
      "iam:GetRole",
      "iam:GetRolePolicy",
      "iam:PutRolePolicy",
      "iam:DeleteRolePolicy",
      "iam:ListRolePolicies",
      "iam:ListAttachedRolePolicies",
      "iam:AttachRolePolicy",
      "iam:DetachRolePolicy",
      "iam:TagRole",
      "iam:UntagRole",
    ]
    resources = [
      aws_iam_role.app.arn,
      aws_iam_role.terraform_operator.arn,
    ]
  }

  statement {
    sid       = "ReadProjectInstanceProfile"
    actions   = ["iam:GetInstanceProfile"]
    resources = [aws_iam_instance_profile.app.arn]
  }
}

resource "aws_iam_role_policy" "terraform_operator_iam_management" {
  name   = "${var.project_name}-terraform-operator-iam-management"
  role   = aws_iam_role.terraform_operator.id
  policy = data.aws_iam_policy_document.terraform_operator_iam_management.json
}

output "terraform_operator_role_arn" {
  value = aws_iam_role.terraform_operator.arn
}

# Sensitive: fetch once with `-raw`, store in a password manager, never
# commit or leave sitting in shell history / a file on the box.
output "terraform_operator_access_key_id" {
  value = aws_iam_access_key.terraform_operator.id
}

output "terraform_operator_secret_access_key" {
  value     = aws_iam_access_key.terraform_operator.secret
  sensitive = true
}
