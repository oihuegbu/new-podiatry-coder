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
# by explicit human action (the account root) -- never by the app's own
# instance-metadata-sourced credentials, which are not a trusted principal in
# its trust policy. The always-on app process can never reach this role's
# permissions no matter what code runs under it. To run terraform from the
# box: generate short-lived credentials locally (`aws sts assume-role
# --role-arn <this role's arn> --role-session-name terraform`, using the
# account root credentials already used to bootstrap this project) and export
# them into that one SSH session only -- never written to disk on the box,
# never available to the always-running app.

data "aws_caller_identity" "current" {}

data "aws_iam_policy_document" "terraform_operator_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "AWS"
      identifiers = ["arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"]
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
