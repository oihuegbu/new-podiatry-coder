# Lets the EC2 box run terraform itself (product-owner decision, 2026-08-11:
# operator-driven infra work should happen from the box like everything
# else, not from the laptop). This is a real widening of the app role's
# permissions beyond what the running application needs -- PowerUserAccess
# grants broad create/modify access to almost all AWS services (explicitly
# EXCLUDING IAM/Organizations management of other principals), plus a
# narrow self-only statement so terraform can manage this role's *own*
# inline policies and attachments (which PowerUserAccess deliberately
# excludes). It cannot create/modify IAM users, groups, or other roles.

resource "aws_iam_role_policy_attachment" "app_terraform_poweruser" {
  role       = aws_iam_role.app.name
  policy_arn = "arn:aws:iam::aws:policy/PowerUserAccess"
}

data "aws_iam_policy_document" "app_self_iam_management" {
  statement {
    sid = "ManageOwnRolePolicies"
    actions = [
      "iam:GetRole",
      "iam:GetRolePolicy",
      "iam:PutRolePolicy",
      "iam:DeleteRolePolicy",
      "iam:ListRolePolicies",
      "iam:ListAttachedRolePolicies",
      "iam:AttachRolePolicy",
      "iam:DetachRolePolicy",
    ]
    resources = [aws_iam_role.app.arn]
  }

  statement {
    sid       = "ReadOwnInstanceProfile"
    actions   = ["iam:GetInstanceProfile"]
    resources = [aws_iam_instance_profile.app.arn]
  }
}

resource "aws_iam_role_policy" "app_self_iam_management" {
  name   = "${var.project_name}-self-iam-management"
  role   = aws_iam_role.app.id
  policy = data.aws_iam_policy_document.app_self_iam_management.json
}
