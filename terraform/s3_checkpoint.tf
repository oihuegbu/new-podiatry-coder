# External anchor for the provenance audit chain's terminal-head checkpoint
# (issue #6, F6-R4-A). The EC2 app role can WRITE new checkpoint records here
# but cannot delete or overwrite prior ones -- that separation of privilege
# is the actual security property this bucket exists for: the same process
# that appends to the local audit DB/witness journal must not also be able
# to erase the external record that would catch a consistent local
# truncation of both.
#
# Object Lock is enabled on the bucket (a one-time, unrevocable capability
# flag) so a real WORM retention policy can be applied later without
# recreating the bucket. No default retention rule is set here -- that is a
# deliberate follow-up decision, not something to default into, since a
# default retention rule would make every object undeletable (even by the
# account owner) for the configured period.

resource "aws_s3_bucket" "provenance_checkpoint" {
  bucket = "${var.project_name}-provenance-checkpoint-${random_id.bucket_suffix.hex}"

  object_lock_enabled = true
}

resource "aws_s3_bucket_versioning" "provenance_checkpoint" {
  bucket = aws_s3_bucket.provenance_checkpoint.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "provenance_checkpoint" {
  bucket                  = aws_s3_bucket.provenance_checkpoint.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "provenance_checkpoint" {
  bucket = aws_s3_bucket.provenance_checkpoint.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Deliberately PutObject + GetObject only -- no DeleteObject,
# DeleteObjectVersion, PutObjectRetention, BypassGovernanceRetention, or
# bucket-policy actions. The app role can append and read; it cannot erase
# or weaken retention on anything already written. Deleting/retention
# changes require the account's own (human) credentials, outside this role.
#
# The explicit Deny statement below is load-bearing, not decorative: this
# role also holds PowerUserAccess (see iam_terraform_access.tf, granted so
# the box can run terraform for infra work) which independently grants full
# s3:DeleteObject etc. on every bucket in the account, including this one.
# IAM allows are additive across policies, so without an explicit Deny here
# PowerUserAccess would silently restore full delete/overwrite/retention-
# bypass access to the one bucket this whole fix depends on being append-
# only -- confirmed by reproduction: `aws s3 rm` on this bucket succeeded
# before this Deny was added. An explicit Deny always wins over any Allow
# from any other attached policy, so this holds regardless of what else is
# ever attached to this role.
data "aws_iam_policy_document" "provenance_checkpoint" {
  statement {
    sid       = "AppendProvenanceCheckpoint"
    actions   = ["s3:PutObject", "s3:GetObject"]
    resources = ["${aws_s3_bucket.provenance_checkpoint.arn}/checkpoints/*"]
  }

  statement {
    sid    = "DenyProvenanceCheckpointTamper"
    effect = "Deny"
    actions = [
      "s3:DeleteObject",
      "s3:DeleteObjectVersion",
      "s3:DeleteBucket",
      "s3:PutBucketPolicy",
      "s3:DeleteBucketPolicy",
      "s3:PutBucketAcl",
      "s3:PutObjectAcl",
      "s3:PutBucketVersioning",
      "s3:PutBucketObjectLockConfiguration",
      "s3:PutObjectRetention",
      "s3:PutObjectLegalHold",
      "s3:BypassGovernanceRetention",
      "s3:PutEncryptionConfiguration",
      "s3:PutBucketPublicAccessBlock",
    ]
    resources = [
      aws_s3_bucket.provenance_checkpoint.arn,
      "${aws_s3_bucket.provenance_checkpoint.arn}/*",
    ]
  }
}

resource "aws_iam_role_policy" "provenance_checkpoint" {
  name   = "${var.project_name}-provenance-checkpoint-policy"
  role   = aws_iam_role.app.id
  policy = data.aws_iam_policy_document.provenance_checkpoint.json
}

output "provenance_checkpoint_bucket" {
  value = aws_s3_bucket.provenance_checkpoint.bucket
}
