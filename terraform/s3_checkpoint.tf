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

# Deliberately PutObject + GetObject (+ read-only ListBucket, see below) --
# no DeleteObject, DeleteObjectVersion, PutObjectRetention,
# BypassGovernanceRetention, or bucket-policy actions. The app role can
# append and read; it cannot erase or weaken retention on anything already
# written. Deleting/retention changes require the account's own (human)
# credentials, outside this role.
#
# The explicit Deny statement below is load-bearing, not decorative. It was
# added when this role still carried PowerUserAccess, which independently
# granted full s3:DeleteObject on every bucket in the account -- confirmed by
# reproduction: `aws s3 rm` on this bucket succeeded before the Deny existed.
# PowerUserAccess has since moved to the separate, root-assumable
# terraform-operator role (iam_terraform_access.tf, F6-R4-A finding B), so the
# Deny is no longer counteracting a live Allow -- but it stays, because an
# explicit Deny outranks any Allow from any policy ever attached to this role
# in future, and that durability is precisely what the anchor's external-trust
# claim rests on.
data "aws_iam_policy_document" "provenance_checkpoint" {
  statement {
    sid       = "AppendProvenanceCheckpoint"
    actions   = ["s3:PutObject", "s3:GetObject"]
    resources = ["${aws_s3_bucket.provenance_checkpoint.arn}/checkpoints/*"]
  }

  # ABSENCE MUST BE OBSERVABLE, or the control fails closed on every single read.
  #
  # S3 answers GetObject for a key that does not exist with 403 AccessDenied --
  # not 404 NoSuchKey -- unless the caller holds s3:ListBucket on the bucket.
  # S3CheckpointAnchor treats anything other than NoSuchKey as UNVERIFIABLE and
  # holds the release, deliberately: folding 403 into "this store was never
  # anchored" is exactly how one permission regression would silently switch the
  # whole control off while every release kept certifying an external anchor.
  #
  # So without this statement the first read of any store raises
  # AnchorUnavailable and every encounter holds. That is not hypothetical: it
  # was PROVED live from the instance role -- once finding B removed
  # PowerUserAccess (which had been supplying s3:ListBucket bucket-wide as a
  # side effect), 9 of the 16 live anchor regressions failed with AccessDenied
  # on a nonexistent key. The fix belongs here, not in the client: "may this
  # principal observe that an object does not exist" is an authorization
  # question, and answering it by treating 403 as absent is the silent-empty-
  # success failure this whole module exists to refuse.
  #
  # Deliberately UNCONDITIONAL rather than scoped with an s3:prefix condition.
  # The 404-vs-403 authorization check S3 performs for GetObject does not carry
  # an s3:prefix request context, so a prefix-conditioned grant does not satisfy
  # it -- the release path would keep failing closed, with the policy looking
  # correct. Listing is read-only metadata over a bucket that holds nothing but
  # checkpoint objects, and it confers no ability to write, overwrite or delete;
  # the Deny below and the prefix scoping on Get/PutObject are untouched.
  statement {
    sid       = "ObserveProvenanceCheckpointAbsence"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.provenance_checkpoint.arn]
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

# Write-once for sequence records, enforced at the storage layer -- not just
# by the Python client always sending IfNoneMatch (F6-R4-A2, issue #6). The
# app role's identity-based policy above grants unconditional PutObject over
# the whole checkpoints/ prefix (S3 has no IAM condition key that means "only
# when this specific key doesn't already exist" -- IfNoneMatch is a REQUEST
# header, not a resource attribute an IAM policy can inspect); a second
# process using the same instance-role credentials could call PutObject
# directly, skip the header, and silently overwrite an already-anchored
# sequence's current version. Versioning keeps the prior bytes recoverable,
# but the release reader reads the CURRENT version, so an unconditional
# overwrite is functionally indistinguishable from tampering to that reader.
#
# This is a resource-based bucket policy (not another identity-based
# statement) because request-header inspection -- "was IfNoneMatch actually
# sent" -- is exactly what S3's documented conditional-write enforcement
# pattern requires:
# https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes-enforce.html
#
# Scoped to */seq/* only. The head.json pointer is deliberately, permanently
# mutable (checkpoint.py's own docstring: "NOT trusted... a starting FLOOR")
# and must stay overwritable without a conditional header.
data "aws_iam_policy_document" "provenance_checkpoint_bucket_policy" {
  statement {
    sid       = "RequireConditionalWriteForSequenceRecords"
    effect    = "Deny"
    actions   = ["s3:PutObject"]
    resources = ["${aws_s3_bucket.provenance_checkpoint.arn}/checkpoints/*/seq/*"]
    principals {
      type        = "*"
      identifiers = ["*"]
    }
    condition {
      test     = "Null"
      variable = "s3:if-none-match"
      values   = ["true"]
    }
  }
}

resource "aws_s3_bucket_policy" "provenance_checkpoint" {
  bucket = aws_s3_bucket.provenance_checkpoint.id
  policy = data.aws_iam_policy_document.provenance_checkpoint_bucket_policy.json
}

output "provenance_checkpoint_bucket" {
  value = aws_s3_bucket.provenance_checkpoint.bucket
}
