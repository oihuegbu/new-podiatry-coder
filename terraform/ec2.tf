data "aws_ami" "al2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

locals {
  user_data = templatefile("${path.module}/templates/user_data.sh.tftpl", {
    bucket     = aws_s3_bucket.artifacts.bucket
    key        = aws_s3_object.app_source.key
    region     = var.aws_region
    secret_arn = aws_secretsmanager_secret.app_env.arn
  })
}

resource "aws_instance" "app" {
  ami                    = data.aws_ami.al2023.id
  instance_type          = var.instance_type
  key_name               = aws_key_pair.app.key_name
  subnet_id              = data.aws_subnets.default.ids[0]
  vpc_security_group_ids = [aws_security_group.app.id]
  iam_instance_profile   = aws_iam_instance_profile.app.name
  user_data              = local.user_data

  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.root_volume_gb
    encrypted             = true
    delete_on_termination = true
  }

  # EBS-backed root volume survives stop/start (only storage is billed while
  # stopped) — matches the "stop between runs" cost model.
  instance_initiated_shutdown_behavior = "stop"

  tags = {
    Name = "${var.project_name}-host"
  }

  lifecycle {
    # user_data embeds the S3 source object's content-hashed key, so every
    # code change would otherwise change user_data and force a replace —
    # destroying the EBS root volume and its ~60-90 min Qdrant embedding
    # volume for what should be a routine code deploy. user_data only runs
    # once at first boot anyway; code updates ship via the documented SSH
    # redeploy path (terraform/README.md), not by re-triggering user_data.
    # AMI is ignored for the same reason — a newer AL2023 AMI publishing
    # shouldn't force-replace a running instance either.
    ignore_changes = [user_data, ami]
  }
}
