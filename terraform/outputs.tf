output "instance_id" {
  value = aws_instance.app.id
}

output "public_ip" {
  value = aws_instance.app.public_ip
}

output "ssh_command" {
  value = "ssh -i ${var.project_name}-key.pem ec2-user@${aws_instance.app.public_ip}"
}

output "ssm_command" {
  value       = "aws ssm start-session --target ${aws_instance.app.id} --region ${var.aws_region}"
  description = "Fallback access if the SSH CIDR goes stale"
}

output "artifact_bucket" {
  value = aws_s3_bucket.artifacts.bucket
}

output "secret_arn" {
  value = aws_secretsmanager_secret.app_env.arn
}

output "start_stop_commands" {
  value = <<-EOT
    Stop (pay storage only, ~$8/mo for ${var.root_volume_gb}GB gp3):
      aws ec2 stop-instances --instance-ids ${aws_instance.app.id} --region ${var.aws_region}
    Start:
      aws ec2 start-instances --instance-ids ${aws_instance.app.id} --region ${var.aws_region}
  EOT
}
