output "instance_id" {
  value = aws_instance.machine.id
}

output "bucket" {
  value = aws_s3_bucket.artifacts.bucket
}

output "image_repository" {
  value = aws_ecr_repository.server.repository_url
}
