variable "region" {
  description = "AWS region for every resource."
  type        = string
  default     = "eu-west-1"
}

variable "instance_type" {
  description = "Machine that runs the stack. 16 GB is the minimum for Airflow, Kafka and kind together."
  type        = string
  default     = "t3.xlarge"
}

variable "disk_gb" {
  description = "Root disk size. It is billed even while the machine is stopped."
  type        = number
  default     = 50
}
