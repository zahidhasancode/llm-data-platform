# Build notes

## Task 1: foundation and AWS

**What was built:** Terraform for one EC2 machine, an S3 bucket, an ECR repository and an IAM role;
Docker Compose with Postgres; CI checks for both.

**Why:** the stack needs more memory than the laptop has, and everything must be removable with
one command, so nothing keeps billing.

**Main tradeoff:** one machine costs far less than EKS and managed services, but it has no high
availability. That is fine for a demonstration and wrong for a real product.

**Tested on 2026-10-31:** `terraform apply` created 10 resources. Postgres started healthy on the
machine. The machine wrote to its bucket, and was refused when it listed all buckets or described
EC2 instances. `terraform destroy` removed everything. Not tested yet: pushing to ECR.

### Interview questions

1. **Why no SSH?** Session Manager gives a shell through the IAM role, so there is no open port and
   no key to lose. Access is logged by AWS.
2. **Why an instance role and not access keys?** The role gives short-lived credentials
   automatically, so no long-lived secret sits on the machine.
3. **Why is `ecr:GetAuthorizationToken` on `*`?** That action does not support limiting to one
   repository. The push and pull actions are limited to this project's repository.
