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

## Task 2: data with Spark

**What was built:** `src/data/spark_job.py` reads JSONL, trims it, rejects bad rows with a reason,
removes duplicate questions, splits by a hash of the question, and writes versioned files.

**Why:** training is only comparable when you can name the exact data behind it.

**Main tradeoff:** Spark is more than 6,000 rows need; plain Python would be simpler and could be
as fast at this size. It is here to show the same job would scale, and the local run proves the
logic, not cluster tuning.

**Measured on 2026-10-31:** 6,000 text-to-SQL rows gave 5,077 train, 306 validation and 617 test,
with 0 rejected, in 7.44 s on an 8 GB M2 laptop. Three runs, one with shuffled input, gave the same
hash. Full details are in `results/data_text2sql.json`.

### Interview questions

1. **Why split by a hash, not at random?** A question keeps its split when data is added, and the
   same question can never leak from training into the test set.
2. **Why reject rows instead of dropping them?** Silent drops hide data problems. A rejected file
   with reasons can be counted and checked.
3. **Why sort before hashing?** Spark gives no row order guarantee, so without sorting the same
   data could produce a different hash.
