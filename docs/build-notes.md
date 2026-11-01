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

## Task 4: MLflow

**What was built:** every finished run records its settings, loss curve, accuracy and adapter, and each
registered version stores its dataset hash. Results with a missing or mismatched hash are refused.

**Main tradeoff:** MLflow only reads results written by real runs and computes nothing, so it cannot
show a number nobody measured.

### Interview questions

1. **Why an alias for the champion?** The server follows the alias, so promotion and rollback change
   one pointer and need no code change.
2. **Why store the dataset hash on the version?** Months later you can name the exact data behind a model.
3. **What is the difference between a run and a model version?** A run is one experiment; a version
   is a run's output that was registered for possible use.

## Task 5: Airflow

**What was built:** a DAG that fetches a run from S3, records it, gates it and promotes it, with
retries and an alert. Tested on AWS: a missing run failed on all 3 attempts, the alert fired, and
later tasks did not run. A real promote and a real block still need real results.

**Main tradeoff:** training stays on Kaggle, so the DAG starts from its results instead of starting
training itself. That is cheap, but the first stage is manual.

### Interview questions

1. **Why is a block a skip, not a failure?** Nothing broke; the gate did its job. A failure should
   mean someone needs to look.
2. **Why compare with the champion as well as the base model?** A model can beat the base and still
   be worse than what is live.
3. **Why separate databases for Airflow and MLflow?** Each manages its own tables, and they clash in one.

## Task 6: Docker and Kubernetes

**What was built:** a FastAPI server, an image in ECR and a two-replica Deployment on `kind`. Tested
on AWS: 30 requests during an update from v1 to v2, 0 failed, both versions seen.

**Main tradeoff:** `kind` on one machine shows Kubernetes behaviour cheaply but gives no real node
failure. CPU serving is slow, about 13 s per request, and only suits a demonstration.

### Interview questions

1. **What makes the update safe?** `maxUnavailable: 0` keeps capacity up, and traffic reaches a new
   pod only after its readiness probe passes.
2. **Why the `preStop` wait?** A pod can keep receiving traffic briefly after it starts to stop, so
   it keeps serving until the Service has removed it.
3. **Why does this test prove little about production?** It is one node, low traffic and 30 requests.

## Task 7: Kafka

**What was built:** the evaluation platform consumes events and stores them. Tested on a real broker:
300 sent, consumer killed with SIGKILL after 52 were stored, exactly 300 stored after restart.

**Main tradeoff:** at-least-once delivery means duplicates can arrive, so the event id is the primary
key and a repeat does nothing.

### Interview questions

1. **Why commit the offset after the database write?** A crash then repeats work instead of losing it.
2. **Why does skipping the commit alone not save a failed event?** The next poll returns the next
   message, and committing that moves past the failure, so the consumer must seek back.
3. **Is this exactly-once?** No. It is at-least-once delivery with an idempotent write.

## Task 3: real training

**What was built:** real QLoRA training on the versioned dataset, on one Kaggle T4. It refuses data
whose hash does not match.

**Measured on 2026-10-31:** 5,077 training rows, 131.6 minutes. On 617 held-out questions, execution
accuracy rose from 60.9% to 69.4%, and valid SQL from 90.4% to 97.7%. One seed only.

## Task 8: closed loop

**What was built:** the server reports whether each answer's SQL runs, and a monitor starts the
pipeline when the valid-SQL rate falls too far.

**Tested on AWS:** 60 healthy events started nothing. 60 planted bad events started the pipeline,
and the champion moved from version 1 to 2.

**Main tradeoff and limit:** valid SQL is a cheap signal, not correctness. The pipeline promoted a
stored run, so the loop does not yet start real training.

### Interview questions

1. **Why require a minimum sample?** A few bad answers should not start an expensive retrain.
2. **Why is valid SQL a weak signal?** A query can run and still answer the wrong question.
3. **What is missing before production?** Automatic training, a rollback, and a cooldown between runs.

## Task 9: failure tests

Consumer killed: 300 of 300 stored. Pod killed: 1 of 18 requests failed. A failed run: 3 attempts,
then an alert. A real weak run (57.4%): blocked for gain and for falling below the champion.

**Found by testing:** the monitor has no cooldown, so a lasting drop starts repeated runs.

### Interview questions

1. **Why did one request fail when a pod died?** It was in progress on that pod. Retries belong in
   the client, and only for requests that are safe to repeat.
2. **Why plant a real weak model instead of fake numbers?** It tests the gate on real output.
3. **What would you fix first?** A cooldown, then making the record step safe to retry.
