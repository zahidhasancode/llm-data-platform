# Architecture

The goal is one connected pipeline: new data becomes a fine-tuned model, a quality gate decides
whether it is promoted, the server runs it, and live quality can start the next round.

```text
raw data → cleaning and versioning → fine-tuning → evaluation → gate → model registry
                                         ↑                               ↓
                              quality drop starts a run           model server
                                         ↑                               ↓
                              evaluation platform  ←  request events (Kafka)
```

Only the data stage is real today. Training and evaluation are still simulated, and the other
parts are planned. The README's status table says what works.

## Where things run

| Part | Where | Why |
|---|---|---|
| Stack | One `t3.xlarge` on AWS | 16 GB is enough for the services together; a laptop with 8 GB is not |
| Fine-tuning | Kaggle T4 | Free GPU, and enough for a small model |
| Datasets, adapters, MLflow files | S3 | They outlive the machine |
| Images | ECR | Kubernetes needs a registry to pull from |

## Decisions

- **One machine, not managed services.** `kind` and Compose cost a fraction of EKS and managed
  Kafka and use the same Kubernetes and Kafka concepts. The tradeoff is no high availability.
- **No inbound network access.** The security group has no inbound rules. Shell access goes
  through Session Manager, so there is no SSH key to lose and no open port.
- **A limited machine role.** It can use this project's bucket and image repository only, so no
  long-lived keys sit on the machine.
- **Everything is in Terraform.** `terraform destroy` removes every resource, so nothing keeps
  billing after the project ends.
