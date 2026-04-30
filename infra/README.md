# Infra (CDK, Python)

Single stack: VPC + Aurora Serverless v2 (Postgres + pgvector) + Valkey
Serverless + ECS Fargate API behind an ALB. See `docs/09-deployment.md`
for the narrative version.

## Prereqs

- AWS CLI v2 (`aws --version` ≥ 2.x), credentials on a profile that can
  CDK-bootstrap and create the resources.
- Node 18+ for the CDK CLI (`npm install -g aws-cdk`).
- Docker running locally — CDK builds the API image from the repo root.

## One-time bootstrap (per AWS account/region)

```bash
cd infra
uv sync --extra infra
export CDK_DEFAULT_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export CDK_DEFAULT_REGION=us-east-1
uv run cdk bootstrap
```

## Deploy

```bash
uv run cdk deploy --require-approval never
```

CDK will print outputs:

```
CanAID.ApiUrl              = http://CanAID-Api...elb.amazonaws.com
CanAID.DbSecretArn         = arn:aws:secretsmanager:...
CanAID.LangFuseSecretArn   = arn:aws:secretsmanager:...
CanAID.LogGroupName        = /aws/ecs/CanAID-...
```

## Post-deploy

1. **Apply RAG schema.**
   ```bash
   # From a bastion or any host that can reach the Aurora endpoint:
   psql "$(aws secretsmanager get-secret-value --secret-id <DbSecretArn> \
         --query SecretString --output text | jq -r .connectionString)" \
       -f ../scripts/sql/01-rag.sql \
       -f ../scripts/sql/02-audit.sql
   ```
2. **Build the index.**
   ```bash
   CANAID_PG_DSN=... uv run python ../scripts/build_index.py --reset
   ```
3. **Populate LangFuse secret** (Console → Secrets Manager → CanAID-LangFuseSecret-*).
4. **Force-redeploy the ECS task** to pick up new secrets:
   ```bash
   aws ecs update-service --cluster CanAID-Cluster --service CanAID-Api --force-new-deployment
   ```
5. **Point Streamlit Cloud at the API.**
   - Repo → Streamlit Cloud → app entrypoint `src/canaid/ui/streamlit_app.py`
   - Add secret `CANAID_API_URL=<the ApiUrl output>`.

## Cost shape

Approximate idle cost (us-east-1, on-demand):

| Resource | ~$/month |
|---|---|
| Aurora Serverless v2 @ 0.5 ACU min | $40 |
| ElastiCache Valkey Serverless | $20 (data + ECPU floor) |
| Fargate task (0.5 vCPU / 1 GB) 24/7 | $15 |
| ALB | $20 |
| NAT gateway + traffic | $35 |
| CloudWatch logs (1mo) | $5 |
| **Total idle** | **~$135/mo** |

Plus Bedrock per-token billing, which is the variable component.

## Tear-down

```bash
uv run cdk destroy
```

Aurora has `removal_policy=SNAPSHOT` — it'll leave a final snapshot in
your account that you can delete manually.
