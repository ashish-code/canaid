"""CanAID infrastructure stack.

What this stack creates:

  * VPC with public + private subnets, NAT gateway, S3 endpoint.
  * Aurora Serverless v2 PostgreSQL 16 + pgvector (RAG index, audit log,
    long-term memory, LangGraph Postgres checkpointer).
  * ElastiCache Serverless (Valkey-compatible Redis) for the turn cache.
  * ECS Fargate cluster + service running the FastAPI image, behind an
    Application Load Balancer.
  * Secrets Manager: DB credentials (auto-created by RDS) and a slot
    for LangFuse keys (populated manually).
  * CloudWatch log group with 30-day retention.
  * IAM task role with: Bedrock invoke-model, Comprehend PII detection,
    Bedrock Guardrails apply, Secrets read.

Deliberately NOT in this stack (yet):

  * Cognito for end-user auth (the demo runs unauthenticated).
  * OpenSearch Serverless — pgvector handles the demo. The Phase 9 doc
    explains the migration path.
  * CloudFront / WAF / certificates — public ALB serves HTTP; for HTTPS
    add a hosted zone + ACM cert + ALB redirect.
  * Bedrock Guardrail resource — a `scripts/setup_guardrail.py` run is
    cleaner than CDK for now, since guardrail iteration is fast and
    we want to recreate it without redeploying the whole app.
"""

from __future__ import annotations

import aws_cdk as cdk
from aws_cdk import Duration, RemovalPolicy, Stack
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_ecs_patterns as ecs_patterns
from aws_cdk import aws_elasticache as elasticache
from aws_cdk import aws_iam as iam
from aws_cdk import aws_logs as logs
from aws_cdk import aws_rds as rds
from aws_cdk import aws_secretsmanager as secrets
from constructs import Construct


class CanAIDStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # --- VPC ---------------------------------------------------------
        vpc = ec2.Vpc(
            self,
            "Vpc",
            max_azs=2,
            nat_gateways=1,
            subnet_configuration=[
                ec2.SubnetConfiguration(name="public", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24),
                ec2.SubnetConfiguration(name="private", subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS, cidr_mask=24),
            ],
            gateway_endpoints={
                "S3": ec2.GatewayEndpointOptions(service=ec2.GatewayVpcEndpointAwsService.S3),
            },
        )

        # --- Aurora Serverless v2 + pgvector -----------------------------
        db_security_group = ec2.SecurityGroup(self, "DbSg", vpc=vpc, description="Aurora SG")
        db_cluster = rds.DatabaseCluster(
            self,
            "Aurora",
            engine=rds.DatabaseClusterEngine.aurora_postgres(
                version=rds.AuroraPostgresEngineVersion.VER_16_3,
            ),
            credentials=rds.Credentials.from_generated_secret("canaid"),
            default_database_name="canaid",
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            security_groups=[db_security_group],
            serverless_v2_min_capacity=0.5,
            serverless_v2_max_capacity=4,
            writer=rds.ClusterInstance.serverless_v2("writer"),
            removal_policy=RemovalPolicy.SNAPSHOT,
            storage_encrypted=True,
        )

        # --- ElastiCache Serverless (Valkey/Redis) -----------------------
        cache_sg = ec2.SecurityGroup(self, "CacheSg", vpc=vpc, description="Valkey SG")
        valkey = elasticache.CfnServerlessCache(
            self,
            "TurnCache",
            engine="valkey",
            serverless_cache_name=cdk.Fn.join("-", [self.stack_name.lower(), "cache"]),
            subnet_ids=[s.subnet_id for s in vpc.private_subnets],
            security_group_ids=[cache_sg.security_group_id],
            cache_usage_limits=elasticache.CfnServerlessCache.CacheUsageLimitsProperty(
                data_storage=elasticache.CfnServerlessCache.DataStorageProperty(
                    maximum=1, unit="GB"
                ),
                ecpu_per_second=elasticache.CfnServerlessCache.ECPUPerSecondProperty(
                    maximum=5000
                ),
            ),
        )

        # --- LangFuse secret slot (populated manually) -------------------
        langfuse_secret = secrets.Secret(
            self,
            "LangFuseSecret",
            description="LangFuse Cloud public/secret keys for CanAID",
            secret_object_value={
                "LANGFUSE_PUBLIC_KEY": cdk.SecretValue.unsafe_plain_text("REPLACE_ME"),
                "LANGFUSE_SECRET_KEY": cdk.SecretValue.unsafe_plain_text("REPLACE_ME"),
            },
        )

        # --- ECS Fargate -------------------------------------------------
        cluster = ecs.Cluster(self, "Cluster", vpc=vpc, container_insights=True)
        log_group = logs.LogGroup(
            self,
            "ApiLogs",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )
        api_image = ecr_assets.DockerImageAsset(
            self,
            "ApiImage",
            directory="..",   # repo root has the Dockerfile
            platform=ecr_assets.Platform.LINUX_AMD64,
        )

        api = ecs_patterns.ApplicationLoadBalancedFargateService(
            self,
            "Api",
            cluster=cluster,
            cpu=512,
            memory_limit_mib=1024,
            desired_count=1,
            min_healthy_percent=50,
            max_healthy_percent=200,
            public_load_balancer=True,
            task_image_options=ecs_patterns.ApplicationLoadBalancedTaskImageOptions(
                image=ecs.ContainerImage.from_docker_image_asset(api_image),
                container_port=8000,
                log_driver=ecs.LogDriver.aws_logs(stream_prefix="api", log_group=log_group),
                environment={
                    "CANAID_LOG_LEVEL": "INFO",
                    "CANAID_API_HOST": "0.0.0.0",
                    "CANAID_API_PORT": "8000",
                    "CANAID_USE_POSTGRES_CHECKPOINTER": "true",
                    "CANAID_REDIS_URL": cdk.Token.as_string(
                        cdk.Fn.join(
                            "",
                            [
                                "rediss://",
                                valkey.attr_endpoint_address,
                                ":",
                                cdk.Token.as_string(valkey.attr_endpoint_port),
                                "/0",
                            ],
                        )
                    ),
                    "AWS_REGION": self.region,
                },
                secrets={
                    "CANAID_PG_DSN": ecs.Secret.from_secrets_manager(
                        db_cluster.secret, field="connectionString"
                    ),
                    "LANGFUSE_PUBLIC_KEY": ecs.Secret.from_secrets_manager(
                        langfuse_secret, field="LANGFUSE_PUBLIC_KEY"
                    ),
                    "LANGFUSE_SECRET_KEY": ecs.Secret.from_secrets_manager(
                        langfuse_secret, field="LANGFUSE_SECRET_KEY"
                    ),
                },
            ),
        )
        api.target_group.configure_health_check(
            path="/health",
            healthy_http_codes="200",
            interval=Duration.seconds(15),
            timeout=Duration.seconds(5),
        )

        # Allow the API to reach the DB and the cache.
        db_security_group.add_ingress_rule(
            peer=api.service.connections.security_groups[0],
            connection=ec2.Port.tcp(5432),
            description="API → Aurora",
        )
        cache_sg.add_ingress_rule(
            peer=api.service.connections.security_groups[0],
            connection=ec2.Port.tcp(6379),
            description="API → ElastiCache",
        )

        # IAM: Bedrock + Comprehend permissions on the task role.
        api.task_definition.task_role.add_to_principal_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                    "bedrock:Converse",
                    "bedrock:ConverseStream",
                    "bedrock:ApplyGuardrail",
                ],
                resources=["*"],   # Bedrock model ARNs are account/region/model — wildcard is conventional
            )
        )
        api.task_definition.task_role.add_to_principal_policy(
            iam.PolicyStatement(
                actions=[
                    "comprehend:DetectPiiEntities",
                    "comprehend:ContainsPiiEntities",
                ],
                resources=["*"],
            )
        )

        # --- outputs -----------------------------------------------------
        cdk.CfnOutput(
            self, "ApiUrl",
            value=f"http://{api.load_balancer.load_balancer_dns_name}",
            description="API URL — point CANAID_API_URL in Streamlit Cloud at this.",
        )
        cdk.CfnOutput(self, "DbSecretArn", value=db_cluster.secret.secret_arn)
        cdk.CfnOutput(self, "LangFuseSecretArn", value=langfuse_secret.secret_arn)
        cdk.CfnOutput(self, "LogGroupName", value=log_group.log_group_name)
