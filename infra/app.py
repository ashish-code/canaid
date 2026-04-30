"""CDK app entry point.

Single stack today (`CanAIDStack`). Phase 10 may split into a network
stack + an app stack so the network can survive app redeploys.
"""

from __future__ import annotations

import os

import aws_cdk as cdk

from canaid_stack import CanAIDStack

app = cdk.App()

env = cdk.Environment(
    account=os.getenv("CDK_DEFAULT_ACCOUNT"),
    region=os.getenv("CDK_DEFAULT_REGION", "us-east-1"),
)

CanAIDStack(
    app,
    "CanAID",
    env=env,
    description="CanAID — multi-agent contact-center chatbot harness.",
)

app.synth()
