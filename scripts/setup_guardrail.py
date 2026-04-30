"""Create the Bedrock Guardrail used by CanAID.

Run once per AWS account/region. Prints the guardrail ID; paste it into
`.env` as `CANAID_GUARDRAIL_ID`.

```bash
uv run python scripts/setup_guardrail.py
```

Re-running is idempotent if you pass `--force` — it'll delete and recreate.
By default the script refuses to overwrite an existing guardrail with the
same name, to avoid accidental destruction in production.

For Phase 9 production deploys, prefer creating the guardrail through the
CDK stack so the IaC owns the resource. This script is for fast local
setup.
"""

from __future__ import annotations

import argparse
import sys

import boto3
from botocore.exceptions import ClientError

from canaid.config import get_settings
from canaid.observability.logging import configure_logging, get_logger

GUARDRAIL_NAME = "canaid-contact-center"


# ---- policy definitions --------------------------------------------------
TOPIC_POLICY = {
    "topicsConfig": [
        {
            "name": "ClinicalAdvice",
            "definition": (
                "Providing medical, clinical, dosage, diagnostic, or treatment "
                "advice for a patient. The contact-center bot must defer all "
                "such questions to a clinician."
            ),
            "examples": [
                "Should I increase the dose of this medication?",
                "Is it safe for a 72-year-old to take this with their other meds?",
                "What's the right way to clean a stage 3 pressure ulcer?",
                "Can you tell me what's wrong with my patient based on these symptoms?",
            ],
            "type": "DENY",
        },
        {
            "name": "SpecificPricing",
            "definition": (
                "Quoting specific dollar prices for products. Contractual pricing "
                "varies and is owned by the sales contact, not the contact-center."
            ),
            "examples": [
                "How much does GLOVE-NTR-M-100 cost?",
                "What's your best price on a case of N95s?",
                "Tell me the unit price of the chlorhexidine.",
            ],
            "type": "DENY",
        },
    ],
}


CONTENT_POLICY = {
    "filtersConfig": [
        {"type": "HATE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
        {"type": "INSULTS", "inputStrength": "HIGH", "outputStrength": "HIGH"},
        {"type": "SEXUAL", "inputStrength": "HIGH", "outputStrength": "HIGH"},
        {"type": "VIOLENCE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
        {"type": "MISCONDUCT", "inputStrength": "HIGH", "outputStrength": "HIGH"},
        # PROMPT_ATTACK only applies to inputs.
        {"type": "PROMPT_ATTACK", "inputStrength": "HIGH", "outputStrength": "NONE"},
    ],
}


# Anonymize most identifiers; BLOCK only the ones that should never appear.
SENSITIVE_INFO_POLICY = {
    "piiEntitiesConfig": [
        {"type": "EMAIL", "action": "ANONYMIZE"},
        {"type": "PHONE", "action": "ANONYMIZE"},
        {"type": "ADDRESS", "action": "ANONYMIZE"},
        {"type": "NAME", "action": "ANONYMIZE"},
        {"type": "URL", "action": "ANONYMIZE"},
        {"type": "AGE", "action": "ANONYMIZE"},
        {"type": "DRIVER_ID", "action": "BLOCK"},
        {"type": "PASSPORT_NUMBER", "action": "BLOCK"},
        {"type": "US_SOCIAL_SECURITY_NUMBER", "action": "BLOCK"},
        {"type": "CA_SOCIAL_INSURANCE_NUMBER", "action": "BLOCK"},
        {"type": "CREDIT_DEBIT_CARD_NUMBER", "action": "BLOCK"},
        {"type": "CREDIT_DEBIT_CARD_CVV", "action": "BLOCK"},
        {"type": "CREDIT_DEBIT_CARD_EXPIRY", "action": "BLOCK"},
        {"type": "PIN", "action": "BLOCK"},
        {"type": "US_BANK_ACCOUNT_NUMBER", "action": "BLOCK"},
        {"type": "US_BANK_ROUTING_NUMBER", "action": "BLOCK"},
    ],
}


CONTEXTUAL_GROUNDING = {
    "filtersConfig": [
        # Only attach grounding to the RAG path in code (per-call). The
        # values below are sane defaults for when grounding is enabled.
        {"type": "GROUNDING", "threshold": 0.7},
        {"type": "RELEVANCE", "threshold": 0.5},
    ],
}


BLOCKED_INPUT_MESSAGING = (
    "I can't help with that one. Want me to connect you with a teammate "
    "who can take it from here?"
)
BLOCKED_OUTPUT_MESSAGING = (
    "Sorry — I had a response ready, but it didn't pass our content policy. "
    "Let me try a different angle, or I can connect you with a teammate."
)


def find_existing(client) -> dict | None:
    paginator = client.get_paginator("list_guardrails")
    for page in paginator.paginate():
        for g in page.get("guardrails", []):
            if g.get("name") == GUARDRAIL_NAME:
                return g
    return None


def main() -> int:
    configure_logging()
    log = get_logger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force", action="store_true",
        help="delete the existing guardrail with this name before creating a new one",
    )
    args = parser.parse_args()

    region = get_settings().aws_region
    client = boto3.client("bedrock", region_name=region)

    existing = find_existing(client)
    if existing and not args.force:
        log.info(
            "guardrail.exists",
            id=existing["id"],
            arn=existing.get("arn"),
            status=existing.get("status"),
        )
        print(f"\nGuardrail '{GUARDRAIL_NAME}' already exists.")
        print(f"  ID:      {existing['id']}")
        print(f"  Status:  {existing.get('status')}")
        print(f"  Region:  {region}")
        print("\nAdd this to your .env:")
        print(f"  CANAID_GUARDRAIL_ID={existing['id']}")
        return 0

    if existing and args.force:
        log.warning("guardrail.deleting", id=existing["id"])
        try:
            client.delete_guardrail(guardrailIdentifier=existing["id"])
        except ClientError as e:
            log.error("guardrail.delete_failed", error=str(e))
            return 1

    log.info("guardrail.creating", name=GUARDRAIL_NAME, region=region)
    resp = client.create_guardrail(
        name=GUARDRAIL_NAME,
        description=(
            "CanAID contact-center: deny clinical advice + specific pricing; "
            "anonymize/block PII; high-strength content filters."
        ),
        topicPolicyConfig=TOPIC_POLICY,
        contentPolicyConfig=CONTENT_POLICY,
        sensitiveInformationPolicyConfig=SENSITIVE_INFO_POLICY,
        blockedInputMessaging=BLOCKED_INPUT_MESSAGING,
        blockedOutputsMessaging=BLOCKED_OUTPUT_MESSAGING,
    )

    print("\nCreated guardrail.")
    print(f"  ID:      {resp['guardrailId']}")
    print(f"  Version: {resp['version']}")
    print(f"  ARN:     {resp.get('guardrailArn')}")
    print("\nAdd this to your .env:")
    print(f"  CANAID_GUARDRAIL_ID={resp['guardrailId']}")
    print(f"  CANAID_GUARDRAIL_VERSION={resp['version']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
