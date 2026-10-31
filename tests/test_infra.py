"""Safety rules for the infrastructure code."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MAIN_TF = (ROOT / "infra" / "terraform" / "main.tf").read_text()


def test_security_group_has_no_inbound_rules() -> None:
    assert not re.search(r"^\s*ingress\s*\{", MAIN_TF, re.MULTILINE)


def test_bucket_blocks_public_access() -> None:
    for setting in (
        "block_public_acls",
        "block_public_policy",
        "ignore_public_acls",
        "restrict_public_buckets",
    ):
        assert re.search(rf"{setting}\s*=\s*true", MAIN_TF)


def test_machine_role_has_no_wildcard_except_the_ecr_login() -> None:
    wildcards = re.findall(r'actions\s*=\s*\[([^\]]*)\]\s*\n\s*resources\s*=\s*\["\*"\]', MAIN_TF)
    assert [a.strip() for a in wildcards] == ['"ecr:GetAuthorizationToken"']


def test_postgres_is_not_published_to_the_network() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    assert compose["services"]["postgres"]["ports"] == ["127.0.0.1:5432:5432"]


def test_compose_has_no_password_in_the_file() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    assert compose["services"]["postgres"]["environment"]["POSTGRES_PASSWORD"].startswith("${")
