"""Shared constants and engine-mapping tables for all Terraform builders."""
from __future__ import annotations

from app.models.schemas import CacheEngine

AWS_PROVIDER_VERSION = ">= 5.73.0"

APP_PORT = 80
RDS_PORT = 5432
REDIS_PORT = 6379
MEMCACHED_PORT = 11211
VALKEY_PORT = 6379

CACHE_ENGINE_RESOURCE_MAP: dict[CacheEngine, dict] = {
    CacheEngine.MEMCACHED: {
        "resource_type": "aws_elasticache_cluster",
        "engine": "memcached",
        "port": MEMCACHED_PORT,
    },
    CacheEngine.REDIS: {
        "resource_type": "aws_elasticache_replication_group",
        "engine": "redis",
        "port": REDIS_PORT,
    },
    CacheEngine.VALKEY: {
        "resource_type": "aws_elasticache_replication_group",
        "engine": "valkey",
        "port": VALKEY_PORT,
    },
}

RDS_ENGINE_MAP: dict[str, str] = {
    "PostgreSQL": "postgres",
    "postgres": "postgres",
    "MySQL": "mysql",
    "mysql": "mysql",
    "MariaDB": "mariadb",
    "mariadb": "mariadb",
}

RDS_ENGINE_PORT_MAP: dict[str, int] = {
    "postgres": 5432,
    "mysql": 3306,
    "mariadb": 3306,
}

CACHE_ENGINE_PORT_MAP: dict[CacheEngine, int] = {
    CacheEngine.REDIS: 6379,
    CacheEngine.VALKEY: 6379,
    CacheEngine.MEMCACHED: 11211,
}

_PARAM_GROUPS: dict[CacheEngine, str] = {
    CacheEngine.REDIS: "default.redis7",
    CacheEngine.VALKEY: "default.valkey7.2",
    CacheEngine.MEMCACHED: "default.memcached1.6",
}


def rds_engine(engine_suggestion: str | None) -> str:
    """Resolve a free-text engine suggestion to a Terraform engine string."""
    if engine_suggestion is None:
        return "postgres"
    if engine_suggestion in RDS_ENGINE_MAP:
        return RDS_ENGINE_MAP[engine_suggestion]
    lowered = engine_suggestion.strip().lower()
    for key, val in RDS_ENGINE_MAP.items():
        if key.lower() == lowered:
            return val
    return "postgres"


def param_group(engine: CacheEngine) -> str:
    return _PARAM_GROUPS.get(engine, "default.redis7")


def comment_block(text: str) -> str:
    lines = text.rstrip()
    wrapped_lines = []
    for ln in lines.split("\n"):
        wrapped_lines.append(f" * {ln}" if ln else " *")
    return "/*\n" + "\n".join(wrapped_lines) + "\n */"
