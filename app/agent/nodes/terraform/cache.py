"""
Cache resource builder (ElastiCache Cluster / Replication Group) and engine mappings.
"""
from __future__ import annotations

from app.models.schemas import CacheEngine

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

CACHE_ENGINE_PORT_MAP: dict[CacheEngine, int] = {
    CacheEngine.REDIS: 6379,
    CacheEngine.VALKEY: 6379,
    CacheEngine.MEMCACHED: 11211,
}

_PARAM_GROUPS = {
    CacheEngine.REDIS: "default.redis7",
    CacheEngine.VALKEY: "default.valkey7.2",
    CacheEngine.MEMCACHED: "default.memcached1.6",
}


def _param_group(engine: CacheEngine) -> str:
    return _PARAM_GROUPS.get(engine, "default.redis7")


def _cache_memcached_cluster(*, cache_instance: str, cache_port: int) -> str:
    return f"""\
resource "aws_elasticache_cluster" "app" {{
  cluster_id = "${{var.app_name}}-cache"
  engine = "memcached"
  node_type = "{cache_instance}"
  num_cache_nodes = 1
  parameter_group_name = "default.memcached1.6"
  port = {cache_port}
  security_group_ids = [aws_security_group.cache.id]
}}
"""


def _cache_replication_group(
    *,
    cache_engine: str,
    cache_instance: str,
    cache_port: int,
    param_group: str,
) -> str:
    return f"""\
resource "aws_elasticache_replication_group" "app" {{
  replication_group_id = "${{var.app_name}}-cache"
  replication_group_description = "Cache tier for ${{var.app_name}}"
  engine = "{cache_engine}"
  node_type = "{cache_instance}"
  num_cache_clusters = 1
  parameter_group_name = "{param_group}"
  port = {cache_port}
  security_group_ids = [aws_security_group.cache.id]
}}
"""
