"""
Database resource builder (RDS DB Instance) and engine mapping helpers.
"""
from __future__ import annotations

RDS_PORT = 5432

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


def _rds_engine(engine_suggestion: str | None) -> str:
    if engine_suggestion is None:
        # Fallback: postgres is the safe default.
        return "postgres"
    if engine_suggestion in RDS_ENGINE_MAP:
        return RDS_ENGINE_MAP[engine_suggestion]
    lowered = engine_suggestion.strip().lower()
    for key, val in RDS_ENGINE_MAP.items():
        if key.lower() == lowered:
            return val
    # Fallback rather than crashing
    return "postgres"


def _rds_instance(*, db_engine: str, db_instance: str) -> str:
    return f"""\
resource "aws_db_instance" "app" {{
  identifier = "${{var.app_name}}-db"
  engine = "{db_engine}"
  instance_class = "{db_instance}"
  allocated_storage = 20
  db_name = "appdb"
  username = var.db_username
  password = var.db_password
  skip_final_snapshot = true
  vpc_security_group_ids = [aws_security_group.db.id]
  publicly_accessible = false
}}
"""
