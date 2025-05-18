import sqlite3
import json
import datetime
from typing import Any, Dict, List, Optional

class EKMDatabase:
    """Simple SQLite-backed storage for experiment results."""

    def __init__(self, db_path: str = "ekm_results.db"):
        self.conn = sqlite3.connect(db_path)
        self._migrate()

    def _migrate(self):
        cur = self.conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER)")
        row = cur.execute("SELECT version FROM schema_version").fetchone()
        version = row[0] if row else 0
        migrations = [self._migration_1]
        while version < len(migrations):
            migrations[version]()
            version += 1
            cur.execute("DELETE FROM schema_version")
            cur.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
            self.conn.commit()

    def _migration_1(self):
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_name TEXT,
                matrix_id TEXT,
                model_name TEXT,
                path TEXT,
                data TEXT,
                created_at TEXT
            )
            """
        )
        self.conn.commit()

    def add_result(self, experiment_name: str, matrix_id: str, model_name: str, path: List[int], data: Dict[str, Any]):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO results (experiment_name, matrix_id, model_name, path, data, created_at) VALUES (?,?,?,?,?,?)",
            (
                experiment_name,
                matrix_id,
                model_name,
                json.dumps(path),
                json.dumps(data),
                datetime.datetime.utcnow().isoformat(),
            ),
        )
        self.conn.commit()

    def query_results(
        self,
        experiment_name: Optional[str] = None,
        matrix_id: Optional[str] = None,
        model_name: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        clauses = []
        params: List[Any] = []
        if experiment_name:
            clauses.append("experiment_name=?")
            params.append(experiment_name)
        if matrix_id:
            clauses.append("matrix_id=?")
            params.append(matrix_id)
        if model_name:
            clauses.append("model_name=?")
            params.append(model_name)
        if after:
            clauses.append("created_at>=?")
            params.append(after)
        if before:
            clauses.append("created_at<=?")
            params.append(before)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        cur = self.conn.cursor()
        query = f"SELECT experiment_name, matrix_id, model_name, path, data, created_at FROM results{where}"
        rows = cur.execute(query, params).fetchall()
        results = []
        for r in rows:
            results.append(
                {
                    "experiment_name": r[0],
                    "matrix_id": r[1],
                    "model_name": r[2],
                    "path": json.loads(r[3]),
                    "data": json.loads(r[4]),
                    "created_at": r[5],
                }
            )
        return results
