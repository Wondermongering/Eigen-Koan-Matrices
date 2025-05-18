import sqlite3
import json
import datetime
from typing import Dict, List, Optional

from eigen_koan_matrix import EigenKoanMatrix


class EKMRepository:
    """SQLite-backed repository for versioning Eigen-Koan Matrices."""

    def __init__(self, db_path: str = "ekm_repository.db"):
        self.conn = sqlite3.connect(db_path)
        self._migrate()

    def _migrate(self):
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS matrices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                matrix_id TEXT,
                version INTEGER,
                name TEXT,
                description TEXT,
                author TEXT,
                json TEXT,
                metrics TEXT,
                created_at TEXT,
                UNIQUE(matrix_id, version)
            )
            """
        )
        self.conn.commit()

    def add_matrix(
        self,
        matrix: EigenKoanMatrix,
        author: str,
        description: str,
        metrics: Optional[Dict[str, float]] = None,
    ) -> int:
        """Add a new version of a matrix to the repository."""
        cur = self.conn.cursor()
        row = cur.execute(
            "SELECT MAX(version) FROM matrices WHERE matrix_id=?",
            (matrix.id,),
        ).fetchone()
        version = (row[0] or 0) + 1
        cur.execute(
            """
            INSERT INTO matrices (matrix_id, version, name, description, author, json, metrics, created_at)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                matrix.id,
                version,
                matrix.name,
                description,
                author,
                matrix.to_json(),
                json.dumps(metrics or {}),
                datetime.datetime.utcnow().isoformat(),
            ),
        )
        self.conn.commit()
        return version

    def get_matrix(self, matrix_id: str, version: Optional[int] = None) -> EigenKoanMatrix:
        """Retrieve a matrix version. Defaults to latest."""
        cur = self.conn.cursor()
        if version is None:
            row = cur.execute(
                "SELECT json FROM matrices WHERE matrix_id=? ORDER BY version DESC LIMIT 1",
                (matrix_id,),
            ).fetchone()
        else:
            row = cur.execute(
                "SELECT json FROM matrices WHERE matrix_id=? AND version=?",
                (matrix_id, version),
            ).fetchone()
        if row is None:
            raise ValueError("Matrix version not found")
        json_str = row[0]
        data = json.loads(json_str)
        matrix = EigenKoanMatrix.from_json(json_str)
        # Preserve original ID from the stored JSON
        matrix.id = data.get("id", matrix.id)
        return matrix

    def list_matrices(self) -> List[Dict[str, object]]:
        """List all matrix versions with metadata."""
        cur = self.conn.cursor()
        rows = cur.execute(
            "SELECT matrix_id, version, name, description, author, metrics, created_at FROM matrices ORDER BY matrix_id, version"
        ).fetchall()
        result: List[Dict[str, object]] = []
        for r in rows:
            result.append(
                {
                    "matrix_id": r[0],
                    "version": r[1],
                    "name": r[2],
                    "description": r[3],
                    "author": r[4],
                    "metrics": json.loads(r[5]) if r[5] else {},
                    "created_at": r[6],
                }
            )
        return result

    def update_metrics(self, matrix_id: str, version: int, metrics: Dict[str, float]):
        """Update effectiveness metrics for a matrix version."""
        cur = self.conn.cursor()
        cur.execute(
            "UPDATE matrices SET metrics=? WHERE matrix_id=? AND version=?",
            (json.dumps(metrics), matrix_id, version),
        )
        if cur.rowcount == 0:
            raise ValueError("Matrix version not found")
        self.conn.commit()
