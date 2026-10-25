"""Use a fresh, dedicated SQLite database for each pytest run."""
import os
from pathlib import Path

TEST_DB = Path("/tmp/rena_pytest_shared.db")

# This file is reserved exclusively for automated tests.
os.environ["RENA_DATABASE_URL"] = f"sqlite:///{TEST_DB}"

# Runs before test modules import the application/database.
TEST_DB.unlink(missing_ok=True)
