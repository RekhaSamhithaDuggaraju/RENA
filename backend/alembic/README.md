# RENA migrations

Production deployments should run Alembic against `RENA_DATABASE_URL`.
The application keeps `Base.metadata.create_all()` for first-run local smoke tests,
while schema evolution belongs in versioned migrations.
