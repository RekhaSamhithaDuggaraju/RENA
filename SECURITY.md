# RENA Security

RENA V22 uses bearer authentication, per-resource project/audit ownership checks, upload allowlists and path containment, and signed certificate verification.

Production requires `RENA_AUTH_SECRET`, `RENA_CERTIFICATE_SECRET`, and `RENA_DATABASE_URL`; the application refuses production startup when they are missing.

Certificates use HMAC signatures held separately from the certificate payload. Public verification exposes only certificate-safe metadata. Private experiment data remains behind authenticated resource access.

Known limitations: production deployment still requires an externally managed worker/process isolation strategy and a configured reverse proxy/rate limiter for internet exposure.
