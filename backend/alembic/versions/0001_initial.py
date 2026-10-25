"""Initial RENA V22 schema."""
from alembic import op
import sqlalchemy as sa
revision='0001_initial'
down_revision=None
branch_labels=None
depends_on=None

def upgrade():
    op.create_table('users',sa.Column('id',sa.String(),primary_key=True),sa.Column('email',sa.String(),nullable=False),sa.Column('password_hash',sa.String(),nullable=False),sa.Column('role',sa.String(),nullable=False),sa.Column('created_at',sa.DateTime()))
    op.create_index('ix_users_email','users',['email'],unique=True)
    op.create_table('projects',sa.Column('id',sa.String(),primary_key=True),sa.Column('name',sa.String(),nullable=False),sa.Column('owner_id',sa.String(),sa.ForeignKey('users.id')),sa.Column('description',sa.Text()),sa.Column('created_at',sa.DateTime()),sa.Column('settings',sa.JSON()))
    op.create_index('ix_projects_owner_id','projects',['owner_id'])
    op.create_table('audits',sa.Column('id',sa.String(),primary_key=True),sa.Column('owner_id',sa.String(),sa.ForeignKey('users.id')),sa.Column('project_id',sa.String(),sa.ForeignKey('projects.id')),sa.Column('algorithm',sa.String(),nullable=False),sa.Column('status',sa.String(),nullable=False),sa.Column('runtime_seconds',sa.Float()),sa.Column('sisa_shards_retrained',sa.Integer()),sa.Column('accuracy_retain',sa.Float()),sa.Column('accuracy_test',sa.Float()),sa.Column('accuracy_forget',sa.Float()),sa.Column('accuracy_original_test',sa.Float()),sa.Column('consensus_verdict',sa.String()),sa.Column('consensus_score',sa.Float()),sa.Column('consensus_agreement',sa.Float()),sa.Column('rena_x_verdict',sa.String()),sa.Column('rena_x_score',sa.Float()),sa.Column('authoritative_verdict',sa.String()),sa.Column('module_results',sa.JSON()),sa.Column('method_trace',sa.JSON()),sa.Column('consensus_detail',sa.JSON()),sa.Column('diagnosis',sa.JSON()),sa.Column('dataset_summary',sa.JSON()),sa.Column('created_at',sa.DateTime()),sa.Column('completed_at',sa.DateTime()))
    op.create_index('ix_audits_owner_id','audits',['owner_id']); op.create_index('ix_audits_project_id','audits',['project_id'])
    op.create_table('assets',sa.Column('id',sa.String(),primary_key=True),sa.Column('project_id',sa.String(),sa.ForeignKey('projects.id'),nullable=False),sa.Column('kind',sa.String(),nullable=False),sa.Column('name',sa.String(),nullable=False),sa.Column('version',sa.String()),sa.Column('path',sa.String(),nullable=False),sa.Column('metadata_json',sa.JSON()),sa.Column('created_at',sa.DateTime()))
    op.create_index('ix_assets_project_id','assets',['project_id'])
    op.create_table('jobs',sa.Column('id',sa.String(),primary_key=True),sa.Column('project_id',sa.String(),sa.ForeignKey('projects.id')),sa.Column('kind',sa.String(),nullable=False),sa.Column('status',sa.String()),sa.Column('progress',sa.JSON()),sa.Column('payload',sa.JSON()),sa.Column('result',sa.JSON()),sa.Column('error',sa.Text()),sa.Column('created_at',sa.DateTime()),sa.Column('started_at',sa.DateTime()),sa.Column('completed_at',sa.DateTime()))
    op.create_index('ix_jobs_project_id','jobs',['project_id'])
    op.create_table('audit_events',sa.Column('id',sa.String(),primary_key=True),sa.Column('project_id',sa.String(),sa.ForeignKey('projects.id')),sa.Column('audit_id',sa.String(),sa.ForeignKey('audits.id')),sa.Column('job_id',sa.String(),sa.ForeignKey('jobs.id')),sa.Column('event_type',sa.String(),nullable=False),sa.Column('message',sa.Text(),nullable=False),sa.Column('details',sa.JSON()),sa.Column('created_at',sa.DateTime()))
    op.create_index('ix_audit_events_project_id','audit_events',['project_id']); op.create_index('ix_audit_events_audit_id','audit_events',['audit_id']); op.create_index('ix_audit_events_job_id','audit_events',['job_id'])
    op.create_table('audit_certificates',sa.Column('id',sa.String(),primary_key=True),sa.Column('audit_id',sa.String(),sa.ForeignKey('audits.id'),nullable=False),sa.Column('certificate_hash',sa.String(),nullable=False),sa.Column('payload',sa.JSON(),nullable=False),sa.Column('signature',sa.String()),sa.Column('created_at',sa.DateTime()))
    op.create_index('ix_audit_certificates_audit_id','audit_certificates',['audit_id'],unique=True); op.create_index('ix_audit_certificates_certificate_hash','audit_certificates',['certificate_hash'],unique=True)
    op.create_table('audit_templates',sa.Column('id',sa.String(),primary_key=True),sa.Column('project_id',sa.String(),sa.ForeignKey('projects.id'),nullable=False),sa.Column('name',sa.String(),nullable=False),sa.Column('config',sa.JSON()),sa.Column('created_at',sa.DateTime()))
    op.create_index('ix_audit_templates_project_id','audit_templates',['project_id'])
    op.create_table('schedules',sa.Column('id',sa.String(),primary_key=True),sa.Column('project_id',sa.String(),sa.ForeignKey('projects.id'),nullable=False),sa.Column('name',sa.String(),nullable=False),sa.Column('cron',sa.String(),nullable=False),sa.Column('payload',sa.JSON()),sa.Column('enabled',sa.Boolean(),nullable=False),sa.Column('last_run_at',sa.DateTime()),sa.Column('next_run_at',sa.DateTime()),sa.Column('created_at',sa.DateTime()))
    op.create_index('ix_schedules_project_id','schedules',['project_id'])
    op.create_table('audit_share_links',sa.Column('id',sa.String(),primary_key=True),sa.Column('audit_id',sa.String(),sa.ForeignKey('audits.id'),nullable=False),sa.Column('token',sa.String(),nullable=False),sa.Column('created_at',sa.DateTime()),sa.Column('expires_at',sa.DateTime()),sa.Column('revoked',sa.Boolean(),nullable=False))
    op.create_index('ix_audit_share_links_audit_id','audit_share_links',['audit_id']); op.create_index('ix_audit_share_links_token','audit_share_links',['token'],unique=True)

def downgrade():
    for name in ['audit_share_links','schedules','audit_templates','audit_certificates','audit_events','jobs','assets','audits','projects','users']:
        op.drop_table(name)
