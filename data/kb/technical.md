# Technical Support & Troubleshooting

## Login problems
If you cannot log in, first reset your password using the "Forgot password" link.
Reset emails arrive within 10 minutes; check spam. Accounts lock after 5 failed
attempts for 15 minutes. SSO users must sign in through their company portal.

## API access
API keys are created in Settings → Developer → API Keys. Keys are rate limited to
100 requests per minute on the Pro plan and 1000 on Enterprise. Rotate keys every
90 days. A 401 error means an invalid or expired key; a 429 means rate limited.

## Outages and status
Check status.insightdesk.example for live incident status. We post updates every
30 minutes during an incident. Subscribe to status alerts for email notifications.

## Data export
You can export your data as CSV or JSON from Settings → Data → Export. Large exports
are emailed as a download link within one hour.
