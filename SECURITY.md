# Security and private artifacts

Do not commit keys, credentials, dataset access tokens, checkpoints, private logs, or research records. Revoke accidentally exposed credentials at the issuing service before removing them from repository history.

This add-on has no checkpoint-loading routine or network download behavior. Its synthetic tests do not need external files. A separately connected detector may load checkpoints using pickle-compatible APIs; use only trusted artifacts. Report a non-sensitive code issue through the repository's issue tracker, but never include secrets in a public report.
