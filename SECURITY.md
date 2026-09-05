# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | ✅                 |
| 0.1.x   | ❌                 |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability:

1. **DO NOT** create a public GitHub issue
2. Email: [SECURITY_EMAIL] or use GitHub Private Vulnerability Reporting
3. Include:
   - Description of vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## Response Timeline

- **Acknowledgment:** Within 48 hours
- **Assessment:** Within 1 week
- **Fix release:** Within 30 days (critical: 72 hours)
- **Disclosure:** Coordinated with reporter

## Security Measures

- All dependencies audited regularly
- No hardcoded credentials or API keys
- User data (screenshots) stored locally only
- Models downloaded from trusted sources (HuggingFace)

## Known Risks

- Screen capture requires OS permissions
- Model files from external sources (verify checksums)
- GPU drivers required for accelerated inference
