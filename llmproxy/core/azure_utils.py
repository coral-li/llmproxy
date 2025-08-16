from typing import Optional
from urllib.parse import urlparse


def _extract_hostname(value: object) -> Optional[str]:
    """Extract a lowercase hostname from a URL or raw hostname string.

    - Accepts full URLs (with or without trailing slash)
    - Accepts raw hostnames (e.g., "example.com", "sub.domain.com")
    - Returns None for invalid or non-string input
    """
    if not isinstance(value, str):
        return None

    candidate = value.strip()
    if not candidate:
        return None

    parsed = urlparse(candidate if "://" in candidate else f"http://{candidate}")
    if not parsed.hostname:
        return None

    return parsed.hostname.lower()


def is_azure_host(url_or_host: object) -> bool:
    """Return True if the hostname ends with '.azure.com'.

    Only matches real subdomains of azure.com (e.g., 'foo.azure.com').
    The bare 'azure.com' hostname is NOT considered a match.
    """
    hostname = _extract_hostname(url_or_host)
    if not hostname:
        return False
    return hostname.endswith(".azure.com")
