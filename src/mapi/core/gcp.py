"""Make Application Default Credentials resolvable on a platform without gcloud.

ADC is a RESOLUTION ORDER, not a single mechanism. Locally it finds the file
`gcloud auth application-default login` writes; on a dyno there is no gcloud,
no home directory state and nothing to log in with, so that path simply does
not exist. What ADC also honours is `GOOGLE_APPLICATION_CREDENTIALS` pointing
at a service-account key -- so the same client code works in both places once
the key is on disk.

The key arrives as a config var because that is the only durable place a
platform gives you, and google-auth wants a PATH rather than a blob. This
writes it once at boot, mode 0600, and points the variable at it.

Deliberately not committed anywhere and never logged: the JSON contains a
private key, and the failure mode of printing it is unrecoverable.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .logging import get_logger

log = get_logger(__name__)

#: The blob a platform can store. Not a Google-defined name -- google-auth
#: reads only the PATH variable -- so this is ours and needs materializing.
JSON_ENV = "GOOGLE_APPLICATION_CREDENTIALS_JSON"
PATH_ENV = "GOOGLE_APPLICATION_CREDENTIALS"


def materialize_adc() -> Path | None:
    """Write the inline service-account key to a file ADC can find.

    Returns the path written, or None when there is nothing to do -- which is
    the normal case locally, where real ADC is already present and must not be
    disturbed. Never raises: a malformed blob degrades to "no credentials",
    which surfaces as a clear provider error at first use rather than a crash
    during boot that takes the whole app down.
    """
    raw = os.environ.get(JSON_ENV, "").strip()
    if not raw:
        return None
    if os.environ.get(PATH_ENV):
        # An explicit path wins: someone mounted a real key deliberately.
        return None
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict) or "private_key" not in parsed:
            raise ValueError("not a service-account key")
    except (ValueError, TypeError) as exc:
        log.error("adc_json_invalid", error=str(exc)[:120])
        return None

    path = Path(tempfile.gettempdir()) / "mapi-adc.json"
    path.write_text(raw)
    path.chmod(0o600)
    os.environ[PATH_ENV] = str(path)
    # The client email is safe to log and is the only part worth seeing: it
    # answers "which identity is this dyno using" without exposing the key.
    log.info("adc_materialized", client_email=parsed.get("client_email", "?"))
    return path


__all__ = ["JSON_ENV", "PATH_ENV", "materialize_adc"]
