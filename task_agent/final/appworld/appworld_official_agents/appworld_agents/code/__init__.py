"""Official AppWorld experiment agent code vendored for local modification."""

from __future__ import annotations

import os

import litellm

os.environ.setdefault("NO_API_KEY", "__empty__")
os.environ.setdefault("LITELLM_LOG", "ERROR")
litellm.drop_params = True
litellm.turn_off_message_logging = True
litellm.suppress_debug_info = True
