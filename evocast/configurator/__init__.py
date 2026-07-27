"""EvoCast configurator subsystem.

Validator-gated task configuration pipeline:

  1. config_validator — deterministic CSV validation
  2. config_compiler  — intent → TFB config + multi-horizon manifest

The v2 agent can help users reason, but compiled configs remain deterministic.
"""

from evocast.configurator.config_validator import validate_intent, ValidationReport
from evocast.configurator.config_compiler import compile_config, compile_multi_horizon_manifest
