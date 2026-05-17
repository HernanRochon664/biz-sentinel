"""Kedro settings for BizSentinel."""

from kedro.config import YamlConfigLoader  # type: ignore[attr-defined]

# Configuration loader class that will be used to load the configuration
# YamlConfigLoader is the default and supports YAML files with templating
CONFIG_LOADER_CLASS = YamlConfigLoader

# Additional configuration for the config loader
# This can include default paths to look for configuration files
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
}