"""Kedro CLI for BizSentinel."""

import logging
from pathlib import Path

from kedro.framework.project import configure_project  # type: ignore[import-untyped]
from kedro.framework.session import KedroSession  # type: ignore[import-untyped]
from kedro.framework.startup import bootstrap_project  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


def main():
    """Main entry point for running Kedro commands."""
    # Bootstrap the project
    project_path = Path.cwd()
    bootstrap_project(project_path)
    
    # Configure the project
    configure_project("biz_sentinel")
    
    # Create a session and run the CLI
    with KedroSession.create(project_path=project_path) as session:
        session.run()


if __name__ == "__main__":
    main()