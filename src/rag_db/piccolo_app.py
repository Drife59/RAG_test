"""
Import all of the Tables subclasses in your app here, and register them with
the APP_CONFIG.
"""

import os

from piccolo.conf.apps import AppConfig

from src.rag_db.tables import Article

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))


APP_CONFIG = AppConfig(
    app_name="rag_db",
    migrations_folder_path=os.path.join(CURRENT_DIRECTORY, "piccolo_migrations"),
    # table_classes=table_finder(
    #    modules=[".tables"],
    #    package=get_package(__name__),
    #    exclude_imported=True,
    # ),
    table_classes=[Article],
    migration_dependencies=[],
    commands=[],
)
