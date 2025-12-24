from piccolo.apps.migrations.auto.migration_manager import MigrationManager


ID = "2025-12-24T20:13:38:730046"
VERSION = "1.30.0"
DESCRIPTION = ""


async def forwards():
    manager = MigrationManager(
        migration_id=ID, app_name="", description=DESCRIPTION
    )

    def run():
        print(f"running {ID}")

    manager.add_raw(run)

    return manager
