from piccolo.conf.apps import AppRegistry
from piccolo.engine.postgres import PostgresEngine

APP_REGISTRY = AppRegistry(apps=["rag_db.piccolo_app"])
DB = PostgresEngine(
    config={
        "host": "localhost",
        "port": 5432,
        "user": "rag_user",
        "password": "rag_59#",
        "database": "rag_db",
    }
)
