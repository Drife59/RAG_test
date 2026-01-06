from piccolo.columns import Text, Varchar
from piccolo.table import Table


class Article(Table, tablename="articles"):
    id = Varchar(length=30, primary_key=True)
    source = Varchar(length=100)
    content = Text()
