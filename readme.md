# Connecting to Heroku PostgreSQL
This guide shows how to access data in two different ways,
using the traditional `psycopg2` library as well as `sqlalchemy`, a library that abstracts SQL code in favor of a more pythonic implementation.

## Requirements
```
pyodbc 
psycopg2
sqlalchemy
```

## Credentials
While this data isn't sensitive, treat the URL with care as anybody could connect to it remotely.

```python
DATABASE_URL="postgres://u2bbja15c4igeb:pedd96ce18167a13cc8d49f93dc8db09b97790d992dfcf068d16b08f4d0edbd0b@c4c161t4pf58h3.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com:5432/d421pu93ggcr06"
```

## Approach 1: Psycopg2 
We can begin by creating a wrapper that helps us handle retrieving data as well as comitting write operations.
```python
import psycopg2 

def execute_sql(sql, return_all=False, commit=False):
    try:

        con = psycopg2.connect(DATABASE_URL, sslmode='require')
        cursor = con.cursor()
        cursor.execute(sql)

        if commit:
            con.commit()

        if return_all:
            out = cursor.fetchall()
            con.close()
            return out

        con.close()

    except psycopg2.OperationalError as e:
        try:
            print(e)
            con.close()
        except Exception:
            pass

```

### Parameters
1. `sql` : `str` - any valid PostgreSQL statement.
2. `return_all`: `bool` - if true, returns the output. Will throw an error if the `sql` parameter doesn't terminate in some kind of `SELECT` statement.
3. `commit` : `bool` - if true, persists any `INSERT`, `UPDATE`, or `DELETE` statements to database state.

### Example
```python
# retreives 10 rows with all columns from the review table
data = execute_sql("SELECT * FROM Review LIMIT 10", return_all=True)

```

## Approach 2: SQL Alchemy 
This approach requires less 'vanilla' SQL by relying on python objects, but also has slightly more rigid requirements.

First, we'll have to transform our URL slightly:
```python
ALCHEMY_URL="postgresql+psycopg2://u2bbja15c4igeb:p3e3829ff3559f72150418faf7ad91c698f73f28c3510cbf7ea9eea2e6d9ebf1a@c9mq4861d16jlm.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com:5432/db1bfsr1040tp7"
```

```python
from sqlalchemy import create_engine

engine = create_engine(ALCHEMY_URL)

reviews_tbl = Table("reviews", MetaData(schema="public"), autoload_with=engine)

# get ten rows 
query = reviews_tbl.select().limit(10)
```

## Endpoints

You can get a full list of tables with the following query:
```sql
select table_name 
from information_schema.tables 
where table_schema = 'public' 
and table_type = 'BASE TABLE'
```

Running this through the `execute_sql` function returns a list of table names that can then be queried for extra discovery.