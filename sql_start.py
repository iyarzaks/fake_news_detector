import json
import pyodbc
from project_utils import *
q1="""create table ARTICLE(
    aID int primary key,
    URL varchar(200),
    header varchar(200),
    domain varchar(200),
    modelResult bit,
    alg1 float,
    alg2 float,
    alg3 float,
    modelWeightedResult float
);"""

q2 = """create table HISTORICAL_INFO(
    time datetime primary key,
    HIST_aID int,
    foreign key (HIST_aID) references ARTICLE on delete cascade
);"""

q3 = """create table EXTROVERTED_WORD(
    word varchar(50) primary key,
    extrovertCount int,
    totalCount int
);"""


""" connect to sql server and build three basic tables in it. 
use when new sql server is configured"""


cursor, cnxn = connect_sql_server()
cursor.execute(q1)
cursor.execute(q2)
cursor.execute(q3)
cnxn.commit()
