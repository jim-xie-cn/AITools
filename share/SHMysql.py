import pandas as pd
import os,json,requests
from sqlalchemy import create_engine,text
import logging as logger

class CPandasMysql:

    def __init__(self,host="127.0.0.1",port=3306,user='root',password='iot_admin',database='iot'):
        self.m_database = database
        self.create_engine(host,port,user,password,database)
    
    def __del__(self):
        self.close()

    def get_database(self):
        return self.m_database

    def create_engine(self, host, port, user, password, database):
        self.m_engine = create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}/{database}",pool_timeout=36000, pool_recycle=36000)
        self.m_conn = self.m_engine.connect()
        connection = self.m_engine.raw_connection()
        cursor = connection.cursor()
        cursor.execute("SET GLOBAL max_allowed_packet=671088640")
        connection.commit()

    def load(self,sql):
        try:
            return pd.read_sql(sql, con=self.m_engine)
        except:
            logger.exception(sql)

    def save(self,table, df, action="append",chunksize=10000):
        ret = 0
        try:
            ret = df.to_sql(table, con=self.m_engine, if_exists=action, chunksize=chunksize,index=False)
        except Exception as e:
            logger.error(f"Error writing column : {e}")
        return ret

    def execute(self,sql):
        try:
            self.m_conn.execute(text(sql))
            self.m_conn.commit()
        except Exception as e:
            try:
                self.m_conn.rollback()
            except:
                pass
            logger.exception(f"excute error: [{e}], sql: {sql}")

    def query(self,sql):
        try:
            rows = self.m_conn.execute(text(sql))
            self.m_conn.commit()
            return rows
        except Exception as e:
            try:
                self.m_conn.rollback()
            except:
                pass
            logger.exception(f"query error: [{e}], sql: {sql}")

    def close(self):
        try:
            self.m_conn.close()
            self.m_engine.dispose()
        except:
            logger.exception(f"close error: [{e}]")

def main():
    test = CPandasMysql()
    sql = "show tables"
    df = test.load(sql)
    test.close()
    #print(df)

if __name__ == '__main__':
    main()
