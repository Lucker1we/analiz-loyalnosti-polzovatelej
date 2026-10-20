import pandas as pd
from sqlalchemy import create_engine
import psycopg
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import phik

import os
from dotenv import load_dotenv

class DataProcessor:

    def __init__(self):
        pass

    def load_db(self, sql_base:bool = False):
        if sql_base:

            db_config = {'user': os.getenv('DB_USER'), # имя пользователя
                'pwd': os.getenv('DB_PWD'), # пароль
                'host': os.getenv('DB_HOST'),
                'port': os.getenv('DB_PORT'), # порт подключения
                'db': os.getenv('DB_NAME') # название базы данных
                }
            
            connection_string = 'postgresql://{}:{}@{}:{}/{}'.format(
                db_config['user'],
                db_config['pwd'],
                db_config['host'],
                db_config['port'],
                db_config['db'],
                )
            
            engine = create_engine(connection_string)

            query = '''
                SELECT p.user_id,
                p.device_type_canonical,
                p.order_id,
                p.created_dt_msk as order_dt,
                p.created_ts_msk as order_ts,
                p.currency_code,
                p.revenue,
                p.tickets_count,
                EXTRACT (DAY from (p.created_dt_msk - LAG(p.created_dt_msk) OVER (PARTITION BY p.user_id ORDER BY p.created_dt_msk))) as days_since_prev,
                e.event_id,
                e.event_name_code as event_name,
                e.event_type_main,
                p.service_name,
                r.region_name,
                c.city_name
                FROM afisha.purchases as p
                LEFT JOIN afisha.events as e USING (event_id)
                LEFT JOIN afisha.city as c USING(city_id)
                LEFT JOIN afisha.regions as r USING(region_id)
                WHERE (device_type_canonical = 'mobile' OR device_type_canonical = 'desktop') AND (e.event_type_main != 'фильм')
                ORDER BY user_id
                '''
            
            df = pd.read_sql_query(query, con=engine)

        else:
            df_orders = pd.read_csv(os.getenv('DB_O_LINK'))
            df_events = pd.read_csv(os.getenv('DB_E_LINK'))
            df = df_orders.merge(df_events, on = 'event_id', how = 'left')
            df = df.rename(columns = {'created_dt_msk': 'order_dt', 'created_ts_msk': 'order_ts'})
            needed_columns = ['user_id', 'device_type_canonical', 'order_id', 'order_dt', 'currency_code', 'revenue', 'tickets_count', 'days_since_prev', 'event_id', 'event_name', 'event_type_main', 'service_name', 'region_name', 'city_name']
            valid_columns = [col for col in needed_columns if col in df.columns]
            df = df[valid_columns]
        return df
    
