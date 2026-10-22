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
            print('Загрузка датасета из базы данных...')

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
            print('Загрузка датасета по ссылке...')
            df_orders = pd.read_csv(os.getenv('DB_O_LINK'))
            df_events = pd.read_csv(os.getenv('DB_E_LINK'))
            df = df_orders.merge(df_events, on = 'event_id', how = 'left')
            df = df.rename(columns = {'created_dt_msk': 'order_dt', 'created_ts_msk': 'order_ts'})
            needed_columns = ['user_id', 'device_type_canonical', 'order_id', 'order_dt', 'order_ts', 'currency_code', 'revenue', 'tickets_count', 'days_since_prev', 'event_id', 'event_name', 'event_type_main', 'service_name', 'region_name', 'city_name']
            valid_columns = [col for col in needed_columns if col in df.columns]
            df = df[valid_columns]
            df['order_dt'] = pd.to_datetime(df['order_dt'])
            df['order_ts'] = pd.to_datetime(df['order_ts'])
        return df
    
    def load_converter_df(self):
        tenge_df = pd.read_csv("../data/final_tickets_tenge_df.csv")
        return tenge_df

    def clean_data(self, df:pd.DataFrame, tenge_df:pd.DataFrame):

        print("Изначальный датасет:")
        print(df.info())
        
        tenge_df['data'] = pd.to_datetime(tenge_df['data'])
        rate_dict = tenge_df.set_index('data')['curs'].to_dict()
        df['revenue_rub'] = np.where(df['currency_code'] == 'kzt',
                             df['revenue'] * (df['order_dt'].map(rate_dict) / 100),
                             df['revenue']
                             )
        
        df['tickets_count'] = df['tickets_count'].astype('int8')
        df['order_id'] = df['order_id'].astype('int32')
        df['event_id'] = df['event_id'].astype('int32')
        df['days_since_prev'] = df['days_since_prev'].astype('Int16')

        for col in ['revenue', 'revenue_rub']:
            df[col] = df[col].astype('float32')

        for col in [ 
            "device_type_canonical",
            "currency_code",
            "event_type_main",
            "service_name",
            "region_name",
            "city_name"
            ]:
                df[col] = df[col].astype('category')


        init_rows = len(df)

        duplcate_cols =  [
            'user_id', 'device_type_canonical', 'order_ts', 'currency_code', 'revenue', 'tickets_count',
            'event_id', 'event_name', 'event_type_main', 'service_name', 'region_name', 'city_name', 'revenue_rub'
        ]

        duplicates = df.duplicated(subset=duplcate_cols).sum()
        df = df.drop_duplicates(subset=duplcate_cols, keep='first')         


        df = df[df['revenue_rub'] >= 0]
        df = df[df['revenue_rub'] <= df['revenue_rub'].quantile(0.99)].reset_index(drop=True)

        print(f"Исходное кол-во заказов: {init_rows}")
        print(f"Количество заказов после фильтрации: {len(df)}")
        print(f'Найдено {duplicates} дубликатов')
        print(f"Удалено анамальных заказов: {init_rows - len(df) - duplicates}")
        print(f"Доля удаленных данных: {(((init_rows - len(df))/init_rows) * 100):.2f}%")

        print('Обновлённый датасет:')
        print(df.info())

        return df


    def create_profiles(self, df:pd.DataFrame):
        df = df.sort_values(by = 'order_ts')

        user_profiles = df.groupby('user_id').agg(
            first_order_dt = ('order_dt', 'first'),
            last_order_dt = ('order_dt', 'last'),
            first_device = ('device_type_canonical', 'first'),
            first_region = ('region_name', 'first'),
            first_event_type = ('event_type_main', 'first'),
            total_orders = ('order_id', 'count'),
            avg_revenue = ('revenue_rub', 'mean'),
            avg_days_between_orders = ('days_since_prev', 'mean')
            ).reset_index()

        user_profiles['is_two'] = np.where(user_profiles['total_orders'] >= 2, 1, 0)

        init_len = len(user_profiles)
        user_profiles = user_profiles[user_profiles['total_orders']<= user_profiles['total_orders'].quantile(0.99)]
        print('Исключено анамальных пользователей пользователей: ', init_len - len(user_profiles))

        print('Общая информация по данным пользовательский профилей:')
        print(user_profiles.info())
        print(user_profiles.head(5))


        print('Базовые показатели по пользователям')
        print(f"Общее число пользователей: {(user_profiles['user_id'].nunique()):,}")
        print(f"Средняя выручка с одного заказа: {(user_profiles['avg_revenue'].mean()):.2f} руб.")
        print(f"Доля пользователей, совершивших 2 и более заказа: {(user_profiles['is_two'].mean()):.2f}")
        for col in ['total_orders', 'avg_days_between_orders']:
            print(f'Статстические показатели по параметру {col}:')
            print(user_profiles[col].describe())
            print()

        return user_profiles


    def retention_analysis(self, user_profiles:pd.DataFrame, show_plots:bool=False):

        print('Изучение пользователей по принзнакам')

        total_users = len(user_profiles)

        for col in ['first_event_type', 'first_device', 'first_region']:
            segment = user_profiles.groupby(col, observed = False).agg(user_count=('user_id', 'count'))
            segment['share'] = (segment['user_count']/total_users).round(2)
            segment = segment.sort_values(by= 'user_count', ascending = False).reset_index()
            print(f"Расчёт пользователей по сегенту {col}: ")
            print(segment)
            print()

        print('Анализ вернувшихся пользователей')

        fig, axes = plt.subplots(3,1, figsize = (12,18))

        for idx, col in enumerate(['first_event_type', 'first_device', 'first_region']):
            segment_come_back = user_profiles.groupby(col, observed = False). agg(come_back =('is_two', 'mean'), user_count =('user_id', 'count')).reset_index()
            if col == 'first_region':
                segment_come_back = segment_come_back.sort_values(by = 'user_count', ascending = False).head(10)
                segment_come_back['first_region'] = segment_come_back['first_region'].cat.remove_unused_categories()
            segment_come_back = segment_come_back.sort_values(by = 'come_back', ascending = False)
            print(f"Доля вернувшихся клиентов по сегменту {col}:")
            print(segment_come_back)
            print()

            sns.barplot(data = segment_come_back, x = 'come_back', y = col, ax = axes[idx], palette='viridis', legend = False)
            axes[idx].set_title(f'Доля совершивших повторный заказ по признаку: {col}')
            axes[idx].set_xlabel(f'Доля вернувшихся пользователей')
            axes[idx].set_ylabel(col)

        plt.tight_layout()
        if show_plots:
            plt.show()

        print('Влияние типа мероприятия на вероятность возвращения пользователя')

        print(user_profiles.groupby(['first_event_type', 'is_two'])['user_id'].count())

        plt.figure(figsize=(12,7))

        sns.countplot(data= user_profiles, x = 'first_event_type', hue = 'is_two', palette= 'viridis', legend = True, alpha= 0.8,)
        plt.title('График распределиня пользователей по типу первого мероприятия')
        plt.xlabel('Тип мероприятия')
        plt.ylabel('Количество пользователей')

        plt.tight_layout()
        if show_plots:
            plt.show()

        print('Влияние региона на вероятность возвращения пользователя')
        plt.figure(figsize=(12,7))

        top_regions = user_profiles['first_region'].value_counts().head(10).index
        df_top_regions = user_profiles[user_profiles['first_region'].isin(top_regions)].copy()
        df_top_regions['first_region'] = df_top_regions['first_region'].cat.remove_unused_categories()

        print(df_top_regions.groupby(['first_region', 'is_two'])['user_id'].count())

        sns.countplot(data = df_top_regions, y = 'first_region', hue = 'is_two', legend = True, alpha = 0.8, palette='viridis')
        plt.title('График распределния пользователей по региону первой покупки')
        plt.xlabel('Регион')
        plt.ylabel('Количество пользователей')

        plt.tight_layout()

        if show_plots:
            plt.show()

        print('Анализ средней выручки и повторными заказами')

        print('График распределения средней выручки с билета')
        plt.figure(figsize = (12,7))

        sns.histplot(data = user_profiles, x = 'avg_revenue', hue = 'is_two', bins = 30, palette='coolwarm', legend=True, alpha = 0.7)

        plt.title('Распределение средней выручки с билета')
        plt.xlabel('Средняя выручка в рублях')
        plt.ylabel('Количество пользователей')

        plt.tight_layout()
        if show_plots:
            plt.show()

        print('Исследование временных характеристик первого зказа и их влияния на повторные покупки')
        plt.figure(figsize=(12,7))

        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        sns.countplot(data = user_profiles.sort_values('order_day_name'), x= 'order_day_name', hue = 'is_two', palette='coolwarm', alpha = 0.7, order= days_order)

        plt.title('Распределение пользователей по дням недели')
        plt.xlabel('День недели первой покупки')
        plt.ylabel('Количество пользователей')

        plt.grid(axis = 'y', linestyle = '--', alpha = 0.5)

        plt.tight_layout()

        print(user_profiles.sort_values('order_day_name').groupby(['order_day_name', 'is_two'])['user_id'].count())

        if show_plots:
            plt.show()

        print('Корреляционный анализ количества покупок и признаков пользователя')

        
        full_phik_matrix = user_profiles[['order_day_name', 'first_device', 'first_region', 'first_event_type', 'avg_revenue', 'avg_days_between_orders', 'total_orders']].phik_matrix() 

        print('Влияние признаков на количество заказов (total_orders):')
        print(full_phik_matrix.loc['total_orders'].drop('total_orders').sort_values(ascending=False))

        terms = [
            (user_profiles['total_orders'] == 1),
            (user_profiles['total_orders'] == 2),
            ((user_profiles['total_orders'] == 3) | (user_profiles['total_orders'] == 4)),
            (user_profiles['total_orders'] > 4)
        ]

        tags = ['1 заказ', '2 заказа', '3-4 заказа', '5 и более заказов']

        user_profiles['total_orders_segment'] = np.select(terms, tags, default='0')

        full_phik_matrix = user_profiles[['order_day_name', 'first_device', 'first_region', 'first_event_type', 'avg_revenue', 'avg_days_between_orders', 'total_orders_segment']].phik_matrix()

        print('Влияние признаков на количество заказов')
        print(full_phik_matrix.loc['total_orders_segment'].drop('total_orders_segment').sort_values(ascending = False))

        plt.figure(figsize = (10,7))

        sns.heatmap(data = full_phik_matrix.loc[['total_orders_segment']].drop('total_orders_segment',axis = 1), annot = True, cmap = 'coolwarm', fmt='.3f', vmin = 0, vmax = 1)

        plt.title('Корреляция признаков пользователя с сегментами активности')

        plt.tight_layout()
        if show_plots:
            plt.show()


    def conclusion(self):
        print('''
Основные результаты

1. Статистический профиль выборки

• Объем данных: Исследовано 21 628 уникальный пользователь (290 849 логов заказов).
• Экономика: Средний чек одного заказа составляет 544.97 рубля
• Частота покупок: Среднее количество заказаов пользователя 6.49

2. Влияние признаков первого заказа на удержание (Retention)

• Тип мероприятия: Главные точки входа по удержанию — театр и выставки. При этом концерты генерируют основной объем трафика, но удерживают аудиторию на среднем уровне.
• Тип устройства: Смартфоны приводят 83% новых пользователей, но доля возврата выше у desktop-версии (64% против 61% у mobile).
• География: Рынок сильно концентрирован — половину трафика обеспечивают Каменевский регион (33%) и Североярская область (17%). Однако аномально высокая доля возврата зафиксирована в небольшом Шанырском регионе (~67%).

3. Влияние временных характеристик

• День недели: Практически не влияет на лояльность клиента (коэффициент = 0.037). Доля возвратов стабильна в любой день.
• Трафик: Суббота — абсолютный пик для привлечения новых пользователей, в то время как в воскресенье наблюдается спад активности.

4. Факторы, связанные с числом покупок

• Драйверы активности: Сильнее всего с объемом заказов связаны метрики avg_days_between_orders (0.429) и avg_revenue (0.288). Чем меньше дней проходит между транзакциями и чем стабильнее чек (средний ценовой сегмент от 300 до 800 руб.), тем выше итоговая активность.
• Неважные факторы: Параметры первого контакта (девайс, день недели, тип события) имеют корреляцию ниже 0.10 и никак не предопределяют лояльность.

Рекомендации

1. Что улучшить (Стратегические приоритеты)

• Удержание в Каменевском и Североярском регионах — ключевой драйвер:
	• Почему: На графике распределения четко видно, что эти два региона суммарно дают львиную долю всей вашей базы. Каменевский регион лидирует с огромным отрывом, и при этом показатель возврата (is_two = 1) там находится на высоком уровне (~63%).
	• Рекомендация: Вместо поиска новых мелких рынков удерживайте лидерство здесь. Любое улучшение возвращения пользователя в Каменевском регионе даже на 1–2% принесет компании больше чистой прибыли, чем стопроцентное удержание во всех мелких округах вместе взятых. Запустите для этих регионов локальные программы лояльности.
• Перезапуск мобильного удержания:
	• Почему: Смартфоны обеспечивают 83% всего входящего трафика, но их возвращение пользователя (61%) уступает десктопу (64%).
	• Рекомендация: Терять клиентов на самой массовой платформе недопустимо. Необходим глубокий аудит мобильного UX/UI: упрощение процесса чекаута, сохранение данных карт, оптимизация скорости загрузки билетов и внедрение push-уведомлений о бронях.
• Театр как эталон удержания:
	• Почему: Театральное направление имеет отличный баланс — большой объем аудитории на графике и один из самых высоких показателей удержания (~63.5%).
	• Рекомендация: Выделите дополнительный бюджет на закупку театрального инвентаря и эксклюзивных партнерств. Эту аудиторию проще и дешевле всего возвращать.

2. Где доработать (Продуктовые изменения)

• Оптимизация воронки Концертов (Высокий трафик, средний возврат):
	• Почему: Концерты — абсолютный лидер по привлечению новичков (почти 6000 пользователей совершили там первую покупку). Однако доля возврата здесь заметно уступает выставкам и театрам.
	• Рекомендация: Концертная аудитория часто бывает «одноразовой» (пришли на конкретного артиста и ушли). Чтобы заставить их вернуться, внедряйте персонализированные товарные рекомендации на основе ИИ (например: «Вы были на рок-концерте, вот скидка 15% на похожую группу в следующем месяце»), привязанные к окну повторной покупки (в среднем 8–16 дней, судя по корреляции).
• Сглаживание «воскресного спада»:
	• Почему: Суббота показывает максимальный пик регистраций и покупок, а в воскресенье активность падает.
	• Рекомендация: Используйте автоматические триггерные рассылки в воскресенье вечером с подборками мероприятий на будущую рабочую неделю, завлекая клиентов скидками «раннего бронирования».

        ''')

            

