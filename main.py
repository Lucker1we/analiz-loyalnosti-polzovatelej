import pandas as pd
from sqlalchemy import create_engine
import psycopg
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import phik

import os
from dotenv import load_dotenv

from src.processing import DataProcessor

def main():
    analytics = DataProcessor()

    df = analytics.load_db(sql_base=True)
    tenge_df = analytics.load_converter_df()
    df = analytics.clean_data(df, tenge_df)
    user_profiles = analytics.create_profiles(df)
    analytics.retention_analysis(user_profiles)
    analytics.conclusion()

if __name__ == '__main__':
    main()