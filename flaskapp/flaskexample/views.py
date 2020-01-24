#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 13:03:04 2020

@author: michaelboles
"""

# # basic setup 

# from flask import render_template
# from flaskexample import app
# # from flask import request

# @app.route('/')
# @app.route('/index')
# def index():
#     user = { 'nickname': 'Miguel' } # fake user
#     return render_template("index.html", title = 'Home', user = user)



from flask import render_template
from flaskexample import app
from sqlalchemy import create_engine
# from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2

# Python code to connect to Postgres
# You may need to modify this based on your OS, 
# as detailed in the postgres dev setup materials.
user = 'michaelboles' #add your Postgres username here      
host = 'localhost'
dbname = 'sample_db'
db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
con = None
con = psycopg2.connect(database = dbname, 
                       user = user, 
                       host = host, 
                       password = 'mypassword') #add your Postgres password here

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html",
       title = 'Home', user = { 'nickname': 'Miguel' },
       )

@app.route('/db')
def birth_page():
    sql_query = """                                                                       
                SELECT * FROM birth_data_table WHERE delivery_method='Cesarean';          
                """
    query_results = pd.read_sql_query(sql_query,con)
    births = ""
    for i in range(0,10):
        births += query_results.iloc[i]['birth_month']
        births += "<br>"
    return births


# @app.route('/input')
# def cesareans_input():
#     return render_template("input.html")

# @app.route('/output')
# def cesareans_output():
#     return render_template("output.html")