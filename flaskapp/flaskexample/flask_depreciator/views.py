# Create my flask app
    
from flask import render_template
from flask import request
from flask_depreciator import app
from sqlalchemy import create_engine
import pandas as pd
import psycopg2

from flask_depreciator.a_Model import ModelIt

# Python code to connect to Postgres
# You may need to modify this based on your OS, 
# as detailed in the postgres dev setup materials.
user = 'michaelboles' #add your Postgres username here      
host = 'localhost'
dbname = 'fit_data_6_clean'
db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
con = None
con = psycopg2.connect(database = dbname, user = user, host = host, password = 'sqlpassword') #add your Postgres password here

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html",
       title = 'Home', user = { 'nickname': 'Miguel' },
       )

@app.route('/db')
def makes_page():
    sql_query = """                                                             
                SELECT * FROM fit_data_6_clean_data_table;                                                                               
                """
    makes_results = pd.read_sql_query(sql_query,con)
    makes = ""
    print(makes_results)
    for i in range(len(makes_results)):
        makes += makes_results.iloc[i]['Make']
        makes += "<br>"
    return makes

@app.route('/db_fancy')
def cesareans_page_fancy():
    sql_query = """
               SELECT index, attendant, birth_month FROM birth_data_table WHERE delivery_method='Cesarean';
                """
    query_results=pd.read_sql_query(sql_query,con)
    births = []
    for i in range(0,query_results.shape[0]):
        births.append(dict(index=query_results.iloc[i]['index'], attendant=query_results.iloc[i]['attendant'], birth_month=query_results.iloc[i]['birth_month']))
    return render_template('cesareans.html',births=births)
    
@app.route('/input')
def cesareans_input():
    return render_template("input.html")

@app.route('/output')
def cesareans_output():
  #pull 'birth_month' from input field and store it
  patient = request.args.get('birth_month')
    #just select the Cesareans  from the birth dtabase for the month that the user inputs
  query = "SELECT index, attendant, birth_month FROM birth_data_table WHERE delivery_method='Cesarean' AND birth_month='%s'" % patient
  print(query)
  query_results=pd.read_sql_query(query,con)
  print(query_results)
  births = []
  for i in range(0,query_results.shape[0]):
      births.append(dict(index=query_results.iloc[i]['index'], attendant=query_results.iloc[i]['attendant'], birth_month=query_results.iloc[i]['birth_month']))
      the_result = ModelIt(patient,births)
      return render_template("output.html", births = births, the_result = the_result)
