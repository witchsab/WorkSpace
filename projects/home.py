from flask import Blueprint, request, render_template

import pandas as pd
import pandas_datareader.stockdata as web
import datetime

home_api = Blueprint('home_api', __name__)

@home_api.route("/pd")
def panda():
    TICK = "KLAC"    
    return getTickData(TICK)


@home_api.route("/pd/query", methods=['GET', 'POST'])
def query (): 
    TICK = request.args.get('TICK')
    return getTickData(TICK)


# Function: get historic data for TICK
def getTickData(TICK):
    
    # TICK = "AMD"
    s = "Hello World! this is my personal digital ocean!"
    start = datetime.datetime(2019,4,1)
    end = datetime.datetime.today()
    att = web.DataReader(TICK, "yahoo", start, end)
    return "<h1 style='color:red'>PANDAS TEST v.1.0</h1><h2>"+TICK+"</h2><p>"+att.to_html()+ "</p>"
