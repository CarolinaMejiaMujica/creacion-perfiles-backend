from fastapi import APIRouter,Response
import pandas as pd
from starlette.responses import JSONResponse
import selenium
import requests, time, random
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome import service
from selenium.webdriver.chrome.service import Service
import os, sys, stat

information = APIRouter()

@information.get("/info/")
def cantidades(link: str):
    abs_path=os.path.abspath(os.path.dirname(__file__))
    filename_path=os.path.join(abs_path, "chromedriver")
    os.chmod(filename_path,stat.S_IRWXU)
    os.environ["PATH"] += os.pathsep + filename_path
    print(os.environ["PATH"])
    ser = Service(filename_path)
    options = webdriver.ChromeOptions()
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    browser = webdriver.Chrome(options=options,service=ser)
    #browser = webdriver.Chrome(executable_path=filename_path)
    browser.get('https://www.linkedin.com/uas/login')
    username='abc.inf.2022@gmail.com'
    password='hackathon/2022'
    elementID=browser.find_element_by_id('username')
    elementID.send_keys(username)
    elementID=browser.find_element_by_id('password')
    elementID.send_keys(password)
    elementID.submit()
    browser.get(link)
    ###Time scroll
    '''SCROLL_PAUSE_TIME=5
    last_height=browser.execute_script("return document.body.scrollHeight")

    for i in range(3):
        browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_TIME)
        new_height=browser.execute_script("return document.body.scrollHeight")
        if new_height==last_height:
            break
        last_height=new_height
    #READ HTML
    src=browser.page_source
    soup=BeautifulSoup(src,'lxml')
    info = {"name":''}
    #NAME
    name_div=soup.find('div',{'class':'pv-text-details__left-panel'})
    name_loc=name_div.find_all('h1')
    name=name_loc[0].get_text().strip()
    info["name"]=name
    #IMAGE
    image=soup.find('img',{'title':name})
    img=image['src']
    info["image"]=img
    #PROFESSION
    profession=name_div.find_all("div")
    profession=profession[1].get_text().strip()
    info["profession"]=profession
    #LOCATION
    location_div=soup.find('div',{'class':'pb2 pv-text-details__left-panel'})
    location=location_div.find('span').get_text().strip()
    info["location"]=location
    #LINK
    link=location_div.find('a')
    #INFO CONTACTO
    info_contact='https://www.linkedin.com' + str(link['href'])
    info["link"]=info_contact
    ###SECCIONES###
    secciones=soup.find_all('section',{'class':"artdeco-card ember-view break-words pb3 mt4"})'''

    return JSONResponse(content={"cantidadTotal":5,"cantidadAnalisis":3})

 