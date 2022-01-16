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
import os
from facebook_scraper import get_profile
import cv2 as cv
import math
import time
import cv2
import urllib.request
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from io import BytesIO
import PIL.Image as pilimage
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow.keras as keras
from tensorflow.keras.utils import plot_model
from tqdm.notebook import tqdm
import math as mt
import itertools

abs_path=os.path.abspath(os.path.dirname(__file__))

information = APIRouter()

def infoLinkedin(link: str):
    abs_path=os.path.abspath(os.path.dirname(__file__))
    filename_path=os.path.join(abs_path, "chromedriver.exe")
    ser = Service(filename_path)
    options = webdriver.ChromeOptions()
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    browser = webdriver.Chrome(options=options,service=ser)
    browser.get('https://www.linkedin.com/uas/login')
    abs_path=os.path.abspath(os.path.dirname(__file__))
    filename_path=os.path.join(abs_path, "chromedriver")
    #os.chmod(filename_path,stat.S_IRWXU)
    #os.environ["PATH"] += os.pathsep + filename_path
    #print(os.environ["PATH"])
    ser = Service(filename_path)
    options = webdriver.ChromeOptions()
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    browser = webdriver.Chrome(options=options,service=ser)
    #browser = webdriver.Chrome(executable_path='/path/to/chromedriver')
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
    SCROLL_PAUSE_TIME=5
    last_height=browser.execute_script("return document.body.scrollHeight")
    for i in range(3):
        browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_TIME)
        new_height=browser.execute_script("return document.body.scrollHeight")
        if new_height==last_height:
            break
        last_height=new_height
    ###INFO###
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
    secciones=soup.find_all('section',{'class':"artdeco-card ember-view break-words pb3 mt4"})
    for sec in secciones:
        section=sec.find('div')['id']
        if section == 'recent_activity':
            continue
        if section == 'about':
            text=sec.find('div',{'class':"display-flex ph5 pv3"})
            span=text.find('span',{'class':"visually-hidden"}).get_text().strip()
            info["about"]=span
        if section == 'experience':
            experiences={}
            exp=sec.find('ul',{'class':"pvs-list ph5 display-flex flex-row flex-wrap"})
            jobs=exp.find_all('li',{'class':"artdeco-list__item pvs-list__item--line-separated pvs-list__item--one-column"})
            for job in jobs:
                company=job.find('a',{'data-field':"experience_company_logo"})
                logo=company.find('img')['src']
                company=company['href']
                texts=job.find_all('span',{'class':'visually-hidden'})
                trabajo=''
                for i,text in zip(range(len(texts)),texts):
                    if i ==0 :
                        trabajo=text.get_text().strip()
                        experiences[text.get_text().strip()]=[]
                    else:
                        if ',' in text.get_text().strip():
                            continue
                        else:
                            experiences[trabajo].append(text.get_text().strip())
                experiences[trabajo].append(company)
                experiences[trabajo].append(logo)
            info["experience"]=experiences
        if section == 'education':
            educations={}
            exp=sec.find('ul',{'class':"pvs-list ph5 display-flex flex-row flex-wrap"})
            edus=exp.find_all('li',{'class':"artdeco-list__item pvs-list__item--line-separated pvs-list__item--one-column"})
            for edu in edus:
                company=edu.find('a',{'class':"optional-action-target-wrapper display-flex"})
                logo=company.find('img')['src']
                company=company['href']
                texts=edu.find_all('span',{'class':'visually-hidden'})
                trabajo=''
                for i,text in zip(range(len(texts)),texts):
                    if i ==0 :
                        trabajo=text.get_text().strip()
                        educations[text.get_text().strip()]=[]
                    else:
                        if '.pdf' in text.get_text().strip():
                            continue
                        else:
                            educations[trabajo].append(text.get_text().strip())
                educations[trabajo].append(company)
                educations[trabajo].append(logo)        
            info["education"]=educations
        if section == "licenses_and_certifications":
            certifications={}
            exp=sec.find('ul',{'class':"pvs-list ph5 display-flex flex-row flex-wrap"})
            certis=exp.find_all('li',{'class':"artdeco-list__item pvs-list__item--line-separated pvs-list__item--one-column"})
            for certi in certis:
                company=certi.find('a',{'data-field':"entity_image_licenses_and_certifications"})
                logo=company.find('img')['src']
                company=company['href']
                texts=certi.find_all('span',{'class':'visually-hidden'})
                trabajo=''
                for i,text in zip(range(len(texts)),texts):
                    if i ==0 :
                        trabajo=text.get_text().strip()
                        certifications[text.get_text().strip()]=[]
                    else:
                        if '.pdf' in text.get_text().strip() or i==3:
                            continue
                        else:
                            certifications[trabajo].append(text.get_text().strip())
                certifications[trabajo].append(company)
                certifications[trabajo].append(logo)        
            info["licenses_and_certifications"]=certifications        
        if section == "volunteering_experience":
            volunteers={}
            exp=sec.find('ul',{'class':"pvs-list ph5 display-flex flex-row flex-wrap"})
            volunts=exp.find_all('li',{'class':"artdeco-list__item pvs-list__item--line-separated pvs-list__item--one-column"})
            for vol in volunts:
                company=vol.find('a',{'data-field':"entity_image_volunteer_experiences"})
                if company is None:
                    pass
                else:
                    logo=company.find('img')['src']
                    company=company['href']
                texts=vol.find_all('span',{'class':'visually-hidden'})
                trabajo=''
                for i,text in zip(range(len(texts)),texts):
                    if i ==0 :
                        trabajo=text.get_text().strip()
                        volunteers[text.get_text().strip()]=[]
                    else:
                        if '.pdf' in text.get_text().strip():
                            continue
                        else:
                            volunteers[trabajo].append(text.get_text().strip())
                if company is None:
                    pass
                else:
                    volunteers[trabajo].append(company)
                    volunteers[trabajo].append(logo)        
            info["volunteering_experience"]=volunteers
        if section == "skills":
            skills=[]
            exp=sec.find('ul',{'class':"pvs-list ph5 display-flex flex-row flex-wrap"})
            skill=exp.find_all('li',{'class':"artdeco-list__item pvs-list__item--line-separated pvs-list__item--one-column"})
            for skil in skill:
                texts=skil.find_all('span',{'class':'visually-hidden'})
                trabajo=''
                for i,text in zip(range(len(texts)),texts):
                    if '·' in text.get_text().strip():
                        continue
                    else:
                        skills.append(text.get_text().strip())
            info["skills"]=skills
        if section == "publications":
            publications={}
            exp=sec.find('ul',{'class':"pvs-list ph5 display-flex flex-row flex-wrap"})
            publication=exp.find_all('li',{'class':"artdeco-list__item pvs-list__item--line-separated pvs-list__item--one-column"})
            for pub in publication:
                texts=pub.find_all('span',{'class':'visually-hidden'})
                trabajo=''
                for i,text in zip(range(len(texts)),texts):
                    if i ==0 :
                        trabajo=text.get_text().strip()
                        publications[text.get_text().strip()]=[]
                    else:
                        publications[trabajo].append(text.get_text().strip())
            info["publications"]=publications
        if section == "projects":
            projects={}
            exp=sec.find('ul',{'class':"pvs-list ph5 display-flex flex-row flex-wrap"})
            project=exp.find_all('li',{'class':"artdeco-list__item pvs-list__item--line-separated pvs-list__item--one-column"})
            for proj in project:
                texts=proj.find_all('span',{'class':'visually-hidden'})
                trabajo=''
                for i,text in zip(range(len(texts)),texts):
                    if i ==0 :
                        trabajo=text.get_text().strip()
                        projects[text.get_text().strip()]=[]
                    else:
                        projects[trabajo].append(text.get_text().strip())
            info["projects"]=projects
        if section == "honors_and_awards":
            awards={}
            exp=sec.find('ul',{'class':"pvs-list ph5 display-flex flex-row flex-wrap"})
            award=exp.find_all('li',{'class':"artdeco-list__item pvs-list__item--line-separated pvs-list__item--one-column"})
            for awa in award:
                texts=awa.find_all('span',{'class':'visually-hidden'})
                trabajo=''
                for i,text in zip(range(len(texts)),texts):
                    if i ==0 :
                        trabajo=text.get_text().strip()
                        awards[text.get_text().strip()]=[]
                    else:
                        awards[trabajo].append(text.get_text().strip())
            info["honors_and_awards"]=awards
        if section == "languages":
            languages=[]
            exp=sec.find('ul',{'class':"pvs-list ph5 display-flex flex-row flex-wrap"})
            language=exp.find_all('li',{'class':"artdeco-list__item pvs-list__item--line-separated pvs-list__item--one-column"})
            for lang in language:
                texts=lang.find_all('span',{'class':'visually-hidden'})
                trabajo=''
                for i,text in zip(range(len(texts)),texts):
                    if i!=0:
                        continue
                    else:
                        languages.append(text.get_text().strip())        
            info["languages"]=languages
        if section == "interests":
            interests=[]
            exp=sec.find('ul',{'class':"pvs-list ph5 display-flex flex-row flex-wrap"})
            interest=exp.find_all('li',{'class':"artdeco-list__item pvs-list__item--line-separated pvs-list__item--two-column"})
            for inte in interest:
                texts=inte.find_all('span',{'class':'visually-hidden'})
                trabajo=''
                for i,text in zip(range(len(texts)),texts):
                    if 'seguidores' in text.get_text().strip():
                        continue
                    else:
                        interests.append(text.get_text().strip())  
            info["interests"]=interests
    
    return info

def infoFacebook(info: dict,profile_id:str):
    abs_path=os.path.abspath(os.path.dirname(__file__))
    filename_path=os.path.join(abs_path, "facebook.com_cookies.txt")
    profile_id = profile_id
    profile = get_profile(profile_id, cookies=filename_path,   friends=True) 
    dictionary=list(profile.keys())
    if 'Basic Info' in dictionary:
        if 'Gender' in list(profile['Basic Info'].keys()):
            info['gender']=profile['Basic Info']['Gender']
    if 'Basic Info' in dictionary:
        if 'Birthday' in list(profile['Basic Info'].keys()):
            info['birthday']=profile['Basic Info']['Birthday']
    if 'Contact Info' in dictionary:
        if 'Facebook' in list(profile['Contact Info'].keys()):
            info['facebook']=profile['Contact Info']['Facebook']
    if 'Relationship' in dictionary:
        info['relationship']=profile['Relationship']
    return info

def age(info:dict):
    abs_path=os.path.abspath(os.path.dirname(__file__))
    def getFaceBox(net, frame, conf_threshold=0.7):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
        net.setInput(blob)
        detections = net.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append([x1, y1, x2, y2])
                cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
        return frameOpencvDnn, bboxes
    faceProto = os.path.join(abs_path, "opencv_face_detector.pbtxt")
    faceModel = os.path.join(abs_path, "opencv_face_detector_uint8.pb")
    ageProto = os.path.join(abs_path, "age_deploy.prototxt")
    ageModel = os.path.join(abs_path, "age_net.caffemodel")
    genderProto = os.path.join(abs_path, "gender_deploy.prototxt")
    genderModel = os.path.join(abs_path, "gender_net.caffemodel")
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(15-20)','(21-24)', '(25-32)','(33-37)', '(38-43)','(44-47)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']
    ageNet = cv.dnn.readNet(ageModel, ageProto)
    genderNet = cv.dnn.readNet(genderModel, genderProto)
    faceNet = cv.dnn.readNet(faceModel, faceProto)
    padding = 20
    def age_gender_detector(frame):
        t = time.time()
        frameFace, bboxes = getFaceBox(faceNet, frame)
        for bbox in bboxes:
            face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            label = "{},{}".format(gender, age)
            cv.putText(frameFace, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
        return age
    url=info["image"]
    url_response = urllib.request.urlopen(url)
    img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, -1)
    output = age_gender_detector(img)
    info['age']=output
    return info

def scrap_facebook(subsets, subsets_comp, browser, img_sel):
    # Lists to store the scraped data in
    names = []
    ids = []
    img_urls = []
    similar = []
    for option_name in subsets:
        link='https://es-la.facebook.com/public/'+option_name
        browser.get(link)
        src=browser.page_source
        soup=BeautifulSoup(src,'lxml')
        profile_containers=soup.find_all('div',{'class':'_3u1 _gli'})
        # Extract data from individual movie container
        for container in profile_containers[:2]:
        # If the movie has Metascore, then extract:
            if container.find('div', class_ = 'clearfix _ikh') is not None:
                name = container.a['title']
                id = container.a['href']
                img_url = container.img['src']
                if(id not in ids):
                    equal = False
                    for i in subsets_comp:
                        if(name==i):
                            similar.append(id)
                            equal = True
                    if(equal==False):
                        names.append(name)
                        ids.append(id)
                        img_urls.append(img_url)

    img_urls.append(img_sel)
    return names,ids,img_urls,similar

def save_images(img_urls, names):
    filenames = []
    i = 0
    for url in tqdm(img_urls):
        response = requests.get(url)
        img = pilimage.open(BytesIO(response.content))
        img = img.resize((320,320))
        name = 'linkedin'
        if(i<len(names)):
            name = '_'.join(names[i])
        loc = os.path.join(abs_path, 'Images/'+name+'_'+str(i)+'.jpg')
        img.save(loc)
        filenames.append(loc)
        i=i+1
    return filenames

def get_features(filenames, target_size, model):
    features = []
    for filename in tqdm(filenames):
        img = keras.preprocessing.image.load_img(filename, target_size=target_size)
        x = keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features.append(model.predict(x).ravel())
    features = np.array(features)
    return features

def get_similarity(filenames, features):
    idx = len(filenames)-1
    f1=features[idx]**2
    f1=np.sum(f1)
    d=[]
    for i in range(len(features)):
        f2 = features[i]**2
        f2 = np.sum(f2)
        dist = features[idx] * features[i]
        dist = np.sum(dist) 
        dist = dist / mt.sqrt(f1*f2)
        d.append((dist,i))
    d_s=sorted(d,reverse=True)
    return d_s[1:]

def get_possible_profiles(name, img_sel):
    name_split = name.split(" ")
    # Possible names 
    subsets=[]
    subsets_comp=[]
    for L in range(0, len(name_split)+1):
        for subset in itertools.combinations(name_split, L):
            if(len(subset)>2):
                subsets.append('-'.join(subset))
                subsets_comp.append(' '.join(subset))
    abs_path=os.path.abspath(os.path.dirname(__file__))
    filename_path=os.path.join(abs_path, "chromedriver.exe")
    ser = Service(filename_path)
    options = webdriver.ChromeOptions()
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    browser = webdriver.Chrome(options=options,service=ser)

    names,ids,img_urls,similar = scrap_facebook(subsets, subsets_comp, browser, img_sel)
       
    model = keras.models.load_model(os.path.join(abs_path, "resnet_train.h5"), compile=False)

    filenames = save_images(img_urls, names)
    
    features = get_features(filenames, (320,320), model)

    d_s = get_similarity(filenames, features)

    l = len(similar)
    first = True
    # Add based on image
    for i in range(5-l):
        if(i<len(d_s)):
            if(first & len(similar)>0):
                if (ids[d_s[i][1]] != similar[0]):
                    similar.append(ids[d_s[i][1]])
            else:
                similar.append(ids[d_s[i][1]])
        else:
            break
        first = False
    return similar


@information.get("/info/")
def infoAll(link:str):
    ##Linkedin
    info=infoLinkedin(link)
    similar = get_possible_profiles(info['name'], info['image'])
    position=similar[0].rfind("/")
    ##Facebook
    profile_id=similar[0][position+1:]
    info2=infoFacebook(info,profile_id)
    ##Age
    info3=age(info)
    return JSONResponse(info3)