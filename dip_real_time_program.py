# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 16:57:08 2022

@author: Mensot
"""
import numpy as np
import cv2
import time
from tensorflow.keras.models import load_model

def main():
    
    yolo = cv2.dnn.readNet("custom-yolov4-tiny-detector_best.weights", "custom-yolov4-tiny-detector.cfg")
    cnn_model = load_model('dip_cnn_32.h5')
  
    triedy_klasifikacia = []
    with open("znacky_triedy.txt", "r") as f:
        triedy_klasifikacia = [riadok.strip() for riadok in f.readlines()]

    nazvy_vrstiev = yolo.getLayerNames()
    vystupne_vrstvy = [nazvy_vrstiev[i - 1] for i in yolo.getUnconnectedOutLayers()]
    farby = np.random.uniform(0, 255, size=(len(triedy_klasifikacia), 3))
    font = cv2.FONT_HERSHEY_TRIPLEX
    cas_zacatia = time.time()
    pocet_snimok = 0
    kamera = cv2.VideoCapture(0)

    while True:
        pocet_snimok +=1
        uspesne, img = kamera.read()
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
        yolo.setInput(blob)
        vystupy = yolo.forward(vystupne_vrstvy)
        trieda_ids = []
        istoty = []
        ramceky = []
        h, w = img.shape[:2]
        for vystup in vystupy:
            for detekcia in vystup:
                skore = detekcia[5:]
                trieda_id = np.argmax(skore)
                istota = skore[trieda_id]
                if istota > 0.5:
                    ramcek = detekcia[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = ramcek.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    ramcek = [x, y, int(width), int(height)]
                    ramceky.append(ramcek)
                    istoty.append(float(istota))
                    trieda_ids.append(trieda_id)
        indexy = cv2.dnn.NMSBoxes(ramceky, istoty, 0.5, 0.4)

        if len(indexy) > 0:
            for i in indexy.flatten():
                (x, y) = (ramceky[i][0], ramceky[i][1])
                (w, h) = (ramceky[i][2], ramceky[i][3])
                
                orezany_img = img[y:y+h, x:x+w]
                if len(orezany_img) > 0:
                    try:
                        orezany_img = cv2.resize(orezany_img, (32, 32))
                        orezany_img =  orezany_img.reshape(-1, 32, 32, 3)
                        predikcia_pole = cnn_model.predict(orezany_img)
                        percento = max(predikcia_pole[0])
                        if percento > 0.8:
                            predikcia = np.argmax(predikcia_pole)
                            percento = "{:.2f}".format(percento*100)
                            popis = str(percento) + "% " + str(triedy_klasifikacia[predikcia]) 
                            farba = farby[predikcia]
                            cv2.rectangle(img, (x, y), (x + w, y + h), farba, 2)
                            cv2.putText(img, popis, (x, y), font, 0.5, farba, 1)
                    except Exception as e:
                        print(str(e))

        cas_uplynuty = time.time() - cas_zacatia
        fps = pocet_snimok/cas_uplynuty
        cv2.putText(img, str(round(fps)) + " FPS", (10, 50), font, 1, (100, 255, 0), 1)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord ('q'):
            break
    cv2.destroyAllWindows()
    
main()
