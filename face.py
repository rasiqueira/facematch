# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 21:16:42 2019

@author: Rodrigo
"""

import face_recognition
import cv2


def main(args):

	camera_port = 0
 
	nFrames = 30
 
	camera = cv2.VideoCapture(camera_port)
	
	file = "c:\\face_recognition\\documento.png"
		
	print ("Digite <ESC> para sair / <s> para Salvar")	
	
	emLoop= True
	 
	while(emLoop):
	
		retval, img = camera.read()
		cv2.imshow('Foto do documento',img)
	
		k = cv2.waitKey(100)
	
		if k == 27:
			emLoop= False
		
		elif k == ord('s'):
			cv2.imwrite(file,img)
			emLoop= False
	
	cv2.destroyAllWindows()
	camera.release()
	return 0

if __name__ == '__main__':  
    import sys
    main(sys.argv)
    
def main(args):

	camera_port = 0
 
	nFrames = 30
 
	camera = cv2.VideoCapture(camera_port)
	
	file = "c:\\face_recognition\\selfie.png"
		
	print ("Digite <ESC> para sair / <s> para Salvar")	
	
	emLoop= True
	 
	while(emLoop):
	
		retval, img = camera.read()
		cv2.imshow('Selfie',img)
	
		k = cv2.waitKey(100)
	
		if k == 27:
			emLoop= False
		
		elif k == ord('s'):
			cv2.imwrite(file,img)
			emLoop= False
	
	cv2.destroyAllWindows()
	camera.release()
	return 0

if __name__ == '__main__':  
    import sys
    main(sys.argv)



    



try:
    known_image = face_recognition.load_image_file ( "C:\\face_recognition\\documento.png" )
    unknown_image = face_recognition.load_image_file ( "C:\\face_recognition\\selfie.png" )

    doc_encoding = face_recognition.face_encodings (known_image)[0] 
    unknown_encoding = face_recognition.face_encodings (unknown_image)[0] 

    resultados = face_recognition.compare_faces ([doc_encoding], unknown_encoding)

    print("As fotos são da mesma pessoa")
    import Algorithmia
    from filestack import Client
    client = Client("Ap51tCL6uTmyewuphkzPUz")
    params = {"mimetype": "image/png"}
    new_filelink = client.upload(filepath="c:\\face_recognition\\selfie.png", params=params)


    input = new_filelink.url
    client = Algorithmia.client('simtcMfVcIi2EcFstBQAJlyiiKe1')
    algo = client.algo('deeplearning/GenderClassification/2.0.0')
    algo.set_options(timeout=300) # optional
    
    if algo.pipe(input).result['results'][0]['gender'][0]['gender'] == 'Male':
        genero = 'masculino'
    else:
        genero = 'feminino'

    print("O gênero provável é: " + genero)
    input = {
            "image": new_filelink.url
            }
    client = Algorithmia.client('simtcMfVcIi2EcFstBQAJlyiiKe1')
    algo = client.algo('deeplearning/AgeClassification/2.0.0')
    algo.set_options(timeout=300) # optional
    print("O range de idade provável é: " + str(algo.pipe(input).result['results'][0]['age'][0]['ageRange']))
       
except:
    print("As fotos não são da mesma pessoa")
    
    

    

