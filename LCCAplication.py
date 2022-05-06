from tkinter import *
from tkinter import messagebox, Canvas, simpledialog
from PIL import Image,ImageTk
import multiprocessing

from sklearn.metrics.pairwise import pairwise_distances
from tensorflow.python.platform import gfile
import tensorflow.compat.v1 as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
import numpy as np
import detect_and_align as detectar_y_alinear
import cv2
import os
import imutils

import pymysql
from datetime import datetime

from DatosPersona import IdPersona, cargar_modelo

class LCCAplication(Frame):
    def __init__(self,root=None, model=None,font=None, id_folder = None, umbral=None):
        super().__init__(root)
        self.root=root
        cap = None
        self.id_folder = id_folder      
        self.umbral = umbral
        self.model = model
        self.font = font
    
        "El primer modelo es el sistema de reconocimiento facial para identificar a alumnos de LCC"
        with tf.Graph().as_default():
                self.sess = tf.Session()
                self.mtcnn = detectar_y_alinear.create_mtcnn(self.sess, None)
                cargar_modelo(self.model)
                self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                self.id_data = IdPersona(id_folder[0], self.mtcnn, self.sess, self.embeddings, self.images_placeholder, self.phase_train_placeholder, self.umbral)

        "El segundo modelo se utilizará como un activador, al no haber botones, se utilizarán gestos de la mano para realizar acciones"
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mpDraw = mp.solutions.drawing_utils
        self.modelo_manos = load_model('mp_hand_gesture')
        self.f = open('gesture.names', 'r')
        self.classNames = self.f.read().split('\n')
        self.f.close()
        
        "Se cargan todos los componentes de tkinter"
        
        
        
        #Variables para datos del alumno identificado
        self.creditos_totales = 383
        self.lcc_matricula = IntVar()
        self.lcc_matricula_identificado = IntVar()
        self.lcc_nombre_identificado = StringVar()
        self.lcc_apellido_identificado = StringVar()
        self.lcc_creditos_identificado = IntVar()
        self.lcc_kardex_identificado = DoubleVar()
        self.lcc_sc_identificado = StringVar()
        self.lcc_pp_identificado = StringVar()
        self.lcc_fecha_ultimoingreso = StringVar()
        self.saludocompleto = StringVar()
        
        #Iniciamos las variables en blanco 
        self.lcc_matricula_identificado.set("")
        self.lcc_nombre_identificado.set("")
        self.lcc_apellido_identificado.set("")
        self.lcc_creditos_identificado.set("")
        self.lcc_kardex_identificado.set("")
        self.lcc_sc_identificado.set("")
        self.lcc_pp_identificado.set("")
        self.lcc_fecha_ultimoingreso.set("")
        self.saludocompleto.set("")
                
        self.canvas = Canvas(
            self.root,
            bg = "#ffffff",
            height = 650,
            width = 1200,
            bd = 0,
            highlightthickness = 0,
            relief = "ridge")
        self.canvas.place(x = 0, y = 0)
        background_img = PhotoImage(file = f"./interfaz/background.png")   
        background = self.canvas.create_image(
            600.0, 301.0,
            image=background_img)
        self.saludo = Label(root, textvariable = self.saludocompleto,
                       font = ("None", int(24.0)),
                       background="#ffffff").place(x=550,y=170)
        self.canvas.create_text(
            865.0, 336.0,
            text = "Creditos ",
            fill = "#000000",
            font = ("None", int(24.0)))
        self.creditos = Label(root, textvariable = self.lcc_creditos_identificado,
                       font = ("None", int(24.0)),
                       background="#ffffff").place(x=735, y=351)
        self.canvas.create_text(
            670.0, 424.5,
            text = "Servicio social: ",
            fill = "#000000",
            font = ("None", int(24.0)))
        self.canvas.create_text(
            1015.5, 424.5,
            text = "Practicas profesionales:",
            fill = "#000000",
            font = ("None", int(24.0)))
        self.servicio_social = Label(root, textvariable = self.lcc_sc_identificado,
                       font = ("None", int(24.0)),
                       background="#ffffff").place(x= 640,y=450)
        self.practicas_profesionales = Label(root, textvariable = self.lcc_pp_identificado,
                       font = ("None", int(24.0)),
                       background="#ffffff").place(x=985,y=450)
        img0 = PhotoImage(file = f"./interfaz/img0.png")
        self.btnFinalizar = Button(
            image = img0,
            borderwidth = 0,
            highlightthickness = 0,
            command = self.finalizar,
            relief = "flat")
        self.btnFinalizar.place(
            x = 967, y = 564,
            width = 174,
            height = 34)
        img1 = PhotoImage(file = f"./interfaz/img1.png")
        self.btnIniciar = Button(
            image = img1,
            borderwidth = 0,
            highlightthickness = 0,
            command = self.iniciar,
            relief = "flat")
        self.btnIniciar.place(
            x = 672, y = 564,
            width = 197,
            height = 34)      
        self.lblVideo = Label(self.root, background="#bababa")
        self.lblVideo.place(
            x = 16, y = 151,
            width = 500,
            height = 490)    
        
        self.CargarTkinter()
       
    def iniciar(self):
        global cap
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.visualizar()
    
    def finalizar(self):
        global cap
        cap.release()
        self.lcc_nombre_identificado.set("")
        self.lcc_apellido_identificado.set("")
        self.lcc_creditos_identificado.set("")
        self.lcc_kardex_identificado.set("")
        self.btnIniciar.config(state="normal")
        self.btnFinalizar.config(state="disabled")
       
    def CargarTkinter(self):
        window.resizable(False, False)
        window.mainloop()
    def visualizar(self):
        if cap is not None:
            ret, frame = cap.read()
            frame = cv2.flip(frame,1)

            if ret == True:
                face_patches, cuadros_delimitadores, puntos_referencia = detectar_y_alinear.detect_faces(frame, self.mtcnn)
                if len(face_patches) > 0:
                    
                    #Codigo por comentar....
                    face_patches = np.stack(face_patches)
                    feed_dict = {self.images_placeholder: face_patches, self.phase_train_placeholder: False}
                    embs = self.sess.run(self.embeddings, feed_dict=feed_dict)

                    personas_reconocidas, distancias_personas_reconocidas = self.id_data.find_matching_ids(embs)
                    """
                    posicion_cara: Todas las caras reconocidas en el frame, será un arreglo de indice 4, donde estará su posición (x,y) y su tamaño
                    persona_reconocida: Es una etiqueta con el nombre de la persona reconocida
                    """
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    for posicion_cara, _, persona_reconocida, distancia in zip(
                        cuadros_delimitadores, puntos_referencia, personas_reconocidas, distancias_personas_reconocidas
                    ):
                        if posicion_cara[2] > 300 and posicion_cara[3] > 300:
                            if persona_reconocida is None:
                                persona_reconocida = "Desconocido"
                                #print("Desconocido! No se pudo reconocer el rostro.")
                            else:
                                xd = 1
                               #print("Hola %s! Acuraccy: %1.4f" % (persona_reconocida, distancia))
                            #Una vez reconocido el rostro, se va a mostrar en la cámara: Su nombre, Un cuadrado identificado al rostro y un pequeño mensaje
                            cv2.rectangle(frame, (posicion_cara[0], posicion_cara[1]), (posicion_cara[2], posicion_cara[3]), (0, 255, 0), 4)
                            cv2.putText(frame, persona_reconocida, (posicion_cara[0] + 15, posicion_cara[1]-5), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
                        else:
                            cv2.putText(frame, "Acercate a la camara", (20,20), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                            cv2.rectangle(frame, (posicion_cara[0], posicion_cara[1]), (posicion_cara[2], posicion_cara[3]), (0, 255, 0), 4)
                #Verificamos gestos de la mano
                x, y, c = frame.shape
                framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.hands.process(framergb)
                className = ''
                if result.multi_hand_landmarks:
                    landmarks = []
                    for handslms in result.multi_hand_landmarks:
                        for lm in handslms.landmark:
                            lmx = int(lm.x * x)
                            lmy = int(lm.y * y)
                            landmarks.append([lmx, lmy])

                        # Drawing landmarks on frames
                        self.mpDraw.draw_landmarks(frame, handslms, self.mpHands.HAND_CONNECTIONS)

                        # Predict gesture
                        prediction = self.modelo_manos.predict([landmarks])
                        # print(prediction)
                        classID = np.argmax(prediction)
                        className = self.classNames[classID]
                #cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                #   1, (0,0,255), 2, cv2.LINE_AA)
                if (className == 'thumbs up'):
                    print("Gracias por tu reseña positiva")
                if (className == 'live long'):
                    print("Buscando tu informacion...")  
                    self.BuscarDatosAlumno()
                    
                if (className == 'thumbs down'):
                    print("Este error ayudara a que la inteligencia artificial se mejore")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                im = Image.fromarray(frame)
                img = ImageTk.PhotoImage(image=im)
                self.lblVideo.configure(image=img)
                self.lblVideo.image = img
                self.lblVideo.after(10, self.visualizar) 
                
            else:
                self.lblVideo.image = ""
                cap.release()   
    
    
    def BuscarDatosAlumno(self):
        db = pymysql.connect(host = 'localhost', 
                         user= 'root',
                         password='xSK!NyF@pU#sD&L', 
                         database='alumnos_lcc',connect_timeout=300)
        cur = db.cursor()
        sql = 'SELECT * FROM alumnos_lcc.alumno INNER JOIN alumnos_lcc.registros ON alumnos_lcc.alumno.idalumno =alumnos_lcc.registros.id_alumno WHERE idalumno = {}'.format('219205955')
        cur.execute(sql)
        results = cur.fetchall()
        if not results:
            messagebox.showinfo(message="Tus datos no están capturados correctamente. Contactate con el administrador", title="¡Error!")
            return
    
        self.lcc_nombre_identificado.set(results[0][1])
        self.lcc_apellido_identificado.set(results[0][2])       
        self.lcc_creditos_identificado.set(str(results[0][3]) + '/{} - {}%'.format(self.creditos_totales, round(results[0][3]/self.creditos_totales*100,2) ))
        self.lcc_kardex_identificado.set(results[0][4])
        self.lcc_fecha_ultimoingreso.set(results[0][6])
        if results[0][3] < int(self.creditos_totales * .8):
            self.lcc_sc_identificado.set("No aplica ")
        if results[0][3] >= int(self.creditos_totales * .8):
            self.lcc_sc_identificado.set("Si aplica")
            
        if results[0][3] < int(self.creditos_totales * .9):
            self.lcc_pp_identificado.set("No aplica ")
        if results[0][3] >= int(self.creditos_totales * .9):
            self.lcc_pp_identificado.set("Si aplica")
        self.saludar()
    def saludar(self):
        self.saludocompleto.set(
            "Hola {} {}\nLa ultima vez que te vi fue el\n{}".format(
                self.lcc_nombre_identificado.get(),
                self.lcc_apellido_identificado.get(),
                self.lcc_fecha_ultimoingreso.get())
        )
        
if __name__ == "__main__":
    cap = None   
    window = Tk()    
    window.geometry("1200x650")
    window.configure(bg = "#ffffff") 
    LCCAplication(window,model='./model/20170512-110547.pb',id_folder=['./ids/'],umbral=1.0  )