from sklearn.metrics.pairwise import pairwise_distances
from tensorflow.python.platform import gfile
import tensorflow.compat.v1 as tf
import numpy as np
import detect_and_align as detectar_y_alinear

import cv2
import os

#Librerias gráficas para presentar una interfaz de usuario...
from tkinter import *
from tkinter import messagebox, Canvas, simpledialog
from PIL import Image,ImageTk
import imutils

#importamos la libreria para acceder a la base de datos

#Modelo preentrenado de clasificador en cascada para detectar una sonrisa...
smileCascade = cv2.CascadeClassifier('haarcascade_smile.xml')



class IdData:
    """Keeps track of known identities and calculates id matches"""

    def __init__(
        self, id_folder, mtcnn, sess, embeddings, images_placeholder, phase_train_placeholder, distancia_umbral
    ):
        print("Cargando todas las personas conocidas: ", end="")
        self.distancia_umbral = distancia_umbral
        self.id_folder = id_folder
        self.mtcnn = mtcnn
        self.id_names = []
        self.embeddings = None

        carpeta_imagenes = []
        os.makedirs(id_folder, exist_ok=True)
        ids = os.listdir(os.path.expanduser(id_folder))
        if not ids:
            return

        for id_name in ids:
            id_dir = os.path.join(id_folder, id_name)
            carpeta_imagenes = carpeta_imagenes + [os.path.join(id_dir, img) for img in os.listdir(id_dir)]

        print("Encontré %d imagenes en total" % len(carpeta_imagenes))
        fotos_alineadas, id_image_paths = self.detect_id_faces(carpeta_imagenes)
        feed_dict = {images_placeholder: fotos_alineadas, phase_train_placeholder: False}
        self.embeddings = sess.run(embeddings, feed_dict=feed_dict)

        if len(id_image_paths) < 5:
            self.print_distance_table(id_image_paths)

    def add_id(self, embedding, new_id, face_patch):
        if self.embeddings is None:
            self.embeddings = np.atleast_2d(embedding)
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])
        self.id_names.append(new_id)
        id_folder = os.path.join(self.id_folder, new_id)
        os.makedirs(id_folder, exist_ok=True)
        filenames = [s.split(".")[0] for s in os.listdir(id_folder)]
        numbered_filenames = [int(f) for f in filenames if f.isdigit()]
        img_number = max(numbered_filenames) + 1 if numbered_filenames else 0
        cv2.imwrite(os.path.join(id_folder, f"{img_number}.jpg"), face_patch)

    def detect_id_faces(self, carpeta_imagenes):
        fotos_alineadas = []
        id_image_paths = []
        for image_path in carpeta_imagenes:
            image = cv2.imread(os.path.expanduser(image_path), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_patches, _, _ = detectar_y_alinear.detect_faces(image, self.mtcnn)
            if len(face_patches) > 1:
                print(
                    "[ALERTA] Se reconocieron varias caras a la vez.: %s" % image_path
                    + "\nEs importante que solo esté una persona en la cámara  "
                    + "Si crees que es un falso negativo, puedes resolverlo incrementando el umbral de la red neuronal"
                )
            fotos_alineadas = fotos_alineadas + face_patches
            id_image_paths += [image_path] * len(face_patches)
            path = os.path.dirname(image_path)
            self.id_names += [os.path.basename(path)] * len(face_patches)

        return np.stack(fotos_alineadas), id_image_paths

    def print_distance_table(self, id_image_paths):
        """Prints distances between id embeddings"""
        distance_matrix = pairwise_distances(self.embeddings, self.embeddings)
        image_names = [path.split("/")[-1] for path in id_image_paths]
        print("Distance matrix:\n{:20}".format(""), end="")
        [print("{:20}".format(name), end="") for name in image_names]
        for path, distance_row in zip(image_names, distance_matrix):
            print("\n{:20}".format(path), end="")
            for distance in distance_row:
                print("{:20}".format("%0.3f" % distance), end="")
        print()

    def find_matching_ids(self, embs):
        if self.id_names:
            personas_reconocidas = []
            distancias_personas_reconocidas = []
            distance_matrix = pairwise_distances(embs, self.embeddings)
            for distance_row in distance_matrix:
                min_index = np.argmin(distance_row)
                if distance_row[min_index] < self.distancia_umbral:
                    personas_reconocidas.append(self.id_names[min_index])
                    distancias_personas_reconocidas.append(distance_row[min_index])
                else:
                    personas_reconocidas.append(None)
                    distancias_personas_reconocidas.append(None)
        else:
            personas_reconocidas = [None] * len(embs)
            distancias_personas_reconocidas = [np.inf] * len(embs)
        return personas_reconocidas, distancias_personas_reconocidas


def load_model(model):
    model_exp = os.path.expanduser(model)
    if os.path.isfile(model_exp):
        print("Cargando el modelo FACENET, llamado: %s" % model_exp)
        with gfile.FastGFile(model_exp, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")
    else:
        raise ValueError("Error, esperado: nombre del modelo no la ruta!")


def main(id_folder, model, umbral):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Iniciar el modelo
            mtcnn = detectar_y_alinear.create_mtcnn(sess, None)

            load_model(model)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # ANCHOR STEP
            id_data = IdData(
                id_folder[0], mtcnn, sess, embeddings, images_placeholder, phase_train_placeholder, umbral
            )

            cap = cv2.VideoCapture(0)

            while True:
                _, frame = cap.read()

                # Buscar caras en la cámara....
                face_patches, cuadros_delimitadores, puntos_referencia = detectar_y_alinear.detect_faces(frame, mtcnn)

                if len(face_patches) > 0:
                    
                    #Codigo por comentar....
                    face_patches = np.stack(face_patches)
                    feed_dict = {images_placeholder: face_patches, phase_train_placeholder: False}
                    embs = sess.run(embeddings, feed_dict=feed_dict)

                    personas_reconocidas, distancias_personas_reconocidas = id_data.find_matching_ids(embs)
                    """
                    posicion_cara: Todas las caras reconocidas en el frame, será un arreglo de indice 4, donde estará su posición (x,y) y su tamaño
                    persona_reconocida: Es una etiqueta con el nombre de la persona reconocida
                    """
                    for posicion_cara, _, persona_reconocida, distancia in zip(
                        cuadros_delimitadores, puntos_referencia, personas_reconocidas, distancias_personas_reconocidas
                    ):
                        if posicion_cara[2] > 350:
                            if persona_reconocida is None:
                                persona_reconocida = "Desconocido"
                                print("Desconocido! No se pudo reconocer el rostro.")
                            else:
                                print("Hola %s! Acuraccy: %1.4f" % (persona_reconocida, distancia))
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            #Una vez reconocido el rostro, se va a mostrar en la cámara: Su nombre, Un cuadrado identificado al rostro y un pequeño mensaje
                            cv2.rectangle(frame, (posicion_cara[0], posicion_cara[1]), (posicion_cara[2], posicion_cara[3]), (0, 255, 0), 4)
                            cv2.putText(frame, persona_reconocida, (posicion_cara[0], posicion_cara[2]-5), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                            
                            #Proceso de sonrisa
                            """
                            ¿Para qué sirve sonreír al momento que se te detectó?
                                En la versión anterior se tenía un botón de activación al momento de detectar un rostro, sin embargo
                                El programa está pensando para que esté puesto en una pantallá en la entrada del departamento de LCC
                                Entonces... por obvias razónes, si el programa tuviera un botón, entonces la pantalla tendría que ser
                                Touch, o tener algún mouse para poder hacer clic en el botón... 
                                
                                Entonces, se me ocurrió la idea de tener una función de activación con un gesto, como una sonrisa
                                aúnque también podría ser un gesto fisico, como levantar el pulgar
                                
                            ¿Cómo el programa logra detectar la sonrisa de la persona?
                                Se implementó un clasificador en cascada preentrenado de OpenCV, este clasificador contiene muchas sonrisas
                                fotografias
                                
                            Posibles problemas:


                            ¿Problema?
                                *El clasificador es un poco ineficiente, ya que, si la persona no está sonriendo, en algún cierto momento
                                 se va a detectar que la persona está sonriendo...
                            ¿Solución?
                                Tener un contador, para que la persona tenga que sonreir al menos 3 segundos para que pueda obtener más 
                                información detallada.


                            ¿Problema?
                                *Si la persona está sonriendo y deja de sonreir, 
                                ¿la información se va a borrar? 
                                ¿Se va a detener el proceso de busqueda de la información?
                                Pueden existir muchas preguntas acerca de esto...
                            ¿Solución?
                                Se me ocurren muchas soluciones, por el momento, la más fácil es tener una variable booleana que...
                                True: Cuando la persona sonríe por más de 3 segundos
                                False: Cuando la persona ya no se detecte
                                
                                Esta solución es debido a que una persona cuando tenga su información lista, dejará de sonreír para poder leer
                                o muchos fenoménos que pueden ocurrir, entonces, lo ideal es que la información desaparezca cuando la persona 
                                deje de estar en el encuadre de la cámara.
                            """
                            cara_detectada = frame[posicion_cara[0]:posicion_cara[0]+posicion_cara[3], posicion_cara[1]:posicion_cara[1]+posicion_cara[2]]
                            sonrisa = smileCascade.detectMultiScale(
                            cara_detectada,
                            scaleFactor= 1.5,
                            minNeighbors=20,
                            minSize=(25, 25),
                            )
                        
                        for i in sonrisa:
                            if len(sonrisa)>1:
                                cv2.putText(frame,"Cargando informacion...",(posicion_cara[0],posicion_cara[1]-30),font,1,(0,255,0), 2, cv2.LINE_AA)
                            else:
                                cv2.putText(frame,"Sonrie, para ver mas info",(posicion_cara[0],posicion_cara[1]-30),font,1,(0,0,255), 2, cv2.LINE_AA)

                cv2.imshow("frame", frame)
                key = cv2.waitKey(1)
                if key == ord("q"):
                    break


            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main(id_folder=['./ids/'], model='./model/20170512-110547.pb', umbral=1.0)
