from tkinter import *
from tkinter import messagebox, Canvas, simpledialog
from PIL import Image,ImageTk

from sklearn.metrics.pairwise import pairwise_distances
from tensorflow.python.platform import gfile
import tensorflow.compat.v1 as tf
import numpy as np
import detect_and_align as detectar_y_alinear
import cv2
import os
import imutils

def btn_clicked():
    print("Button Clicked")


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
            id_data = IdData(id_folder[0], mtcnn, sess, embeddings, images_placeholder, phase_train_placeholder, umbral)
            visualizar()
            
def iniciar():
    print("xD")
    btnIniciar.config(state="disabled")
    btnFinalizar.config(state="normal")
    global cap
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    main(id_folder=['./ids/'], model='./model/20170512-110547.pb', umbral=1.0)
    

def visualizar(mtcnn,id_folder, umbral,sess,id_data,images_placeholder,embeddings,phase_train_placeholder):
    global cap
    if cap is not None:
        ret, frame = cap.read()
        if ret == True:
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                #Si la coincidencia es mayor al 50%, se muestra la matricula del usuario
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=im)
            lblVideo.configure(image=img)
            lblVideo.image = img
            lblVideo.after(10, visualizar(mtcnn,id_folder, umbral,sess,id_data,images_placeholder,embeddings,phase_train_placeholder))
        else:
            lblVideo.image = ""
            cap.release()
        
        
        
def finalizar():
    global cap
    cap.release()
    lcc_matricula_identificado.set("")
    lcc_nombre_identificado.set("")
	
    lcc_apellido_identificado.set("")
    lcc_creditos_identificado.set("")
    lcc_kardex_identificado.set("")
    btnIniciar.config(state="normal")
    btnFinalizar.config(state="disabled")

cap = None   
window = Tk()
#Variables para datos del alumno identificado
lcc_matricula = IntVar()
lcc_matricula_identificado = IntVar()
lcc_nombre_identificado = StringVar()
lcc_apellido_identificado = StringVar()
lcc_creditos_identificado = IntVar()
lcc_kardex_identificado = DoubleVar()
lcc_sc_identificado = StringVar()


#Iniciamos las variables en blanco 
lcc_matricula_identificado.set("")
lcc_nombre_identificado.set("")
lcc_apellido_identificado.set("")
lcc_creditos_identificado.set("")
lcc_kardex_identificado.set("")
lcc_sc_identificado.set("")


window.geometry("1200x650")
window.configure(bg = "#ffffff")
canvas = Canvas(
    window,
    bg = "#ffffff",
    height = 650,
    width = 1200,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge")
canvas.place(x = 0, y = 0)

canvas.create_text(
    865, 240.5,
    text = "Hola Jesus Martin Garcia Encinas\nLa ultima vez que te vi fue el 05/05/22",
    fill = "#000000",
    font = ("None", int(24.0)))

canvas.create_text(
    865.0, 336.0,
    text = "Creditos ",
    fill = "#000000",
    font = ("None", int(24.0)))

canvas.create_text(
    866.0, 371.5,
    text = "225/383  -> 59%",
    fill = "#000000",
    font = ("None", int(24.0)))

canvas.create_text(
    670.0, 424.5,
    text = "Servicio social: ",
    fill = "#000000",
    font = ("None", int(24.0)))

canvas.create_text(
    1015.5, 424.5,
    text = "Practicas profesionales:",
    fill = "#000000",
    font = ("None", int(24.0)))

canvas.create_text(
    670.0, 468.0,
    text = "Faltan creditos",
    fill = "#000000",
    font = ("None", int(24.0)))

canvas.create_text(
    1015.5, 468.0,
    text = "Faltan creditos",
    fill = "#000000",
    font = ("None", int(24.0)))

background_img = PhotoImage(file = f"background.png")
background = canvas.create_image(
    600.0, 301.0,
    image=background_img)

img0 = PhotoImage(file = f"img0.png")
btnFinalizar = Button(
    image = img0,
    borderwidth = 0,
    highlightthickness = 0,
    command = finalizar,
    relief = "flat")

btnFinalizar.place(
    x = 967, y = 564,
    width = 174,
    height = 34)

img1 = PhotoImage(file = f"img1.png")
btnIniciar = Button(
    image = img1,
    borderwidth = 0,
    highlightthickness = 0,
    command = iniciar,
    relief = "flat")

btnIniciar.place(
    x = 672, y = 564,
    width = 197,
    height = 34)

lblVideo = Label(window, background="#bababa")
lblVideo.place(
    x = 16, y = 151,
    width = 500,
    height = 490)


if __name__ == "__main__":
    #main(id_folder=['./ids/'], model='./model/20170512-110547.pb', umbral=1.0)
    window.resizable(False, False)
    window.mainloop()

