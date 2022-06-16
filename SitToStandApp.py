# SitToStandApp.py
# ESTE SCRIPT DETECTA LA FLEXION DE RODILLA

import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees
from tkinter import Tk     
from tkinter.filedialog import askdirectory, askopenfilename

def filter_u_EMA(stack_ini, alfa):
    stack_filter = np.zeros(0)
    while (np.size(stack_filter)<np.size(stack_ini)):
        if(np.size(stack_filter)<2):
            stack_filter = np.append(stack_filter,stack_ini[np.size(stack_filter)])
        else:
            stack_filter = np.append(stack_filter,
            alfa * stack_ini[np.size(stack_filter)-1] + (1-alfa)*stack_filter[np.size(stack_filter)-2])
    return stack_filter

def derivate_stack(stack_ini, delta_t):
    stack_der = np.zeros(0)
    while (np.size(stack_der)<np.size(stack_ini)):
        if(np.size(stack_der)<2):
            stack_der = np.append(stack_der,0)
        else:
            stack_der = np.append(stack_der,
            (stack_ini[np.size(stack_der)] - stack_ini[np.size(stack_der)-1]) / delta_t) 

    return stack_der



mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
# Abro una ventana para seleccionar el archivo
video_file = askopenfilename() # show an "Open" dialog box and return the path to the selected file

#Procesamiento de la dir del archivo para obtener datos como 
#extensión de archivo, nombre de archivo y carpeta
V_video_file = video_file.split("/")
video_path = video_file[0:len(video_file)-len(V_video_file[-1])]
V_video_file_name = V_video_file[-1].split(".")
video_file_name = V_video_file[-1][0:len(V_video_file[-1])-len(V_video_file_name[-1])-1]
video_file_extension = V_video_file_name[-1]

# Carga de archivo
print( "Se analizará: " + video_file + " ...")
cap = cv2.VideoCapture(video_file)

# Datos del video cargado
FPS_original = cap.get(5)  #ej. 25.0 
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count/FPS_original
delta_t = 1/FPS_original
width_original  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float "width"
height_original = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float "height"
resolution_original =  (int(width_original), int(height_original))  #ej. (640, 480)
#fourcc_original = cap.get(cv2.CAP_PROP_FOURCC) #No se utilizar esta funcion
print("\n-FPS_original:" + str(FPS_original) + 
    "\n-frame_count_original: " + str(frame_count) + 
    "\n-duration_original:" + str(duration) + "s" 
    + "\n-resolution_original: " + str(int(width_original)) + "x" + str(int(height_original)) + "\n")

# cv2.CAP_PROP_FRAME_WIDTH   # 3
# cv2.CAP_PROP_FRAME_HEIGHT  # 4
# cv2.CAP_PROP_FPS           # 5
# cv2.CAP_PROP_FRAME_COUNT   # 7

# Nombre Y ruta del video generado para guardar como RESULTADO
video_path = askdirectory()
video_path_result = video_path + "/Videos/Videos Resultados/"
video_file_result = video_path_result + video_file_name + "_resultado.mp4"

# Datos para el video generado para guardar
scale_percent = 700 * 100 / height_original # Porcentaje de escalado para el video a guardar (será el mismo para el video original a mostrar)
FPS_result = FPS_original   #ej. 25.0
width_result = int(width_original * scale_percent / 100)
height_result = int(height_original * scale_percent / 100)
resolution_result = (width_result, height_result)   #ej. (640, 480)
frame_count_result = 0
duration_result = 0

# Creacion de los objetos para el guardado del video prosesado
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') #*'mpv4"
outVideoWriter = cv2.VideoWriter(video_file_result, fourcc, FPS_result, resolution_result) # (name.mp4, fourcc, FPS, resolution)

# Vector de angulos de la rodilla (knee)
V_angles_knee = np.zeros(0)
V_vel_angles_knee = np.zeros(0)
#V_angles_knee_filter = np.zeros(0)
#V_vel_angles_knee_filter = np.zeros(0)
alfa = 0.1
# Vector de tiempos para cada frame
V_time = np.zeros(0)

# Inicio de While True para reproduccion y analisis
with mp_pose.Pose(static_image_mode=False) as pose:
    while True:
        ret, frame = cap.read()
                
        if ret == False:
            break      
        # np.append(V_time, round(cap.get(cv2.CAP_PROP_POS_MSEC)/1000, 4))
        # #print(round(cap.get(cv2.CAP_PROP_POS_MSEC)/1000, 4)
        V_time= np.append(V_time, frame_count_result/FPS_result)
        frame_count_result = frame_count_result + 1
     
        # Reescalado de la imagen/imagenes del video
        # scale_percent = 50  # --> Definido arriba para determinar la escritura del video resultado
        width = int(frame.shape[1] * scale_percent / 100)   # Otra opcion: height, width, layers = frame.shape
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)

        resized_frame = cv2.resize(frame, dim, interpolation= cv2.INTER_AREA)   # resized_frame será el nuevo "frame" que se trabaja
        
        #Utilizo el primer frame como pantalla de carga
        if (frame_count_result == 1):
                loading_page = resized_frame

        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        results = pose.process (frame_rgb)

        # Adquiero coordenadas de los marcadores
        if results.pose_landmarks is not None:
            # Landmark 24
            x1 = int(results.pose_landmarks.landmark[24].x * width)
            y1 = int(results.pose_landmarks.landmark[24].y * height)

            # Landmark 26
            x2 = int(results.pose_landmarks.landmark[26].x * width)
            y2 = int(results.pose_landmarks.landmark[26].y * height)

            # Landmark 28
            x3 = int(results.pose_landmarks.landmark[28].x * width)
            y3 = int(results.pose_landmarks.landmark[28].y * height)

            # Calculo de angulo:
            p1 = np.array([x1, y1])
            p2 = np.array([x2, y2])
            p3 = np.array([x3, y3])

            l1 = np.linalg.norm(p2 - p3)
            l2 = np.linalg.norm(p1 - p3)
            l3 = np.linalg.norm(p1 - p2)

            # Calcular el ángulo (teorema del coseno) y lo agrego a V_angles_knee
            angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
            
            V_angles_knee = np.append(V_angles_knee, angle)

#            if (np.size(V_angles_knee)<=2):
#                V_angles_knee_filter = np.append(V_angles_knee_filter, angle)

#            vel_angles_knee = 0
            # Calcular la velocidad angular y lo agrego a V_vel_angles_knee
#            if (np.size(V_angles_knee) > 2):
#                angles_count = np.size(V_angles_knee)
#                V_angles_knee_filter = np.append(V_angles_knee_filter, 
#                alfa * angle + (1-alfa)* V_angles_knee_filter[angles_count-2])

#                vel_angles_knee = ( V_angles_knee[angles_count-1] - V_angles_knee[angles_count-2] ) / delta_t
#                V_vel_angles_knee = np.append(V_vel_angles_knee, vel_angles_knee)
#                if(np.size(V_vel_angles_knee_filter)<2):
#                    V_vel_angles_knee_filter = np.append(V_vel_angles_knee_filter, 19)
#                else:
#                    V_vel_angles_knee_filter = np.append(V_vel_angles_knee_filter, 
#                    alfa * vel_angles_knee + (1-alfa)* V_vel_angles_knee_filter[np.size(V_vel_angles_knee_filter)-2])
               
            # Visualización de segmentos de muslo y pierna
            aux_image = np.zeros(resized_frame.shape, np.uint8)

            cv2.line(aux_image, (x1, y1), (x2, y2), (100, 255, 0), 20)
            cv2.line(aux_image, (x2, y2), (x3, y3), (100, 255, 0), 28)
            cv2.line(aux_image, (x1, y1), (x3, y3), (100, 255, 0), 5)
            contours = np.array([[x1, y1], [x2, y2], [x3, y3]])
            cv2.fillPoly(aux_image, pts=[contours], color=(128, 200, 250))

            output = cv2.addWeighted(resized_frame, 1, aux_image, 0.8, 0)   #output es el frame ya procesado

            cv2.circle(output, (x1, y1), 6, (0, 255, 255), 4)
            cv2.circle(output, (x2, y2), 6, (128, 0, 255), 4)
            cv2.circle(output, (x3, y3), 6, (255, 191, 0), 4)

            cv2.putText(output, str(int(angle)), (x2, y2 - 30), 1, 1.5, (128, 0, 250), 2)   # Agrego el angulo en el video
            cv2.putText(output, "Angulo en grados,", (10, height - 40), 4, 0.75, (20, 20, 20), 2) # Agrego info en el video
            cv2.putText(output, "Pulse ESPACIO para finalizar.", (10, height - 10), 4, 0.75, (20, 20, 20), 2) # Agrego info en el video
            #cv2.putText(output, "Velocidad : "+str(round(vel_angles_knee,2)) + " grad/s", (10, height - 70), 4, 0.75, (20, 20, 20), 2) # Agrego info en el video
            
            # Guardado del video resultante
            outVideoWriter.write(output)
            
            #Pantalla de carga
            cv2.putText(loading_page, "Cargando: ", (10, height - 40), 4, 0.75, (20, 20, 20), 2) # Agrego info en el video
            cv2.putText(loading_page, "|", (int((frame_count_result/frame_count)*(width-10)), height - 10), 4, 0.75, (0, 100, 0), 2) # Agrego info en el video
            # Muestro pantalla de carga  
            cv2.imshow("Frame", loading_page)  #Muestro video original reescalada
            if cv2.waitKey(1) & 0xFF == ord(' '):
                break


# Guardo los angulos medidos

V_angles_knee_filter = filter_u_EMA(V_angles_knee,alfa) #Filtro lo ang
V_vel_angles_knee = derivate_stack(V_angles_knee_filter,delta_t) #Calculo la velocidad con el ang filtrado
data = V_vel_angles_knee
kernel_size = 20
kernel = np.ones(kernel_size) / kernel_size
V_vel_angles_knee_filter = np.convolve(data, kernel, mode='same')
#V_vel_angles_knee_filter = filter_u_EMA(V_vel_angles_knee,alfa) #Filtro la vel
print(np.size(V_time))
print(np.size(V_angles_knee_filter))
print(np.size(V_vel_angles_knee))
#print(np.size(V_vel_angles_knee_filter))

V_ang_and_vel = np.stack(( V_time[:-2],V_angles_knee_filter[:-2], V_vel_angles_knee_filter[:-2]),1)
data_path = video_path+"/Datos/"
with open(data_path+'datos_ang_' + video_file_name + '.csv', 'wb') as h:
    np.savetxt(h, V_ang_and_vel, delimiter=',', fmt='%0.3f', header="Time (s),Ang (°),Vel (°/s)")


#Calculo de la potencia muscular
#time_stand_prom Tiempo promedio que se tarda en ir desde 90 grados a 180 grados (levantarse)
#femur_lenght largo del femur [m]
#time_stand_prom = time_stand # Para esta primer prueba será una toma simple, no promediada
#femur_lenght = 0.4 # Estimada de momento
#Pmean = 2.773 – 6.228 × time_stand_prom + 18.224 * femur_lenght

# Datos del video resultado generado
# FPS, resolution y factor de escala ya se determinaron/seteados antes
duration_result = frame_count_result/FPS_result
print("\n-FPS_result: " + str(FPS_result) + 
    "\n-width_result: " + str(width_result) + 
    "\n-height_result: " + str(height_result) + 
    "\n-Resize Escala: " + str(scale_percent) +
    "\n-frame_count_result: " + str(frame_count_result) + 
    "\n-duration_result: " + str(duration_result))

# RELEASE
outVideoWriter.release()

# Muestro imagenes/video
video_out = cv2.VideoCapture(video_file_result)
while(video_out.isOpened()):
    ret, frame = video_out.read()
    if ret==True:
        
        cv2.imshow('frame',frame)

        if cv2.waitKey(int(delta_t*1000)) & 0xFF == ord(' '):
            break
    else:
        break

# RELEASE
cap.release()
video_out.release()
cv2.destroyAllWindows()
