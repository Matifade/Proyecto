import os
os.environ["QT_LOGGING_RULES"] = "*.warning=false"

import cv2
import numpy as np
from collections import deque, Counter  # <-- Esta es la línea que falta

def detectar_piel(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    min_YCrCb = np.array([0, 133, 77], np.uint8)
    max_YCrCb = np.array([255, 173, 127], np.uint8)
    
    mask = cv2.inRange(ycrcb, min_YCrCb, max_YCrCb)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return cv2.GaussianBlur(mask, (5, 5), 0)

def clasificar_gesto(contour):
    hull = cv2.convexHull(contour, returnPoints=False)
    if len(hull) < 3:
        return "Desconocido"

    try:
        defects = cv2.convexityDefects(contour, hull)
    except cv2.error:
        return "Desconocido"

    if defects is None:
        return "Puño"

    valles_dedos = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i].flatten()
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])

        a = np.linalg.norm(np.array(end) - np.array(start))
        b = np.linalg.norm(np.array(far) - np.array(start))
        c = np.linalg.norm(np.array(end) - np.array(far))

        if 2 * b * c == 0:
            continue
        angle = np.arccos(np.clip((b**2 + c**2 - a**2) / (2 * b * c), -1.0, 1.0)) * (180 / np.pi)

        if angle <= 90 and d > 3000:
            valles_dedos += 1

    x, y, w, h = cv2.boundingRect(contour)
    relacion_aspecto = float(h) / (w if w > 0 else 1)

    if valles_dedos == 0:
        if relacion_aspecto > 1.3:
            return "Pulgar Arriba"
        return "Puño"
    elif valles_dedos == 1:
        return "Tijera"
    elif valles_dedos >= 3:
        return "Mano Extendida"
    
    return "Gesto no reconocido"

cap = cv2.VideoCapture(0)

# Buffer para guardar los últimos 12 fotogramas
historial = deque(maxlen=12)
gesto_estable_anterior = ""

print("Iniciando detección... Presiona 'q' para salir.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (300, 100), (600, 400), (0, 255, 0), 2)
    roi = frame[100:400, 300:600]

    mask = detectar_piel(roi)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    gesto_instantaneo = "Sin mano"

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 2000:
            gesto_instantaneo = clasificar_gesto(max_contour)

    # Agregar la lectura al historial
    historial.append(gesto_instantaneo)

    # Obtener el gesto dominante en la ventana de fotogramas
    conteo = Counter(historial)
    gesto_dominante, frecuencia = conteo.most_common(1)[0]

    # Exigir que el gesto se repita al menos en 8 de los últimos 12 frames para darlo por válido
    if frecuencia >= 8 and gesto_dominante not in ["Desconocido", "Gesto no reconocido"]:
        gesto_actual = gesto_dominante
    else:
        gesto_actual = gesto_estable_anterior

    # Actualizar la terminal en una SOLA LÍNEA sobreescribiendo el texto
    if gesto_actual != gesto_estable_anterior:
        print(f"\rGesto detectado: {gesto_actual:<20}", end="", flush=True)
        gesto_estable_anterior = gesto_actual

    cv2.putText(frame, f"Gesto: {gesto_actual}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Camara", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nPrograma finalizado.")