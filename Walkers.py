import cv2

# Crie nosso classificador de corpos
body_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Inicie a captura de vídeo para o arquivo de vídeo
cap = cv2.VideoCapture('walking.avi')

# Faça o loop assim que o vídeo for carregado com sucesso
while True:
    # Leia o primeiro quadro
    ret, frame = cap.read()

    # Se o quadro não for lido corretamente, encerre o loop
    if not ret:
        print("Fim do vídeo ou erro ao carregar o arquivo.")
        break

    # Converta cada quadro em escala de cinza
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Passe o quadro para nosso classificador de corpos
    bodies = body_classifier.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5)

    # Extraia as caixas delimitadoras para quaisquer corpos identificados
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Exiba o quadro
    cv2.imshow('Detecção de Corpos', frame)

    # Saia do loop ao pressionar a barra de espaço (tecla 32)
    if cv2.waitKey(1) == 32:
        break

# Libere os recursos e feche as janelas
cap.release()
cv2.destroyAllWindows()
