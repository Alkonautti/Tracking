from ultralytics import YOLO #importataan YOLO kirjasto ultralyticsiltä
import cv2 #importataan cv2 kirjasto
import mediapipe as mp #importataan mediapipe kirjasto muuttujaksi mp

model = YOLO('yolov8n.pt') #YOLO kirjastosta käyttöön yolov8n.pt ja laitetaan se model objektiksi
mp_drawing = mp.solutions.drawing_utils #kutsutaan metodia drawing_utils mediapipe kirjastosta ja laitetaan se muuttujaksi mp_drawing
mp_drawing_styles = mp.solutions.drawing_styles #sama tässä kun ylemmässä mutta drawing_styles
mp_hands = mp.solutions.hands #ja tässä kanssa mutta hands metodi

cap = cv2.VideoCapture(0) #cv2 kirjastosta käyttöön videocapture metodi, joka etsii ensimmäisen (0) kameran tai video feedin mikä liitetty koneeseen. voi olla myös video tiedosto jos laittaa kansiopolun.
hands=mp_hands.Hands() #metodi muuttujaksi

cv2.namedWindow('AR_TEST', cv2.WINDOW_NORMAL) #parametri joka vaikuttaa kun myöhemmin koodissa luodaan ikkuna videolle

ret = True #alustetaan muuttuja True arvoksi

while ret: #loopataan kun muuttuja ret on True
    ret, image = cap.read() #luetaan yksittäinen kuva video inputista, jos ei löydy niin ret = False, lopetetaan while loop
    
    track_results = model.track(image, persist=True) #aloitetaan objectien trakcing, persist=True että se on jatkuvaa. Lisätään array track_resultsiin kun tunnistetaan objekti
    frame_ = track_results[0].plot() #otetaan track_results arraysta ensimmäinen indeksi ja plotataan. tämä siirretään muuttujaan frame_. frame_ on yksittäinen kuva feedista
    
    #image = cv2.cvtColor(cv2.flip(frame_, 1), cv2.COLOR_BGR2RGB)
    hand_results = hands.process(frame_) #frame_ muuttujassa on kuva jossa on object tracking jo tehty ja aloitetaan prosessointi että tunnistetaan käsi. lisätään hand_results arrayhyn
    #frame_ = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    
    if hand_results.multi_hand_landmarks: #jos käsi tunnistettu ja myös nivel kohdat
        for hand_landmarks in hand_results.multi_hand_landmarks: #käydään hand_arrayn kaikki indeksit läpi                                                                   
            mp_drawing.draw_landmarks(                                                 
            frame_,
            hand_landmarks,mp_hands.HAND_CONNECTIONS) #piirretään frame_ muuttujaan kädessä olevat punaiset pisteet ja viivat sen mukaan mikä frame on muuttujassa frame_
            
       
    cv2.imshow('AR_TEST', frame_) #avataan ja päivitetään ikkunaa jossa kuva näytetään, nimetään ikkuna "AR_TEST"
        
    if cv2.waitKey(25) & 0xFF == ord('q'): #wait time 25ms ja jos painetaan näppäintä Q niin suljetaan ikkuna
        break
        
cap.release() #poistetaan käytöstä kameran tai tiedoston feedi
cv2.destroyAllWindows() #suljetaan kaikki ikkunat mitä luotu
