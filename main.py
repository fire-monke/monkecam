import cv2
import mediapipe as mp

# Default camera (index O)
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5
)

def cover(img, w, h):
    ih, iw = img.shape[:2]

    # ratio to fill the frame
    scale = max(w / iw, h / ih)
    
    # resize
    new_w = int(iw * scale)
    new_h = int(ih * scale)
    resized = cv2.resize(img, (new_w, new_h))

    # center
    x_start = (new_w - w) // 2
    y_start = (new_h - h) // 2

    return resized[y_start:y_start+h, x_start:x_start+w]


# region INIT IMAGES TO SHOW
img_ragebait = cv2.imread("ragebait.jpg")
img_ragebait = cover(img_ragebait, 600, 400)

img_allo_salam = cv2.imread("allo-salam.jpg")
img_allo_salam = cover(img_allo_salam, 400, 600)

img_thinking_monkey = cv2.imread("thinking-monke.jpg")
img_thinking_monkey = cover(img_thinking_monkey, 800, 400)

# end region
# region CHECK LANDMARKS FCTs
def isHandClosed(handLms):
    # Fingers Ids (tip, pip)
    finger_ids = [(8,6), (12,10), (16,14), (20,18)]
    fingers_folded = 0

    for tip, pip in finger_ids:
        if handLms.landmark[tip].y > handLms.landmark[pip].y:
            fingers_folded += 1

    return fingers_folded >= 4

def isMonkeyThinking(hand, face):
    # between top of the top lip (0) and top of the chin (200)
    if not(face.landmark[0].y < hand.landmark[8].y < face.landmark[200].y and face.landmark[202].x < hand.landmark[8].x < face.landmark[273].x):
        return False

    # 20 is the nb of landmark on 1 hand
    for id in range(0, 21):
        if hand.landmark[8].y > hand.landmark[id].y:
            return False
    return True

def handUnderFace(hand, face):
    if not(face.landmark[150].x < hand.landmark[9].x < face.landmark[379].x):
        return False
    
    # 20 is the nb of landmark on 1 hand
    for id in range(0, 21):
        # landmark 200 is located between the bottom of the chin and the lower lip.
        if face.landmark[200].y > hand.landmark[id].y:
            return False
    return True
# end region

current_state = None
before_state = None

cv2.imshow("Yr self", img_allo_salam)

while True:
    success, image = cap.read()
    if not success:
        break
    image = cv2.flip(image, 1)

    # Convert BGR to RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False

    # Process
    h_results = hands.process(rgb)
    f_result = face_mesh.process(rgb)

    rgb.flags.writeable = True

    current_state = None
    # ===== HANDS =====
    if h_results.multi_hand_landmarks:
        for hand in h_results.multi_hand_landmarks:

            if f_result.multi_face_landmarks:
                # with one face
                face = f_result.multi_face_landmarks[0]
                if isMonkeyThinking(hand, face):
                    current_state = "thinking_monke"
                
                if isHandClosed(hand):
                    if handUnderFace(hand, face):
                        current_state = "ragebait"

    if current_state != before_state:
        match current_state:
            case "thinking_monke":
                img_to_show = img_thinking_monkey
            case "ragebait":
                img_to_show = img_ragebait
            case _:
                img_to_show = img_allo_salam

        cv2.imshow("Yr self", img_to_show)
        before_state = current_state
    
    cv2.imshow("Camera", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the cam
cap.release()
# Close all windows
cv2.destroyAllWindows()