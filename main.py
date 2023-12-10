import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from PyQt5 import QtCore, QtGui, QtWidgets

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_faces=5,
    refine_landmarks=True
)
hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    enable_segmentation=True,
    model_complexity=2
)


class TransparentWindow(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setGeometry(0, 0, 1920, 1080)
        self.pen_color = QtGui.QColor(0, 255, 0, 255)
        self.landmarks_enabled = True
        self.bounding_box_enabled = True
        self.hand_detection_enabled = True
        self.pose_detection_enabled = True
        self.face_oval_enabled = True
        self.face_mesh_numbers_enabled = False

        self.initUI()

    def initUI(self):
        self.config_window = ConfigWindow()
        self.config_window.colorChanged.connect(self.update_color)
        self.config_window.toggleLandmarksChanged.connect(self.update_landmarks)
        self.config_window.toggleBoundingBoxChanged.connect(self.update_bounding_box)
        self.config_window.toggleHandDetectionChanged.connect(self.update_hand_detection)
        self.config_window.togglePoseDetectionChanged.connect(self.update_pose_detection)
        self.config_window.toggleFaceOvalChanged.connect(self.update_face_oval)
        self.config_window.toggleFaceMeshNumbersChanged.connect(self.update_face_mesh_numbers)
        self.config_window.show()

    def update_color(self, new_color):
        self.pen_color = QtGui.QColor(new_color.red(), new_color.green(), new_color.blue(), 255)

    def update_landmarks(self, state):
        self.landmarks_enabled = state

    def update_bounding_box(self, state):
        self.bounding_box_enabled = state

    def update_hand_detection(self, state):
        self.hand_detection_enabled = state

    def update_pose_detection(self, state):
        self.pose_detection_enabled = state

    def update_face_oval(self, state):
        self.face_oval_enabled = state

    def update_face_mesh_numbers(self, state):
        self.face_mesh_numbers_enabled = state

    def update_overlay(self, faces, hands, pose_landmarks):
        transparent_image = np.zeros((1080, 1920, 4), dtype=np.uint8)

        if self.landmarks_enabled and faces:
            for landmarks in faces:
                for index, lm in enumerate(landmarks):
                    x, y = int(lm.x * 1920), int(lm.y * 1080)
                    cv2.circle(transparent_image, (x, y), 1,
                               (self.pen_color.red(), self.pen_color.green(), self.pen_color.blue(), 255), -1)
                    if self.face_mesh_numbers_enabled:
                        cv2.putText(transparent_image, str(index), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                                    (self.pen_color.red(), self.pen_color.green(), self.pen_color.blue(), 255), 1,
                                    cv2.LINE_AA)

                if self.face_oval_enabled:
                    for connection in mp_face_mesh.FACEMESH_FACE_OVAL:
                        x1, y1 = int(landmarks[connection[0]].x * 1920), int(landmarks[connection[0]].y * 1080)
                        x2, y2 = int(landmarks[connection[1]].x * 1920), int(landmarks[connection[1]].y * 1080)
                        cv2.line(transparent_image, (x1, y1), (x2, y2),
                                 (self.pen_color.red(), self.pen_color.green(), self.pen_color.blue(), 255), 2)

        if self.hand_detection_enabled and hands:
            for hand_landmarks in hands:
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * 1920), int(lm.y * 1080)
                    cv2.circle(transparent_image, (x, y), 3, (255, 0, 0, 255), -1)
                for connection in mp_hands.HAND_CONNECTIONS:
                    x1, y1 = int(hand_landmarks.landmark[connection[0]].x * 1920), int(
                        hand_landmarks.landmark[connection[0]].y * 1080)
                    x2, y2 = int(hand_landmarks.landmark[connection[1]].x * 1920), int(
                        hand_landmarks.landmark[connection[1]].y * 1080)
                    cv2.line(transparent_image, (x1, y1), (x2, y2), (0, 255, 0, 255), 2)

        if self.bounding_box_enabled and faces:
            left, top, right, bottom = 1920, 1080, 0, 0
            for landmarks in faces:
                for lm in landmarks:
                    x, y = int(lm.x * 1920), int(lm.y * 1080)
                    if x < left:
                        left = x
                    if x > right:
                        right = x
                    if y < top:
                        top = y
                    if y > bottom:
                        bottom = y
            cv2.rectangle(transparent_image, (left, top), (right, bottom),
                          (self.pen_color.red(), self.pen_color.green(), self.pen_color.blue(), 255), 2)

        if self.pose_detection_enabled and pose_landmarks:
            for lm in pose_landmarks.landmark:
                x, y = int(lm.x * 1920), int(lm.y * 1080)
                cv2.circle(transparent_image, (x, y), 4, (255, 0, 0, 255), -1)
            for connection in mp_pose.POSE_CONNECTIONS:
                x1, y1 = int(pose_landmarks.landmark[connection[0]].x * 1920), int(
                    pose_landmarks.landmark[connection[0]].y * 1080)
                x2, y2 = int(pose_landmarks.landmark[connection[1]].x * 1920), int(
                    pose_landmarks.landmark[connection[1]].y * 1080)
                cv2.line(transparent_image, (x1, y1), (x2, y2),
                         (self.pen_color.red(), self.pen_color.green(), self.pen_color.blue(), 255), 2)

                x_values = [int(lm.x * 1920) for lm in pose_landmarks.landmark]
                y_values = [int(lm.y * 1080) for lm in pose_landmarks.landmark]

                min_x = min(x_values)
                max_x = max(x_values)
                min_y = min(y_values)
                max_y = max(y_values)

                cv2.rectangle(transparent_image, (min_x, min_y), (max_x, max_y),
                              (self.pen_color.red(), self.pen_color.green(), self.pen_color.blue(), 255), 2)

        h, w, ch = transparent_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(transparent_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGBA8888)
        self.setPixmap(QtGui.QPixmap.fromImage(convert_to_Qt_format))


class ConfigWindow(QtWidgets.QWidget):
    colorChanged = QtCore.pyqtSignal(QtGui.QColor)
    toggleLandmarksChanged = QtCore.pyqtSignal(bool)
    toggleBoundingBoxChanged = QtCore.pyqtSignal(bool)
    toggleHandDetectionChanged = QtCore.pyqtSignal(bool)
    togglePoseDetectionChanged = QtCore.pyqtSignal(bool)
    toggleFaceOvalChanged = QtCore.pyqtSignal(bool)
    toggleFaceMeshNumbersChanged = QtCore.pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Settings")
        self.setGeometry(100, 100, 300, 350)

        self.pen_color = QtGui.QColor(0, 255, 0, 255)
        self.landmarks_enabled = True
        self.bounding_box_enabled = True
        self.hand_detection_enabled = True
        self.pose_detection_enabled = True
        self.face_oval_enabled = True
        self.face_mesh_numbers_enabled = False
        self.initUI()

    def initUI(self):
        layout = QtWidgets.QVBoxLayout()

        self.landmarks_checkbox = QtWidgets.QCheckBox('Landmarks', self)
        self.landmarks_checkbox.setChecked(self.landmarks_enabled)
        self.landmarks_checkbox.stateChanged.connect(self.landmarks_changed)
        layout.addWidget(self.landmarks_checkbox)

        self.bounding_box_checkbox = QtWidgets.QCheckBox('Bounding Box', self)
        self.bounding_box_checkbox.setChecked(self.bounding_box_enabled)
        self.bounding_box_checkbox.stateChanged.connect(self.bounding_box_changed)
        layout.addWidget(self.bounding_box_checkbox)

        self.hand_detection_checkbox = QtWidgets.QCheckBox('Hand Detection', self)
        self.hand_detection_checkbox.setChecked(self.hand_detection_enabled)
        self.hand_detection_checkbox.stateChanged.connect(self.hand_detection_changed)
        layout.addWidget(self.hand_detection_checkbox)

        self.pose_detection_checkbox = QtWidgets.QCheckBox('Pose Detection', self)
        self.pose_detection_checkbox.setChecked(self.pose_detection_enabled)
        self.pose_detection_checkbox.stateChanged.connect(self.pose_detection_changed)
        layout.addWidget(self.pose_detection_checkbox)

        self.face_oval_checkbox = QtWidgets.QCheckBox('Face Oval', self)
        self.face_oval_checkbox.setChecked(self.face_oval_enabled)
        self.face_oval_checkbox.stateChanged.connect(self.face_oval_changed)
        layout.addWidget(self.face_oval_checkbox)

        self.face_mesh_numbers_checkbox = QtWidgets.QCheckBox('Face Mesh Numbers', self)
        self.face_mesh_numbers_checkbox.setChecked(self.face_mesh_numbers_enabled)
        self.face_mesh_numbers_checkbox.stateChanged.connect(self.face_mesh_numbers_changed)
        layout.addWidget(self.face_mesh_numbers_checkbox)

        self.color_button = QtWidgets.QPushButton('Set Color', self)
        self.color_button.clicked.connect(self.show_color_dialog)
        layout.addWidget(self.color_button)

        self.setLayout(layout)

    def landmarks_changed(self, state):
        self.toggleLandmarksChanged.emit(state == QtCore.Qt.Checked)

    def bounding_box_changed(self, state):
        self.toggleBoundingBoxChanged.emit(state == QtCore.Qt.Checked)

    def hand_detection_changed(self, state):
        self.toggleHandDetectionChanged.emit(state == QtCore.Qt.Checked)

    def pose_detection_changed(self, state):
        self.togglePoseDetectionChanged.emit(state == QtCore.Qt.Checked)

    def face_oval_changed(self, state):
        self.toggleFaceOvalChanged.emit(state == QtCore.Qt.Checked)

    def face_mesh_numbers_changed(self, state):
        self.toggleFaceMeshNumbersChanged.emit(state == QtCore.Qt.Checked)

    def show_color_dialog(self):
        color_dialog = QtWidgets.QColorDialog(self.pen_color)
        if color_dialog.exec_():
            new_color = color_dialog.currentColor()
            self.pen_color = new_color
            self.colorChanged.emit(new_color)


app = QtWidgets.QApplication([])
overlay = TransparentWindow()
overlay.show()

while True:
    screenshot = pyautogui.screenshot()
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results_face = face_mesh.process(frame)
    faces = [face_landmarks.landmark for face_landmarks in
             results_face.multi_face_landmarks] if results_face.multi_face_landmarks else []

    results_hands = hands.process(frame)
    hands_landmarks = [hand_landmarks for hand_landmarks in
                       results_hands.multi_hand_landmarks] if results_hands.multi_hand_landmarks else []

    results_pose = pose.process(frame)
    pose_landmarks = results_pose.pose_landmarks

    overlay.update_overlay(faces, hands_landmarks, pose_landmarks)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

face_mesh.close()
hands.close()
pose.close()
app.exit()
