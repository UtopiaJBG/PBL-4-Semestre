import sys
import os
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QLineEdit, QMessageBox, QDialog, QFormLayout,
    QGroupBox, QCheckBox
)
from PyQt5.QtGui import QFont, QIcon, QPixmap
from PyQt5.QtCore import Qt, QTimer

import threading
import socket
import pyqtgraph as pg
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.fft import fft, fftfreq

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
from scipy.stats import hypergeom
from tensorflow.keras.models import load_model
import joblib
import mplcursors

# Ensure the working directory is correct
os.chdir("C:/Users/ppoli/Desktop/einstein/4_semestre/PBL")

class Dashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Painel Médico")
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.initUI()
        self.apply_styles()

    def initUI(self):
        main_layout = QVBoxLayout()

        header_label = QLabel()
        header_label.setAlignment(Qt.AlignCenter)
        header_label.setObjectName("header_label")
        pixmap = QPixmap('images/background.png')
        pixmap = pixmap.scaled(int(pixmap.width()*0.25), int(pixmap.height()*0.25), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        header_label.setPixmap(pixmap)
        main_layout.addWidget(header_label)
        main_layout.addSpacing(20)

        doctor_group = QGroupBox("Responsável")
        doctor_layout = QVBoxLayout()
        selector_layout = QHBoxLayout()
        doctor_label = QLabel("Selecione:")
        self.doctor_combo = QComboBox()
        self.load_doctors()
        selector_layout.addWidget(doctor_label)
        selector_layout.addWidget(self.doctor_combo)
        doctor_layout.addLayout(selector_layout)
        doctor_button_layout = QHBoxLayout()
        add_doctor_btn = QPushButton("Adicionar")
        remove_doctor_btn = QPushButton("Remover")
        add_doctor_btn.clicked.connect(self.add_new_doctor)
        remove_doctor_btn.clicked.connect(self.remove_doctor)
        doctor_button_layout.addStretch()
        doctor_button_layout.addWidget(add_doctor_btn)
        doctor_button_layout.addWidget(remove_doctor_btn)
        doctor_layout.addLayout(doctor_button_layout)
        doctor_group.setLayout(doctor_layout)
        main_layout.addWidget(doctor_group)

        patient_group = QGroupBox("Paciente")
        patient_layout = QVBoxLayout()
        selector_layout = QHBoxLayout()
        patient_label = QLabel("Selecione:")
        self.patient_combo = QComboBox()
        self.load_patients()
        selector_layout.addWidget(patient_label)
        selector_layout.addWidget(self.patient_combo)
        patient_layout.addLayout(selector_layout)
        patient_button_layout = QHBoxLayout()
        add_patient_btn = QPushButton("Adicionar")
        remove_patient_btn = QPushButton("Remover")
        add_patient_btn.clicked.connect(self.add_new_patient)
        remove_patient_btn.clicked.connect(self.remove_patient)
        patient_button_layout.addStretch()
        patient_button_layout.addWidget(add_patient_btn)
        patient_button_layout.addWidget(remove_patient_btn)
        patient_layout.addLayout(patient_button_layout)
        patient_group.setLayout(patient_layout)
        main_layout.addWidget(patient_group)
        main_layout.addSpacing(20)

        action_buttons_layout = QHBoxLayout()
        collect_data_btn = QPushButton("Coletar Dados")
        collect_data_btn.clicked.connect(self.open_collect_data_window)
        analyse_data_btn = QPushButton("Analisar Dados")
        analyse_data_btn.clicked.connect(self.open_analyse_data_window)
        action_buttons_layout.addStretch()
        action_buttons_layout.addWidget(collect_data_btn)
        action_buttons_layout.addWidget(analyse_data_btn)
        main_layout.addLayout(action_buttons_layout)

        main_layout.setContentsMargins(40, 30, 40, 30)
        self.setLayout(main_layout)
        self.setFixedSize(pixmap.width() + 80, pixmap.height() + 400)

    def apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #f7f7f7;
            }
            QLabel#header_label {
                color: #2c3e50;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #d3d3d3;
                border-radius: 5px;
                margin-top: 20px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 10px;
            }
            QLabel {
                font-size: 14px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QComboBox {
                padding: 5px;
                font-size: 14px;
            }
            QLineEdit {
                padding: 5px;
                font-size: 14px;
            }
            QMessageBox {
                font-size: 14px;
            }
        """)

    def load_doctors(self):
        self.doctor_combo.clear()
        doctors = self.read_list_from_file("data/doctors.txt")
        self.doctor_combo.addItems(doctors)

    def load_patients(self):
        self.patient_combo.clear()
        patients = self.read_list_from_file("data/patients.txt")
        self.patient_combo.addItems(patients)

    def read_list_from_file(self, filename):
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        if not os.path.exists(filename):
            return []
        with open(filename, "r", encoding="latin-1") as f:
            lines = f.read().splitlines()
        return lines

    def write_list_to_file(self, filename, items):
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        with open(filename, "w", encoding="latin-1") as f:
            for item in items:
                f.write(item + "\n")

    def write_item_to_file(self, filename, item):
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        with open(filename, "a", encoding="latin-1") as f:
            f.write(item + "\n")

    def add_new_doctor(self):
        name, ok = self.get_name_dialog("Adicionar novo responsável")
        if ok and name.strip():
            doctors = self.read_list_from_file("data/doctors.txt")
            if name in doctors:
                QMessageBox.warning(self, "Erro", "Responsável já existe.")
                return
            self.write_item_to_file("data/doctors.txt", name)
            self.load_doctors()
        elif ok:
            QMessageBox.warning(self, "Erro", "Nome não pode ser vazio.")

    def add_new_patient(self):
        name, ok = self.get_name_dialog("Adicionar novo paciente")
        if ok and name.strip():
            patients = self.read_list_from_file("data/patients.txt")
            if name in patients:
                QMessageBox.warning(self, "Erro", "Paciente já existe.")
                return
            self.write_item_to_file("data/patients.txt", name)
            self.load_patients()
        elif ok:
            QMessageBox.warning(self, "Erro", "Nome não pode ser vazio.")

    def remove_doctor(self):
        current_doctor = self.doctor_combo.currentText()
        if not current_doctor:
            QMessageBox.information(self, "Informação", "Nenhum responsável selecionado para remover.")
            return
        reply = QMessageBox.question(self, "Confirmar remoção",
                                     f"Você tem certeza que deseja remover o responsável '{current_doctor}'? Todas as suas coletas serão perdidas!",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            doctors = self.read_list_from_file("data/doctors.txt")
            doctors.remove(current_doctor)
            self.write_list_to_file("data/doctors.txt", doctors)
            self.load_doctors()
            QMessageBox.information(self, "Sucesso", f"Responsável '{current_doctor}' foi removido.")

    def remove_patient(self):
        current_patient = self.patient_combo.currentText()
        if not current_patient:
            QMessageBox.information(self, "Informação", "Nenhum paciente selecionado para remover.")
            return
        reply = QMessageBox.question(self, "Confirmar remoção",
                                     f"Você tem certeza que deseja remover o paciente '{current_patient}'? Todos os seus dados serão perdidos!",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            patients = self.read_list_from_file("data/patients.txt")
            patients.remove(current_patient)
            self.write_list_to_file("data/patients.txt", patients)
            self.load_patients()
            QMessageBox.information(self, "Sucesso", f"Paciente '{current_patient}' foi removido.")

    def get_name_dialog(self, title):
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setWindowIcon(QIcon('images/logo.ico'))
        layout = QVBoxLayout(dialog)
        form_layout = QFormLayout()
        name_edit = QLineEdit()
        form_layout.addRow("Nome:", name_edit)
        layout.addLayout(form_layout)
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        ok_button = QPushButton("Confirmar")
        cancel_button = QPushButton("Cancelar")
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        result = dialog.exec_()
        name = name_edit.text()
        return name, result == QDialog.Accepted

    def open_collect_data_window(self):
        doctor_name = self.doctor_combo.currentText()
        patient_name = self.patient_combo.currentText()
        if not doctor_name or not patient_name:
            QMessageBox.warning(self, "Aviso", "Selecione um responsável e um paciente.")
            return
        self.collect_data_window = CollectDataWindow(doctor_name, patient_name)
        self.collect_data_window.show()

    def open_analyse_data_window(self):
        doctor_name = self.doctor_combo.currentText()
        if not doctor_name:
            QMessageBox.warning(self, "Aviso", "Selecione um responsável.")
            return
        self.analyse_data_window = AnalyseDataWindow(doctor_name)
        self.analyse_data_window.show()

class DataReceiver(threading.Thread):
    def __init__(self, host, port, data_callback):
        super().__init__()
        self.host = host
        self.port = port
        self.data_callback = data_callback  # Function to process received data
        self.client = None
        self.running = True

    def run(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.settimeout(5.0)  # Timeout to avoid blocking
        try:
            self.client.connect((self.host, self.port))
            print(f"Connected to server {self.host}:{self.port}")
            buffer = ""
            while self.running:
                try:
                    data = self.client.recv(1024).decode('utf-8')
                    if not data:
                        break
                    buffer += data
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        self.data_callback(line.strip())
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"Data reception error: {e}")
                    break
        except Exception as e:
            print(f"Connection error: {e}")
        finally:
            self.client.close()
            print("Connection closed.")

    def stop(self):
        self.running = False
        if self.client:
            self.client.close()

class CollectDataWindow(QWidget):
    def __init__(self, doctor_name, patient_name):
        super().__init__()
        self.doctor_name = doctor_name
        self.patient_name = patient_name
        self.setWindowTitle("Coletar Dados")
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.setMinimumSize(800, 600)  # Adjusted to accommodate the graphs
        self.is_recording = False  # Flag to indicate if recording is in progress
        self.initUI()
        self.apply_styles()
        self.setup_timers()
        # Start the data receiver
        self.data_lock = threading.Lock()
        self.acc_data = [[], []]  # Accelerometer data [ace_mao, ace_ant]
        self.pot_data = [[], []]  # Potentiometer data [ang_pun, ang_cot]
        self.time_data = []
        self.data_receiver = DataReceiver('192.168.4.1', 5005, self.process_data)
        self.data_receiver.start()
        # Load the trained model and scaler
        self.neural_net_dir = os.path.join(os.path.dirname(__file__), 'neural_net')
        try:
            self.model_mao = load_model(os.path.join(self.neural_net_dir, 'trained_model_mao.keras'))
            self.scaler_mao = joblib.load(os.path.join(self.neural_net_dir, 'scaler_mao.save'))
        except Exception as e:
            QMessageBox.warning(self, "Erro", f"Erro ao carregar o modelo da mão: {e}")
            self.model_mao = None
            self.scaler_mao = None
        try:
            self.model_ant = load_model(os.path.join(self.neural_net_dir, 'trained_model_ant.keras'))
            self.scaler_ant = joblib.load(os.path.join(self.neural_net_dir, 'scaler_ant.save'))
        except Exception as e:
            QMessageBox.warning(self, "Erro", f"Erro ao carregar o modelo do antebraço: {e}")
            self.model_ant = None
            self.scaler_ant = None

    def initUI(self):
        main_layout = QVBoxLayout()

        # Information label
        info_label = QLabel(f"Responsável: {self.doctor_name}    Paciente: {self.patient_name}")
        info_label.setAlignment(Qt.AlignCenter)
        info_font = QFont("Arial", 16, QFont.Bold)
        info_label.setFont(info_font)
        main_layout.addWidget(info_label)
        main_layout.addSpacing(10)

        # AVC Checkbox
        self.avc_checkbox = QCheckBox("AVC")
        main_layout.addWidget(self.avc_checkbox)
        main_layout.addSpacing(10)

        # Intervention selection
        intervention_group = QGroupBox("Intervenção")
        intervention_layout = QVBoxLayout()
        selector_layout = QHBoxLayout()
        intervention_label = QLabel("Selecione:")
        self.intervention_combo = QComboBox()
        self.load_interventions()
        self.intervention_combo.currentIndexChanged.connect(self.on_intervention_changed)
        selector_layout.addWidget(intervention_label)
        selector_layout.addWidget(self.intervention_combo)
        intervention_layout.addLayout(selector_layout)
        intervention_group.setLayout(intervention_layout)
        main_layout.addWidget(intervention_group)

        # Area for graphs
        self.graphs_layout = QVBoxLayout()
        main_layout.addLayout(self.graphs_layout)

        main_layout.addStretch()

        # Record / Stop Recording button
        self.record_btn = QPushButton("Gravar")
        record_layout = QHBoxLayout()
        record_layout.addStretch()
        self.record_btn.clicked.connect(self.toggle_recording)
        record_layout.addWidget(self.record_btn)
        record_layout.addStretch()
        main_layout.addLayout(record_layout)

        self.setLayout(main_layout)

        # Initialize the graphs
        self.init_graphs()
        self.on_intervention_changed()  # Update the visibility of graphs

    def load_interventions(self):
        interventions = ["Min/Max", "Destreza"]
        self.intervention_combo.clear()
        self.intervention_combo.addItems(interventions)

    def apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #f7f7f7;
            }
            QLabel {
                font-size: 14px;
            }
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                border: 1px solid #d3d3d3;
                border-radius: 5px;
                margin-top: 10px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 10px;
            }
            QPushButton {
                background-color: #2ecc71;
                color: white;
                border: none;
                padding: 10px 30px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
            QComboBox {
                padding: 5px;
                font-size: 14px;
            }
            QCheckBox {
                font-size: 14px;
            }
        """)

    def init_graphs(self):
        self.accelerometer_graphs = []
        self.potentiometer_graphs = []
        self.graph_widgets = []
        self.graph_rows = []  # To keep references to the row widgets
        self.acc_row_widget = None
        self.pot_row_widget = None
        self.update_graphs()  # Initialize the graphs

    def update_graphs(self):
        # Clear existing graphs
        for row_widget in self.graph_rows:
            self.graphs_layout.removeWidget(row_widget)
            row_widget.setParent(None)
            row_widget.deleteLater()
        self.graph_rows.clear()
        self.graph_widgets.clear()
        self.accelerometer_graphs.clear()
        self.potentiometer_graphs.clear()

        # Create a widget for the accelerometer graphs row
        self.acc_row_widget = QWidget()
        acc_layout = QHBoxLayout(self.acc_row_widget)
        for i in range(2):
            graph_container = QWidget()
            graph_container_layout = QVBoxLayout(graph_container)
            label_text = "Aceleração mão" if i == 0 else "Aceleração antebraço"
            label = QLabel(label_text)
            label.setAlignment(Qt.AlignCenter)
            label.setFont(QFont("Arial", 12, QFont.Bold))
            plot_widget = pg.PlotWidget()
            graph_container_layout.addWidget(label)
            graph_container_layout.addWidget(plot_widget)
            acc_layout.addWidget(graph_container)
            self.graph_widgets.append(plot_widget)
            self.accelerometer_graphs.append(plot_widget)
        self.graphs_layout.addWidget(self.acc_row_widget)
        self.graph_rows.append(self.acc_row_widget)

        # Create a widget for the potentiometer graphs row
        self.pot_row_widget = QWidget()
        pot_layout = QHBoxLayout(self.pot_row_widget)
        for i in range(2):
            graph_container = QWidget()
            graph_container_layout = QVBoxLayout(graph_container)
            label_text = "Ângulo punho" if i == 0 else "Ângulo cotovelo"
            label = QLabel(label_text)
            label.setAlignment(Qt.AlignCenter)
            label.setFont(QFont("Arial", 12, QFont.Bold))
            plot_widget = pg.PlotWidget()
            graph_container_layout.addWidget(label)
            graph_container_layout.addWidget(plot_widget)
            pot_layout.addWidget(graph_container)
            self.graph_widgets.append(plot_widget)
            self.potentiometer_graphs.append(plot_widget)
        self.graphs_layout.addWidget(self.pot_row_widget)
        self.graph_rows.append(self.pot_row_widget)

    def setup_timers(self):
        # Timer to update the graphs
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(50)  # Updates every 50 ms

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        # Start recording
        self.is_recording = True
        self.record_btn.setText("Parar Gravação")
        # Clear previous data
        with self.data_lock:
            self.acc_data = [[], []]  # [ace_mao, ace_ant]
            self.pot_data = [[], []]  # [ang_pun, ang_cot]
            self.time_data = []

        # Initialize min/max variables if "Min/Max" is selected
        if self.intervention_combo.currentText() == "Min/Max":
            self.cot_ang_min = None
            self.cot_ang_max = None
            self.pun_ang_min = None
            self.pun_ang_max = None

        QMessageBox.information(self, "Gravação Iniciada", "Iniciando a gravação dos dados.")

    def stop_recording(self):
        # Stop recording
        self.is_recording = False
        self.record_btn.setText("Gravar")

        # Save data based on the selected intervention
        if self.intervention_combo.currentText() == "Min/Max":
            self.save_minmax_data()
        elif self.intervention_combo.currentText() == "Destreza":
            self.save_accelerometer_data()

        QMessageBox.information(self, "Gravação Finalizada", "Dados gravados com sucesso.")

    def process_data(self, data):
        try:
            values = data.strip().split(',')
            with self.data_lock:
                # Update time
                current_time = self.time_data[-1] + 0.05 if self.time_data else 0
                self.time_data.append(current_time)

                num_values = len(values)
                index = 0

                # Update potentiometer data (ang_pun, ang_cot)
                if num_values >= index + 2:
                    ang_pun = float(values[index])
                    ang_cot = float(values[index + 1])
                    self.pot_data[0].append(ang_pun)
                    self.pot_data[1].append(ang_cot)

                    # Update min/max values if recording and "Min/Max" is selected
                    if self.is_recording and self.intervention_combo.currentText() == "Min/Max":
                        # Update min/max for elbow angle (ang_cot)
                        if self.cot_ang_min is None or ang_cot < self.cot_ang_min:
                            self.cot_ang_min = ang_cot
                        if self.cot_ang_max is None or ang_cot > self.cot_ang_max:
                            self.cot_ang_max = ang_cot
                        # Update min/max for wrist angle (ang_pun)
                        if self.pun_ang_min is None or ang_pun < self.pun_ang_min:
                            self.pun_ang_min = ang_pun
                        if self.pun_ang_max is None or ang_pun > self.pun_ang_max:
                            self.pun_ang_max = ang_pun

                    index += 2

                # Update accelerometer data (ace_mao, ace_ant)
                if num_values >= index + 2:
                    ace_mao = float(values[index])
                    ace_ant = float(values[index + 1])
                    self.acc_data[0].append(ace_mao)
                    self.acc_data[1].append(ace_ant)
                    index += 2

        except Exception as e:
            print(f"Erro ao processar dados: {e}")

    def update_data(self):
        with self.data_lock:
            # Update accelerometer graphs
            for i, plot_widget in enumerate(self.accelerometer_graphs):
                plot_widget.plot(self.time_data, self.acc_data[i], clear=True)
                plot_widget.setLabel('left', 'Amplitude')
                plot_widget.setLabel('bottom', 'Tempo', units='s')
            # Update potentiometer graphs
            for i, plot_widget in enumerate(self.potentiometer_graphs):
                plot_widget.plot(self.time_data, self.pot_data[i], clear=True)
                plot_widget.setLabel('left', 'Amplitude')
                plot_widget.setLabel('bottom', 'Tempo', units='s')

    def save_minmax_data(self):
        # Ensure the directory exists
        entries_dir = os.path.join('data', 'entries')
        if not os.path.exists(entries_dir):
            os.makedirs(entries_dir)
        # Define the file path
        filepath = os.path.join(entries_dir, 'minmax.csv')
        # Get the date in day/month/year format
        date_str = datetime.now().strftime("%d/%m/%Y")
        # Get AVC status
        avc_status = '1' if self.avc_checkbox.isChecked() else '0'
        # Prepare values with default empty strings if None
        cot_ang_min = str(self.cot_ang_min) if self.cot_ang_min is not None else ''
        cot_ang_max = str(self.cot_ang_max) if self.cot_ang_max is not None else ''
        pun_ang_min = str(self.pun_ang_min) if self.pun_ang_min is not None else ''
        pun_ang_max = str(self.pun_ang_max) if self.pun_ang_max is not None else ''
        # Prepare the data as a dictionary
        data = {
            'paciente': [self.patient_name.strip().replace(';', ' ')],
            'avc': [avc_status],
            'data': [date_str],
            'cot_ang_min': [cot_ang_min],
            'cot_ang_max': [cot_ang_max],
            'pun_ang_min': [pun_ang_min],
            'pun_ang_max': [pun_ang_max]
        }
        df = pd.DataFrame(data)
        # Write to CSV file
        if not os.path.isfile(filepath):
            df.to_csv(filepath, sep=';', index=False)
        else:
            df.to_csv(filepath, sep=';', mode='a', index=False, header=False)
        print(f"Min/max data saved to {filepath}")

    def save_accelerometer_data(self):
        # Ensure the directory exists
        entries_dir = os.path.join('data', 'entries')
        if not os.path.exists(entries_dir):
            os.makedirs(entries_dir)

        # Define the file path for 'destreza.csv'
        filepath = os.path.join(entries_dir, 'destreza.csv')

        # Get the date in day/month/year format
        date_str = datetime.now().strftime("%d/%m/%Y")
        # Get AVC status
        avc_status = '1' if self.avc_checkbox.isChecked() else '0'

        # Process ace_mao and ace_ant
        mao_bands = self.process_accelerometer_signal(self.acc_data[0], self.time_data)
        ant_bands = self.process_accelerometer_signal(self.acc_data[1], self.time_data)

        # Ensure bands have 10 elements
        if len(mao_bands) != 10 or len(ant_bands) != 10:
            QMessageBox.warning(self, "Erro", "Erro ao processar dados do acelerômetro.")
            return

        # Prepare the input for classification
        # For hand (mao)
        X_mao_new = np.array(mao_bands).reshape(1, -1)
        X_mao_new_scaled = self.scaler_mao.transform(X_mao_new)

        # For forearm (ant)
        X_ant_new = np.array(ant_bands).reshape(1, -1)
        X_ant_new_scaled = self.scaler_ant.transform(X_ant_new)

        # Predict the classes
        # Predict the class for hand
        y_mao_pred = self.model_mao.predict(X_mao_new_scaled)
        class_mao_pred = np.argmax(y_mao_pred, axis=1)[0]  # Get the class with the highest probability

        # Predict the class for forearm
        y_ant_pred = self.model_ant.predict(X_ant_new_scaled)
        class_ant_pred = np.argmax(y_ant_pred, axis=1)[0]  # Get the class with the highest probability

        # Prepare the data as a dictionary
        data = {
            'paciente': [self.patient_name.strip().replace(';', ' ')],
            'avc': [avc_status],
            'data': [date_str],
            'classe_mao': [class_mao_pred],
            'classe_ant': [class_ant_pred],
        }
        data.update({f"mao_b{i+1}": [mao_bands[i]] for i in range(10)})
        data.update({f"ant_b{i+1}": [ant_bands[i]] for i in range(10)})

        df = pd.DataFrame(data)
        # Write to CSV file
        if not os.path.isfile(filepath):
            df.to_csv(filepath, sep=';', index=False)
        else:
            df.to_csv(filepath, sep=';', mode='a', index=False, header=False)
        print(f"Data saved to {filepath}")

    def process_accelerometer_signal(self, signal_data, time_data):
        # Convert lists to numpy arrays
        signal_data = np.array(signal_data)
        time_data = np.array(time_data)

        if len(signal_data) < 2 or len(time_data) < 2:
            return [0]*10

        # Polynomial regression of degree 10
        coeffs = np.polyfit(time_data, signal_data, deg=10)
        poly_func = np.poly1d(coeffs)

        # Calculate the derivative of the polynomial function
        derivative_func = np.polyder(poly_func)

        # Evaluate the derivative
        derivative_values = derivative_func(time_data)

        # Perform FFT
        N = len(derivative_values)
        T = time_data[1] - time_data[0] if len(time_data) > 1 else 1.0  # Sampling interval
        yf = fft(derivative_values)
        xf = fftfreq(N, T)[:N//2]

        # Calculate magnitude
        magnitude = 2.0/N * np.abs(yf[0:N//2])

        # Define frequency bands from 0Hz to 20Hz divided into 10 intervals
        freq_bands = np.linspace(0, 20, 11)
        band_values = []

        for i in range(10):
            # Get indices for the current frequency band
            idx_band = np.where((xf >= freq_bands[i]) & (xf < freq_bands[i+1]))
            # Calculate the integral (area under the curve) for the current band
            if idx_band[0].size > 0:
                band_integral = np.trapz(magnitude[idx_band], xf[idx_band])
            else:
                band_integral = 0
            band_values.append(band_integral)

        return band_values

    def closeEvent(self, event):
        # Stop the timer and data receiver when the window is closed
        self.timer.stop()
        self.data_receiver.stop()
        self.data_receiver.join()
        event.accept()

    def on_intervention_changed(self):
        selected_intervention = self.intervention_combo.currentText()
        if selected_intervention == "Min/Max":
            self.show_potentiometer_plots()
            self.hide_accelerometer_plots()
        elif selected_intervention == "Destreza":
            self.show_accelerometer_plots()
            self.hide_potentiometer_plots()
        else:
            # Show all graphs if needed
            self.show_potentiometer_plots()
            self.show_accelerometer_plots()

    def show_potentiometer_plots(self):
        if self.pot_row_widget:
            self.pot_row_widget.show()

    def hide_potentiometer_plots(self):
        if self.pot_row_widget:
            self.pot_row_widget.hide()

    def show_accelerometer_plots(self):
        if self.acc_row_widget:
            self.acc_row_widget.show()

    def hide_accelerometer_plots(self):
        if self.acc_row_widget:
            self.acc_row_widget.hide()

class AnalyseDataWindow(QWidget):
    def __init__(self, doctor_name):
        super().__init__()
        self.doctor_name = doctor_name
        self.setWindowTitle("Analisar Dados")
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.setMinimumSize(400, 200)
        self.initUI()
        self.apply_styles()

    def initUI(self):
        self.main_layout = QVBoxLayout()

        # Display only the doctor's name
        info_label = QLabel(f"Responsável: {self.doctor_name}")
        info_label.setAlignment(Qt.AlignCenter)
        info_font = QFont("Arial", 16, QFont.Bold)
        info_label.setFont(info_font)
        self.main_layout.addWidget(info_label)
        self.main_layout.addSpacing(20)

        # Analysis selector
        analysis_layout = QHBoxLayout()
        analysis_label = QLabel("Análise:")
        self.analysis_combo = QComboBox()
        self.analysis_combo.addItems(["Boxplot - Min/Max", "Destreza"])
        analysis_layout.addWidget(analysis_label)
        analysis_layout.addWidget(self.analysis_combo)
        self.main_layout.addLayout(analysis_layout)
        self.analysis_combo.currentIndexChanged.connect(self.on_analysis_changed)

        # Patient selector
        patient_layout = QHBoxLayout()
        patient_label = QLabel("Paciente:")
        self.patient_combo = QComboBox()
        patient_layout.addWidget(patient_label)
        patient_layout.addWidget(self.patient_combo)
        self.main_layout.addLayout(patient_layout)

        # Analyze button
        analyze_layout = QHBoxLayout()
        analyze_layout.addStretch()
        self.analyze_button = QPushButton("Analisar")
        self.analyze_button.clicked.connect(self.perform_analysis)
        analyze_layout.addWidget(self.analyze_button)
        analyze_layout.addStretch()
        self.main_layout.addLayout(analyze_layout)

        # Initialize selections
        self.setLayout(self.main_layout)
        self.load_patients()  # Load initial data

    def apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #f7f7f7;
            }
            QLabel {
                font-size: 14px;
            }
            QPushButton {
                background-color: #e67e22;
                color: white;
                border: none;
                padding: 10px 30px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #d35400;
            }
            QComboBox {
                padding: 5px;
                font-size: 14px;
            }
        """)

    def on_analysis_changed(self):
        self.load_patients()

    def load_patients(self):
        selected_analysis = self.analysis_combo.currentText()

        if selected_analysis == "Boxplot - Min/Max":
            data_file = os.path.join('data', 'entries', 'minmax.csv')
        elif selected_analysis == "Destreza":
            data_file = os.path.join('data', 'entries', 'destreza.csv')
        else:
            QMessageBox.warning(self, "Aviso", "Selecione uma análise válida.")
            self.patient_combo.clear()
            return

        if not os.path.exists(data_file):
            QMessageBox.warning(self, "Aviso", f"O arquivo {data_file} não existe.")
            self.patient_combo.clear()
            return

        # Load unique patients from the CSV file
        try:
            df = pd.read_csv(data_file, sep=';')
            unique_patients = df['paciente'].dropna().unique()
            self.patient_combo.clear()
            self.patient_combo.addItems(unique_patients.astype(str))
        except Exception as e:
            QMessageBox.warning(self, "Erro", f"Erro ao carregar pacientes: {e}")
            self.patient_combo.clear()
            return

    def perform_analysis(self):
        selected_patient = self.patient_combo.currentText()
        selected_analysis = self.analysis_combo.currentText()

        if not selected_patient:
            QMessageBox.warning(self, "Aviso", "Selecione um paciente.")
            return

        if selected_analysis == "Boxplot - Min/Max":
            data_file = os.path.join('data', 'entries', 'minmax.csv')
            if not os.path.exists(data_file):
                QMessageBox.warning(self, "Aviso", f"O arquivo {data_file} não existe.")
                return
            try:
                # Read data
                df = pd.read_csv(data_file, sep=';', header=0)

                # Convert 'avc' column to numeric
                df['avc'] = pd.to_numeric(df['avc'], errors='coerce')

                # Verify that 'data' column exists
                if 'data' not in df.columns:
                    QMessageBox.warning(self, "Erro", "A coluna 'data' não foi encontrada no arquivo CSV.")
                    return

                # Create temporary DataFrame
                # Include records for the selected patient and all patients with avc == 0
                df_patient = df[df['paciente'] == selected_patient].copy()
                df_control = df[df['avc'] == 0].copy()
                df_control['paciente'] = 'Controle'  # Label for control group
                # Combine dataframes
                df_combined = pd.concat([df_patient, df_control], ignore_index=True)

                # Strip leading/trailing whitespaces from 'data' column
                df_combined['data'] = df_combined['data'].astype(str).str.strip()

                # Parse dates using the correct format
                df_combined['data'] = pd.to_datetime(df_combined['data'], format="%d/%m/%Y", errors='coerce')

                # Check for NaT values (unparseable dates)
                if df_combined['data'].isnull().any():
                    invalid_dates = df_combined[df_combined['data'].isnull()]
                    QMessageBox.warning(
                        self,
                        "Erro",
                        f"As seguintes datas são inválidas e não puderam ser processadas:\n{invalid_dates['data']}"
                    )
                    return

                # Sort by date
                df_combined.sort_values('data', inplace=True)
                # Convert 'data' back to string for plotting
                df_combined['data_str'] = df_combined['data'].dt.strftime("%d/%m/%Y")
                # Replace date with 'Controle' for control group
                df_combined.loc[df_combined['paciente'] == 'Controle', 'data_str'] = 'Controle'

                # Determine the order of dates with 'Controle' at the end
                date_series = df_combined[df_combined['data_str'] != 'Controle']['data_str']
                sorted_dates = pd.to_datetime(date_series, format="%d/%m/%Y", dayfirst=True).sort_values().dt.strftime("%d/%m/%Y")
                date_order = sorted_dates.unique().tolist()
                date_order.append('Controle')  # Place 'Controle' at the end

                # Plotting
                self.show_boxplots(df_combined, date_order, selected_patient)
            except Exception as e:
                QMessageBox.warning(self, "Erro", f"Erro ao processar dados: {e}")
                return
        elif selected_analysis == "Destreza":
            data_file = os.path.join('data', 'entries', 'destreza.csv')
            if not os.path.exists(data_file):
                QMessageBox.warning(self, "Aviso", f"O arquivo {data_file} não existe.")
                return
            try:
                # Process destreza data
                self.process_destreza_data(selected_patient, data_file)
            except Exception as e:
                QMessageBox.warning(self, "Erro", f"Erro ao processar dados: {e}")
                return
        else:
            QMessageBox.warning(self, "Aviso", "Selecione uma análise válida.")
            return

    def process_destreza_data(self, selected_patient, data_file):
        # Read data
        df = pd.read_csv(data_file, sep=';', header=0)

        # Verify that 'data' column exists
        if 'data' not in df.columns:
            QMessageBox.warning(self, "Erro", "A coluna 'data' não foi encontrada no arquivo CSV.")
            return

        # Verify that 'classe_mao' and 'classe_ant' columns exist
        if 'classe_mao' not in df.columns or 'classe_ant' not in df.columns:
            QMessageBox.warning(self, "Erro", "As colunas 'classe_mao' e/ou 'classe_ant' não foram encontradas no arquivo CSV.")
            return

        # Subset data for selected patient
        df_patient = df[df['paciente'] == selected_patient].copy()

        if df_patient.empty:
            QMessageBox.warning(self, "Aviso", "Não há dados para o paciente selecionado.")
            return

        # Parse dates
        df_patient['data'] = df_patient['data'].astype(str).str.strip()
        df_patient['data'] = pd.to_datetime(df_patient['data'], format="%d/%m/%Y", errors='coerce')

        # Check for NaT values (unparseable dates)
        if df_patient['data'].isnull().any():
            invalid_dates = df_patient[df_patient['data'].isnull()]
            QMessageBox.warning(
                self,
                "Erro",
                f"As seguintes datas são inválidas e não puderam ser processadas:\n{invalid_dates['data']}"
            )
            return

        # Process data for classe_mao and classe_ant
        df_patient['date_str'] = df_patient['data'].dt.strftime("%d/%m/%Y")

        # Prepare data for classe_mao
        group_mao = df_patient.groupby(['date_str', 'classe_mao']).size().reset_index(name='counts')
        total_counts_mao = group_mao.groupby('date_str')['counts'].sum().reset_index(name='total_counts')
        group_mao = pd.merge(group_mao, total_counts_mao, on='date_str')
        group_mao['percentage'] = group_mao['counts'] / group_mao['total_counts'] * 100
        group_mao['classe'] = group_mao['classe_mao'].astype(int)
        group_mao['type'] = 'Mão'

        # Prepare data for classe_ant
        group_ant = df_patient.groupby(['date_str', 'classe_ant']).size().reset_index(name='counts')
        total_counts_ant = group_ant.groupby('date_str')['counts'].sum().reset_index(name='total_counts')
        group_ant = pd.merge(group_ant, total_counts_ant, on='date_str')
        group_ant['percentage'] = group_ant['counts'] / group_ant['total_counts'] * 100
        group_ant['classe'] = group_ant['classe_ant'].astype(int)
        group_ant['type'] = 'Antebraço'

        # Combine data
        group_combined = pd.concat([group_mao, group_ant], ignore_index=True)

        # Sort dates
        date_order = pd.to_datetime(group_combined['date_str'], format="%d/%m/%Y", dayfirst=True).sort_values().dt.strftime("%d/%m/%Y").unique()

        # Create ordered categorical type for date_str
        group_combined['date_str'] = pd.Categorical(group_combined['date_str'], categories=date_order, ordered=True)

        # Sort data
        group_combined.sort_values(['type', 'date_str'], inplace=True)

        # Plot
        self.show_destreza_plots(group_combined, date_order)

    def show_destreza_plots(self, df, date_order):
        # Create a new window for the plots
        self.plot_window = QWidget()
        self.plot_window.setWindowTitle("Destreza - Evolução das Classes")
        self.plot_window.setWindowIcon(QIcon('images/logo.ico'))
        layout = QVBoxLayout(self.plot_window)

        # Create a matplotlib figure with two subplots
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        fig.tight_layout(pad=4.0)

        # Separate data for mão and antebraço
        df_mao = df[df['type'] == 'Mão']
        df_ant = df[df['type'] == 'Antebraço']

        # Plotting for classe_mao
        sns.scatterplot(
            data=df_mao,
            x='date_str',
            y='classe',
            size='percentage',
            sizes=(50, 300),
            hue=None,
            color='red',
            legend=False,
            ax=axs[0]
        )
        axs[0].set_title('Evolução das Classes - Mão')
        axs[0].set_xlabel('Data')
        axs[0].set_ylabel('Classe')
        axs[0].set_yticks([0, 1, 2, 3])
        axs[0].set_yticklabels(['0', '1', '2', '3'])
        axs[0].set_xticklabels(date_order, rotation=45)
        axs[0].invert_yaxis()

        # Plotting for classe_ant
        sns.scatterplot(
            data=df_ant,
            x='date_str',
            y='classe',
            size='percentage',
            sizes=(50, 300),
            hue=None,
            color='red',
            legend=False,
            ax=axs[1]
        )
        axs[1].set_title('Evolução das Classes - Antebraço')
        axs[1].set_xlabel('Data')
        axs[1].set_ylabel('Classe')
        axs[1].set_yticks([0, 1, 2, 3])
        axs[1].set_yticklabels(['0', '1', '2', '3'])
        axs[1].set_xticklabels(date_order, rotation=45)
        axs[1].invert_yaxis()

        # Add the matplotlib figure to the PyQt5 window
        self.canvas = FigureCanvas(fig)
        layout.addWidget(self.canvas)

        self.plot_window.show()

    def show_boxplots(self, df, date_order, selected_patient):
        # Create a new window for the plots and store it as an instance variable
        self.plot_window = QWidget()
        self.plot_window.setWindowTitle("Boxplots - Min/Max")
        self.plot_window.setWindowIcon(QIcon('images/logo.ico'))
        layout = QVBoxLayout(self.plot_window)

        # Create a matplotlib figure
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        fig.tight_layout(pad=3.0)

        # Define the columns and titles for the plots
        columns = ['cot_ang_min', 'cot_ang_max', 'pun_ang_min', 'pun_ang_max']
        titles = ['Ângulo Mínimo Cotovelo', 'Ângulo Máximo Cotovelo', 'Ângulo Mínimo Punho', 'Ângulo Máximo Punho']

        # Define palette
        palette = {'Controle': 'lightblue', selected_patient: 'lightgreen'}

        for i, ax in enumerate(axs.flat):
            col = columns[i]
            title = titles[i]
            sns.boxplot(
                x='data_str', y=col, hue='paciente', data=df,
                ax=ax, palette=palette,
                order=date_order
            )
            ax.set_title(title)
            ax.set_xlabel('Data')
            ax.set_ylabel('Ângulo (graus)')
            ax.legend(title='Paciente', loc='best')
            ax.tick_params(axis='x', rotation=45)

        # Add the matplotlib figure to the PyQt5 window
        self.canvas = FigureCanvas(fig)
        layout.addWidget(self.canvas)

        self.plot_window.show()



if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Set the application icon
    app.setWindowIcon(QIcon('images/logo.ico'))  # Ensure 'images/logo.ico' exists

    # For Windows: set the AppUserModelID so the icon appears on the taskbar
    if sys.platform == "win32":
        import ctypes
        myappid = 'mycompany.myproduct.subproduct.version'  # Replace with a unique identifier
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    dashboard = Dashboard()
    dashboard.show()
    sys.exit(app.exec_())
