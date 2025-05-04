import os
import time
import tensorflow as tf
import streamlit as st

class ModelLoader:
    """
    Class untuk menangani proses pemuatan model dari file .h5 berdasarkan
    konfigurasi arsitektur, optimizer, dan learning rate.
    """

    def __init__(self, predictor, evaluator):
        """
        Inisialisasi ModelLoader dengan dependency ke ImagePredictor dan ModelEvaluator.
        
        Parameters:
            predictor (ImagePredictor): Instance dari kelas ImagePredictor
            evaluator (ModelEvaluator): Instance dari kelas ModelEvaluator
        """
        self.predictor = predictor
        self.evaluator = evaluator
        self.model_dir = "models"
        self.current_model = None

    def get_model_filename(self, arch, opt, lr):
        """
        Menghasilkan nama file model berdasarkan konfigurasi.
        
        Parameters:
            arch (str): Arsitektur model (misal: DenseNet201)
            opt (str): Optimizer (misal: Adam, SGD)
            lr (str): Learning rate (misal: 0.001)
        
        Returns:
            str: Nama file model (contoh: DenseNet201_Adam_0.001.h5)
        """
        return f"{arch}_{opt}_{lr}.h5"

    def safe_load_model(self, model_path, max_retries=3, delay=0.5):
        """
        Memuat model TensorFlow (.h5) dengan retry jika terjadi error.

        Parameters:
            model_path (str): Path lengkap ke file model
            max_retries (int): Jumlah maksimum percobaan
            delay (float): Waktu tunggu antar percobaan dalam detik
        
        Returns:
            model (tf.keras.Model or None): Model yang berhasil dimuat, atau None jika gagal
        """
        for attempt in range(max_retries):
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
                return model
            except Exception as e:
                print(f"[Percobaan {attempt+1}] Gagal memuat model: {e}")
                time.sleep(delay)
        return None

    def load_model(self, arch, opt, lr):
        """
        Prosedur utama untuk memuat model dari konfigurasi user.

        Parameters:
            arch (str): Arsitektur model
            opt (str): Optimizer
            lr (str): Learning rate

        Efek Samping:
            - Mengatur session state pada Streamlit
            - Memperbarui model pada predictor dan evaluator
        """
        # Validasi input
        if arch == "-" or opt == "-" or lr == "-":
            st.session_state.model_loaded = False
            st.session_state.error_msg = "🚨 Konfigurasi model ada yang kosong! Lengkapi konfigurasi model di sidebar."
            return

        # Buat path file model
        filename = self.get_model_filename(arch, opt, lr)
        model_path = os.path.join(self.model_dir, filename)

        # Validasi file model tersedia
        if not os.path.exists(model_path):
            st.session_state.model_loaded = False
            st.session_state.error_msg = f"🚨 File model tidak ditemukan di direktori!"
            return

        # Load model dengan aman
        model = self.safe_load_model(model_path)
        if model is None:
            st.session_state.model_loaded = False
            st.session_state.error_msg = f"🚨 Gagal memuat model setelah beberapa percobaan. Periksa file atau coba lagi."
            return

        # Update model pada komponen lainnya
        self.predictor.update_model(model, arch)
        self.evaluator.set_model_info(model)
        self.current_model = (arch, opt, lr)

        # Update session state
        st.session_state.model_loaded = True
        st.session_state.model_filename = filename
        st.session_state.model_instance = model
        st.session_state.architecture = arch
        st.session_state.optimizer = opt
        st.session_state.learning_rate = lr
        st.session_state.error_msg = ""
        st.toast("✅ Model berhasil dimuat!")
