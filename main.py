import streamlit as st
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

from image_prediction import ImagePredictor
from model_evaluation import ModelEvaluator
from model_loader import ModelLoader


class MainApp:
    def __init__(self):
        self.predictor = ImagePredictor()
        self.evaluator = ModelEvaluator()
        self.model_loader = ModelLoader(self.predictor, self.evaluator)

        self.architectures = ["-", "DenseNet201", "VGG19", "InceptionV3"]
        self.optimizers = ["-", "Adam", "SGD"]
        self.learning_rates = ["-", "0.001", "0.0001"]

        # Restore model jika ada
        if "model_instance" in st.session_state:
            self.predictor.update_model(
                st.session_state["model_instance"],
                st.session_state["architecture"]
            )

    def load_css(self):
        if os.path.exists("styles.css"):
            with open("styles.css") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    def render_sidebar(self):
        st.sidebar.markdown("## 🔧 Konfigurasi Model")

        architecture = st.sidebar.selectbox("Arsitektur", self.architectures, key="arch")
        optimizer = st.sidebar.selectbox("Optimizer", self.optimizers, key="opt")
        lr = st.sidebar.selectbox("Learning Rate", self.learning_rates, key="lr")

        if st.sidebar.button("📥 Muat Model"):
            st.session_state["architecture"] = architecture
            st.session_state["optimizer"] = optimizer
            st.session_state["learning_rate"] = lr
            self.model_loader.load_model(architecture, optimizer, lr)

        st.sidebar.markdown("---")
        if st.sidebar.button("📈 Hasil Pemodelan Klasifikasi"):
            st.session_state.page = "Hasil Pemodelan Klasifikasi"
        if st.sidebar.button("🔍 Prediksi PCOS"):
            st.session_state.page = "Prediksi"

    def render_status_page(self):
        if not st.session_state.get("model_loaded", False):
            st.warning("⚠️ Model belum dimuat! Silakan pilih konfigurasi model di sidebar.")
            if st.session_state.get("error_msg"):
                st.error(st.session_state.error_msg)
        else:
            st.success("✅ Model berhasil dimuat! Silakan lanjut ke Hasil Pemodelan Klasifikasi atau Prediksi PCOS.")

    def require_model_loaded(self):
        if not st.session_state.get("model_loaded", False):
            self.render_status_page()
            return False
        return True

    def render_prediction_page(self):
        if not self.require_model_loaded():
            return

        st.markdown("<h1 style='color: #EC7FA9;text-align: center;margin-bottom: 30px;'>🔍 Prediksi PCOS 💫</h1>", unsafe_allow_html=True)
        st.markdown(
            "<p class='cute-subtitle'>Aplikasi ini menggunakan <b>Convolutional Neural Network (CNN)</b> untuk menganalisis citra ultrasonografi "
            "ovarium dan mendeteksi <b>Polycystic Ovary Syndrome (PCOS)</b> secara cepat dan akurat. Dengan deteksi dini, pasien bisa mendapatkan "
            "penanganan lebih awal untuk kesehatan yang lebih baik! ❤️</p>",
            unsafe_allow_html=True
        )
        st.markdown("---")

        st.markdown(
            f"<div style='background-color:#f5f5f5;padding:10px;border-radius:10px;margin-bottom:20px;'>"
            f"<b>📌 Model Aktif:</b> {st.session_state.get("architecture")} | <b>Optimizer:</b> {st.session_state.get("optimizer")} | <b>Learning Rate:</b> {st.session_state.get("learning_rate")}"
            f"</div>",
            unsafe_allow_html=True
        )

        uploaded_files = st.file_uploader("📤 Upload gambar dalam format JPG/JPEG/PNG (maksimal 10 MB per gambar).", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        if uploaded_files and st.button("🚀 Prediksi Gambar"):
            for uploaded_file in uploaded_files:
                if not self.predictor.validate_uploaded_file(uploaded_file):
                    st.error(f"🚨 Ukuran file '{uploaded_file.name}' melebihi 10 MB!")
                    continue

                image = Image.open(uploaded_file)
                st.image(image, caption=uploaded_file.name, use_container_width=True)

                with st.spinner("🔄 Sedang memproses gambar..."):
                    hasil_prediksi = self.predictor.predict_image(image)
                    st.markdown("### 🩺 Hasil Prediksi")
                    st.markdown(
                        f"<div class='result-card'><h2>{hasil_prediksi}</h2></div>",
                        unsafe_allow_html=True
                    )
                    st.markdown("---")

            st.success("✅ Prediksi selesai!")

    def render_evaluation_page(self):
        if not self.require_model_loaded():
            return

        st.markdown("<h1 style='color: #EC7FA9;text-align: center;margin-bottom: 30px;'>📈 Hasil Pemodelan Klasifikasi ✨</h1>", unsafe_allow_html=True)
        st.markdown(
            "<p class='cute-subtitle'>Model ini dibuat menggunakan metode <b>Convolutional Neural Network (CNN)</b> "
            "dengan tiga arsitektur berbeda, yaitu <b>DenseNet201, VGG19, dan InceptionV3</b>. Terdapat dua kombinasi "
            "parameter yang digunakan, yaitu <b>optimizer (Adam & SGD)</b> serta <b>learning rate (0.001 & 0.0001)</b> 💡</p>",
            unsafe_allow_html=True
        )
        st.markdown("---")

        st.markdown(
            f"<div style='background-color:#f5f5f5;padding:10px;border-radius:10px;margin-bottom:20px;'>"
            f"<b>📌 Model Aktif:</b> {st.session_state.get("architecture")} | <b>Optimizer:</b> {st.session_state.get("optimizer")} | <b>Learning Rate:</b> {st.session_state.get("learning_rate")}"
            f"</div>",
            unsafe_allow_html=True
        )

        model_filename = st.session_state.model_filename
        eval_data_path = os.path.join("evaluation_data", model_filename.replace(".h5", ".npz"))

        if not os.path.exists(eval_data_path):
            st.error("🚨 File hasil pemodelan klasifikasi tidak ditemukan di direktori!")
            return

        eval_data = np.load(eval_data_path, allow_pickle=True)
        cm, report_dict, acc, precision, recall, f1_score = self.evaluator.generate_evaluation_data(eval_data)
        accuracy, val_accuracy, loss, val_loss = self.evaluator.plot_accuracy_loss(eval_data)

        st.markdown("### 📉 Accuracy & Loss")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(accuracy, label='Train Accuracy')
        ax1.plot(val_accuracy, label='Val Accuracy')
        ax1.set_title('Accuracy')
        ax1.legend()
        ax2.plot(loss, label='Train Loss')
        ax2.plot(val_loss, label='Val Loss')
        ax2.set_title('Loss')
        ax2.legend()
        st.pyplot(fig)

        st.markdown("### 📊 Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.evaluator.class_labels, yticklabels=self.evaluator.class_labels)
        ax_cm.set_xlabel("Prediksi")
        ax_cm.set_ylabel("Aktual")
        st.pyplot(fig_cm)

        st.markdown("### 🎯 Metrik Evaluasi")
        st.markdown(f"- Akurasi: `{acc:.2f}%`")
        st.markdown(f"- Presisi: `{precision:.2f}%`")
        st.markdown(f"- Recall: `{recall:.2f}%`")
        st.markdown(f"- F1 Score: `{f1_score:.2f}%`")

        st.markdown("### 📋 Classification Report")
        report_df = DataFrame(report_dict).transpose().rename(columns={
            "precision": "Precision", "recall": "Recall", "f1-score": "F1-Score", "support": "Support"
        })
        st.dataframe(report_df.style.format({"Precision": "{:.2f}", "Recall": "{:.2f}", "F1-Score": "{:.2f}", "Support": "{:.0f}"}))

        st.markdown("---")
        st.success("✅ Evaluasi selesai!")

    def run(self):
        self.load_css()
        self.render_sidebar()

        if "page" not in st.session_state:
            st.session_state.page = "Status"

        match st.session_state.page:
            case "Status": self.render_status_page()
            case "Hasil Pemodelan Klasifikasi": self.render_evaluation_page()
            case "Prediksi": self.render_prediction_page()

if __name__ == "__main__":
    MainApp().run()
