import numpy as np

class ImagePredictor:
    """
    Class untuk memproses dan memprediksi gambar USG menggunakan model CNN.
    """

    def __init__(self):
        # Inisialisasi variabel model dan arsitektur yang digunakan
        self.model = None
        self.architecture = None

    def update_model(self, model, architecture):
        """
        Memperbarui model dan arsitektur yang akan digunakan untuk prediksi.

        Parameters:
            model: Model TensorFlow/Keras yang telah dilatih
            architecture (str): Nama arsitektur model (misal: "VGG19", "InceptionV3")
        """
        self.model = model
        self.architecture = architecture

    def validate_uploaded_file(self, uploaded_file):
        """
        Memvalidasi file yang diunggah agar tidak melebihi 10MB.

        Parameters:
            uploaded_file: File gambar yang diunggah melalui Streamlit

        Returns:
            bool: True jika file valid, False jika terlalu besar atau tidak ada file
        """
        max_size = 10 * 1024 * 1024  # 10MB
        return uploaded_file is not None and uploaded_file.size <= max_size

    def preprocess_image(self, image):
        """
        Melakukan preprocessing pada gambar:
        - Konversi ke RGB
        - Resize ke ukuran input model (224x224 atau 299x299)
        - Normalisasi piksel ke range 0-1
        - Tambahkan dimensi batch

        Parameters:
            image (PIL.Image.Image): Gambar yang akan diproses

        Returns:
            np.ndarray: Gambar hasil preprocessing dalam bentuk array (1, H, W, 3)
        """
        # Tentukan ukuran target berdasarkan arsitektur
        target_size = (299, 299) if self.architecture == "InceptionV3" else (224, 224)

        # Proses gambar
        img = image.convert("RGB").resize(target_size)
        img_array = np.array(img) / 255.0  # Normalisasi ke [0, 1]
        return np.expand_dims(img_array, axis=0)  # Tambah dimensi batch

    def predict_image(self, image):
        """
        Melakukan prediksi terhadap gambar USG menggunakan model yang sudah dimuat.

        Parameters:
            image (PIL.Image.Image): Gambar USG yang akan diprediksi

        Returns:
            str: Hasil prediksi ("Positif PCOS" atau "Negatif PCOS")
        """
        # Validasi model
        if self.model is None:
            raise ValueError("Model belum dimuat. Gunakan `update_model()` terlebih dahulu.")
        
        if image is None:
            raise ValueError("Gambar tidak boleh kosong.")

        # Preprocessing dan prediksi
        img_array = self.preprocess_image(image)
        prediction = self.model.predict(img_array)[0][0]  # Ambil probabilitas

        # Hitung probabilitas
        prob_pos = prediction
        prob_neg = 1 - prediction

        # Tampil di terminal
        print(f"Probabilitas Positif PCOS: {prob_pos:.4f}")
        print(f"Probabilitas Negatif PCOS: {prob_neg:.4f}")

        # Return hasil klasifikasi saja ke UI
        return "👉 Positif PCOS 🟥" if prob_pos > 0.5 else "👉 Negatif PCOS 🟩"