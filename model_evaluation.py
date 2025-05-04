from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

class ModelEvaluator:
    """
    Class untuk mengevaluasi performa model klasifikasi biner,
    termasuk perhitungan metrik dan ekstraksi data history training.
    """

    def __init__(self):
        # Simpan model dan label klasifikasi (0: Negatif, 1: Positif)
        self.model = None
        self.class_labels = ["Negatif", "Positif"]

    def set_model_info(self, model):
        """
        Menyimpan model yang sedang digunakan untuk referensi evaluasi.

        Parameters:
            model: Model deep learning (TensorFlow/Keras) yang sedang aktif
        """
        self.model = model

    def generate_evaluation_data(self, eval_data):
        """
        Menghitung metrik evaluasi dari hasil prediksi dan label aktual.

        Parameters:
            eval_data (dict): Dictionary yang berisi:
                - 'y_true': List/array label sebenarnya
                - 'y_pred': List/array hasil prediksi model

        Returns:
            tuple:
                - confusion_matrix (np.ndarray)
                - classification_report (dict)
                - accuracy (float)
                - precision (float)
                - recall (float)
                - f1_score (float)
        """
        y_true = eval_data['y_true']
        y_pred = eval_data['y_pred']

        # Hitung confusion matrix dan classification report
        cm = confusion_matrix(y_true, y_pred)
        report_dict = classification_report(
            y_true, y_pred, target_names=self.class_labels, output_dict=True
        )
        
        # Hitung metrik utama (dalam persen)
        accuracy = accuracy_score(y_true, y_pred) * 100
        precision = precision_score(y_true, y_pred) * 100
        recall = recall_score(y_true, y_pred) * 100
        f1 = f1_score(y_true, y_pred) * 100

        return cm, report_dict, accuracy, precision, recall, f1

    def plot_accuracy_loss(self, eval_data):
        """
        Mengambil data history training (accuracy dan loss) untuk keperluan visualisasi.

        Parameters:
            eval_data (dict): Dictionary yang berisi:
                - 'history': History callback dari model training

        Returns:
            tuple:
                - train_accuracy (list of float)
                - val_accuracy (list of float)
                - train_loss (list of float)
                - val_loss (list of float)
        """
        history = eval_data['history'].item()  # .item() diperlukan jika history disimpan sebagai np.object
        return (
            history['accuracy'],
            history['val_accuracy'],
            history['loss'],
            history['val_loss']
        )
