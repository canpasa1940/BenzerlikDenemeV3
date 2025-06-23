import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import warnings
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
import librosa
from feature_extractor import extract_from_file, extract_features

class AudioClassifier:
    def __init__(self, model_path="my_enhanced_audio_model.h5", 
                 scaler_path="scaler.pkl", 
                 label_encoder_path="label_encoder.pkl"):
        """Ses sınıflandırıcı ve benzerlik analizi sınıfı"""
        warnings.filterwarnings("ignore")
        
        # Model ve ön işleme araçlarını yükle
        from tensorflow import keras
        self.model = keras.models.load_model(model_path)
        
        # PKL dosyalarını joblib ile yükle
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(label_encoder_path)
        
        self.classes = self.label_encoder.classes_
        print(f"Model yüklendi. Sınıflar: {self.classes}")
        
        # Referans ses veritabanı
        self.reference_database = []
        
    def predict_single(self, audio_file):
        """Tek bir ses dosyasını sınıflandır"""
        # Özellik çıkar
        features = extract_from_file(audio_file)
        if features is None:
            return None, None, None
            
        # DataFrame'e çevir ve sırala
        feature_names = [f"mfcc{i+1:02d}" for i in range(20)] + [
            "rms_mean", "rms_std", "zcr_mean", "centroid_mean", 
            "bandwidth_mean", "rolloff_mean", "flatness_mean", "flux_mean"
        ] + [f"contrast_b{i+1}" for i in range(7)] + [
            "onset_mean", "onset_std", "onset_max", "onset_sum",
            "attack_time", "attack_slope", "hpi_ratio"
        ]
        
        feature_vector = np.array([features[name] for name in feature_names]).reshape(1, -1)
        
        # Normalize et
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Tahmin yap
        prediction = self.model.predict(feature_vector_scaled)
        predicted_class_idx = np.argmax(prediction[0])
        predicted_class = self.classes[predicted_class_idx]
        confidence = prediction[0][predicted_class_idx]
        
        return predicted_class, confidence, feature_vector_scaled[0]
    
    def add_to_database(self, audio_file, predicted_class, features):
        """Sesi referans veritabanına ekle"""
        self.reference_database.append({
            'filename': audio_file.name if hasattr(audio_file, 'name') else str(audio_file),
            'class': predicted_class,
            'features': features
        })
    
    def find_similar_sounds(self, target_features, target_class, top_k=5):
        """Benzer sesleri bul"""
        if len(self.reference_database) == 0:
            return []
            
        similarities = []
        
        for ref in self.reference_database:
            # Aynı sınıftan olanları öncelendir
            if ref['class'] == target_class:
                # Cosine similarity hesapla
                cos_sim = cosine_similarity([target_features], [ref['features']])[0][0]
                
                # Euclidean distance hesapla  
                euc_dist = euclidean_distances([target_features], [ref['features']])[0][0]
                
                similarities.append({
                    'filename': ref['filename'],
                    'class': ref['class'],
                    'cosine_similarity': cos_sim,
                    'euclidean_distance': euc_dist,
                    'features': ref['features']
                })
        
        # Cosine similarity'ye göre sırala (yüksekten düşüğe)
        similarities = sorted(similarities, key=lambda x: x['cosine_similarity'], reverse=True)
        
        return similarities[:top_k]
    
    def get_pca_visualization_data(self, target_features, target_class):
        """PCA ile 2D görselleştirme verisi hazırla"""
        if len(self.reference_database) < 2:
            return None, None, None, None
            
        # Aynı sınıftan sesleri al
        same_class_features = []
        same_class_names = []
        
        for ref in self.reference_database:
            if ref['class'] == target_class:
                same_class_features.append(ref['features'])
                same_class_names.append(ref['filename'])
        
        if len(same_class_features) < 2:
            return None, None, None, None
            
        # Target ses ile birleştir
        all_features = same_class_features + [target_features]
        all_names = same_class_names + ['Yüklenen Ses']
        
        # PCA uygula
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(all_features)
        
        return pca_features, all_names, pca.explained_variance_ratio_, pca
    
    def classify_multiple_files(self, audio_files):
        """Birden fazla ses dosyasını sınıflandır"""
        results = []
        
        for audio_file in audio_files:
            result = self.predict_single(audio_file)
            if result[0] is not None:
                predicted_class, confidence, features = result
                
                # Veritabanına ekle
                self.add_to_database(audio_file, predicted_class, features)
                
                results.append({
                    'filename': audio_file.name if hasattr(audio_file, 'name') else str(audio_file),
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'features': features
                })
        
        return results
    
    def get_database_summary(self):
        """Veritabanı özetini döndür"""
        if not self.reference_database:
            return {}
            
        df = pd.DataFrame(self.reference_database)
        summary = df['class'].value_counts().to_dict()
        return summary 