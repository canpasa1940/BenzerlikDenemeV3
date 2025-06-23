import numpy as np
import pandas as pd
import pickle
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from feature_extractor import extract_features_from_file
import os

class AudioClassifier:
    def __init__(self, model_path="my_enhanced_audio_model.h5", 
                 scaler_path="scaler.pkl", 
                 label_encoder_path="label_encoder.pkl"):
        """
        Ses sınıflandırıcı ve benzerlik analizi sınıfı
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.label_encoder_path = label_encoder_path
        
        # Model ve preprocessor'ları yükle
        self.load_model_and_preprocessors()
        
        # Özellik sırası (notebook'taki sırayla aynı olmalı)
        self.feature_columns = self._get_feature_columns()
        
        # Referans veri seti için boş liste
        self.reference_data = []
        
    def _get_feature_columns(self):
        """Özellik sütunlarının sırasını belirle"""
        columns = []
        # MFCC özellikleri
        for i in range(20):
            columns.append(f"mfcc{i+1:02d}")
        
        # Diğer özellikler
        columns.extend([
            "rms_mean", "rms_std", "zcr_mean", "centroid_mean",
            "bandwidth_mean", "rolloff_mean", "flatness_mean", "flux_mean"
        ])
        
        # Contrast band özellikleri
        for i in range(7):
            columns.append(f"contrast_b{i+1}")
            
        # Onset ve attack özellikleri
        columns.extend([
            "onset_mean", "onset_std", "onset_max", "onset_sum",
            "attack_time", "attack_slope", "hpi_ratio"
        ])
        
        return columns
    
    def load_model_and_preprocessors(self):
        """Model ve ön işleyicileri yükle"""
        try:
            # TensorFlow modelini yükle
            from tensorflow import keras
            self.model = keras.models.load_model(self.model_path)
            print(f"✅ Model yüklendi: {self.model_path}")
            
            # Scaler'ı yükle
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scaler = joblib.load(self.scaler_path)
            print(f"✅ Scaler yüklendi: {self.scaler_path}")
            
            # Label encoder'ı yükle
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.label_encoder = joblib.load(self.label_encoder_path)
            print(f"✅ Label encoder yüklendi: {self.label_encoder_path}")
            
            # Sınıfları al
            self.classes = self.label_encoder.classes_
            print(f"📊 Mevcut sınıflar: {list(self.classes)}")
            
        except Exception as e:
            print(f"❌ Model yükleme hatası: {e}")
            raise
    
    def predict_audio(self, file_path):
        """
        Ses dosyasını sınıflandır
        """
        # Özellikleri çıkar
        features, error = extract_features_from_file(file_path)
        if error:
            return None, f"Özellik çıkarma hatası: {error}"
        
        try:
            # DataFrame'e çevir ve sırala
            feature_df = pd.DataFrame([features])
            feature_df = feature_df[self.feature_columns]
            
            # Ölçeklendir
            features_scaled = self.scaler.transform(feature_df)
            
            # Tahmin yap
            predictions = self.model.predict(features_scaled, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            confidence = float(predictions[0][predicted_class_idx])
            
            # Tüm sınıf olasılıklarını al
            class_probabilities = {}
            for i, class_name in enumerate(self.classes):
                class_probabilities[class_name] = float(predictions[0][i])
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'class_probabilities': class_probabilities,
                'features': features,
                'features_scaled': features_scaled.flatten()
            }, None
            
        except Exception as e:
            return None, f"Sınıflandırma hatası: {e}"
    
    def add_to_reference(self, file_path, prediction_result, original_filename=None):
        """
        Referans veri setine yeni sample ekle (duplicate kontrolü ile)
        """
        if prediction_result:
            # Orijinal dosya adını kullan, yoksa path'ten al
            display_name = original_filename if original_filename else os.path.basename(file_path)
            
            # Aynı dosya zaten var mı kontrol et
            existing = any(ref['display_name'] == display_name for ref in self.reference_data)
            if existing:
                print(f"⚠️ Dosya zaten referans veri setinde: {display_name}")
                return
            
            print(f"✅ Referans veri setine ekleniyor: {display_name}")
            self.reference_data.append({
                'file_path': file_path,
                'display_name': display_name,
                'predicted_class': prediction_result['predicted_class'],
                'confidence': prediction_result['confidence'],
                'features': prediction_result['features'],
                'features_scaled': prediction_result['features_scaled']
            })
    
    def find_similar_sounds(self, target_features_scaled, target_class=None, top_k=5, exclude_self=None):
        """
        Benzer sesleri bul
        """
        if not self.reference_data:
            return []
        
        similarities = []
        
        for ref_data in self.reference_data:
            # Kendisiyle karşılaştırmayı atla
            if exclude_self and ref_data['display_name'] == exclude_self:
                continue
                
            # Sadece aynı sınıftan örnekleri karşılaştır (isteğe bağlı)
            if target_class and ref_data['predicted_class'] != target_class:
                continue
            
            # Cosine similarity hesapla
            cos_sim = cosine_similarity(
                [target_features_scaled],
                [ref_data['features_scaled']]
            )[0][0]
            
            # Euclidean distance hesapla
            eucl_dist = euclidean_distances(
                [target_features_scaled],
                [ref_data['features_scaled']]
            )[0][0]
            
            similarities.append({
                'file_path': ref_data['file_path'],
                'display_name': ref_data['display_name'],
                'predicted_class': ref_data['predicted_class'],
                'confidence': ref_data['confidence'],
                'cosine_similarity': cos_sim,
                'euclidean_distance': eucl_dist,
                'features_scaled': ref_data['features_scaled']
            })
        
        # Cosine similarity'ye göre sırala (yüksekten düşüğe)
        similarities.sort(key=lambda x: x['cosine_similarity'], reverse=True)
        
        return similarities[:top_k]
    
    def visualize_similarities(self, target_features, similar_sounds):
        """
        Benzerlik görselleştirmesi
        """
        if not similar_sounds:
            return None
        
        # PCA ile 2D'ye indirge
        all_features = [target_features]
        labels = ['Hedef Ses']
        
        for sim in similar_sounds:
            all_features.append(sim['features_scaled'])
            labels.append(f"{os.path.basename(sim['file_path'])} ({sim['predicted_class']})")
        
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(all_features)
        
        # Plotly ile interaktif görselleştirme
        fig = go.Figure()
        
        # Hedef ses
        fig.add_trace(go.Scatter(
            x=[features_2d[0, 0]],
            y=[features_2d[0, 1]],
            mode='markers',
            name='Hedef Ses',
            marker=dict(size=15, color='red', symbol='star'),
            text=['Hedef Ses'],
            textposition="top center"
        ))
        
        # Benzer sesler
        for i, sim in enumerate(similar_sounds, 1):
            fig.add_trace(go.Scatter(
                x=[features_2d[i, 0]],
                y=[features_2d[i, 1]],
                mode='markers',
                name=f"Benzer #{i}",
                marker=dict(size=10, color='blue'),
                text=[f"{os.path.basename(sim['file_path'])}<br>Sınıf: {sim['predicted_class']}<br>Benzerlik: {sim['cosine_similarity']:.3f}"],
                textposition="top center"
            ))
        
        fig.update_layout(
            title="Ses Benzerlik Analizi (PCA 2D Projeksiyon)",
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} varyans)",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} varyans)",
            showlegend=True
        )
        
        return fig
    
    def get_class_statistics(self):
        """
        Referans veri setindeki sınıf istatistikleri
        """
        if not self.reference_data:
            return {}
        
        class_counts = {}
        class_confidences = {}
        
        for data in self.reference_data:
            class_name = data['predicted_class']
            confidence = data['confidence']
            
            if class_name not in class_counts:
                class_counts[class_name] = 0
                class_confidences[class_name] = []
            
            class_counts[class_name] += 1
            class_confidences[class_name].append(confidence)
        
        # Ortalama güven skorları
        class_avg_confidence = {}
        for class_name, confidences in class_confidences.items():
            class_avg_confidence[class_name] = np.mean(confidences)
        
        return {
            'class_counts': class_counts,
            'class_avg_confidence': class_avg_confidence,
            'total_samples': len(self.reference_data)
        } 