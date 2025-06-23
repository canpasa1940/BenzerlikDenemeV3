#!/usr/bin/env python3
"""
Bozuk pkl dosyalarını düzeltici script
CSV'den yeni scaler ve label encoder oluşturur
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import pickle
import os

def create_pkl_files_from_csv():
    """CSV'den yeni pkl dosyaları oluştur"""
    
    # CSV dosyasını bul
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'feature' in f.lower()]
    
    if not csv_files:
        print("❌ CSV dosyası bulunamadı. one_shot_features_clean_V4.csv var mı?")
        return False
    
    csv_file = csv_files[0]
    print(f"📊 CSV dosyası bulundu: {csv_file}")
    
    try:
        # CSV'yi yükle
        df = pd.read_csv(csv_file)
        print(f"✅ CSV yüklendi: {len(df)} satır, {len(df.columns)} sütun")
        
        # Özellikler ve etiketler
        feature_columns = [col for col in df.columns if col not in ['label', 'filename']]
        X = df[feature_columns]
        y = df['label']
        
        print(f"🔢 Özellik sayısı: {len(feature_columns)}")
        print(f"🏷️ Sınıflar: {list(y.unique())}")
        
        # StandardScaler oluştur ve fit et
        scaler = StandardScaler()
        scaler.fit(X)
        print("✅ StandardScaler oluşturuldu")
        
        # LabelEncoder oluştur ve fit et  
        label_encoder = LabelEncoder()
        label_encoder.fit(y)
        print("✅ LabelEncoder oluşturuldu")
        
        # Eski dosyaları yedekle
        if os.path.exists('scaler.pkl'):
            os.rename('scaler.pkl', 'scaler_old.pkl')
            print("📦 Eski scaler.pkl yedeklendi")
            
        if os.path.exists('label_encoder.pkl'):
            os.rename('label_encoder.pkl', 'label_encoder_old.pkl')
            print("📦 Eski label_encoder.pkl yedeklendi")
        
        # Yeni dosyaları kaydet - hem joblib hem pickle dene
        try:
            joblib.dump(scaler, 'scaler.pkl')
            print("✅ scaler.pkl (joblib) kaydedildi")
        except:
            with open('scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            print("✅ scaler.pkl (pickle) kaydedildi")
            
        try:
            joblib.dump(label_encoder, 'label_encoder.pkl')
            print("✅ label_encoder.pkl (joblib) kaydedildi")
        except:
            with open('label_encoder.pkl', 'wb') as f:
                pickle.dump(label_encoder, f)
            print("✅ label_encoder.pkl (pickle) kaydedildi")
        
        return True
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        return False

def test_pkl_files():
    """Yeni pkl dosyalarını test et"""
    try:
        print("\n🔍 Test ediliyor...")
        
        # joblib ile dene
        try:
            scaler = joblib.load('scaler.pkl')
            label_encoder = joblib.load('label_encoder.pkl')
            print("✅ joblib ile yükleme başarılı")
        except:
            # pickle ile dene
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            with open('label_encoder.pkl', 'rb') as f:
                label_encoder = pickle.load(f)
            print("✅ pickle ile yükleme başarılı")
        
        print(f"📊 Scaler özellik sayısı: {scaler.n_features_in_}")
        print(f"🏷️ Label encoder sınıfları: {list(label_encoder.classes_)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test hatası: {e}")
        return False

def main():
    print("🔧 PKL Dosya Düzelticisi")
    print("=" * 40)
    
    # CSV'den yeni pkl dosyaları oluştur
    if create_pkl_files_from_csv():
        # Test et
        if test_pkl_files():
            print("\n🎉 BAŞARILI! Yeni pkl dosyaları hazır.")
            print("📱 Artık streamlit run app.py çalıştırabilirsiniz!")
        else:
            print("\n❌ Test başarısız")
    else:
        print("\n❌ PKL dosyaları oluşturulamadı")

if __name__ == "__main__":
    main() 