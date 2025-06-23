#!/usr/bin/env python3
"""
Model ve pickle dosyaları düzeltme scripti
"""

import os
import sys
import warnings
import numpy as np

def recreate_model_h5():
    """
    Basit bir model oluştur ve kaydet
    """
    print("🔧 Yeni uyumlu model oluşturuluyor...")
    
    try:
        import tensorflow as tf
        
        # Basit bir model oluştur
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(42,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(7, activation='softmax')  # 7 sınıf
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Model kaydet
        model.save('my_enhanced_audio_model.h5')
        print("✅ Yeni model kaydedildi: my_enhanced_audio_model.h5")
        
        return True
        
    except Exception as e:
        print(f"❌ Model oluşturma hatası: {e}")
        return False

def recreate_scaler():
    """
    Scaler dosyasını yeniden oluştur
    """
    print("🔧 Yeni scaler oluşturuluyor...")
    
    try:
        from sklearn.preprocessing import StandardScaler
        import joblib
        
        # Dummy veri ile scaler oluştur
        dummy_data = np.random.randn(100, 42)  # 42 özellik
        
        scaler = StandardScaler()
        scaler.fit(dummy_data)
        
        # Scaler kaydet
        joblib.dump(scaler, 'scaler.pkl')
        print("✅ Yeni scaler kaydedildi: scaler.pkl")
        
        return True
        
    except Exception as e:
        print(f"❌ Scaler oluşturma hatası: {e}")
        return False

def recreate_label_encoder():
    """
    Label encoder dosyasını yeniden oluştur
    """
    print("🔧 Yeni label encoder oluşturuluyor...")
    
    try:
        from sklearn.preprocessing import LabelEncoder
        import joblib
        
        # 7 sınıf için label encoder oluştur
        classes = ['Bass', 'Clap', 'Cymbal', 'Hat', 'Kick', 'Rims', 'Snare']
        
        label_encoder = LabelEncoder()
        label_encoder.fit(classes)
        
        # Label encoder kaydet
        joblib.dump(label_encoder, 'label_encoder.pkl')
        print("✅ Yeni label encoder kaydedildi: label_encoder.pkl")
        print(f"📊 Sınıflar: {list(label_encoder.classes_)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Label encoder oluşturma hatası: {e}")
        return False

def backup_existing_files():
    """
    Mevcut dosyaları yedekle
    """
    files_to_backup = [
        'my_enhanced_audio_model.h5',
        'scaler.pkl', 
        'label_encoder.pkl'
    ]
    
    for file in files_to_backup:
        if os.path.exists(file):
            backup_name = f"{file}.backup"
            os.rename(file, backup_name)
            print(f"📦 {file} -> {backup_name} (yedeklendi)")

def main():
    """Ana düzeltme fonksiyonu"""
    print("🔧 Model ve Pickle Dosyaları Düzeltme Scripti")
    print("=" * 60)
    
    print(f"🐍 Python versiyonu: {sys.version}")
    print(f"📁 Çalışma dizini: {os.getcwd()}")
    
    # Mevcut dosyaları yedekle
    print("\n📦 DOSYA YEDEKLEME")
    print("=" * 30)
    backup_existing_files()
    
    # Yeni dosyalar oluştur
    print("\n🔧 YENİ DOSYALAR OLUŞTURULUYOR")
    print("=" * 40)
    
    success_count = 0
    
    if recreate_model_h5():
        success_count += 1
        
    if recreate_scaler():
        success_count += 1
        
    if recreate_label_encoder():
        success_count += 1
    
    print("\n" + "=" * 60)
    print("📊 SONUÇLAR")
    print("=" * 60)
    
    if success_count == 3:
        print("✅ Tüm dosyalar başarıyla oluşturuldu!")
        print("🚀 Artık uygulamayı test edebilirsiniz:")
        print("   streamlit run app.py")
    else:
        print(f"⚠️ {success_count}/3 dosya oluşturuldu")
        print("❌ Bazı dosyalar oluşturulamadı")
    
    print("\n💡 NOT:")
    print("Bu script basit/dummy verilerle dosyaları yeniden oluşturdu.")
    print("Gerçek eğitilmiş model için orijinal eğitim verilerinizi kullanmanız gerekir.")

if __name__ == "__main__":
    main() 