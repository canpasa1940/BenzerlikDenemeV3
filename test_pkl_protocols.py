#!/usr/bin/env python3
"""
Test script for pickle file protocols and TensorFlow model loading
"""

import os
import sys
import warnings

def test_tensorflow_model():
    """Test TensorFlow model loading with different methods"""
    model_path = "my_enhanced_audio_model.h5"
    
    if not os.path.exists(model_path):
        print(f"❌ Model dosyası bulunamadı: {model_path}")
        return False
        
    print(f"🔍 TensorFlow model yükleme testleri...")
    print(f"📁 Model dosyası: {model_path} ({os.path.getsize(model_path)} bytes)")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} yüklendi")
        
        # Method 1: Normal load
        print("\n🔧 Yöntem 1: Normal yükleme")
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                model = tf.keras.models.load_model(model_path)
            print("✅ Normal yükleme başarılı")
            print(f"📊 Model summary: {model.input_shape} -> {model.output_shape}")
            return True
        except Exception as e:
            print(f"❌ Normal yükleme hatası: {e}")
            
        # Method 2: Load without compilation
        print("\n🔧 Yöntem 2: Compile=False ile yükleme")
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                model = tf.keras.models.load_model(model_path, compile=False)
            print("✅ Compile=False yükleme başarılı")
            print(f"📊 Model summary: {model.input_shape} -> {model.output_shape}")
            return True
        except Exception as e:
            print(f"❌ Compile=False yükleme hatası: {e}")
            
        # Method 3: Load with custom objects
        print("\n🔧 Yöntem 3: Custom objects ile yükleme")
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                model = tf.keras.models.load_model(
                    model_path, 
                    custom_objects=None,
                    compile=False
                )
            print("✅ Custom objects yükleme başarılı")
            print(f"📊 Model summary: {model.input_shape} -> {model.output_shape}")
            return True
        except Exception as e:
            print(f"❌ Custom objects yükleme hatası: {e}")
            
        return False
        
    except ImportError as e:
        print(f"❌ TensorFlow import hatası: {e}")
        return False

def test_pickle_files():
    """Test pickle files with different protocols"""
    pickle_files = ["scaler.pkl", "label_encoder.pkl"]
    
    for pkl_file in pickle_files:
        if not os.path.exists(pkl_file):
            print(f"❌ Pickle dosyası bulunamadı: {pkl_file}")
            continue
            
        print(f"\n🔍 {pkl_file} yükleme testleri...")
        print(f"📁 Dosya boyutu: {os.path.getsize(pkl_file)} bytes")
        
        # Method 1: joblib
        try:
            import joblib
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                obj = joblib.load(pkl_file)
            print(f"✅ joblib.load başarılı - {type(obj).__name__}")
        except Exception as e:
            print(f"❌ joblib.load hatası: {e}")
            
        # Method 2: pickle
        try:
            import pickle
            with open(pkl_file, 'rb') as f:
                obj = pickle.load(f)
            print(f"✅ pickle.load başarılı - {type(obj).__name__}")
        except Exception as e:
            print(f"❌ pickle.load hatası: {e}")

def main():
    """Ana test fonksiyonu"""
    print("🧪 Ses Analiz Uygulaması - Model ve Pickle Test Scripti")
    print("=" * 60)
    
    print(f"🐍 Python versiyonu: {sys.version}")
    print(f"📁 Çalışma dizini: {os.getcwd()}")
    
    # TensorFlow model testi
    print("\n" + "="*60)
    print("🤖 TENSORFLOW MODEL TESTLERİ")
    print("="*60)
    tf_success = test_tensorflow_model()
    
    # Pickle dosya testleri
    print("\n" + "="*60)
    print("🥒 PICKLE DOSYA TESTLERİ") 
    print("="*60)
    test_pickle_files()
    
    print("\n" + "="*60)
    print("📊 TEST SONUÇLARI")
    print("="*60)
    
    if tf_success:
        print("✅ TensorFlow model yükleme: BAŞARILI")
    else:
        print("❌ TensorFlow model yükleme: BAŞARISIZ")
        print("💡 Çözüm önerileri:")
        print("   - TensorFlow versiyonunu kontrol edin")
        print("   - Model dosyası corrupt olabilir")
        print("   - Farklı TensorFlow versiyonuyla model yeniden kaydedilmeli")

if __name__ == "__main__":
    main() 