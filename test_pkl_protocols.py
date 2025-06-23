#!/usr/bin/env python3
"""
PKL dosyalarını farklı yöntemlerle açmaya çalışır
"""

import pickle
import joblib
import sys
import warnings

def try_open_pickle(filename):
    """Farklı yöntemlerle pickle dosyasını açmayı dene"""
    print(f"\n🔍 {filename} dosyasını test ediliyor...")
    
    methods = [
        ("joblib.load", lambda f: joblib.load(f)),
        ("pickle.load (rb)", lambda f: pickle.load(open(f, 'rb'))),
        ("pickle.load (rb, protocol=2)", lambda f: pickle.load(open(f, 'rb'), encoding='latin1')),
        ("pickle.load (rb, encoding=bytes)", lambda f: pickle.load(open(f, 'rb'), encoding='bytes')),
        ("pickle.load (rb, fix_imports=False)", lambda f: pickle.load(open(f, 'rb'), fix_imports=False)),
    ]
    
    for method_name, method_func in methods:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                obj = method_func(filename)
            print(f"✅ {method_name}: BAŞARILI!")
            
            # Obje tipini ve özelliklerini göster
            print(f"   📊 Tip: {type(obj)}")
            if hasattr(obj, 'classes_'):
                print(f"   🏷️ Sınıflar: {list(obj.classes_)}")
            if hasattr(obj, 'n_features_in_'):
                print(f"   🔢 Özellik sayısı: {obj.n_features_in_}")
            if hasattr(obj, 'mean_') and hasattr(obj, 'scale_'):
                print(f"   📈 Scaler: mean shape={obj.mean_.shape}, scale shape={obj.scale_.shape}")
            
            return obj, method_name
            
        except Exception as e:
            print(f"❌ {method_name}: {e}")
    
    return None, None

def fix_audio_classifier():
    """Audio classifier'ı bozuk pkl dosyalarıyla çalışacak şekilde düzenle"""
    
    # Scaler'ı test et
    scaler, scaler_method = try_open_pickle('scaler.pkl')
    if not scaler:
        print("❌ Scaler açılamadı!")
        return False
    
    # Label encoder'ı test et
    label_encoder, encoder_method = try_open_pickle('label_encoder.pkl')
    if not label_encoder:
        print("❌ Label encoder açılamadı!")
        return False
    
    print(f"\n🎉 ÇÖZÜM BULUNDU!")
    print(f"📦 Scaler: {scaler_method}")
    print(f"📦 Label Encoder: {encoder_method}")
    
    # audio_classifier.py'ı güncelle
    update_code = f'''
    # OTOMATIK OLUŞTURULAN ÇÖZÜM
    def load_model_and_preprocessors(self):
        """Model ve ön işleyicileri yükle - Düzeltilmiş versiyon"""
        try:
            # TensorFlow modelini yükle
            from tensorflow import keras
            self.model = keras.models.load_model(self.model_path)
            print(f"✅ Model yüklendi: {{self.model_path}}")
            
            # Scaler'ı yükle - Çalışan yöntem: {scaler_method}
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                '''
    
    if 'joblib' in scaler_method:
        update_code += 'self.scaler = joblib.load(self.scaler_path)'
    elif 'latin1' in scaler_method:
        update_code += '''with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f, encoding='latin1')'''
    elif 'bytes' in scaler_method:
        update_code += '''with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f, encoding='bytes')'''
    elif 'fix_imports=False' in scaler_method:
        update_code += '''with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f, fix_imports=False)'''
    else:
        update_code += '''with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)'''
    
    update_code += f'''
            print(f"✅ Scaler yüklendi: {{self.scaler_path}}")
            
            # Label encoder'ı yükle - Çalışan yöntem: {encoder_method}
            '''
    
    if 'joblib' in encoder_method:
        update_code += 'self.label_encoder = joblib.load(self.label_encoder_path)'
    elif 'latin1' in encoder_method:
        update_code += '''with open(self.label_encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f, encoding='latin1')'''
    elif 'bytes' in encoder_method:
        update_code += '''with open(self.label_encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f, encoding='bytes')'''
    elif 'fix_imports=False' in encoder_method:
        update_code += '''with open(self.label_encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f, fix_imports=False)'''
    else:
        update_code += '''with open(self.label_encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)'''
    
    update_code += '''
            print(f"✅ Label encoder yüklendi: {self.label_encoder_path}")
            
            # Sınıfları al
            self.classes = self.label_encoder.classes_
            print(f"📊 Mevcut sınıflar: {list(self.classes)}")
            
        except Exception as e:
            print(f"❌ Model yükleme hatası: {e}")
            raise
    '''
    
    print("\n📝 DÜZELTME KODU:")
    print("="*60)
    print(update_code)
    print("="*60)
    
    return True

def main():
    print("🔧 PKL Protokol Test Edici")
    print("=" * 40)
    
    fix_audio_classifier()

if __name__ == "__main__":
    main() 