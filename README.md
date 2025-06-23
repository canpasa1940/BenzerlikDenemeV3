# 🎵 One-Shot Ses Sınıflandırıcı

Bu uygulama, ses dosyalarını sınıflandırmak ve benzerlik analizi yapmak için geliştirilmiş bir web uygulamasıdır.

## 🌟 Özellikler

- **Ses Sınıflandırma**: WAV dosyalarını 7 farklı kategoriye ayırır (Bass, Clap, Cymbal, Hat, Kick, Rims, Snare)
- **Toplu İşlem**: Birden fazla ses dosyasını aynı anda işler
- **Benzerlik Analizi**: Yüklenen sesler arasında benzerlik karşılaştırması yapar
- **Görselleştirme**: Dalga formları, güven skorları ve benzerlik haritaları
- **Gerçek Zamanlı**: Anında sonuç verir

## 🎯 Nasıl Kullanılır

1. **Ses Dosyası Yükle**: Sol panelden WAV dosyalarınızı seçin
2. **Sınıflandır**: Otomatik olarak ses sınıflandırması yapılır
3. **Sonuçları İncele**: Sağ panelde sonuçları ve grafikleri görün
4. **Benzerlik Analizi**: Birden fazla ses yükleyerek karşılaştırma yapın

## 📊 Desteklenen Ses Türleri

- **Bass**: Bas davul sesleri
- **Clap**: El çırpma sesleri
- **Cymbal**: Zil sesleri
- **Hat**: Hi-hat sesleri
- **Kick**: Kick davul sesleri
- **Rims**: Rim shot sesleri
- **Snare**: Snare davul sesleri

## 🔧 Teknik Detaylar

- **Model**: TensorFlow/Keras tabanlı derin öğrenme modeli
- **Özellik Çıkarımı**: MFCC, RMS, ZCR, spektral özellikler
- **Benzerlik Metriği**: Cosine similarity ve Euclidean distance
- **Framework**: Streamlit web framework

## 📝 Lisans

Bu proje eğitim amaçlı geliştirilmiştir. 