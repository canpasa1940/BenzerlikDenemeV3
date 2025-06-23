import streamlit as st
import os
import tempfile
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from audio_classifier import AudioClassifier
import librosa
import io
import matplotlib.pyplot as plt

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="🎵 One-Shot Ses Sınıflandırıcı",
    page_icon="🎵",
    layout="wide"
)

# CSS stili
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .similarity-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Ana başlık
st.markdown("""
<div class="main-header">
    <h1>🎵 One-Shot Ses Sınıflandırıcı ve Benzerlik Analizi</h1>
    <p>Ses dosyalarınızı yükleyin, sınıflandırın ve benzer sesler bulun!</p>
</div>
""", unsafe_allow_html=True)

# Session state'i başlat
if 'classifier' not in st.session_state:
    try:
        with st.spinner('🤖 Model yükleniyor...'):
            st.session_state.classifier = AudioClassifier()
        st.success("✅ Model başarıyla yüklendi!")
    except Exception as e:
        st.error(f"❌ Model yükleme hatası: {e}")
        st.stop()

if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# Sidebar - Konfigürasyon
with st.sidebar:
    st.header("⚙️ Ayarlar")
    
    # Sistem kontrolü
    if st.button("🔄 Tüm Verileri Temizle", type="secondary"):
        # Tüm session state'i temizle
        for key in list(st.session_state.keys()):
            if key != 'classifier':  # Classifier'ı koru
                del st.session_state[key]
        # Classifier'daki referans verileri temizle
        st.session_state.classifier.reference_data = []
        st.success("✅ Tüm veriler temizlendi!")
        st.rerun()
    
    # Benzerlik analizi ayarları
    st.subheader("🔍 Benzerlik Analizi")
    similarity_threshold = st.slider("Benzerlik Eşiği", 0.0, 1.0, 0.7, 0.05)
    top_k_similar = st.slider("Gösterilecek Benzer Ses Sayısı", 1, 10, 5)
    same_class_only = st.checkbox("Sadece Aynı Sınıftan Benzer Sesler", value=True)
    
    # Görselleştirme ayarları
    st.subheader("📊 Görselleştirme")
    show_confidence_chart = st.checkbox("Güven Skorları Grafiği", value=True)
    show_similarity_plot = st.checkbox("Benzerlik Haritası", value=True)
    
    # İstatistikler
    st.subheader("📈 İstatistikler")
    st.write(f"**Referans Veri Seti:** {len(st.session_state.classifier.reference_data)} ses")
    
    if st.session_state.classifier.reference_data:
        stats = st.session_state.classifier.get_class_statistics()
        st.write("**Sınıf Dağılımı:**")
        for class_name, count in stats['class_counts'].items():
            st.write(f"• {class_name}: {count}")
    else:
        st.write("*Henüz referans ses yok*")

# Ana içerik alanı
col1, col2 = st.columns([2, 3])

with col1:
    st.header("📁 Ses Dosyası Yükle")
    
    # Dosya yükleme - Multiple files
    uploaded_files = st.file_uploader(
        "WAV dosyalarını seçin:",
        type=['wav'],
        accept_multiple_files=True,
        help="Birden fazla WAV dosyası seçebilirsiniz. Toplu işlem yapılacak."
    )
    
    if uploaded_files:
        st.write(f"📊 **{len(uploaded_files)} dosya seçildi**")
        
        # Batch processing seçenekleri
        col_a, col_b = st.columns(2)
        with col_a:
            auto_process = st.checkbox("🚀 Otomatik işle", value=True)
        with col_b:
            show_waveforms = st.checkbox("📈 Dalga formları göster", value=False)
        
        # Batch processing butonu
        if st.button("🎯 Tüm Dosyaları Sınıflandır", type="primary") or auto_process:
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Sonuçları sakla
            batch_results = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                # Progress update
                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"🔍 İşleniyor: {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
                
                # Geçici dosya oluştur
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    temp_path = tmp_file.name
                
                # Ses dosyasını oynat (sadece ilk dosya)
                if i == 0:
                    st.audio(uploaded_file)
                
                # Dalga formu göster (isteğe bağlı)
                if show_waveforms and i < 3:  # İlk 3 dosya için
                    try:
                        signal, sr = librosa.load(temp_path, sr=22050)
                        fig, ax = plt.subplots(figsize=(10, 2))
                        librosa.display.waveshow(signal, sr=sr, ax=ax)
                        ax.set_title(f"Dalga Formu - {uploaded_file.name}")
                        ax.set_xlabel("Zaman (s)")
                        ax.set_ylabel("Genlik")
                        st.pyplot(fig)
                        plt.close()
                    except Exception as e:
                        st.warning(f"Dalga formu gösterilemedi ({uploaded_file.name}): {e}")
                
                # Sınıflandırma yap
                result, error = st.session_state.classifier.predict_audio(temp_path)
                
                if error:
                    st.error(f"❌ Hata ({uploaded_file.name}): {error}")
                else:
                    # Referans veri setine ekle (orijinal dosya adıyla)
                    st.session_state.classifier.add_to_reference(temp_path, result, uploaded_file.name)
                    
                    # Ses dosyasını session state'te sakla
                    if 'audio_files' not in st.session_state:
                        st.session_state.audio_files = {}
                    st.session_state.audio_files[uploaded_file.name] = uploaded_file.getvalue()
                    
                    # Batch results'a ekle
                    batch_results.append({
                        'file_name': uploaded_file.name,
                        'predicted_class': result['predicted_class'],
                        'confidence': result['confidence'],
                        'result': result
                    })
                
                # Geçici dosyayı temizle
                try:
                    os.unlink(temp_path)
                except:
                    pass
            
            # İşlem tamamlandı
            progress_bar.progress(1.0)
            status_text.text("✅ Tüm dosyalar işlendi!")
            
            # Son result'ı kaydet (en son işlenen dosya)
            if batch_results:
                st.session_state.last_result = {
                    'file_name': batch_results[-1]['file_name'],
                    'result': batch_results[-1]['result']
                }
                st.session_state.batch_results = batch_results
                
            st.success(f"🎉 {len(batch_results)} dosya başarıyla analiz edildi!")
    
    # Tek dosya benzerlik analizi
    st.markdown("---")
    st.subheader("🎯 Tek Dosya Benzerlik Analizi")
    st.write("Mevcut veri setine karşı tek bir dosyayı analiz edin:")
    
    single_file = st.file_uploader(
        "Benzerlik analizi için tek WAV dosyası:",
        type=['wav'],
        accept_multiple_files=False,
        key="single_file_upload",
        help="Bu dosya mevcut veri setiyle karşılaştırılacak."
    )
    
    if single_file and len(st.session_state.classifier.reference_data) > 0:
        if st.button("🔍 Benzerlik Analizi Yap", type="secondary"):
            # Geçici dosya oluştur
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(single_file.read())
                temp_path = tmp_file.name
            
            # Sınıflandırma yap
            result, error = st.session_state.classifier.predict_audio(temp_path)
            
            if error:
                st.error(f"❌ Hata: {error}")
            else:
                # Session state'e kaydet (tek dosya analizi için - veri setine ekleme!)
                st.session_state.single_file_result = {
                    'file_name': single_file.name,
                    'result': result
                }
                
                # Ses dosyasını session state'te sakla
                if 'audio_files' not in st.session_state:
                    st.session_state.audio_files = {}
                st.session_state.audio_files[single_file.name] = single_file.getvalue()
                
                st.success(f"✅ {single_file.name} analiz edildi!")
                st.info(f"📊 Veri setinde {len(st.session_state.classifier.reference_data)} referans ses var.")
                

            
            # Geçici dosyayı temizle
            try:
                os.unlink(temp_path)
            except:
                pass
    elif single_file and len(st.session_state.classifier.reference_data) == 0:
        st.warning("⚠️ Önce toplu analiz yaparak referans veri seti oluşturun.")

with col2:
    st.header("📊 Sonuçlar")
    
    # Batch sonuçları varsa önce onları göster
    if 'batch_results' in st.session_state and st.session_state.batch_results:
        st.subheader("📋 Toplu İşlem Sonuçları")
        
        # Özet tablo
        batch_df = pd.DataFrame(st.session_state.batch_results)
        batch_df = batch_df[['file_name', 'predicted_class', 'confidence']]
        batch_df['confidence'] = batch_df['confidence'].apply(lambda x: f"{x:.1%}")
        batch_df.columns = ['Dosya Adı', 'Sınıf', 'Güven']
        
        st.table(batch_df)
        
        # Benzerlik analizi için dosya seçimi
        st.subheader("🔍 Benzerlik Analizi İçin Dosya Seçin")
        selected_file = st.selectbox(
            "Hangi dosyanın benzerlik analizini yapmak istiyorsunuz?",
            options=[r['file_name'] for r in st.session_state.batch_results],
            index=len(st.session_state.batch_results)-1  # Son dosya varsayılan
        )
        
        # Seçilen dosyanın sonucunu session state'e kaydet
        if selected_file:
            selected_result = next(r for r in st.session_state.batch_results if r['file_name'] == selected_file)
            st.session_state.last_result = {
                'file_name': selected_result['file_name'],
                'result': selected_result['result']
            }
            
            # Seçilen dosya bilgisi
            st.info(f"🎯 Seçilen dosya: **{selected_file}** (Sınıf: **{selected_result['predicted_class']}**, Güven: **{selected_result['confidence']:.1%}**)")
        
        # Sınıf dağılımı
        class_counts = pd.DataFrame(st.session_state.batch_results)['predicted_class'].value_counts()
        
        col_x, col_y = st.columns(2)
        with col_x:
            st.write("**Sınıf Dağılımı:**")
            fig_pie = px.pie(
                values=class_counts.values, 
                names=class_counts.index,
                title="Sınıf Dağılımı"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col_y:
            st.write("**Güven Skorları:**")
            conf_data = [r['confidence'] for r in st.session_state.batch_results]
            fig_hist = px.histogram(
                x=conf_data, 
                nbins=10,
                title="Güven Skoru Dağılımı"
            )
            fig_hist.update_xaxes(title="Güven Skoru")
            fig_hist.update_yaxes(title="Dosya Sayısı")
            st.plotly_chart(fig_hist, use_container_width=True)
    
    # Son analiz sonuçlarını göster (toplu analiz veya tek dosya)
    result_to_show = None
    if 'single_file_result' in st.session_state:
        result_to_show = st.session_state.single_file_result
        col_info, col_clear = st.columns([3, 1])
        with col_info:
            st.info("🎯 **Tek Dosya Analizi Sonucu**")
        with col_clear:
            if st.button("🗑️ Temizle", key="clear_single"):
                del st.session_state.single_file_result
                st.rerun()
    elif 'last_result' in st.session_state:
        result_to_show = st.session_state.last_result
        st.info("📋 **Toplu Analiz - Seçilen Dosya**")
    
    if result_to_show:
        result_data = result_to_show
        result = result_data['result']
        
        # Sınıflandırma sonuçları
        st.markdown(f"""
        <div class="result-container">
            <h3>🎯 Sınıflandırma Sonucu</h3>
            <h2 style="color: #667eea;">{result['predicted_class']}</h2>
            <p><strong>Güven Skoru:</strong> {result['confidence']:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Tüm sınıf olasılıkları
        if show_confidence_chart:
            st.subheader("📈 Tüm Sınıf Olasılıkları")
            prob_df = pd.DataFrame(list(result['class_probabilities'].items()), 
                                 columns=['Sınıf', 'Olasılık'])
            prob_df = prob_df.sort_values('Olasılık', ascending=True)
            
            fig = px.bar(prob_df, x='Olasılık', y='Sınıf', orientation='h',
                        title="Sınıf Olasılıkları",
                        color='Olasılık',
                        color_continuous_scale='viridis')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Benzerlik analizi
        st.subheader("🔍 Benzerlik Analizi")
        st.write(f"**Analiz edilen dosya:** {result_data['file_name']}")
        
        # Hedef sesi çal
        if 'audio_files' in st.session_state and result_data['file_name'] in st.session_state.audio_files:
            st.write("🎵 **Hedef Ses:**")
            st.audio(st.session_state.audio_files[result_data['file_name']], format='audio/wav')
        
        if len(st.session_state.classifier.reference_data) > 1:
            target_class = result['predicted_class'] if same_class_only else None
            
            # Tek dosya analizi ise kendini dahil et (doğruluk kontrolü için)
            exclude_self = None if 'single_file_result' in st.session_state and result_data == st.session_state.single_file_result else result_data['file_name']
            
            similar_sounds = st.session_state.classifier.find_similar_sounds(
                result['features_scaled'], 
                target_class=target_class,
                top_k=top_k_similar,
                exclude_self=exclude_self
            )
            
            if similar_sounds:
                st.write(f"**En benzer {len(similar_sounds)} ses:**")
                
                for i, sim in enumerate(similar_sounds, 1):
                    if sim['cosine_similarity'] >= similarity_threshold:
                        file_name = sim.get('display_name', os.path.basename(sim['file_path']))
                        
                        # Aynı dosya ise özel işaret ekle
                        is_same_file = file_name == result_data['file_name']
                        same_file_indicator = " 🎯 (AYNI DOSYA)" if is_same_file else ""
                        
                        col_sim_info, col_sim_play = st.columns([3, 1])
                        
                        with col_sim_info:
                            st.markdown(f"""
                            <div class="similarity-card">
                                <strong>#{i} - {file_name}{same_file_indicator}</strong><br>
                                <strong>Sınıf:</strong> {sim['predicted_class']}<br>
                                <strong>Benzerlik:</strong> {sim['cosine_similarity']:.3f}<br>
                                <strong>Güven:</strong> {sim['confidence']:.1%}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_sim_play:
                            # Benzer sesi çal
                            if 'audio_files' in st.session_state and file_name in st.session_state.audio_files:
                                st.audio(st.session_state.audio_files[file_name], format='audio/wav')
                            else:
                                st.write("🔇 Ses yok")
                
                # Görselleştirme
                if show_similarity_plot and len(similar_sounds) > 0:
                    st.subheader("🗺️ Benzerlik Haritası")
                    try:
                        fig = st.session_state.classifier.visualize_similarities(
                            result['features_scaled'], similar_sounds
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Görselleştirme hatası: {e}")
            else:
                st.info("Belirlenen eşik değerinde benzer ses bulunamadı.")
        else:
            st.info("Benzerlik analizi için daha fazla ses dosyası yükleyin.")

# Alt bilgi
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🎵 One-Shot Ses Sınıflandırıcı | Tez Projesi</p>
    <p>Desteklenen format: WAV | Sampling Rate: 22.05 kHz</p>
</div>
""", unsafe_allow_html=True) 