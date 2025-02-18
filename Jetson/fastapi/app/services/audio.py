from faster_whisper import WhisperModel
import threading
import numpy as np
import noisereduce as nr
import webrtcvad
import librosa  # 다운샘플링을 위해 추가됨
from app.utils import logging_config

BEAM_S = 5
logger = logging_config.app_logger

class Audio_record:
    def __init__(self):
        import speech_recognition as sr
        
        # 하드웨어 샘플레이트: 마이크가 지원하는 주파수 (예: 48000Hz)
        self.hardware_sample_rate = 48000  
        # Whisper가 기대하는 샘플레이트
        self.target_sample_rate = 16000

        self.chunk_duration_ms = 30  # 청크 길이 (ms)
        self.vad_sec = 1             # 무음 지속 시간 (초) 이상이면 녹음 종료
        
        # 녹음 시 하드웨어 샘플레이트 기준으로 청크 크기 계산
        self.chunk_size = int(self.hardware_sample_rate * self.chunk_duration_ms / 1000)
        self.recognizer = sr.Recognizer()
        # 하드웨어 샘플레이트로 마이크를 초기화 (device_index는 환경에 맞게 조정)
        self.microphone = sr.Microphone(
            device_index=0, 
            # device_index=24, 
            sample_rate=self.hardware_sample_rate,
            chunk_size=self.chunk_size
        )
        self.buffer = []
        self.recording = False
        self.vad = webrtcvad.Vad(1)  # VAD 민감도 (0~3, 3이 가장 민감)
        self.adjust_noise()

    def adjust_noise(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            self.recognizer.energy_threshold += 100

    def record_start(self):
        if not self.recording:
            self.record_thread = threading.Thread(target=self._record_start)
            self.record_thread.start()

    def _record_start(self):
        self.recording = True
        self.buffer = []
        # 하드웨어 샘플레이트 기준으로 무음 감지 처리
        no_voice_target_cnt = self.vad_sec * 1000  
        no_voice_cnt = 0
        with self.microphone as source:
            while self.recording:
                chunk = source.stream.read(self.chunk_size)
                self.buffer.append(chunk)
                if self._vad(chunk, self.hardware_sample_rate):
                    no_voice_cnt = 0
                else:
                    no_voice_cnt += self.chunk_duration_ms
                if no_voice_cnt >= no_voice_target_cnt:
                    self.recording = False

    def _vad(self, chunk, sample_rate):
        # 청크가 bytes인 경우 int16 배열로 변환
        if isinstance(chunk, bytes):
            chunk = np.frombuffer(chunk, dtype=np.int16)
        # 하드웨어 샘플레이트 기준 청크 크기 확인
        expected_size = int(sample_rate * self.chunk_duration_ms / 1000)
        if len(chunk) != expected_size:
            raise ValueError("Chunk size must be exactly 10ms, 20ms, or 30ms")
        return self.vad.is_speech(chunk.tobytes(), sample_rate)

    def record_stop(self, denoise_value):
        # 녹음 종료 후 스레드 대기
        self.recording = False
        self.record_thread.join()
        # 버퍼에 저장된 청크들을 하나의 오디오 데이터로 결합 (int16)
        audio_data = np.frombuffer(b''.join(self.buffer), dtype=np.int16)
        # 다운샘플링: 하드웨어 샘플레이트(48000Hz) -> 타깃 샘플레이트(16000Hz)
        # 먼저 float32로 변환 (librosa는 float32를 기대함)
        audio_data = audio_data.astype(np.float32)
        audio_data_resampled = librosa.resample(audio_data, orig_sr=self.hardware_sample_rate, target_sr=self.target_sample_rate)
        return self._denoise_process(audio_data_resampled, self.target_sample_rate, denoise_value)

    def _denoise_process(self, audio_data, sample_rate, denoise_value):
        # 노이즈 감소 처리 후 Whisper 모델에 전달 가능한 float32 배열로 변환
        denoised = nr.reduce_noise(y=audio_data, sr=sample_rate, prop_decrease=denoise_value)
        audio_denoise = denoised.astype(np.float32) / 32768.0
        return {'audio_denoise': audio_denoise, 'sample_rate': sample_rate}


class Custom_faster_whisper:
    def __init__(self):
        print('Custom_faster_whisper 초기화 성공')

    def set_model(self, model_name):
        print("모델 로딩 시작")
        self.model = WhisperModel(model_name, device="cuda", compute_type="float16")
        print("모델 로딩 완료")
        return model_name

    def run(self, audio, language='ko'):
        print(1)
        """
        audio: WAV 파일 경로나 NumPy 배열 형태의 오디오 데이터
        language: 'ko', 'en' 등 언어 설정 (명시하지 않으면 내부에서 자동 감지)
        """
        segments, info = self.model.transcribe(
            audio,
            beam_size=BEAM_S,
            word_timestamps=True,
            language=language
        )
        dic_list = []
        for segment in segments:
            if segment.no_speech_prob > 0.6:
                continue
            for word in segment.words:
                dic_list.append([word.word])
        result_txt = self._make_txt(dic_list)
        if result_txt:
            logger.info(f"Transcribed text: {result_txt}")
        return dic_list, result_txt

    def _make_txt(self, dic_list):
        return ''.join([dic[0] for dic in dic_list])
