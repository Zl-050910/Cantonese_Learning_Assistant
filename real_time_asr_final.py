#!/usr/bin/env python3
"""
实时语音识别系统（最终版）
使用修复配置，解决音量阈值问题，添加字词级/句子级置信度展示
"""

import multiprocessing as mp
import queue
import time
import signal
import sys
import traceback

# 使用修复版配置
from config_fixed import *


def audio_worker(audio_queue, vad_queue, stop_event, pause_event, resume_event):
    """音频采集和 VAD 工作进程"""
    print("[音频进程] 启动")
    
    # 在子进程中导入和初始化
    from audio_capture import AudioCapture
    import webrtcvad
    import numpy as np

    # 创建VAD检测器
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    frame_size = int(SAMPLE_RATE * VAD_FRAME_DURATION_MS / 1000)
    silence_frames_threshold = int(VAD_SILENCE_DURATION_MS / VAD_FRAME_DURATION_MS)
    
    # 音量阈值
    volume_threshold = 10 ** (VOLUME_THRESHOLD_DB / 20)  # 转换为线性值
    
    capture = AudioCapture(block_size=BLOCK_SIZE)
    
    # VAD状态
    speech_buffer = []
    silence_frames = 0
    is_speech_active = False
    consecutive_speech_frames = 0

    def calculate_volume(audio_frame):
        """计算音频帧的音量（RMS）"""
        if len(audio_frame) == 0:
            return 0.0
        # 转换为 float 并归一化
        if audio_frame.dtype != np.float32:
            audio_float = audio_frame.astype(np.float32) / 32768.0
        else:
            audio_float = audio_frame
        # 计算 RMS
        rms = np.sqrt(np.mean(audio_float ** 2))
        return rms
    
    try:
        capture.start()
        print(f"[音频进程] 音频采集已开始，帧大小: {frame_size}，音量阈值: {volume_threshold:.5f}")
        
        while not stop_event.is_set():
            # 检查是否需要暂停 VAD 检测
            if pause_event.is_set():
                # 等待恢复信号或超时
                resume_event.wait(timeout=0.5)
                if resume_event.is_set():
                    resume_event.clear()
                    pause_event.clear()
                continue
            
            # 正常状态：采集音频并进行 VAD 检测
            block = capture.read_block(timeout=0.1)
            if block is None:
                continue
            
            # 转换到 int16
            if block.dtype != np.int16:
                block_int16 = (block * 32767).astype(np.int16)
            else:
                block_int16 = block

            # 计算音量
            volume = calculate_volume(block)
            
            # VAD 检测
            try:
                audio_bytes = block_int16.tobytes()
                frame_is_speech = vad.is_speech(audio_bytes, SAMPLE_RATE)
            except:
                frame_is_speech = False
            
            # 音量过滤
            if frame_is_speech and volume < volume_threshold:
                # print(f"VAD: 音量过低 ({volume:.4f} < {volume_threshold:.4f})，忽略")
                frame_is_speech = False

            if frame_is_speech:
                consecutive_speech_frames += 1
                speech_buffer.append(block_int16)
                silence_frames = 0
                if not is_speech_active:
                    # 需要连续检测到足够多的语音帧才开始语音段
                    if consecutive_speech_frames >= 2:  # 2帧（60ms）连续语音
                        is_speech_active = True
                        print(f"VAD: 语音开始 (连续{consecutive_speech_frames}帧，音量: {volume:.4f})")
                    # else:
                    #     print(f"VAD: 检测到短暂语音 ({consecutive_speech_frames}帧)，等待更多确认")
            else:
                consecutive_speech_frames = 0
                if is_speech_active:
                    silence_frames += 1
                    speech_buffer.append(block_int16)  # 静音帧也加入缓冲
                    if silence_frames >= silence_frames_threshold:
                        # 静音超时，语音段结束
                        if speech_buffer:
                            segment = np.concatenate(speech_buffer)
                            segment_duration = len(segment) / SAMPLE_RATE * 1000  # ms
                            if segment_duration < MIN_SPEECH_DURATION_MS:
                                # print(f"VAD: 语音段过短 ({segment_duration:.0f}ms < {MIN_SPEECH_DURATION_MS}ms)，忽略")
                                speech_buffer.clear()
                                silence_frames = 0
                                is_speech_active = False
                            else:
                                # 放入队列供识别进程处理
                                try:
                                    vad_queue.put_nowait(segment)
                                    print(f"[音频进程] 发送语音段，时长: {segment_duration:.0f}ms")
                                    # 检测到语音段后立即自动暂停
                                    pause_event.set()
                                except queue.Full:
                                    print("[音频进程] VAD 队列已满，丢弃语音段")
                                speech_buffer.clear()
                                silence_frames = 0
                                is_speech_active = False
                        else:
                            is_speech_active = False

            # 将原始音频块也放入队列（用于调试或备用）
            try:
                audio_queue.put_nowait(block)
            except queue.Full:
                pass
    
    except KeyboardInterrupt:
        # 忽略键盘中断，安静退出
        pass
    except Exception as e:
        print(f"[音频进程] 错误: {e}")
        traceback.print_exc()
    finally:
        capture.stop()
        print("[音频进程] 停止")


def recognition_worker(vad_queue, result_queue, stop_event):
    """语音识别工作进程"""
    print("[识别进程] 启动")
    
    # 在子进程中导入和初始化
    from whisper_recognizer import WhisperRecognizer
    recognizer = WhisperRecognizer()
    
    while not stop_event.is_set():
        try:
            # 从 VAD 队列获取语音段
            segment = vad_queue.get(timeout=1.0)
            if segment is None:
                continue
            
            print(f"[识别进程] 收到语音段，长度: {len(segment)} 样本")
            # 转录
            text, segments, confidence_info = recognizer.transcribe(segment, language=LANGUAGE)
            
            if text:
                print(f"[识别进程] 识别结果: {text} (置信度: {confidence_info['overall_confidence']:.3f})")
                # 将结果放入结果队列
                result_queue.put({
                    'text': text,
                    'segments': segments,
                    'confidence': confidence_info,
                    'timestamp': time.time()
                })
            else:
                print("[识别进程] 未识别到文本")
                
        except queue.Empty:
            continue
        except KeyboardInterrupt:
            # 忽略键盘中断，安静退出
            pass
        except Exception as e:
            print(f"[识别进程] 错误: {e}")
            traceback.print_exc()
    print("[识别进程] 停止")


class RealTimeASRFinal:
    def __init__(self):
        """初始化系统"""
        print("[系统] 开始初始化...")
        
        self.running = False
        self.should_stop_by_command = False
        
        # 进程间通信
        self.audio_queue = mp.Queue(maxsize=MAX_QUEUE_SIZE * 10)
        self.vad_queue = mp.Queue(maxsize=MAX_QUEUE_SIZE)
        self.result_queue = mp.Queue()
        self.stop_event = mp.Event()
        self.pause_event = mp.Event()
        self.resume_event = mp.Event()
        
        # 创建进程
        self.audio_proc = mp.Process(target=audio_worker,
                                     args=(self.audio_queue, self.vad_queue, self.stop_event,
                                           self.pause_event, self.resume_event))
        self.recognizer_proc = mp.Process(target=recognition_worker,
                                          args=(self.vad_queue, self.result_queue, self.stop_event))
        
        print("[系统] 初始化完成")

    def process_user_input(self, text):
        """处理用户输入的接口函数"""
        print(f"[处理接口] 收到用户输入: {text}")
        
        # 检测"结束"文本
        if "结束" in text:
            print("[处理接口] 检测到'结束'指令，准备停止系统...")
            self.should_stop_by_command = True
            return "收到结束指令，系统即将关闭"
        
        # 简单的回显处理
        response = f"我听到你说: {text}"
        print(f"[处理接口] 处理结果: {response}")
        
        return response

    def start_interaction(self):
        """开始人机互动"""
        print("=== 实时语音识别系统（最终版） ===")
        print(f"模型: {MODEL_PATH}")
        print(f"采样率: {SAMPLE_RATE} Hz")
        print(f"音频块大小: {BLOCK_SIZE} 样本 ({BLOCK_DURATION_MS} ms)")
        print(f"VAD 敏感度: {VAD_AGGRESSIVENESS} (0-3)")
        print(f"音量阈值: {VOLUME_THRESHOLD_DB} dB (线性值: {10 ** (VOLUME_THRESHOLD_DB / 20):.5f})")
        print(f"最小语音时长: {MIN_SPEECH_DURATION_MS} ms")
        print("请开始说话，系统将自动识别并处理")
        print("说'结束'可以退出系统，或按 Ctrl+C 退出")
        print("")
        
        # 启动进程
        print("[系统] 启动音频进程...")
        self.audio_proc.start()
        print("[系统] 启动识别进程...")
        self.recognizer_proc.start()
        self.running = True
        
        # 检查进程状态
        time.sleep(1)
        print(f"[系统] 音频进程状态: {'活跃' if self.audio_proc.is_alive() else '已停止'}")
        print(f"[系统] 识别进程状态: {'活跃' if self.recognizer_proc.is_alive() else '已停止'}")
        if not self.audio_proc.is_alive() or not self.recognizer_proc.is_alive():
            print("[系统] 有进程启动失败，系统退出")
            self.stop()
            return
        
        # 设置信号处理
        def signal_handler(sig, frame):
            print(f"\n收到信号 {sig}，正在停止系统...")
            self.stop()
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            conversation_count = 0
            print("[系统] 进入主循环，等待用户输入...")
            while self.running:
                try:
                    # 获取识别结果
                    result = self.result_queue.get(timeout=1.0)
                    conversation_count += 1
                    print(f"\n=== 第 {conversation_count} 轮对话 ===")
                    print(f"用户: {result['text']}")
                    print(f"整体置信度: {result['confidence']['overall_confidence']:.3f}")
                    
                    # 显示词汇级置信度
                    if ('word_confidences' in result['confidence'] and
                        result['confidence']['word_confidences']):
                        print("\n词汇级置信度详情:")
                        for word_info in result['confidence']['word_confidences']:
                            confidence = word_info['confidence']
                            time_range = f"{word_info['start']:.1f}s-{word_info['end']:.1f}s"
                            confidence_marker = "[高]" if confidence > 0.7 else "[中]" if confidence > 0.5 else "[低]"
                            print(f"  {confidence_marker} '{word_info['word']}': {confidence:.3f} ({time_range})")
                    
                    # 调用处理接口
                    response = self.process_user_input(result['text'])
                    
                    if response:
                        print(f"系统: {response}")
                    
                    # 检查是否需要通过指令停止
                    if self.should_stop_by_command:
                        print("[系统] 收到结束指令，准备停止系统...")
                        break
                    
                    # 处理完成后，通知音频进程恢复采集
                    print("[系统] 处理完成，通知音频进程恢复采集")
                    self.resume_event.set()
                    print("等待下一句话...")
                    
                except queue.Empty:
                    # 检查进程是否存活
                    if not self.audio_proc.is_alive() or not self.recognizer_proc.is_alive():
                        print("[系统] 工作进程异常退出")
                        break
        except KeyboardInterrupt:
            print("\n用户主动退出")
        except Exception as e:
            print(f"[系统] 主循环异常: {e}")
            traceback.print_exc()
        finally:
            self.stop()

    def stop(self):
        """停止系统"""
        if self.running:
            print("[系统] 开始停止系统...")
            self.running = False
            self.stop_event.set()
            self.resume_event.set()
            
            # 等待进程结束
            self.audio_proc.join(timeout=2.0)
            self.recognizer_proc.join(timeout=2.0)
            if self.audio_proc.is_alive():
                print("[系统] 音频进程仍在运行，强制终止")
                self.audio_proc.terminate()
            if self.recognizer_proc.is_alive():
                print("[系统] 识别进程仍在运行，强制终止")
                self.recognizer_proc.terminate()
            print("系统已停止")


def main():
    """主函数"""
    mp.freeze_support()
    print("=== 启动实时语音识别系统（最终版） ===")
    try:
        asr_system = RealTimeASRFinal()
        asr_system.start_interaction()
    except Exception as e:
        print(f"[主程序] 系统启动失败: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
