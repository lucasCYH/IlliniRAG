import os
import subprocess
import concurrent.futures
import wave
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from backend import db

_available_voices_cache = None

def get_available_voices():
    """Scan available voices in macOS TTS to support graceful degradation."""
    global _available_voices_cache
    if _available_voices_cache is None:
        try:
            res = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, check=True)
            lines = res.stdout.splitlines()
            _available_voices_cache = []
            for line in lines:
                parts = line.split()
                if parts:
                    _available_voices_cache.append(parts[0])
        except Exception:
            _available_voices_cache = []
    return _available_voices_cache

def select_voice(preferred_voice, fallback_choices):
    """Select a voice from system list, falling back gracefully if preferred voice is missing."""
    available = get_available_voices()
    if not available:
        return preferred_voice, None
    if preferred_voice in available:
        return preferred_voice, None
    
    # Try fallbacks
    for fb in fallback_choices:
        if fb in available:
            return fb, f"找不到語音 '{preferred_voice}'，已優雅降級使用 '{fb}'"
            
    # Try first generic english voice
    for av in available:
        if av.lower().startswith(("en", "daniel", "fred", "samantha", "alex")):
            return av, f"找不到語音 '{preferred_voice}' 及其備用選項，已自動使用系統預設語音 '{av}'"
            
    # Return preferred anyway if no matches found
    return preferred_voice, None

def synthesize_line(index: int, voice: str, text: str) -> str:
    """Synthesize a single line of speech to AIFF, convert it to WAV, and return the path."""
    aiff_path = f"/tmp/pod_{index:03d}.aiff"
    wav_path = f"/tmp/pod_{index:03d}.wav"
    
    print(f"[Podcast Sync] Synthesizing sequence {index:03d} using {voice or 'Default'}...")
    
    cmd = ["say"]
    if voice:
        cmd.extend(["-v", voice])
    cmd.extend(["-o", aiff_path, text])
    
    # Run the blocking say command
    subprocess.run(cmd, check=True)
    # Convert AIFF to WAVE (16-bit Little Endian)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16", aiff_path, wav_path], check=True)
    
    # Cleanup intermediate AIFF file
    try:
        os.remove(aiff_path)
    except OSError:
        pass
        
    return wav_path

def generate_podcast_audio(doc_ids=None):
    llm = Ollama(model="llama3.1", temperature=0.7)
    texts = db.get_parent_chunks_text_by_docs(doc_ids)
    full_text = "\n\n".join(texts)[:20000] 
    
    if not full_text:
        return None, "No documents found to generate a podcast."
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert podcast scriptwriter. Generate a short, engaging 2-host podcast script discussing the following document context. Format strictly as:\nHost A: [dialogue]\nHost B: [dialogue]\nKeep it under 10 lines total."),
        ("human", "Context:\n{context}\n\nPlease write the script.")
    ])
    
    chain = prompt | llm
    script = chain.invoke({"context": full_text})
    
    # Parse script
    lines = script.split('\n')
    tasks = []
    
    # Select voices with graceful degradation fallbacks
    female_fallbacks = ["Samantha", "Tessa", "Victoria", "Karen", "Moira", "Fiona", "Veena"]
    male_fallbacks = ["Alex", "Daniel", "Fred", "Oliver", "Rishi", "Albert"]
    
    voice_a, warn_a = select_voice("Samantha", female_fallbacks)
    voice_b, warn_b = select_voice("Alex", male_fallbacks)
    
    warning_msg = ""
    if warn_a or warn_b:
        warnings = [w for w in [warn_a, warn_b] if w]
        warning_msg = "、".join(warnings) + "。建議至 macOS「系統設定 > 輔助使用 > 語音內容」下載高品質語音包以獲得最佳體驗。"
        print(f"[Podcast Sync] Voice degradation warnings: {warning_msg}")

    # ProcessPoolExecutor/ThreadPoolExecutor for non-blocking concurrent generation
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("Host A:"):
                text = line.replace("Host A:", "").strip()
                tasks.append(executor.submit(synthesize_line, i, voice_a, text))
            elif line.startswith("Host B:"):
                text = line.replace("Host B:", "").strip()
                tasks.append(executor.submit(synthesize_line, i, voice_b, text))
                
        # Wait for all thread pool processes to finish and gather output paths
        results = []
        for future in concurrent.futures.as_completed(tasks):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"[Podcast Sync] Task failed: {e}")
                
    if not results:
        return None, "Failed to generate podcast audio.", warning_msg
        
    # Sort files strictly by sequence index to guarantee timeline ordering
    results.sort() 
    
    # Merge WAV files sequentially using the Python standard library 'wave'
    combined_wav = "podcast.wav"
    try:
        with wave.open(combined_wav, "wb") as output:
            first = True
            for wav_file in results:
                with wave.open(wav_file, "rb") as infile:
                    if first:
                        # Copy wave file headers/parameters from the first chunk
                        output.setparams(infile.getparams())
                        first = False
                    output.writeframes(infile.readframes(infile.getnframes()))
                
                # Cleanup individual segment wav files
                try:
                    os.remove(wav_file)
                except OSError:
                    pass
                    
        print(f"[Podcast Sync] Successfully merged {len(results)} audio segments into '{combined_wav}'")
    except Exception as e:
        print(f"[Podcast Sync] Error merging wav files: {e}")
        return None, f"Failed to merge segments: {e}", warning_msg
        
    clean_script = "\n".join([line.strip() for line in lines if line.strip().startswith(("Host A:", "Host B:"))])
    return combined_wav, clean_script, warning_msg
