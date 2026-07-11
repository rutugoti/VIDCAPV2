import subprocess
import os
import glob

def get_video_duration(video_path):
    """
    Query ffprobe to get the exact duration of the video.
    """
    cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 \"{video_path}\""
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    try:
        return float(res.stdout.strip())
    except Exception as e:
        print(f"Error reading video duration: {e}")
        return 10.0  # Fallback default duration

def extract_dynamic_keyframes(video_path, output_dir, target_count=10):
    """
    Extracts up to target_count frames based on scene change detection.
    Falls back to uniform sampling if too few scene transitions occur.
    """
    os.makedirs(output_dir, exist_ok=True)
    duration = get_video_duration(video_path)
    
    # Stage 1: Attempt Scene Change Detection (threshold = 0.2)
    # Using cross-platform clean quoting (no single quotes nested inside double quotes)
    cmd = (
        f"ffmpeg -y -i \"{video_path}\" "
        f"-vf \"select=gt(scene\\,0.2),scale=768:-1\" "
        f"-vsync vfr -qscale:v 5 \"{output_dir}/frame_%03d.jpg\""
    )
    subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    frames = sorted(glob.glob(f"{output_dir}/frame_*.jpg"))
    
    # Stage 2: Fallback to uniform sampling if under 6 frames are detected
    if len(frames) < 6:
        # Clear existing dynamic frames
        for f in frames:
            try:
                os.remove(f)
            except:
                pass
            
        fps_val = target_count / max(duration, 0.1)
        uniform_cmd = (
            f"ffmpeg -y -i \"{video_path}\" "
            f"-vf \"fps=fps={fps_val:.4f},scale=768:-1\" "
            f"-qscale:v 5 \"{output_dir}/frame_%03d.jpg\""
        )
        subprocess.run(uniform_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        frames = sorted(glob.glob(f"{output_dir}/frame_*.jpg"))
        
    if len(frames) <= target_count:
        return frames
    step = len(frames) / target_count
    return [frames[int(i * step)] for i in range(target_count)]
