import base64
import json
from openai import OpenAI
from src.config import (
    GROQ_API_KEY, VISION_MODEL, TEXT_MODEL, 
    VISION_INSTRUCTION, STYLE_INSTRUCTION
)
from src.pipeline.schemas import StyledCaptions

# Initialize the Groq Client
groq_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def generate_styled_captions(keyframes, audio_transcript=None, scheduler=None):
    print(f"DEBUG: Number of keyframes extracted: {len(keyframes)}")
    
    def get_narrative_for_batch(batch_frames, batch_index_str=""):
        image_contents = []
        for path in batch_frames:
            b64 = encode_image(path)
            image_contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
            
        prompt_text = f"Analyze these sequential keyframes {batch_index_str} and write a neutral, factual description of the scene."
        if audio_transcript:
            prompt_text += f" Incorporate information from this audio transcript: '{audio_transcript}'"
            
        messages = [
            {"role": "system", "content": VISION_INSTRUCTION},
            {"role": "user", "content": [{"type": "text", "text": prompt_text}] + image_contents}
        ]
        
        # API-call level rate limit wait
        if scheduler:
            est_tokens = len(batch_frames) * 1600 + 400
            scheduler.wait_for_token_slot(est_tokens, 0)
            
        response = groq_client.chat.completions.create(
            model=VISION_MODEL,
            messages=messages,
            temperature=0.1
        )
        return response.choices[0].message.content

    if len(keyframes) <= 5:
        narrative = get_narrative_for_batch(keyframes)
    else:
        # Split into two batches
        half1 = keyframes[:5]
        half2 = keyframes[5:]
        print("DEBUG: Splitting keyframes into 2 vision batches due to model image constraints.")
        narrative1 = get_narrative_for_batch(half1, "(first half of the video)")
        narrative2 = get_narrative_for_batch(half2, "(second half of the video)")
        narrative = f"Chronological Description of First Half:\n{narrative1}\n\nChronological Description of Second Half:\n{narrative2}"
        
    print("DEBUG: Combined Vision Narrative:\n", narrative)

    # 3. Stage 2: Styled Captioning (llama-3.3-70b-versatile)
    style_messages = [
        {"role": "system", "content": STYLE_INSTRUCTION},
        {"role": "user", "content": f"Scene description:\n{narrative}"}
    ]
    
    # Try Tier 1: Groq llama-3.3-70b structured parse
    try:
        if scheduler:
            scheduler.wait_for_token_slot(0, 600)
            
        response = groq_client.beta.chat.completions.parse(
            model=TEXT_MODEL,
            messages=style_messages,
            response_format=StyledCaptions,
            temperature=0.7
        )
        parsed = response.choices[0].message.parsed.dict()
        if isinstance(parsed, dict) and "captions" in parsed:
            parsed = parsed["captions"]
        return parsed
    except Exception as e:
        print(f"Tier 1 (Groq 70B Structured) failed: {e}. Trying Tier 2...")
        
    # Try Tier 2: Groq llama-3.3-70b raw JSON completion
    try:
        if scheduler:
            scheduler.wait_for_token_slot(0, 600)
            
        response = groq_client.chat.completions.create(
            model=TEXT_MODEL,
            messages=style_messages + [{"role": "system", "content": "Output strictly valid JSON matching the schema."}],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        content = response.choices[0].message.content
        print("RAW TIER 2 RESPONSE:", content)
        parsed = json.loads(content)
        print("PARSED TIER 2 DICT:", parsed)
        if isinstance(parsed, dict) and "captions" in parsed:
            parsed = parsed["captions"]
        return parsed
    except Exception as e:
        print(f"Tier 2 failed: {e}. Returning safe hardcoded fallbacks.")
        
    # Tier 3: Dynamic Visual-Grounding Fallbacks (Grounded in the Stage 1 narrative)
    first_sentence = narrative.split(".")[0] + "." if narrative else "A sequence of visual events."
    return {
        "formal": first_sentence,
        "sarcastic": f"Apparently, we are looking at: {first_sentence} Quite a thrill.",
        "humorous_tech": f"System log: {first_sentence} No critical execution errors.",
        "humorous_non_tech": f"Look at this: {first_sentence} Simple as that."
    }
