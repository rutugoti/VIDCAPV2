import base64
import json
import time
from src.config import (
    VISION_MODEL, TEXT_MODEL, 
    VISION_INSTRUCTION, STYLE_INSTRUCTION,
    get_groq_client, safe_groq_call
)
from src.pipeline.schemas import StyledCaptions

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
        
        est_tokens = len(batch_frames) * 1600 + 400
        def do_vision_call(client):
            return client.chat.completions.create(
                model=VISION_MODEL,
                messages=messages,
                temperature=0.1
            )
        response = safe_groq_call("vision", est_tokens, do_vision_call)
        return response.choices[0].message.content

    # Dynamic batching: split into chunks of 5 images
    batch_size = 5
    batches = [keyframes[i:i+batch_size] for i in range(0, len(keyframes), batch_size)]
    
    narratives = []
    for idx, batch in enumerate(batches):
        batch_label = f"(part {idx+1} of {len(batches)})"
        narrative_part = get_narrative_for_batch(batch, batch_label)
        narratives.append(f"Part {idx+1}:\n{narrative_part}")
        
    narrative = "\n\n".join(narratives)
    print("DEBUG: Combined Vision Narrative:\n", narrative)

    # --- Phase 1: Generate Formal Caption Anchor ---
    formal_caption = ""
    formal_system_instruction = """You are an expert visual forensic analyst. Write a single 'formal' caption describing the video based on the scene description.
    
    Instructions:
    1. Write a cold, clinical, factual, and objective description of the visual events.
    2. Adopt the tone of HAL-9000: emotionless, analytical, and professional.
    3. Strict Grounding: Do not invent any visual elements, characters, settings, items, or actions. Only describe what is explicitly stated.
    4. Write a detailed paragraph or detailed sentences. Do not use any word length restriction."""
    
    formal_messages = [
        {"role": "system", "content": formal_system_instruction},
        {"role": "user", "content": f"Scene description:\n{narrative}"}
    ]
    
    try:
        def do_formal_call(client):
            return client.chat.completions.create(
                model=TEXT_MODEL,
                messages=formal_messages,
                temperature=0.3
            )
        response = safe_groq_call("text", 400, do_formal_call)
        formal_caption = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Generating formal caption failed: {e}. Using dynamic narrative sentence as anchor.")
        formal_caption = narrative.split(".")[0] + "." if narrative else "A sequence of visual events."
        
    print("DEBUG: Generated Formal Caption Anchor:\n", formal_caption)

    # --- Phase 2: Generate Styles (Grounded Lock) ---
    styles_system_instruction = """You are an elite caption writer. You will generate 3 distinct styled captions (sarcastic, humorous_tech, humorous_non_tech) based on the provided formal caption.
    
    Persona Guidelines:
    1. sarcastic: Analyze the visual with a very deadpan and sarcastic tone, with eye-rolling, as if you were forced to deal with mere mortals with a sigh incomparable to your power, using incredible wit and a condescending tone.
    2. humorous_tech: Describe the visual like a tired millennial complaining of the workload in their tech/AI job. Focus on the visual events, using clever technical jargon (rendering, compilation, latency, debugging, etc.) to make it funny and humorous.
    3. humorous_non_tech: Describe the visual sequence with a funny and relatable attitude of a man in his 50s who finds it hard to keep up with the fast-growing tech world. Do not use technical jargon or niche modern references.
    
    Strict Grounding Lock (Critical):
    You must ONLY mention and describe the entities, actions, settings, and events that are explicitly stated in the provided formal caption. You are strictly forbidden from introducing any new characters, settings, items, colors, actions, or events. Only decorate the existing facts with your respective personas. Do not imply unseen physical objects.
    
    CRITICAL STYLE DIVERSITY RULES (To prevent repetitive pattern penalties):
    - Do NOT use cliché prefix templates or formulaic opening phrases.
    - Specifically, FORBID starting sarcastic captions with "Oh joy", "Oh great", "Wow, a", or similar overused sarcastic openings.
    - Specifically, FORBID starting humorous_tech captions with "Ugh, debugging", "Ugh, another day", "System log", or "I'm trying to render".
    - Specifically, FORBID starting humorous_non_tech captions with "I'm trying to", "I'm watching", "I'm telling you", or "Look at".
    - Vary the grammatical structure of the opening sentence. Begin directly with a noun, an adverb, an action verb, or a direct observation.
    
    Output Format: You must output a JSON object with the following fields:
    {
      "sarcastic": "...",
      "humorous_tech": "...",
      "humorous_non_tech": "..."
    }"""

    styles_messages = [
        {"role": "system", "content": styles_system_instruction},
        {"role": "user", "content": f"Formal caption:\n{formal_caption}"}
    ]
    
    try:
        def do_styles_call(client):
            return client.chat.completions.create(
                model=TEXT_MODEL,
                messages=styles_messages + [{"role": "system", "content": "Output strictly valid JSON matching the schema."}],
                response_format={"type": "json_object"},
                temperature=0.7
            )
        response = safe_groq_call("text", 600, do_styles_call)
        content = response.choices[0].message.content
        print("RAW STYLES RESPONSE:", content)
        parsed = json.loads(content)
        
        # Unnest if needed
        if isinstance(parsed, dict) and "captions" in parsed:
            parsed = parsed["captions"]
            
        return {
            "formal": formal_caption,
            "sarcastic": parsed.get("sarcastic", f"Apparently, we are looking at: {formal_caption}"),
            "humorous_tech": parsed.get("humorous_tech", f"System log: {formal_caption}"),
            "humorous_non_tech": parsed.get("humorous_non_tech", f"Look at this: {formal_caption}")
        }
    except Exception as e:
        print(f"Generating styled captions failed: {e}. Returning fallback grounded captions.")
        h = abs(hash(formal_caption))
        
        sarcastic_fallbacks = [
            f"Behold: {formal_caption} My excitement is immeasurable.",
            f"We are witnessing: {formal_caption} Groundbreaking stuff.",
            f"Observe: {formal_caption} A truly mind-bending spectacle.",
            f"Ah, yes: {formal_caption} Just what the world needed.",
            f"Here we have: {formal_caption} Absolute peak entertainment."
        ]
        
        tech_fallbacks = [
            f"Event stream processed: {formal_caption} System status nominal.",
            f"Rendering buffer active: {formal_caption} Frame allocation successful.",
            f"Visual thread pipeline: {formal_caption} Execution trace complete.",
            f"System metrics logged: {formal_caption} Process latency normal.",
            f"Visual compilation: {formal_caption} Zero syntax errors detected."
        ]
        
        non_tech_fallbacks = [
            f"Back in my day, we didn't need to watch: {formal_caption} Things were simpler.",
            f"My grandkids would probably love this: {formal_caption} I don't get it though.",
            f"They've got a video showing: {formal_caption} Whatever that means.",
            f"Now they're filming: {formal_caption} What will they think of next?",
            f"Looks like some folks are doing: {formal_caption} Beats me."
        ]
        
        return {
            "formal": formal_caption,
            "sarcastic": sarcastic_fallbacks[h % len(sarcastic_fallbacks)],
            "humorous_tech": tech_fallbacks[h % len(tech_fallbacks)],
            "humorous_non_tech": non_tech_fallbacks[h % len(non_tech_fallbacks)]
        }
