from pydantic import BaseModel, Field

class StyledCaptions(BaseModel):
    formal: str = Field(description="Factual and objective caption from the clinical perspective of HAL-9000.")
    sarcastic: str = Field(description="Deadpan, witty, and condescending sarcastic caption.")
    humorous_tech: str = Field(description="Humorous caption using software/IT jargon from a tired millennial.")
    humorous_non_tech: str = Field(description="Relatable everyday humor caption from a 50-year-old grandfather.")
