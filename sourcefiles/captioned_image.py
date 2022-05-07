from typing import List

from pydantic import BaseModel


class CaptionedImage(BaseModel):
    class_id: str
    class_name: str
    image_path: str
    captions: List[str]
