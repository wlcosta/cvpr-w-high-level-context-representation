'''
mediapipe_utils.py
Created on 2023 02 22 12:28:05
Description: File containing utils methods for processing mediapipe faces

Author: Will <wlc2@cin.ufpe.br>
'''
from typing import Union, Tuple
import math
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

def get_face_locations_mediapipe(image):
  image_rows, image_cols, _ = image.shape
  with mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5) as face_detection:
    results = face_detection.process(image)
  if not results.detections:
    return None, None
  relative_bounding_box = results.detections[0].location_data.relative_bounding_box
  rect_start_point = _normalized_to_pixel_coordinates(
    relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
    image_rows)
  rect_end_point = _normalized_to_pixel_coordinates(
    relative_bounding_box.xmin + relative_bounding_box.width,
    relative_bounding_box.ymin + relative_bounding_box.height, image_cols,
    image_rows)

  return rect_start_point, rect_end_point