import cv2

from src.config import BOARD_IMG_PATH, BOARD_POS, DRAW_MATCHING_POINTS
from src.renderer import BoardRenderer

# Initialize board renderer
renderer = BoardRenderer(BOARD_IMG_PATH, BOARD_POS,
                         draw_matching_points=DRAW_MATCHING_POINTS)

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    # read the current frame
    ret, origin_frame = cap.read()
    if not ret:
        print("Unable to capture video")
        exit(0)

    draw = renderer.render(origin_frame)

    cv2.imshow('frame', draw)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
