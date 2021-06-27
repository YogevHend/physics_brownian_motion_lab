import numpy as np
import cv2


BGR_WHITE = (255, 255, 255)
BGR_RED = (0, 0, 255)
BGR_BLACK = (0, 0, 0)
GRAYSCALE_WHITE = 255

def get_single_frame_from_video(
    path,
    frame_number=0,
):
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()

    frame_counter = 0
    try:
        while(cap.isOpened()):
            ret, frame = cap.read()

            if frame is not None:
                if frame_counter == frame_number:
                    break
            else:
                break
            
            frame_counter += 1
    except Exception as exc:
        raise exc
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return frame


def frame_to_bnw(
    frame,
    thresh,
    thresh_maxval=255,
):
    gray_frame = cv2.cvtColor(
        src=frame,
        code=cv2.COLOR_BGR2GRAY,
    )

    ret, bnw_frame = cv2.threshold(
        src=gray_frame,
        thresh=thresh,
        maxval=thresh_maxval,
        type=cv2.THRESH_BINARY,
    )

    return bnw_frame

def put_frame_number_on_frame(
    frame,
    frame_number,
    color=BGR_WHITE,
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = '{frame_number}'.format(
        frame_number=frame_number,
    )
    upper_left_corner = (50, 50)

    cv2.putText(
        img=frame,
        text=text,
        org=upper_left_corner,
        fontFace=font,
        fontScale=1,
        color=color,
        thickness=1,
        lineType=cv2.LINE_AA,
    )

    return frame

def put_text_on_frame(
    frame,
    text,
    position,
    color=BGR_WHITE,
    font_size_scale=1,
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = '{text}'.format(
        text=text,
    )
    upper_left_corner = position

    cv2.putText(
        img=frame,
        text=text,
        org=upper_left_corner,
        fontFace=font,
        fontScale=font_size_scale,
        color=color,
        thickness=1,
        lineType=cv2.LINE_AA,
    )

    return frame
