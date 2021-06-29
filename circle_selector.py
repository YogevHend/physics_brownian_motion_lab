import tqdm
import xlsxwriter
import numpy as np
import cv2


class CircleSelector:
    def __init__(
        self,
    ):
        self.detected_circles = []
        self.current_circle_points = []

    frames = []

def parse_video(
    path,
    new_file_name,
    xlsx_path,
    frame_range_start,
    frame_numbers_to_check,
    frame_range_end=None,
    add_frame_number=True,
):
    cap = cv2.VideoCapture(path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    out = cv2.VideoWriter(
        new_file_name,
        cv2.VideoWriter_fourcc(*'MJPG'),
        10,
        size,
    )
    if frame_range_end is None:
        frame_range_end = int(
            cap.get(
                propId=cv2.CAP_PROP_FRAME_COUNT,
            )
        )
    total_frames = frame_range_end - frame_range_start

    selector = CircleSelector()
    try:
        with tqdm.tqdm(total=total_frames) as pbar:
            while(cap.isOpened()):
                ret, frame = cap.read()
                frame_number = cap.get(
                    cv2.CAP_PROP_POS_FRAMES,
                )

                if frame is None or frame_number > frame_range_end:
                    break

                if frame_number < frame_range_start:
                    continue

                if frame is not None:
                    if frame_number in frame_numbers_to_check:
                        circle = get_circle_by_clicking(
                            frame=frame,
                            selector=selector,
                        )
                        center = (int(circle[0][0]), int(circle[0][1]))
                        radius = int(circle[1])
                        cv2.circle(frame, center, radius, (0, 255, 0), 1)
                        detected_circle = {
                            'circle': circle,
                            'frame_number': frame_number,
                        }
                        selector.detected_circles.append(detected_circle)
                    if add_frame_number:
                        frame = put_frame_number_on_frame(
                            frame=frame,
                            frame_number=frame_number,
                        )
                    out.write(frame)
                else:
                    break
                pbar.update(1)

            pbar.update(1)

        if frame_number < frame_range_start:
            raise ValueError(
                'Reached end of video before finding desired start frame')
        elif frame_number < frame_range_end:
            raise ValueError(
                'Reached end of video before finding desired end frame')
    except:
        raise
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    with open(xlsx_path, 'w') as fp:
        workbook = xlsxwriter.Workbook(xlsx_path)
        worksheet = workbook.add_worksheet('circles')
        worksheet.write(0, 0, 'frame number')
        worksheet.write(0, 1, 'circle radius')
        worksheet.write(0, 2, 'circle center x')
        worksheet.write(0, 3, 'circle center y')
        for index, detected_circle in enumerate(selector.detected_circles):
            worksheet.write(index + 1, 0, detected_circle['frame_number'])
            worksheet.write(index + 1, 1, detected_circle['circle'][1])
            worksheet.write(index + 1, 2, detected_circle['circle'][0][0])
            worksheet.write(index + 1, 3, detected_circle['circle'][0][1])
        workbook.close()

def get_circle_by_clicking(
    frame,
    selector,
):
    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            x = int(x)
            y = int(y)

            font = cv2.FONT_HERSHEY_SIMPLEX
            text = 'x'
            color = (0,0,255,)

            if len(selector.current_circle_points) < 3:
                selector.current_circle_points.append((x,y))

                cv2.putText(
                    img=frame,
                    text=text,
                    org=(x - 1,y - 1),
                    fontFace=font,
                    fontScale=0.1,
                    color=color,
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )
                cv2.imshow(
                    winname='image',
                    mat=frame,
                )
                width = frame.shape[0]
                height = frame.shape[1]
                magnify_by = 3
                cv2.resizeWindow(
                    winname='image',
                    width=width * magnify_by,
                    height=height * magnify_by,
                )
                cv2.waitKey(
                    delay=1,
                )
                if len(selector.current_circle_points) == 3:
                    cv2.destroyAllWindows()

    cv2.namedWindow(
        winname='image',
        flags=cv2.WINDOW_KEEPRATIO,
    )
    cv2.imshow(
        winname='image',
        mat=frame,
    )
    width = frame.shape[0]
    height = frame.shape[1]
    magnify_by = 3
    cv2.resizeWindow(
        winname='image',
        width=width * magnify_by,
        height=height * magnify_by,
    )
    cv2.waitKey(
        delay=1,
    )
    cv2.setMouseCallback('image', click_event)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    p1 = selector.current_circle_points[0]
    p2 = selector.current_circle_points[1]
    p3 = selector.current_circle_points[2]
    selector.current_circle_points = []

    return define_circle(p1,p2,p3)

def define_circle(p1, p2, p3):
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return (None, np.inf)

    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return ((cx, cy), radius)

def put_frame_number_on_frame(
    frame,
    frame_number,
    color=(255, 255, 255),
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

def main():
    # Replace these with the correct paths!
    vid_path = 'D:\\physics_lab_videos\\day4\\blue_dye_35w_17g.avi'
    parsed_path = 'D:\\physics_lab_videos\\day4\\blue_dye_35w_17g_parsed.avi'
    xlsx_path = 'D:\\physics_lab_videos\\day4\\blue_dye_35w_17g.xlsx'

    # The ranges defined here will determine when the script asks the user to select circles
    frame_numbers_to_check = list(range(50,131,2)) # Every two frames between frame 50 and 131...
    frame_numbers_to_check += list(range(130,201,5)) # Every five frames between frame 130 and 201...
    frame_numbers_to_check += list(range(200,301,10))
    frame_numbers_to_check += list(range(300,501,20))
    frame_numbers_to_check += list(range(500,801,30))
    frame_numbers_to_check += list(range(800,1001,40))
    frame_numbers_to_check += list(range(1000,2201,100))
    parse_video(
        path=vid_path,
        new_file_name=parsed_path,
        xlsx_path=xlsx_path,
        frame_numbers_to_check=frame_numbers_to_check,
        frame_range_start=0,
        frame_range_end=None,
    )
