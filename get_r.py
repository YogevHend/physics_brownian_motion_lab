import os.path
import tqdm
import xlsxwriter
import numpy as np
import cv2
import utils

def parse_video(
    path,
    new_file_name,
    xlsx_path,
    frame_range_start,
    frame_range_end=None,
    add_frame_number=True,
):
    first_frame = utils.get_single_frame_from_video(
        path=path,
        frame_number=frame_range_start,
    )
    prev_circles = initialize_circles(
        frame=first_frame,
        add_index=True,
    )

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
    all_circles = []
    if frame_range_end is None:
        frame_range_end = int(
            cap.get(
                propId=cv2.CAP_PROP_FRAME_COUNT,
            )
        )
    total_frames = frame_range_end - frame_range_start

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
                    circles, frame = find_circles(
                        frame,
                        prev_circles,
                    )
                    all_circles.append(circles)
                    if add_frame_number:
                        frame = utils.put_frame_number_on_frame(
                            frame=frame,
                            frame_number=frame_number,
                        )
                        frame = utils.put_text_on_frame(
                            frame=frame,
                            text='C Len: {length}'.format(
                                length=len(
                                    circles,
                                ),
                            ),
                            position=(200, 50),
                            color=utils.BGR_RED,
                            font_size_scale=2,
                        )
                    out.write(frame)
                    prev_circles = circles
                else:
                    break
                pbar.update(1)

            pbar.update(1)

        if frame_number < frame_range_start:
            raise ValueError('Reached end of video before finding desired start frame')
        elif frame_number < frame_range_end:
            raise ValueError('Reached end of video before finding desired end frame')
    except:
        raise
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    with open(xlsx_path, 'w') as fp:
        workbook = xlsxwriter.Workbook(xlsx_path)
        worksheets = {}
        for frame_index, circles in enumerate(all_circles):
            frame_number = frame_index + 1
            for circle in circles:
                worksheet_name = 'mass ' + str(circle['index'])
                if worksheet_name not in worksheets:
                    worksheets[worksheet_name] = workbook.add_worksheet(
                        name=worksheet_name,
                    )
                    worksheets[worksheet_name].write(0, 0, 'frame number')
                    worksheets[worksheet_name].write(0, 1, 'enclosing circle radius')
                    worksheets[worksheet_name].write(0, 2, 'enclosing circle center x')
                    worksheets[worksheet_name].write(0, 3, 'enclosing circle center y')

                worksheet = worksheets[worksheet_name]            
                worksheet.write(frame_number, 0, frame_number)
                worksheet.write(frame_number, 1, circle['radius'])
                worksheet.write(frame_number, 2, circle['center'][0])
                worksheet.write(frame_number, 3, circle['center'][1])
        workbook.close()

def find_circles(
    frame,
    prev_circles,
):
    img = frame.copy()

    new_circles = initialize_circles(
        frame=frame,
        add_index=False,
    )

    detected_circles = []
    for circle in prev_circles:
        closest_circle = get_closest_circle_from_new_circles(
            center=circle['center'],
            radius=circle['radius'],
            new_circles=new_circles,
        )
        if closest_circle is None:
            continue

        new_circles.remove(
            closest_circle,
        )

        closest_circle['index'] = circle['index']
        detected_circles.append(
            closest_circle,
        )

        img = annotate_circle(
            img=img,
            circle=closest_circle,
        )

    return detected_circles, img


def initialize_circles(
    frame,
    add_index=False,
):
    frame = cv2.bilateralFilter(frame, 9, 75, 75)
    bnw_frame = utils.frame_to_bnw(
        frame,
        165,
        thresh_maxval=255,
    )
    all_contours, hierarchy = cv2.findContours(
        image=bnw_frame,
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_NONE,
    )
    hierarchy = hierarchy[0]
    # This finds the correct hierarchy for the contours (it's kind of a weird structure)
    parents = [
        inner_hierarchy[3] for inner_hierarchy in hierarchy
    ]
    # The common parent is the one that appears most
    common_parent = max(
        set(parents),
        key=parents.count,
    )

    circles = []
    for index, component in enumerate(zip(all_contours, hierarchy)):
        contour = component[0]
        (x, y), radius = cv2.minEnclosingCircle(contour)
        current_hierarchy = component[1]
        # Skip contours that aren't of circles (those will have a different common parent)
        if current_hierarchy[3] != common_parent:
            continue
        circle = {
            'radius': radius,
            'center': (x, y),
            'contour': contour,
        }
        if add_index:
            circle['index'] = index

        circles.append(
            circle,
        )

    return circles

def get_closest_circle_from_new_circles(
    center,
    radius,
    new_circles,
):
    radius_epsilon = 8
    distance_epsilon = 30

    similar_circles = [
        circle for circle in new_circles 
        if abs(circle['radius'] - radius) < radius_epsilon
    ]
    if len(similar_circles) == 0:
        return None

    for circle in similar_circles:
        x_delta = center[0] - circle['center'][0]
        y_delta = center[1] - circle['center'][1]

        distance_norm = (x_delta**2 + y_delta**2)**0.5
        circle['dist'] = distance_norm
    
    closest_circle = min(
        similar_circles,
        key=lambda circle: circle['dist'],
    )

    if closest_circle['dist'] < distance_epsilon:
        return closest_circle
    
    return None


def annotate_circle(
    img,
    circle,
):
    center = circle['center']
    center = (int(center[0]), int(center[1]))

    radius = int(circle['radius'])
    text_offset = int(radius * 1.5)

    cv2.drawContours(img, [circle['contour']], -1, (127, 0, 127), 2)
    cv2.circle(img, center, radius, (0, 255, 0), 2)

    text = str(circle['index'])
    img = utils.put_text_on_frame(
        frame=img,
        text=text,
        position=(center[0] + text_offset, center[1] + text_offset),
        color=utils.BGR_RED,
        font_size_scale=1,
    )

    return img

def main():
    path_to_videos = 'D:\\physics_lab_videos\\day3\\'
    unparsed_dir_path = os.path.join(path_to_videos, 'unparsed')
    parsed_dir_path = os.path.join(path_to_videos, 'parsed')
    data_dir_path = os.path.join(path_to_videos, 'data')

    file_names_and_frame_ranges = {
        'day_3_35p_wo_take_1_26.2-26.4_.avi': ['all'],
        # 'day_3_35p_wo_take_1_29.8-35_.avi': ['all'],
        # 'day_3_35p_wo_take_1_45.avi': ['all'],
        # 'day_3_35p_wo_take_1_55.avi': [[1,300], [350,1450]],
        # 'day_3_35p_wo_take_1_65.avi': [[1,1015], [1077,1450]],
        # 'day_3_35p_wo_take_1_75.avi': [[1,278],[320,690],[750,1310]],
        # 'day_3_35p_wo_take_1_85.avi': [[50,550],[695,850],[880,1350]],
        # 'day_3_35p_wo_take_1_95.avi': ['all'],
        # 'day_3_35p_wo_take_1_100.avi': [[1,740], [918,1280]],
    }
    # file_names_and_frame_ranges = {
    #     '20percent_with_oil.avi': ['all'],
    #     '35percent_with_oil.avi': ['all'],
    #     '50percent_with_oil.avi': ['all'],
    #     '50percent_with_oil3.avi': ['all'],
    #     '50percent_with_oil_2.avi': ['all'],
    #     '85percent_with_oil.avi': ['all'],
    # }
    for path in file_names_and_frame_ranges:
        vid_path = os.path.join(unparsed_dir_path, path)
        parsed_path = os.path.join(parsed_dir_path, path.split('.')[0] + '_parsed.avi')
        data_path = os.path.join(data_dir_path, path.split('.')[0] + '_parsed.xlsx')

        print('Starting now:')
        print(vid_path)
        print(parsed_path)
        print(data_path)
        frame_ranges = file_names_and_frame_ranges[path]
        for frame_range in frame_ranges:
            if frame_range == 'all':
                frame_range_start = 0
                frame_range_end = None

                parse_video(
                    path=vid_path,
                    new_file_name=parsed_path,
                    xlsx_path=data_path,
                    frame_range_start=0,
                    frame_range_end=None,
                )
            else:
                frame_range_start = frame_range[0]
                frame_range_end = frame_range[1]
                new_file_name = '{parsed_file_name}_{start_frame}_to_{end_frame}.avi'.format(
                    parsed_file_name=parsed_path.split('.avi')[0],
                    start_frame=frame_range_start,
                    end_frame=frame_range_end,
                )
                new_xlsx_name = '{parsed_file_name}_{start_frame}_to_{end_frame}.xlsx'.format(
                    parsed_file_name=data_path.split('.xlsx')[0],
                    start_frame=frame_range_start,
                    end_frame=frame_range_end,
                )
                parse_video(
                    path=vid_path,
                    new_file_name=new_file_name,
                    xlsx_path=new_xlsx_name,
                    frame_range_start=frame_range_start,
                    frame_range_end=frame_range_end,
                )
