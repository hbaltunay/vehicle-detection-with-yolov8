import cv2
import numpy as np
from PIL import Image
from typing import Callable
from supervision.draw.utils import draw_polygon
from supervision.detection.core import Detections
from supervision.draw.color import Color, ColorPalette
from supervision.detection.utils import polygon_to_mask
from supervision.utils.video import get_video_frames_generator, VideoInfo, VideoSink
from supervision.detection.tools.polygon_zone import PolygonZone, PolygonZoneAnnotator


def to_center_base(detections: Detections) -> None:
    """
    :param detections: It is the object of the results detected by the frame given to the model.
    :return: Setting the frame of the detected object to the center.
    """
    res = []
    for arr in detections.xyxy:
        x = int((arr[0] + arr[2]) // 2)
        y = int((arr[1] + arr[3]) // 2)

        c = int((arr[3] - arr[1]) // 2)

        res.append([x, y + c, x, y + c])
    detections.xyxy = np.array(res)


def detection_filter(detections: Detections, CLASS_ID: list, ID: str) -> None:
    """
    :param detections: It is the object of the results detected by the frame given to the model.
    :param CLASS_ID: List of desired classes.
    :param ID: Name of the Detections object
    :return: Removing unwanted class_ids and tracker_ids in object.
    """

    if ID == "class":
        mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
    elif ID == "tracker":
        mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
    else:
        raise Exception("Choose from 'class' or 'tracker' modes!")

    detections.xyxy = detections.xyxy[mask]
    detections.confidence = detections.confidence[mask]
    detections.class_id = detections.class_id[mask]
    detections.tracker_id = (detections.tracker_id[mask] if detections.tracker_id is not None else None)


def create_zone(
    polygons: list,
    video_info: VideoInfo
) -> tuple[list, list]:
    """
    :param polygons: Coordinates of polygonal areas.
    :param video_info: Information about the video.
    :return: List of objects.
    """

    zones = list()
    zone_annotators = list()
    for poly in polygons:
        zone = PolygonZone(polygon=poly, frame_resolution_wh=video_info.resolution_wh)
        zone_annotator = PolygonZoneAnnotator(zone=zone, color=Color.white(), thickness=2, text_thickness=2,
                                              text_scale=1)
        zones.append(zone)
        zone_annotators.append(zone_annotator)
    return zones, zone_annotators


def frame_draw(
    scene: np.ndarray,
    crop: np.ndarray,
    coords: dict[str, list],
    i: int,
    color: tuple[int, int, int]
) -> np.ndarray:
    """
    :param scene: The image to be processed.
    :param crop: Cropped image.
    :param coords: Areas to display additional vehicle images.
    :param i: Operation queue value.
    :param color: Color values for the frame of the cropped image.
    :return: Processed image.
    """

    coord = coords[str(i)]

    x1, y1, x2, y2 = coord[0], coord[1], coord[2], coord[3]

    cv2.rectangle(
        img=scene,
        pt1=(x1, y1),
        pt2=(x2, y2),
        color=color,
        thickness=8,
    )

    img = Image.fromarray(crop, "RGB")
    img = img.resize((175, 175))
    img = np.array(img)

    scene[y1:y2, x1:x2] = img

    return scene


def process_video(
    source_path: str,
    target_path: str,
    polygons: list,
    coords: dict,
    callback: Callable,
) -> None:
    """
    :param source_path: The path of the source video
    :param target_path: The path of the video to save after processing.
    :param polygons: Coordinates of polygonal areas.
    :param coords: Areas to display additional vehicle images.
    :param callback: Function to be applied to each frame.
    :return: The final video frames is saved.
    """

    source_video_info = VideoInfo.from_video_path(video_path=source_path)

    with VideoSink(target_path=target_path, video_info=source_video_info) as sink:

        zones, _ = create_zone(polygons=polygons, video_info=source_video_info)
        colorp = ColorPalette.default()

        for index, frame in enumerate(
                get_video_frames_generator(source_path=source_path)
        ):

            org_frame = frame.copy()
            frame, res_xyxy, detections = callback(frame, zones)

            for i, polygon in enumerate(polygons):

                mask = polygon_to_mask(
                    polygon=polygon,
                    resolution_wh=(source_video_info.resolution_wh[0], source_video_info.resolution_wh[1]),
                )

                if detections.xyxy.tolist():

                    control = mask[detections.xyxy[:, 1], detections.xyxy[:, 0]].astype(int)

                    idx = np.where(control == 1)[0].tolist()

                    num = 0 if i < 2 else -1

                    if idx:
                        xyxy = res_xyxy[idx]
                        xy = tuple(xyxy[xyxy.argsort(axis=0)[num][3]].astype(int))
                        cv2.rectangle(
                            img=frame,
                            pt1=xy[:2],
                            pt2=xy[2:],
                            color=(255, 255, 255),
                            thickness=4,
                        )
                        crop = org_frame[xy[1]:xy[3], xy[0]:xy[2]]
                        color = colorp.by_idx(i).as_bgr()
                        frame = frame_draw(frame, crop, coords, i, color)

                    else:
                        crop = np.ones((175, 175, 3))
                        color = colorp.by_idx(i).as_bgr()
                        frame = frame_draw(frame, crop, coords, i, color)

                else:
                    crop = np.ones((175, 175, 3))
                    color = colorp.by_idx(i).as_bgr()
                    frame = frame_draw(frame, crop, coords, i, color)

                frame = draw_polygon(
                    scene=frame,
                    polygon=polygon,
                    color=colorp.by_idx(i),
                    thickness=4,
                )

            sink.write_frame(frame=frame)
            print(index + 1, "/", source_video_info.total_frames, "Frame Completed.")
