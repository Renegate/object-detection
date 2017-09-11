import cv2

CLASS_META = {
    1: ('vehicle', [200, 100, 100]), # R
    2: ('pedestrian', [100, 200, 100]), # G
    3: ('cyclist', [100, 100, 200]), # B
    20: ('traffic lights', [175, 150, 0]), # Y
}

class Visualizer(object):

    def draw(self, img, compacted, show=False, wait_ms=15000, img_name='image'):

        for top_left_x, top_left_y, bot_right_x, bot_right_y, \
            cls, score in compacted:

            label, color = CLASS_META.get(cls)

            cv2.rectangle(img, (top_left_x, top_left_y), (bot_right_x, bot_right_y),
                          color, 2)
            cv2.putText(img, label, (top_left_x, top_left_y - 3),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
            cv2.putText(img, str(score), (top_left_x, bot_right_y + 15),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)

        if show == True:
            cv2.imshow(img_name, img)
            cv2.waitKey(wait_ms)
