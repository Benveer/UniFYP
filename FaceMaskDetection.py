from FaceDetection import *
import ResultsToDataFrame
import FaceMaskDetectionModels
from imutils.video import VideoStream

# Give the configuration and weight files for the model and load the network
# using them.
cfgDir = "cfg/yolov3-face.cfg"
weightsDir = "model-weights/yolov3-wider_16000.weights"

net = cv2.dnn.readNetFromDarknet(cfgDir, weightsDir)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def _main_image():
    wind_name = 'FYP Mask Detection'
    cv2.namedWindow(wind_name, cv2.WINDOW_NORMAL)

    for i in range(102):
        i += 1
        frame = cv2.imread(
            "TestSet/Person" + str(i) + ".jpg")

        frame = cv2.resize(frame, (960, 540))

        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                     [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(get_outputs_names(net))

        # Remove the bounding boxes with low confidence
        faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)

        print('[i] ==> # detected faces: {}'.format(len(faces)))
        print('#' * 60)


        face_mask_image_class_resultsTWO = 0

        face_mask_mouth_mask_resultsTWO = 0

        face_mask_gray_scaling_resultsTWO = 0

        face_mask_decisionTWO = 0

        for (sx, sy, sw, sh) in faces:

            roi_color = frame[sy:sy + sh, sx:sx + sw]

            face_mask_image_class_results = FaceMaskDetectionModels.image_classification(roi_color)
            face_mask_image_class_resultsTWO += face_mask_image_class_results

            face_mask_mouth_mask_results = FaceMaskDetectionModels.mouth_detection(
                roi_color)
            face_mask_mouth_mask_resultsTWO += face_mask_mouth_mask_results

            face_mask_gray_scaling_results = FaceMaskDetectionModels.gray_scaling(
                roi_color, 80)
            face_mask_gray_scaling_resultsTWO += face_mask_gray_scaling_results


            face_mask_decision = (face_mask_mouth_mask_results + face_mask_image_class_results + face_mask_gray_scaling_results) / 3


            if 0.5 <= face_mask_decision <= 1:
                cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 5)
                cv2.putText(frame, "Mask", (sx + sw, sy + sh + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GREEN, 2)
                face_mask_decision = 1
            else:
                cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 5)
                cv2.putText(frame, "No Mask", (sx + sw, sy + sh + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)
                face_mask_decision = 0


            face_mask_decisionTWO += face_mask_decision


            ResultsToDataFrame.record_to_df(i,
                                            face_mask_decisionTWO,
                                            face_mask_mouth_mask_resultsTWO,
                                            face_mask_image_class_resultsTWO,
                                            face_mask_gray_scaling_resultsTWO,

                                            )
            print("picture number:", i)


        # Save the output video to file

        cv2.imshow(wind_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('[i] ==> Interrupted by user!')
            break

    cv2.destroyAllWindows()

    print('==> All done!')
    print('***********************************************************')


if __name__ == '__main__':
    _main_image()
