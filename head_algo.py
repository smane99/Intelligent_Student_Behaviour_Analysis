import datetime
import cv2
import dlib
import numpy as np
from imutils import face_utils
import record


face_landmark_path = 'shape_predictor_68_face_landmarks.dat'
K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])

reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]


def get_head_pose(shape):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle



def main():
    # return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to connect to camera.")
        return
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_landmark_path)
    sum1,sum2,sum3=0,0,0
    d={'left':0,'right':0,'up':0,'down':0,'center':0}
    count = 0
    h = datetime.datetime.now()
    h=h.strftime("%M")
    h=int(h)+1
    active_count=0
    inactive_count=0
    x=0
    y=0
    row=1
    l=[]
    l1=[]
    l2=[]
    while cap.isOpened():
        count += 1
        ret, frame = cap.read()
        if ret:
            face_rects = detector(frame, 0)

            if len(face_rects) > 0:
                shape = predictor(frame, face_rects[0])
                shape = face_utils.shape_to_np(shape)

                reprojectdst, euler_angle = get_head_pose(shape)

                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

                for start, end in line_pairs:
                    cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))

                cv2.putText(frame, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)
                x=euler_angle[0, 0]
                cv2.putText(frame, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)
                y=euler_angle[1, 0]
                cv2.putText(frame, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)
                z=euler_angle[2, 0]
                if x<0 and y<0 and z>0:
                    d['up']+=1
                elif y<0 and z<0 and x>0:
                    d['left']+=1
                elif x<0 and z<0 and y>0:
                    d['right']+=1
                elif x>0 and y>0 and z>0:
                    d['down']+=1
                else:
                    d['center']+=1
            cv2.imshow("demo", frame)
            if (x>=-20 and x<=20) and (y>=-20 and y<=20):
               active_count+=1
            else:
               inactive_count+=1  
            if cv2.waitKey(1) & 0xFF == ord('q'):
                for k,v in d.items():
                    print(k)
                    print(v)
                    record.record1(l,l1,l2)
                break
        hs = datetime.datetime.now()
        h2="he is active for 1 min"
        h3="he is inactive for 1 min"

        if hs.strftime("%M") == str(h):
           print(count)
           print("hii")
           if active_count >= inactive_count:
               print("he is active for 1 min")
               l.append(h2)
           else:
               print("he is inactive for 1 min")
               l.append(h3)
           print(active_count,inactive_count)
           l1.append(active_count)
           l2.append(inactive_count)
           active_count=0
           inactive_count=0
           count=0
           h=h+1

if __name__ == '__main__':
    main()
