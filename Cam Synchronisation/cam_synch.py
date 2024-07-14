import cv2
import numpy as np
from apriltag import apriltag

def detect_frame(video_path,bool):
    cap = cv2.VideoCapture(video_path)
    count = 0
    parallelframe = []
    value = 0

    while True:
        ret, image = cap.read()
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("Image", image_gray)
        #print(count)
        #cv2.waitKey(0)
        #if bool:
                #image = cv2.rotate(image, cv2.ROTATE_180)

        detector = apriltag("tagCustom48h12")
        
        data = detector.detect(image_gray)
        if data:
            data_dict = data[0]
            center_x, center_y = data[0]["center"]
            lb,rb,rt,lt = data[0]['lb-rb-rt-lt']
            # Calculate the coordinates for the square
            #square_points = np.array([lb, rb, rt, lt], dtype=np.int32)
            # Draw the square on the image
            
            #cv2.polylines(image, [square_points], isClosed=True, color=(255, 255, 0), thickness=5)
            #cv2.circle(image, (int(center_x),int(center_y)), 3, (255, 0, 0), -1)

            cv2.circle(image, (int(lb[0]),int(lb[1])), 3, (255, 0, 0), -1)
            cv2.circle(image, (int(rb[0]),int(rb[1])), 3, (255, 0, 0), -1)
            cv2.circle(image, (int(rt[0]),int(rt[1])), 3, (255, 0, 0), -1)
            cv2.circle(image, (int(lt[0]),int(lt[1])), 3, (255, 0, 0), -1)

            #Making a camera horizonal normal line
            x1, y1 = 0, 640  
            x2, y2 = 720, 640  

            dx, dy = x2 - x1, y2 - y1
            if bool:
                normal_vector = (-dy, -dx)
            else:
                normal_vector = (-dy, dx)
            

            normal_vector_cam = np.array(normal_vector) / np.linalg.norm(normal_vector)


            adx, ady = lb[0] -rb[0], lb[1] - rb[1]

            normal_vector_april = (ady, -adx)
            normal_vector_april =  np.array(normal_vector_april) / np.linalg.norm(normal_vector_april)

            scale_factor = 100  # Adjust scale factor as needed
            endpoint = tuple(np.round(np.array((int(center_x),int(center_y))) + -scale_factor * normal_vector_april).astype(int))
            # Draw the line representing the normal vector on the image
            color = (0, 255, 0)  # Green color in BGR format
            thickness = 2
            rb = tuple(map(int, rb))
            # Define starting point (x, y)

            cv2.line(image, (int(center_x),int(center_y)), endpoint, color, thickness)

            # Calculate endpoint (straight up parallel with the vertical axis)
            endpoint = (int(center_x), 0)  # Set y-coordinate to 0 for straight up

            # Define line color (red) and thickness
            color = (0, 0, 255)  # BGR format: (blue, green, red)
            thickness = 1  # Adjust thickness as needed

            # Draw the dashed line on the image
            cv2.line(image, (int(center_x), int(center_y)), endpoint, color, thickness, cv2.LINE_AA)

            dot_product = np.dot(normal_vector_cam, normal_vector_april)
            if  bool:
                value = 870
            if dot_product  > 0.99:
                return count, image

                

            

        # intel 387
        if 0xFF == ord('q'):
            break
        
        #cv2.imshow("Frame", image)
        #cv2.waitKey(0)
      
        count +=1



if __name__ == "__main__":
   detect_frame("SjakkLitenIRTobii.mp4", False)
    
