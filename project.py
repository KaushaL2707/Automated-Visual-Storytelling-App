import streamlit as st
import ultralytics 
from ultralytics import YOLO
import cv2
import tempfile
import time
from langchain_community.llms import Ollama

st.title("Automated Visual Storytelling")

st.write("""
Welcome to the Automated Visual Storytelling App! You can upload a photo, video, or use your webcam to detect objects and ganerate story.
""")

llm = Ollama(model="llama3")
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
MAX_VIDEO_FILE_SIZE = 200 * 1024 * 1024  # 200MB

class ObjTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.detected_object_names = set()
        self.stop_signal = False

    def reset(self):
        self.detected_object_names.clear()
        self.stop_signal = False

    def detect_objects_in_image(self, image):
        results = self.model.track(image, persist=True)
        stframe = st.empty()

        for result in results:
            for detection in result.boxes:
                class_id = detection.cls.item()
                if class_id not in self.detected_object_names:
                    self.detected_object_names.add(class_id)
                    class_name = self.model.names[class_id]
                    st.write(f"Detected object class: {class_name}")

        annotated_image = result.plot()
        stframe.image(annotated_image, channels="BGR", use_column_width=True)

    def track_objects_in_video(self, video_path, time_limit=None):
        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()
        start_time = time.time()

        while cap.isOpened() and not self.stop_signal:
            success, frame = cap.read()
            if success:
                results = self.model.track(frame, persist=True)

                for result in results:
                    for detection in result.boxes:
                        class_id = detection.cls.item()
                        if class_id not in self.detected_object_names:
                            self.detected_object_names.add(class_id)
                            class_name = self.model.names[class_id]
                            st.write(f"Detected object class: {class_name}")

                annotated_frame = result.plot()
                stframe.image(annotated_frame, channels="BGR", use_column_width=True)

                if time_limit and (time.time() - start_time) > time_limit:
                    self.stop_signal = True

                time.sleep(0.01)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()

    def story_generator(self, story_category, detected_object_names):
        query = f"Tell me a story involving these objects: {', '.join(detected_object_names)} in the context of {story_category}. Ensure the story is engaging and does not repeat the names or use the same plot. For educational contexts, focus on the knowledge and learning aspects related to the objects rather than discussing teachers or classroom settings."
        response = llm.invoke(query)
        st.write("Story:", response)


option = st.selectbox('Select input type:', ('Photo', 'Video', 'Webcam'))

story_category = st.selectbox('Select story base:', ('Education', 'Entertainment', 'Digital Marketing', 'Assistive Technologies'))

if option == 'Photo':
    uploaded_files_photo = st.file_uploader("Choose a photo...", type=["jpeg", "jpg", "png"], accept_multiple_files=True)
    
    if uploaded_files_photo:
        valid_files = []
        all_detected_objects = set()  # Create a set to store unique object names
        
        for uploaded_file_photo in uploaded_files_photo:
            if uploaded_file_photo.size > MAX_FILE_SIZE:
                st.error(f"The file {uploaded_file_photo.name} is too large. Please upload an image smaller than 5MB.")
            else:
                valid_files.append(uploaded_file_photo)
        
        if valid_files:
            if st.button('Start'):
                for uploaded_file_photo in valid_files:
                    with tempfile.NamedTemporaryFile(delete=False) as tfile:
                        tfile.write(uploaded_file_photo.read())
                        image_path = tfile.name

                        image = cv2.imread(image_path)
                        tracker = ObjTracker(model_path="yolov8n.pt")
                        tracker.detect_objects_in_image(image)
                        all_detected_objects.update(tracker.detected_object_names)  # Add objects from each image
                
                tracker.story_generator(story_category, [tracker.model.names[i] for i in all_detected_objects])

elif option == 'Video':
    uploaded_file_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    
    if uploaded_file_video is not None:  
        if uploaded_file_video.size > MAX_VIDEO_FILE_SIZE:
            st.error("The uploaded video is too large. Please upload a video smaller than 200MB.")
        else:
            if st.button('Start'):
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file_video.read())
                video_path = tfile.name

                tracker = ObjTracker(model_path="yolov8n.pt")

                stop_button = st.button('Stop Video')
            
                if stop_button:
                    tracker.stop_signal = True
                    video_path = None
                    tracker.reset()
                    st.write("Detected Objects:")
                    
                if video_path:
                   tracker.track_objects_in_video(video_path)
                   tracker.story_generator(story_category, [tracker.model.names[i] for i in tracker.detected_object_names])
                
elif option == 'Webcam':
    time_limit = st.number_input('Set time limit for webcam (seconds):', min_value=10, max_value=600, value=60)
    
    if st.button('Start'):
        tracker = ObjTracker(model_path="yolov8n.pt")

        stop_button = st.button('Stop Webcam')

        if stop_button:
           tracker.stop_signal = True
           tracker.reset()

        tracker.track_objects_in_video(0, time_limit)  

#        st.write("Detected object names:", list(tracker.detected_object_names))
        tracker.story_generator(story_category, [tracker.model.names[i] for i in tracker.detected_object_names])
