import pygame
import cv2
import dlib
import numpy as np
from skimage.metrics import structural_similarity as ssim

pygame.init()
screen = pygame.display.set_mode((800, 800))  # Increase height by 200 pixels

detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture(0)

reference_images = []
capture_images = False

def resize_image(image, width, height):
    dim = (width, height)
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

font = pygame.font.Font(None, 36)  # Create a font object

face_percentage = 0
circle_percentage = 0

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            capture_images = True

    if capture_images and len(reference_images) < 5:
        reference_images.append(resize_image(gray, 800, 600))
        capture_images = False

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        face_percentage = (w * h) / (800 * 600) * 100

        if any(ssim(resize_image(gray, 800, 600), ref_img) > 0.66 for ref_img in reference_images):
            cv2.circle(frame, (x + w//2, y + h//2), max(w, h), (0, 255, 0), 2)
            circle_percentage = (np.pi * (max(w, h) / 2) ** 2) / (800 * 600) * 100
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.rot90(frame)  # Rotate the frame 90 degrees counterclockwise
    frame = pygame.surfarray.make_surface(frame)
    screen.blit(frame, (0, 0))

    # Add a white rectangle at the bottom
    pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(0, 600, 800, 200))

    # Add text
    text_surface = font.render(f'Face (the rectangle): {face_percentage:.2f}% and Circle: {circle_percentage:.2f}%', True, (0, 0, 0))
    screen.blit(text_surface, (0, 650))

    pygame.display.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            cap.release()
            cv2.destroyAllWindows()
            exit()