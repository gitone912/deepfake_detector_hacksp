import os
import shutil
import cv2
import numpy as np
import tensorflow as tf
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from .forms import VideoUploadForm
from tensorflow.keras.preprocessing import image 
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import base64
from io import BytesIO
from PIL import Image
from django.contrib import messages

# Load the model once when the server starts
model = tf.keras.models.load_model('models/deepfake_detection_model.h5')

def FrameCapture(path):
    vidObj = cv2.VideoCapture(path)
    count = 0
    success = 1

    frames_dir = os.path.join(settings.MEDIA_ROOT, 'frames')
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    while success:
        success, img = vidObj.read()
        if not success:
            break
        if count % 20 == 0:
            cv2.imwrite(os.path.join(frames_dir, f"frame{count}.jpg"), img)
        count += 1


def evaluate_frames(directory):
    total_confidence = 0
    num_frames = 0
    results = []

    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            img_path = os.path.join(directory, filename)
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            confidence = model.predict(img_array)[0][0]
            total_confidence += confidence
            num_frames += 1

            if confidence >= 0.5:
                results.append((filename, "Fake", confidence))
            else:
                results.append((filename, "Real", confidence))

    if num_frames > 0:
        average_confidence = total_confidence / num_frames
        overall_prediction = "The video is predicted as a deepfake." if average_confidence >= 0.5 else "The video is predicted as real."
    else:
        average_confidence = 0
        overall_prediction = "No frames found."

    return results, average_confidence, overall_prediction

def upload_video(request):
    media_dir = settings.MEDIA_ROOT

    # Check if the media directory exists
    if os.path.exists(media_dir):
        # Delete the media directory and all its contents
        shutil.rmtree(media_dir)

    # Create the media directory again
    os.makedirs(media_dir)
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video = form.cleaned_data['video']
            fs = FileSystemStorage()
            video_path = fs.save(video.name, video)
            video_full_path = os.path.join(settings.MEDIA_ROOT, video_path)

            FrameCapture(video_full_path)

            frames_dir = os.path.join(settings.MEDIA_ROOT, 'frames')
            results, avg_confidence, overall_prediction = evaluate_frames(frames_dir)

            return render(request, 'results.html', {
                'results': results,
                'average_confidence': avg_confidence,
                'overall_prediction': overall_prediction
            })
    else:
        form = VideoUploadForm()
    return render(request, 'landing_page.html', {'form': form})

@csrf_exempt
def detect_images_api(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed'}, status=405)
    
    try:
        # Get JSON data from request
        data = json.loads(request.body)
        required_fields = ['image1', 'image2', 'image3', 'image4']
        
        # Check if all required images are present
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return JsonResponse({
                'error': f'Missing required image fields: {", ".join(missing_fields)}',
                'message': 'Please provide all four images as base64 encoded strings with fields: image1, image2, image3, image4'
            }, status=400)
        
        predictions = []
        total_confidence = 0
        
        # Process each image
        for field in required_fields:
            try:
                # Get base64 string and remove header if present
                base64_string = data[field]
                if ',' in base64_string:
                    base64_string = base64_string.split(',')[1]
                
                # Convert base64 to image
                img_data = base64.b64decode(base64_string)
                img = Image.open(BytesIO(img_data))
                
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize and preprocess
                img = img.resize((224, 224))
                img_array = np.array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array.astype('float32') / 255.0
                
                # Get prediction
                confidence = float(model.predict(img_array)[0][0])
                total_confidence += confidence
                
                predictions.append({
                    'field_name': field,
                    'prediction': 'fake' if confidence >= 0.5 else 'real',
                    'confidence': confidence
                })
                
            except Exception as e:
                return JsonResponse({
                    'error': f'Error processing {field}',
                    'details': str(e)
                }, status=400)
        
        average_confidence = total_confidence / 4
        overall_result = 'fake' if average_confidence >= 0.5 else 'real'
        
        return JsonResponse({
            'overall_result': overall_result,
            'average_confidence': average_confidence,
            'individual_results': predictions
        })
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def register_view(request):
    if request.method == 'POST':
        # Dummy registration - always succeeds
        messages.success(request, 'Registration successful!')
        return redirect('subscription')
    return render(request, 'register.html')

def login_view(request):
    # Simple dummy login view
    if request.method == 'POST':
        # Add your authentication logic here
        return redirect('revenue_dashboard')
    return render(request, 'app/login.html')

def revenue_dashboard(request):
    # Dummy revenue dashboard view
    context = {
        'total_revenue': 1000,  # dummy data
        'transactions': []  # dummy data
    }
    return render(request, 'app/revenue_dashboard.html')

def subscription_view(request):
    # Dummy subscription plans
    subscriptions = [
        {
            'name': 'Basic',
            'price': '$9.99/month',
            'features': ['10 detections/month', 'Basic support', 'Standard accuracy']
        },
        {
            'name': 'Pro',
            'price': '$24.99/month',
            'features': ['50 detections/month', 'Priority support', 'High accuracy', 'API access']
        },
        {
            'name': 'Enterprise',
            'price': '$99.99/month',
            'features': ['Unlimited detections', '24/7 support', 'Highest accuracy', 'API access', 'Custom integration']
        }
    ]
    return render(request, 'subscription.html', {'subscriptions': subscriptions})
