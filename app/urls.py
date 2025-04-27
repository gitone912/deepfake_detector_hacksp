from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_video, name='upload_video'),
    path('login/', views.login_view, name='login'),
    path('dashboard/revenue/', views.revenue_dashboard, name='revenue_dashboard'),
    path('api/detect', views.detect_images_api, name='detect_images_api'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
