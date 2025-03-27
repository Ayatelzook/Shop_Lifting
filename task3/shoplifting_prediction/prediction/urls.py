from django.urls import path
from .views import upload_video

urlpatterns = [
    path('predict/', upload_video, name='predict'),
]
