from django.urls import path
from .views import predict_view

urlpatterns = [
    path('iris-cls-api', predict_view, name='predict'),
]
