from django.contrib import admin
from django.urls import path,include
from . import views

urlpatterns =[
path('',views.home,name='home'),
path('playsong',views.playSong,name='playSong'),
path('mood',views.mood,name='mood'),
path('mood/facerecog',views.faceRecog,name='faceRecog'),
path('mood/voicerecog',views.voiceRecog,name='voiceRecog'),
path('extract',views.extractFeatures,name='extractFeatures'),
]