from django.db import models

# Create your models here.\

class Song(models.Model):
    songid = models.AutoField(primary_key=True)
    songname = models.CharField(max_length=200)
    media = models.CharField(max_length=500)
    img = models.CharField(max_length=500)
    genre = models.CharField(max_length=20, default='rock')

    def __str__(self):
        return self.songname
    
