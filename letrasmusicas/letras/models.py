from django.db import models
from datetime import datetime
from django.contrib.auth.models import User


# Create your models here.
class Letra(models.Model):
    nm_letra = models.CharField(max_length=255)
    ds_letra = models.TextField()
    ds_genero = models.CharField(max_length=255)
