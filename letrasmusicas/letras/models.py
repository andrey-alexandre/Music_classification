from django.db import models
from datetime import datetime
from django.contrib.auth.models import User


# Create your models here.
class Gravadora(models.Model):
    nm_gravadora = models.CharField(max_length=255)

    class Meta:
        db_table = 'gravadora'
        constraints = [
            models.UniqueConstraint(fields=['nm_gravadora'], name='unique gravadora')
        ]


class Disco(models.Model):
    nm_disco = models.CharField(max_length=255)
    id_gravadora = models.ForeignKey(Gravadora, on_delete=models.CASCADE)

    class Meta:
        db_table = 'disco'
        constraints = [
            models.UniqueConstraint(fields=['nm_disco'], name='unique disco')
        ]


class Autor(models.Model):
    nm_autor = models.CharField(max_length=255)
    ds_site = models.CharField(max_length=255)

    class Meta:
        db_table = 'autor'
        constraints = [
            models.UniqueConstraint(fields=['nm_autor'], name='unique autor')
        ]


class Letra(models.Model):
    nm_letra = models.CharField(max_length=255)
    ds_letra = models.TextField()
    ds_genero = models.CharField(max_length=255)
    dt_insert = models.DateTimeField(default=datetime.now, blank=True)
    id_usuario = models.ForeignKey(User, on_delete=models.CASCADE)

    class Meta:
        db_table = "letra"
        constraints = [
            models.UniqueConstraint(fields=['nm_letra'], name='unique letra')
        ]