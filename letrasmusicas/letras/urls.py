# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 08:24:35 2021

@author: Andrey
"""

from django.urls import path

from . import views

urlpatterns = [
    path('', views.cria_letras, name='cria_letras'),
]
