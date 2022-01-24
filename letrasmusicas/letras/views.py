from django.shortcuts import render, redirect
from datetime import datetime
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from .models import Letra


def index(request):
    letra_info = Letra.objects

    # paginator = Paginator(letra_info, 1)
    # page = request.GET.get('page')
    # page_letras = paginator.get_page(page)

    return render(request, 'letras/cria_letra.html', context={'letras': letra_info})


def cria_letras(request):
    if request.method == 'POST':
        nm_letra = request.POST['nome_letra']
        ds_letra = request.POST['ds_letra']
        ds_genero = request.POST['ds_genero']

        recipe = Letra.objects.create(nm_letra=nm_letra, ds_letra=ds_letra, ds_genero=ds_genero, dt_insert=datetime.now()
                                      )
        recipe.save()
        return redirect('cria_letras')
    else:
        return render(request, 'letras/cria_letra.html')
