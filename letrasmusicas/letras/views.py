from django.shortcuts import render
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from .models import Letra


def index(request):
    letra_info = Letra.objects.order_by('-insert_date').filter(published=True)

    paginator = Paginator(letra_info, 1)
    page = request.GET.get('page')
    page_letras = paginator.get_page(page)

    return render(request, 'letras/index.html', context={'letras': page_letras})
