from django.urls import path
from . import views
urlpatterns = [
    path("", views.germany, name="index"),
    path("germany", views.germany, name="index"),
    path("portugal", views.portugal, name="index"),
    path("belgium", views.belgium, name="index"),
    path("netherlands", views.netherlands, name="index"),
]