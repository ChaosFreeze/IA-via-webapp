from django.conf.urls import url
from . import views

app_name = 'bill'

urlpatterns = [
    url(r'^$',views.index,name='index'),
    url(r'^bill/',views.cal_amount,name='cal_amount'),
    url(r'^test/$',views.test_iterations,name='test_iterations')
]
