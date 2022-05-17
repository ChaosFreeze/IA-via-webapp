# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.utils.encoding import python_2_unicode_compatible
from django.db import models
from django.utils import timezone
# Create your models here.

@python_2_unicode_compatible
class Customer(models.Model):
    customer_id = models.AutoField(primary_key=True,default="1")
    customer_name = models.CharField(max_length=200,default="")
    customer_mob_no = models.CharField(max_length=200,default="")
    customer_email =  models.EmailField(max_length=200,default="")

    def __str__(self):
        return self.customer_name

@python_2_unicode_compatible
class Product(models.Model):
	product_id = models.AutoField(primary_key=True)
	product_name = models.CharField(max_length=200)
	product_detail = models.CharField(max_length=200)
	product_price = models.IntegerField(default=0)
	
	def __str__(self):
		return self.product_name

@python_2_unicode_compatible
class Cust_bill(models.Model):
	customer = models.ForeignKey('Customer', on_delete=models.CASCADE)
	shop_name = "APNI_DUKAAN"
	purchase_date = models.DateTimeField('purchased date')
	total_items = models.IntegerField(default=0)
	amount = models.IntegerField(default=0)
	
	def __str__(self):
		return self.amount


@python_2_unicode_compatible
class Inventory(models.Model):
	product = models.OneToOneField('Product', on_delete=models.CASCADE)
	item_present = models.IntegerField(default=0)
	
	def __str__(self):
		return str(self.item_present)