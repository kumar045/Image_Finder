from django.db import models

# Create your models here.
class ImageRetrieval(models.Model):
    folder_path = models.CharField(max_length=140)
    number_of_image_to_search = models.IntegerField()
    number_of_result = models.IntegerField()

    

